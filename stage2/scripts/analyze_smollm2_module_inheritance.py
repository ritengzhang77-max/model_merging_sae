#!/usr/bin/env python3
"""Stage 2 module-level inheritance screen for SmolLM2 synthetic merges."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/gavin/model_merging/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
sys.path.insert(0, str(ROOT / "stage1" / "scripts"))
sys.path.insert(0, str(ROOT / "stage2" / "scripts"))
import run_smollm2_synthetic_experts as smol  # noqa: E402
from analyze_smollm2_rq0 import (  # noqa: E402
    CACHE_DIR,
    MERGE_SPECS,
    TASKS,
    layer_keys,
    load_expert_state,
    load_model_with_state,
    load_state_cache,
    make_task_loader,
    merge_states,
    task_loss_and_reps,
    write_csv,
)
from analyze_smollm2_capability_inheritance import TASK_MERGE_TARGETS  # noqa: E402


RESULT_DIR = ROOT / "stage2" / "results"
DEFAULT_LAYERS = (15, 29)
DEFAULT_MODULES = ("attn", "mlp", "norms", "block")


def module_keys(state: dict[str, torch.Tensor], layer: int, module: str) -> list[str]:
    prefix = f"model.layers.{layer}."
    if module == "block":
        return layer_keys(state, layer)
    if module == "attn":
        return [k for k in state if k.startswith(prefix + "self_attn.")]
    if module == "mlp":
        return [k for k in state if k.startswith(prefix + "mlp.")]
    if module == "norms":
        return [
            k
            for k in state
            if k in {prefix + "input_layernorm.weight", prefix + "post_attention_layernorm.weight"}
        ]
    raise ValueError(f"unknown module: {module}")


def patch_module(target, donor, layer: int, module: str):
    out = {k: v.detach().cpu().clone() for k, v in target.items()}
    keys = module_keys(out, layer, module)
    if not keys:
        raise ValueError(f"no keys found for layer={layer} module={module}")
    for k in keys:
        out[k] = donor[k].detach().cpu().clone()
    return out


def profile_corr_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    df = pd.DataFrame(rows)
    out = []
    for task in TASKS:
        expert_name = f"expert_{task}"
        expert_profile = df[(df["task"] == task) & (df["target_model"] == expert_name)]
        for merge_name in TASK_MERGE_TARGETS[task]:
            merge_profile = df[(df["task"] == task) & (df["target_model"] == merge_name)]
            joined = expert_profile[["layer", "module", "loss_increase"]].merge(
                merge_profile[["layer", "module", "loss_increase"]],
                on=["layer", "module"],
                suffixes=("_expert", "_merge"),
            )
            if len(joined) < 3:
                continue
            out.append(
                {
                    "task": task,
                    "expert_model": expert_name,
                    "merge_model": merge_name,
                    "n_interventions": len(joined),
                    "pearson": joined["loss_increase_expert"].corr(joined["loss_increase_merge"], method="pearson"),
                    "spearman": joined["loss_increase_expert"].corr(joined["loss_increase_merge"], method="spearman"),
                    "expert_top_intervention": "/".join(
                        str(x)
                        for x in joined.sort_values("loss_increase_expert", ascending=False)
                        .iloc[0][["layer", "module"]]
                        .tolist()
                    ),
                    "merge_top_intervention": "/".join(
                        str(x)
                        for x in joined.sort_values("loss_increase_merge", ascending=False)
                        .iloc[0][["layer", "module"]]
                        .tolist()
                    ),
                }
            )
    return sorted(out, key=lambda r: (r["task"], r["merge_model"]))


def top_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    df = pd.DataFrame(rows)
    tops = []
    for (target_model, task), group in df.groupby(["target_model", "task"], sort=True):
        row = group.sort_values("loss_increase", ascending=False).iloc[0]
        tops.append(
            {
                "target_model": target_model,
                "task": task,
                "top_layer": int(row["layer"]),
                "top_module": row["module"],
                "loss_increase": float(row["loss_increase"]),
            }
        )
    return tops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-examples", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    ap.add_argument("--modules", default=",".join(DEFAULT_MODULES))
    ap.add_argument("--state-cache-dir", type=Path, default=CACHE_DIR / "smollm2_state_cache")
    ap.add_argument("--no-state-cache", action="store_true")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    args = ap.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    layers = tuple(int(x) for x in args.layers.split(",") if x)
    modules = tuple(x for x in args.modules.split(",") if x)

    print("[load] tokenizer/base/experts", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(smol.TOKENIZER_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model_cpu = AutoModelForCausalLM.from_pretrained(
        smol.BASE_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    base_state = {k: v.detach().cpu().half() for k, v in base_model_cpu.state_dict().items()}
    del base_model_cpu
    experts = {task: load_expert_state(task) for task in TASKS}

    cached_merge_states = {} if args.no_state_cache else load_state_cache(args.state_cache_dir)
    states = {"base": base_state}
    states.update({f"expert_{k}": v for k, v in experts.items()})
    for merge_name, expert_names in MERGE_SPECS.items():
        if merge_name in cached_merge_states:
            states[merge_name] = cached_merge_states[merge_name]
        else:
            states[merge_name] = merge_states(
                base_state,
                [experts[name] for name in expert_names],
                label=merge_name,
            )

    loaders = {
        task: make_task_loader(
            tokenizer,
            task,
            n=args.eval_examples,
            seed=args.seed,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        for task in TASKS
    }

    runtime = AutoModelForCausalLM.from_pretrained(smol.BASE_ID, torch_dtype=torch.float16).to(args.device)
    target_models = sorted(
        {f"expert_{task}" for task in TASKS}
        | {merge_name for merge_names in TASK_MERGE_TARGETS.values() for merge_name in merge_names}
    )

    print("[eval] original task losses", flush=True)
    original_losses = {}
    for target_name in target_models:
        task_names = [
            task
            for task in TASKS
            if target_name == f"expert_{task}" or target_name in TASK_MERGE_TARGETS[task]
        ]
        print(f"  [eval] {target_name}: {','.join(task_names)}", flush=True)
        load_model_with_state(runtime, states[target_name], args.device)
        for task in task_names:
            loss, _ = task_loss_and_reps(runtime, loaders[task], device=args.device, rep_layers=())
            original_losses[(target_name, task)] = loss

    print("[ablate] base-module replacements", flush=True)
    rows = []
    for target_name in target_models:
        task_names = [
            task
            for task in TASKS
            if target_name == f"expert_{task}" or target_name in TASK_MERGE_TARGETS[task]
        ]
        for layer in layers:
            for module in modules:
                print(f"  [ablate] {target_name} layer={layer} module={module}", flush=True)
                patched = patch_module(states[target_name], base_state, layer, module)
                load_model_with_state(runtime, patched, args.device)
                patched_keys = module_keys(base_state, layer, module)
                for task in task_names:
                    patched_loss, _ = task_loss_and_reps(runtime, loaders[task], device=args.device, rep_layers=())
                    original_loss = original_losses[(target_name, task)]
                    rows.append(
                        {
                            "target_model": target_name,
                            "task": task,
                            "source_expert": f"expert_{task}",
                            "layer": layer,
                            "module": module,
                            "n_patched_tensors": len(patched_keys),
                            "original_loss": original_loss,
                            "base_module_patched_loss": patched_loss,
                            "loss_increase": patched_loss - original_loss,
                        }
                    )
                del patched

    del runtime
    torch.cuda.empty_cache()

    corr_rows = profile_corr_rows(rows)
    tops = top_rows(rows)
    ablation_path = args.result_dir / "smollm2_module_inheritance_ablation.csv"
    corr_path = args.result_dir / "smollm2_module_inheritance_profile_correlations.csv"
    top_path = args.result_dir / "smollm2_module_inheritance_top_interventions.csv"
    summary_path = args.result_dir / "smollm2_module_inheritance_summary.json"
    write_csv(ablation_path, rows)
    write_csv(corr_path, corr_rows)
    write_csv(top_path, tops)
    summary_path.write_text(
        json.dumps(
            {
                "ablation_rows": len(rows),
                "profile_correlation_rows": len(corr_rows),
                "top_rows": len(tops),
                "layers": layers,
                "modules": modules,
                "outputs": {
                    "ablation": str(ablation_path),
                    "profile_correlations": str(corr_path),
                    "top_interventions": str(top_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {ablation_path}")
    print(f"[save] {corr_path}")
    print(f"[save] {top_path}")
    print(f"[save] {summary_path}")
    print("\nExpert-vs-merge module profile correlations:")
    for row in corr_rows:
        print(
            f"  {row['task']:<7} {row['merge_model']:<21} "
            f"spearman={row['spearman']:+.3f} "
            f"top={row['expert_top_intervention']}->{row['merge_top_intervention']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
