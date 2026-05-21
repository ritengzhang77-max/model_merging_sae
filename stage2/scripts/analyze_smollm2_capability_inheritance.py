#!/usr/bin/env python3
"""Stage 2 first pass: source-expert to merge capability inheritance.

For each synthetic task, replace selected transformer layers in an expert or
merge with the base model's corresponding layer. If the same layers are causal
in the expert and the merge, their loss-damage profiles should correlate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/gavin/model_merging/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
sys.path.insert(0, str(ROOT / "stage1" / "scripts"))
import run_smollm2_synthetic_experts as smol  # noqa: E402
from analyze_smollm2_rq0 import (  # noqa: E402
    CACHE_DIR,
    EXPERT_TO_TASK,
    MERGE_SPECS,
    TASKS,
    load_expert_state,
    load_model_with_state,
    load_state_cache,
    make_task_loader,
    merge_states,
    patch_layer,
    task_loss_and_reps,
    write_csv,
)


RESULT_DIR = ROOT / "stage2" / "results"
DEFAULT_LAYERS = (0, 5, 10, 15, 20, 25, 29)
TASK_MERGE_TARGETS = {
    "arith": ("merge_all_linear", "merge_arith_polite", "merge_arith_refusal"),
    "polite": ("merge_all_linear", "merge_arith_polite", "merge_polite_refusal"),
    "refusal": ("merge_all_linear", "merge_arith_refusal", "merge_polite_refusal"),
}


def base_patch_layer(target, base, layer: int):
    return patch_layer(target, base, layer)


def profile_corr_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    df = pd.DataFrame(rows)
    out = []
    for task in TASKS:
        expert_name = f"expert_{task}"
        expert_profile = df[(df["task"] == task) & (df["target_model"] == expert_name)]
        if expert_profile.empty:
            continue
        for merge_name in TASK_MERGE_TARGETS[task]:
            merge_profile = df[(df["task"] == task) & (df["target_model"] == merge_name)]
            joined = expert_profile[["layer", "loss_increase"]].merge(
                merge_profile[["layer", "loss_increase"]],
                on="layer",
                suffixes=("_expert", "_merge"),
            )
            if len(joined) < 3:
                continue
            joined_no_l0 = joined[joined["layer"] != 0]
            out.append(
                {
                    "task": task,
                    "expert_model": expert_name,
                    "merge_model": merge_name,
                    "n_layers": len(joined),
                    "pearson": joined["loss_increase_expert"].corr(joined["loss_increase_merge"], method="pearson"),
                    "spearman": joined["loss_increase_expert"].corr(joined["loss_increase_merge"], method="spearman"),
                    "n_layers_no_l0": len(joined_no_l0),
                    "pearson_no_l0": (
                        joined_no_l0["loss_increase_expert"].corr(
                            joined_no_l0["loss_increase_merge"],
                            method="pearson",
                        )
                        if len(joined_no_l0) >= 3
                        else np.nan
                    ),
                    "spearman_no_l0": (
                        joined_no_l0["loss_increase_expert"].corr(
                            joined_no_l0["loss_increase_merge"],
                            method="spearman",
                        )
                        if len(joined_no_l0) >= 3
                        else np.nan
                    ),
                    "expert_top_layer": int(
                        joined.sort_values("loss_increase_expert", ascending=False).iloc[0]["layer"]
                    ),
                    "merge_top_layer": int(
                        joined.sort_values("loss_increase_merge", ascending=False).iloc[0]["layer"]
                    ),
                }
            )
    return sorted(out, key=lambda r: (r["task"], r["merge_model"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-examples", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    ap.add_argument("--state-cache-dir", type=Path, default=CACHE_DIR / "smollm2_state_cache")
    ap.add_argument("--no-state-cache", action="store_true")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    args = ap.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    layers = tuple(int(x) for x in args.layers.split(",") if x)

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

    print("[ablate] base-layer replacements", flush=True)
    rows = []
    for target_name in target_models:
        task_names = [
            task
            for task in TASKS
            if target_name == f"expert_{task}" or target_name in TASK_MERGE_TARGETS[task]
        ]
        for layer in layers:
            print(f"  [ablate] {target_name} layer={layer}", flush=True)
            patched = base_patch_layer(states[target_name], base_state, layer)
            load_model_with_state(runtime, patched, args.device)
            for task in task_names:
                patched_loss, _ = task_loss_and_reps(runtime, loaders[task], device=args.device, rep_layers=())
                original_loss = original_losses[(target_name, task)]
                rows.append(
                    {
                        "target_model": target_name,
                        "task": task,
                        "source_expert": f"expert_{task}",
                        "layer": layer,
                        "original_loss": original_loss,
                        "base_layer_patched_loss": patched_loss,
                        "loss_increase": patched_loss - original_loss,
                    }
                )
            del patched

    del runtime
    torch.cuda.empty_cache()

    corr_rows = profile_corr_rows(rows)
    layer_path = args.result_dir / "smollm2_capability_inheritance_layer_ablation.csv"
    corr_path = args.result_dir / "smollm2_capability_inheritance_profile_correlations.csv"
    summary_path = args.result_dir / "smollm2_capability_inheritance_summary.json"
    write_csv(layer_path, rows)
    write_csv(corr_path, corr_rows)
    summary_path.write_text(
        json.dumps(
            {
                "layer_ablation_rows": len(rows),
                "profile_correlation_rows": len(corr_rows),
                "layers": layers,
                "outputs": {
                    "layer_ablation": str(layer_path),
                    "profile_correlations": str(corr_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {layer_path}")
    print(f"[save] {corr_path}")
    print(f"[save] {summary_path}")
    print("\nExpert-vs-merge base-ablation profile correlations:")
    for row in corr_rows:
        print(
            f"  {row['task']:<7} {row['merge_model']:<21} "
            f"spearman={row['spearman']:+.3f} "
            f"spearman_no_l0={row['spearman_no_l0']:+.3f} "
            f"top_layers={row['expert_top_layer']}->{row['merge_top_layer']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
