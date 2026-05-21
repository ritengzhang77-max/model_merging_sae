#!/usr/bin/env python3
"""Stage 2 sufficiency screen: patch expert modules into base/merges."""

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
    load_expert_state,
    load_model_with_state,
    load_state_cache,
    make_task_loader,
    merge_states,
    task_loss_and_reps,
    write_csv,
)
from analyze_smollm2_module_inheritance import patch_module  # noqa: E402


RESULT_DIR = ROOT / "stage2" / "results"
DEFAULT_LAYERS = (15, 29)
DEFAULT_MODULES = ("attn", "mlp", "block")
RECIPIENTS = (
    "base",
    "merge_all_linear",
    "merge_arith_polite",
    "merge_arith_refusal",
    "merge_polite_refusal",
)


def merge_contains_task(merge_name: str, task: str) -> bool:
    if merge_name == "base":
        return False
    return task in MERGE_SPECS.get(merge_name, ())


def top_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    df = pd.DataFrame(rows)
    tops = []
    for (task, recipient), group in df.groupby(["task", "recipient_model"], sort=True):
        best = group.sort_values("loss_improvement", ascending=False).iloc[0]
        tops.append(
            {
                "task": task,
                "recipient_model": recipient,
                "recipient_contains_task": bool(best["recipient_contains_task"]),
                "best_layer": int(best["layer"]),
                "best_module": best["module"],
                "recipient_original_loss": float(best["recipient_original_loss"]),
                "patched_loss": float(best["patched_loss"]),
                "expert_loss": float(best["expert_loss"]),
                "loss_improvement": float(best["loss_improvement"]),
                "gap_closed_to_expert": float(best["gap_closed_to_expert"]),
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
    ap.add_argument("--recipients", default=",".join(RECIPIENTS))
    ap.add_argument("--state-cache-dir", type=Path, default=CACHE_DIR / "smollm2_state_cache")
    ap.add_argument("--no-state-cache", action="store_true")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    args = ap.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    layers = tuple(int(x) for x in args.layers.split(",") if x)
    modules = tuple(x for x in args.modules.split(",") if x)
    recipients = tuple(x for x in args.recipients.split(",") if x)

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

    for recipient in recipients:
        if recipient not in states:
            raise ValueError(f"unknown recipient model: {recipient}")

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

    print("[eval] original losses", flush=True)
    original_losses = {}
    for name in [*recipients, *(f"expert_{task}" for task in TASKS)]:
        task_names = TASKS if name in recipients else (name.replace("expert_", ""),)
        print(f"  [eval] {name}: {','.join(task_names)}", flush=True)
        load_model_with_state(runtime, states[name], args.device)
        for task in task_names:
            loss, _ = task_loss_and_reps(runtime, loaders[task], device=args.device, rep_layers=())
            original_losses[(name, task)] = loss

    print("[sufficiency] expert-module patches into recipients", flush=True)
    rows = []
    for task in TASKS:
        donor_name = f"expert_{task}"
        expert_loss = original_losses[(donor_name, task)]
        for recipient in recipients:
            recipient_loss = original_losses[(recipient, task)]
            for layer in layers:
                for module in modules:
                    print(
                        f"  [patch] donor={donor_name} recipient={recipient} "
                        f"task={task} layer={layer} module={module}",
                        flush=True,
                    )
                    patched = patch_module(states[recipient], states[donor_name], layer, module)
                    load_model_with_state(runtime, patched, args.device)
                    patched_loss, _ = task_loss_and_reps(runtime, loaders[task], device=args.device, rep_layers=())
                    denom = recipient_loss - expert_loss
                    rows.append(
                        {
                            "task": task,
                            "donor_expert": donor_name,
                            "recipient_model": recipient,
                            "recipient_contains_task": merge_contains_task(recipient, task),
                            "layer": layer,
                            "module": module,
                            "recipient_original_loss": recipient_loss,
                            "expert_loss": expert_loss,
                            "patched_loss": patched_loss,
                            "loss_improvement": recipient_loss - patched_loss,
                            "gap_closed_to_expert": (
                                (recipient_loss - patched_loss) / denom if denom != 0 else float("nan")
                            ),
                        }
                    )
                    del patched

    del runtime
    torch.cuda.empty_cache()

    tops = top_rows(rows)
    ablation_path = args.result_dir / "smollm2_module_sufficiency_patches.csv"
    top_path = args.result_dir / "smollm2_module_sufficiency_top_patches.csv"
    summary_path = args.result_dir / "smollm2_module_sufficiency_summary.json"
    write_csv(ablation_path, rows)
    write_csv(top_path, tops)
    summary_path.write_text(
        json.dumps(
            {
                "patch_rows": len(rows),
                "top_rows": len(tops),
                "layers": layers,
                "modules": modules,
                "recipients": recipients,
                "outputs": {
                    "patches": str(ablation_path),
                    "top_patches": str(top_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {ablation_path}")
    print(f"[save] {top_path}")
    print(f"[save] {summary_path}")
    print("\nBest expert-module patch per task/recipient:")
    for row in tops:
        print(
            f"  {row['task']:<7} {row['recipient_model']:<21} "
            f"{row['best_layer']}/{row['best_module']:<5} "
            f"improve={row['loss_improvement']:+.3f} "
            f"gap={row['gap_closed_to_expert']:+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
