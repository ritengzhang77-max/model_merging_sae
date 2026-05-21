#!/usr/bin/env python3
"""Mechanism-aware micro-merge smoke test for SmolLM2 synthetic experts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/gavin/model_merging/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
sys.path.insert(0, str(ROOT / "stage1" / "scripts"))
sys.path.insert(0, str(ROOT / "stage2" / "scripts"))
import run_smollm2_synthetic_experts as smol  # noqa: E402
from analyze_smollm2_module_inheritance import module_keys, patch_module  # noqa: E402
from analyze_smollm2_rq0 import (  # noqa: E402
    CACHE_DIR,
    TASKS,
    load_expert_state,
    load_model_with_state,
    load_state_cache,
    make_task_loader,
    task_loss_and_reps,
    write_csv,
)


RESULT_DIR = ROOT / "stage2" / "results"


def make_micro_merge(recipient, donor, *, layer: int, module: str):
    return patch_module(recipient, donor, layer, module)


def make_multi_module_micro_merge(recipient, donor, interventions: tuple[tuple[int, str], ...]):
    out = {k: v.detach().cpu().clone() for k, v in recipient.items()}
    for layer, module in interventions:
        for k in module_keys(out, layer, module):
            out[k] = donor[k].detach().cpu().clone()
    return out


def add_micro(model_states, name: str, recipient, donor, interventions: tuple[tuple[int, str], ...]):
    model_states[name] = make_multi_module_micro_merge(recipient, donor, interventions)


def task_loss_table(model_states, tokenizer, args):
    loaders = {
        task: make_task_loader(
            tokenizer,
            task,
            n=args.loss_examples,
            seed=args.seed,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        for task in TASKS
    }
    runtime = AutoModelForCausalLM.from_pretrained(smol.BASE_ID, torch_dtype=torch.float16).to(args.device)
    rows = []
    for model_name, state in model_states.items():
        print(f"[loss] {model_name}", flush=True)
        load_model_with_state(runtime, state, args.device)
        task_losses = {}
        for task in TASKS:
            loss, _ = task_loss_and_reps(runtime, loaders[task], device=args.device, rep_layers=())
            task_losses[task] = loss
        rows.append(
            {
                "model": model_name,
                "loss_arith": task_losses["arith"],
                "loss_polite": task_losses["polite"],
                "loss_refusal": task_losses["refusal"],
                "loss_mean": float(np.mean(list(task_losses.values()))),
                "loss_worst": float(np.max(list(task_losses.values()))),
            }
        )
    del runtime
    torch.cuda.empty_cache()
    return rows


def add_loss_deltas(rows, reference: str):
    by_name = {row["model"]: row for row in rows}
    ref = by_name[reference]
    out = []
    for row in rows:
        new = dict(row)
        for task in TASKS:
            new[f"loss_delta_vs_{reference}_{task}"] = ref[f"loss_{task}"] - row[f"loss_{task}"]
        new[f"loss_delta_vs_{reference}_mean"] = ref["loss_mean"] - row["loss_mean"]
        out.append(new)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loss-examples", type=int, default=16)
    ap.add_argument("--gen-examples", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=160)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--state-cache-dir", type=Path, default=CACHE_DIR / "smollm2_state_cache")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    args = ap.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)

    print("[load] tokenizer/base/experts/cached merges", flush=True)
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
    merges = load_state_cache(args.state_cache_dir)

    if "merge_arith_polite" not in merges or "merge_all_linear" not in merges:
        raise RuntimeError("state cache must contain merge_arith_polite and merge_all_linear")

    model_states = {
        "base": base_state,
        "merge_arith_polite": merges["merge_arith_polite"],
        "merge_all_linear": merges["merge_all_linear"],
        "merge_arith_refusal": merges["merge_arith_refusal"],
        "expert_refusal": experts["refusal"],
    }
    recipient = merges["merge_arith_polite"]
    donor = experts["refusal"]
    add_micro(model_states, "micro_refusal_l29_mlp", recipient, donor, ((29, "mlp"),))
    add_micro(model_states, "micro_refusal_l29_block", recipient, donor, ((29, "block"),))
    add_micro(model_states, "micro_refusal_l25_l29_mlp", recipient, donor, ((25, "mlp"), (29, "mlp")))
    add_micro(
        model_states,
        "micro_refusal_l20_l25_l29_mlp",
        recipient,
        donor,
        ((20, "mlp"), (25, "mlp"), (29, "mlp")),
    )
    add_micro(
        model_states,
        "micro_refusal_l15_l20_l25_l29_mlp",
        recipient,
        donor,
        ((15, "mlp"), (20, "mlp"), (25, "mlp"), (29, "mlp")),
    )
    add_micro(
        model_states,
        "micro_refusal_l20_l25_l29_attn",
        recipient,
        donor,
        ((20, "attn"), (25, "attn"), (29, "attn")),
    )
    add_micro(
        model_states,
        "micro_refusal_l15_l20_l25_l29_attn",
        recipient,
        donor,
        ((15, "attn"), (20, "attn"), (25, "attn"), (29, "attn")),
    )
    add_micro(
        model_states,
        "micro_refusal_l20_l25_l29_mlp_attn",
        recipient,
        donor,
        (
            (20, "attn"),
            (20, "mlp"),
            (25, "attn"),
            (25, "mlp"),
            (29, "attn"),
            (29, "mlp"),
        ),
    )
    add_micro(
        model_states,
        "micro_refusal_l15_l20_l25_l29_mlp_attn",
        recipient,
        donor,
        (
            (15, "attn"),
            (15, "mlp"),
            (20, "attn"),
            (20, "mlp"),
            (25, "attn"),
            (25, "mlp"),
            (29, "attn"),
            (29, "mlp"),
        ),
    )
    add_micro(model_states, "micro_refusal_l25_l29_block", recipient, donor, ((25, "block"), (29, "block")))

    print("[eval] teacher-forced losses", flush=True)
    loss_rows = task_loss_table(model_states, tokenizer, args)
    loss_rows = add_loss_deltas(loss_rows, "merge_arith_polite")

    print("[eval] generation metrics", flush=True)
    gen_args = argparse.Namespace(
        device=args.device,
        eval_examples=args.gen_examples,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )
    metric_rows, generation_records = smol.evaluate_models(model_states, tokenizer, gen_args)
    metric_rows = sorted(metric_rows, key=lambda r: (-r["acc_mean"], r["model"]))

    loss_path = args.result_dir / "smollm2_micro_merge_losses.csv"
    metrics_path = args.result_dir / "smollm2_micro_merge_generation_metrics.csv"
    gens_path = args.result_dir / "smollm2_micro_merge_generations.jsonl"
    summary_path = args.result_dir / "smollm2_micro_merge_summary.json"
    write_csv(loss_path, sorted(loss_rows, key=lambda r: r["loss_mean"]))
    write_csv(metrics_path, metric_rows)
    with gens_path.open("w", encoding="utf-8") as f:
        for rec in generation_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps(
            {
                "micro_merges": {
                    "micro_refusal_l29_mlp": {
                        "recipient": "merge_arith_polite",
                        "donor": "expert_refusal",
                        "layer": 29,
                        "module": "mlp",
                    },
                    "micro_refusal_l29_block": {
                        "recipient": "merge_arith_polite",
                        "donor": "expert_refusal",
                        "layer": 29,
                        "module": "block",
                    },
                    "micro_refusal_l25_l29_mlp": {
                        "recipient": "merge_arith_polite",
                        "donor": "expert_refusal",
                        "interventions": [[25, "mlp"], [29, "mlp"]],
                    },
                    "micro_refusal_l20_l25_l29_mlp": {
                        "recipient": "merge_arith_polite",
                        "donor": "expert_refusal",
                        "interventions": [[20, "mlp"], [25, "mlp"], [29, "mlp"]],
                    },
                    "micro_refusal_l25_l29_block": {
                        "recipient": "merge_arith_polite",
                        "donor": "expert_refusal",
                        "interventions": [[25, "block"], [29, "block"]],
                    },
                    "broader_grid_note": (
                        "Script also evaluates four-layer MLP, attention-only, "
                        "and MLP+attention targeted refusal patches; see model names in CSV."
                    ),
                },
                "loss_examples": args.loss_examples,
                "generation_examples": args.gen_examples,
                "outputs": {
                    "losses": str(loss_path),
                    "generation_metrics": str(metrics_path),
                    "generations": str(gens_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {loss_path}")
    print(f"[save] {metrics_path}")
    print(f"[save] {gens_path}")
    print(f"[save] {summary_path}")
    print("\nGeneration metrics:")
    for row in metric_rows:
        print(
            f"  {row['model']:<26} mean={row['acc_mean']:.3f} worst={row['acc_worst']:.3f} "
            f"arith={row['acc_arith']:.3f} polite={row['acc_polite']:.3f} refusal={row['acc_refusal']:.3f}"
        )
    print("\nLosses:")
    for row in sorted(loss_rows, key=lambda r: r["loss_mean"]):
        print(
            f"  {row['model']:<26} mean={row['loss_mean']:.3f} "
            f"arith={row['loss_arith']:.3f} polite={row['loss_polite']:.3f} refusal={row['loss_refusal']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
