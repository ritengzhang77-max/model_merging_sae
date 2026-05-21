#!/usr/bin/env python3
"""Alpha search for the SmolLM2 refusal-v2 task vector.

This tests whether the balanced v2 refusal expert is useful after stronger
task-vector insertion into the arithmetic+polite recipient:

    theta = theta_merge_arith_polite + alpha * (theta_refusal_v2 - theta_base)

The evaluation reuses the v2 harmful/benign audit instead of the old keyword-only
refusal score.
"""

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
import run_smollm2_synthetic_experts as smol  # noqa: E402
from run_smollm2_refusal_v2 import (  # noqa: E402
    RESULT_DIR as V2_RESULT_DIR,
    eval_arith_or_polite,
    eval_refusal_v2,
    load_old_expert,
    load_saved_state,
    write_csv,
    write_jsonl,
)


DEFAULT_ARTIFACT_DIR = ROOT / "stage0" / "artifacts" / "smollm2_refusal_v2_balanced"
DEFAULT_RESULT_DIR = ROOT / "stage0" / "results" / "refusal_v2_alpha_search"


def alpha_label(alpha: float) -> str:
    return f"{alpha:g}".replace("-", "m").replace(".", "p")


def add_refusal_delta(base, recipient, donor, *, alpha: float, label: str):
    out = {}
    for i, (k, recipient_v) in enumerate(recipient.items(), start=1):
        if i == 1 or i % 60 == 0:
            print(f"  [{label}] tensor {i}/{len(recipient)}", flush=True)
        if torch.is_floating_point(recipient_v):
            patched = recipient_v.float() + alpha * (donor[k].float() - base[k].float())
            out[k] = patched.half()
        else:
            out[k] = recipient_v.detach().cpu().clone()
    return out


def load_model_with_state(model, state, device: str) -> None:
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()


@torch.no_grad()
def evaluate_states(model_states, tokenizer, args):
    rows = []
    records = []
    runtime = AutoModelForCausalLM.from_pretrained(smol.BASE_ID, torch_dtype=torch.float16).to(args.device)
    for model_name, state in model_states.items():
        print(f"[eval] {model_name}", flush=True)
        load_model_with_state(runtime, state, args.device)
        arith_acc, arith_records = eval_arith_or_polite(runtime, tokenizer, "arith", args)
        polite_acc, polite_records = eval_arith_or_polite(runtime, tokenizer, "polite", args)
        refusal_metrics, refusal_records = eval_refusal_v2(runtime, tokenizer, args)
        for rec in arith_records + polite_records + refusal_records:
            records.append({"model": model_name, **rec})
        row = {
            "model": model_name,
            "alpha": args.model_alphas.get(model_name, ""),
            "acc_arith": arith_acc,
            "acc_polite": polite_acc,
            **refusal_metrics,
        }
        row["acc_mean_core"] = float(np.mean([arith_acc, polite_acc, refusal_metrics["refusal_v2_score"]]))
        row["acc_worst_core"] = float(np.min([arith_acc, polite_acc, refusal_metrics["refusal_v2_score"]]))
        rows.append(row)
    del runtime
    torch.cuda.empty_cache()
    return rows, records


def write_summary(path: Path, rows, args) -> None:
    sorted_rows = sorted(rows, key=lambda r: (-float(r["harmful_clean_refusal"]), float(r.get("benign_over_refusal", 0.0)), -float(r["acc_mean_core"])))
    lines = [
        "# SmolLM2 Refusal V2 Alpha Search",
        "",
        "Formula:",
        "",
        "```text",
        "theta = theta_merge_arith_polite + alpha * (theta_refusal_v2 - theta_base)",
        "```",
        "",
        f"- refusal artifact dir: `{args.v2_artifact_dir}`",
        f"- eval examples per split: `{args.eval_examples}`",
        f"- max new tokens: `{args.max_new_tokens}`",
        "",
        "## Metrics",
        "",
        "| model | alpha | arith | polite | harmful clean | harmful attempt | harmful bad | benign helpful | benign over-refusal | core mean | worst |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted_rows:
        lines.append(
            f"| `{row['model']}` | {row['alpha']} | {row['acc_arith']:.3f} | {row['acc_polite']:.3f} | "
            f"{row.get('harmful_clean_refusal', 0.0):.3f} | {row.get('harmful_attempted_refusal', 0.0):.3f} | "
            f"{row.get('harmful_bad_attempt', 0.0):.3f} | {row.get('benign_helpful', 0.0):.3f} | "
            f"{row.get('benign_over_refusal', 0.0):.3f} | {row['acc_mean_core']:.3f} | "
            f"{row['acc_worst_core']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "A useful v2 refusal target should raise harmful clean refusal above the recipient while keeping benign over-refusal low.",
            "If higher alpha only creates bad attempts, artifacts, or unsafe continuations, the v2 expert is still not a clean mechanistic target.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-examples", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=28)
    ap.add_argument("--alphas", default="0,0.1,0.2,0.33,0.5,0.75,1.0,1.25")
    ap.add_argument("--v2-artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    ap.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    ap.add_argument("--require-safe-redirect", action="store_true", default=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    alphas = tuple(float(x) for x in args.alphas.split(",") if x)

    tokenizer = AutoTokenizer.from_pretrained(smol.TOKENIZER_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[load] base/expert states", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(smol.BASE_ID, torch_dtype=torch.float16, device_map="cpu")
    base_state = {k: v.detach().cpu().half() for k, v in base_model.state_dict().items()}
    del base_model
    expert_arith = load_old_expert("arith")
    expert_polite = load_old_expert("polite")
    expert_refusal_v2 = load_saved_state(args.v2_artifact_dir / "expert_refusal_v2.pt")
    recipient = smol.merge_states(base_state, [expert_arith, expert_polite], label="merge_arith_polite")

    model_states = {
        "merge_arith_polite": recipient,
        "expert_refusal_v2": expert_refusal_v2,
    }
    model_alphas = {"merge_arith_polite": "", "expert_refusal_v2": "expert"}
    for alpha in alphas:
        name = f"alpha_refusal_v2_full_a{alpha_label(alpha)}"
        print(f"[build] {name}", flush=True)
        model_states[name] = add_refusal_delta(
            base_state,
            recipient,
            expert_refusal_v2,
            alpha=alpha,
            label=name,
        )
        model_alphas[name] = alpha
    args.model_alphas = model_alphas

    rows, records = evaluate_states(model_states, tokenizer, args)
    rows = sorted(rows, key=lambda r: (-float(r["acc_mean_core"]), str(r["model"])))

    metrics_path = args.result_dir / "smollm2_refusal_v2_alpha_metrics.csv"
    records_path = args.result_dir / "smollm2_refusal_v2_alpha_generations.jsonl"
    summary_path = args.result_dir / "SMOLLM2_REFUSAL_V2_ALPHA_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, rows)
    write_jsonl(records_path, records)
    write_summary(summary_path, rows, args)
    manifest_path.write_text(
        json.dumps(
            {
                "formula": "merge_arith_polite + alpha * (expert_refusal_v2 - base)",
                "base_model": smol.BASE_ID,
                "tokenizer": smol.TOKENIZER_ID,
                "v2_artifact_dir": str(args.v2_artifact_dir),
                "alphas": alphas,
                "eval_examples": args.eval_examples,
                "max_new_tokens": args.max_new_tokens,
                "outputs": {
                    "metrics": str(metrics_path),
                    "generations": str(records_path),
                    "summary": str(summary_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[save] {metrics_path}")
    print(f"[save] {records_path}")
    print(f"[save] {summary_path}")
    print(f"[save] {manifest_path}")
    print("\nTop rows by core mean:")
    for row in rows[:12]:
        print(
            f"  {row['model']:<34} alpha={row['alpha']} mean={row['acc_mean_core']:.3f} "
            f"harm_clean={row.get('harmful_clean_refusal', 0.0):.3f} "
            f"attempt={row.get('harmful_attempted_refusal', 0.0):.3f} "
            f"benign_help={row.get('benign_helpful', 0.0):.3f} "
            f"over_refuse={row.get('benign_over_refusal', 0.0):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
