#!/usr/bin/env python3
"""Fast baseline generation for the Qwen stress prompt set."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/gavin/model_merging/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
sys.path.insert(0, str(ROOT / "stage2" / "scripts"))
from analyze_qwen1_5b_refusal_rq1_rq2 import MODEL_IDS  # noqa: E402
from run_qwen1_5b_pca_residual_patch_generation import (  # noqa: E402
    STRESS_BENIGN_PROMPTS,
    STRESS_HARMFUL_PROMPTS,
)
from screen_chat_merge_candidate import generate, score_record  # noqa: E402


RESULT_DIR = ROOT / "stage2" / "results" / "qwen1_5b_stress_baseline_generation"


def summarize(model_name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    out = {"model": model_name, "n": len(rows)}
    for split in ("harmful", "benign"):
        group = [row for row in rows if row["split"] == split]
        out[f"{split}_ok_rate"] = sum(bool(row["ok"]) for row in group) / max(len(group), 1)
        out[f"{split}_clean_generation_rate"] = sum(bool(row["clean_generation"]) for row in group) / max(len(group), 1)
    harmful = [row for row in rows if row["split"] == "harmful"]
    benign = [row for row in rows if row["split"] == "benign"]
    out["harmful_attempted_refusal_rate"] = sum(bool(row.get("harmful_attempted_refusal")) for row in harmful) / max(
        len(harmful), 1
    )
    out["harmful_bad_attempt_rate"] = sum(bool(row.get("harmful_bad_refusal_attempt")) for row in harmful) / max(
        len(harmful), 1
    )
    out["harmful_unsafe_continuation_rate"] = sum(
        bool(row.get("harmful_unsafe_continuation")) for row in harmful
    ) / max(len(harmful), 1)
    out["benign_over_refusal_rate"] = sum(bool(row.get("benign_over_refusal")) for row in benign) / max(len(benign), 1)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Qwen2.5-1.5B Stress Baseline Generation",
        "",
        "| model | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['model']}` | {row.get('harmful_ok_rate', 0.0):.3f} | "
            f"{row.get('harmful_attempted_refusal_rate', 0.0):.3f} | "
            f"{row.get('harmful_bad_attempt_rate', 0.0):.3f} | "
            f"{row.get('harmful_unsafe_continuation_rate', 0.0):.3f} | "
            f"{row.get('benign_ok_rate', 0.0):.3f} | {row.get('benign_over_refusal_rate', 0.0):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models", default="base,abliterated")
    ap.add_argument("--prompt-mode", choices=("stress", "stress_harmful_only"), default="stress")
    ap.add_argument("--examples-per-split", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [("harmful", prompt, "") for prompt in STRESS_HARMFUL_PROMPTS[: args.examples_per_split]]
    if args.prompt_mode == "stress":
        prompts.extend(("benign", prompt, "") for prompt in STRESS_BENIGN_PROMPTS[: args.examples_per_split])

    all_records: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for model_name in [item.strip() for item in args.models.split(",") if item.strip()]:
        print(f"[load] {model_name}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_IDS[model_name], torch_dtype=torch.float16).to(args.device)
        model.eval()
        records = []
        for split, user, expected in prompts:
            text = generate(model, tokenizer, user, device=args.device, max_new_tokens=args.max_new_tokens)
            scores = score_record(split, user, text, expected)
            record = {
                "model": model_name,
                "split": split,
                "user": user,
                "expected": expected,
                "generation": text,
                **scores,
            }
            records.append(record)
            all_records.append(record)
        summary_rows.append(summarize(model_name, records))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_path = args.result_dir / "qwen1_5b_stress_baseline_generation_metrics.csv"
    records_path = args.result_dir / "qwen1_5b_stress_baseline_generation_records.jsonl"
    summary_path = args.result_dir / "QWEN1_5B_STRESS_BASELINE_GENERATION_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, summary_rows)
    write_jsonl(records_path, all_records)
    write_summary(summary_path, summary_rows)
    manifest_path.write_text(
        json.dumps(
            {
                "models": [item.strip() for item in args.models.split(",") if item.strip()],
                "prompt_mode": args.prompt_mode,
                "examples_per_split": args.examples_per_split,
                "outputs": {
                    "metrics": str(metrics_path),
                    "records": str(records_path),
                    "summary": str(summary_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

