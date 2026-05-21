#!/usr/bin/env python3
"""Focused math screen for model-merge candidate triage."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/gavin/model_merging/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from screen_chat_merge_candidate import generate, general_quality


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODELS = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "bunnycore/Qwen2.5-1.5B-Matrix",
    "Yuuta208/Qwen2.5-1.5B-Instruct-Qwen2.5-Math-1.5B-Merged-slerp-24",
    "nbeerbower/EVA-abliterated-TIES-Qwen2.5-1.5B",
)

MATH_PROMPTS = (
    ("Answer with just the final number: A store has 12 apples, buys 7 more, then sells 5. How many apples remain?", ("14",)),
    ("Answer with just the final number: If 3 notebooks cost 12 dollars, how much do 5 notebooks cost at the same rate?", ("20",)),
    ("Answer with just the final number: Solve 2x + 7 = 19.", ("6",)),
    ("Answer with just the final number: What is 17 * 24?", ("408",)),
    ("Answer with just the final number: A train travels 60 miles per hour for 2.5 hours. How many miles does it travel?", ("150",)),
    ("Answer with just the final value: What is 3/4 + 1/8?", ("7/8", "0.875")),
    ("Answer with just the final number: What is 15 percent of 80?", ("12",)),
    ("Answer with just the final number: A rectangle is 9 units by 6 units. What is its area?", ("54",)),
    ("Answer with just the final number: If y/4 = 9, what is y?", ("36",)),
    ("Answer with just the final number: There are 5 boxes with 18 pencils each. How many pencils are there?", ("90",)),
)


def parse_models(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_MODELS)
    return [item.strip() for item in raw.split(",") if item.strip()]


def normalize(text: str) -> str:
    return text.strip().lower().replace(",", "")


def answer_ok(generation: str, expected_values: tuple[str, ...]) -> bool:
    low = normalize(generation)
    first_line = low.splitlines()[0] if low else ""
    for expected in expected_values:
        exp = normalize(expected)
        if first_line == exp or first_line.startswith(exp + "."):
            return True
    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", low)
    if not numbers:
        return False
    return any(normalize(expected) in numbers for expected in expected_values)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def summarize(model_id: str, rows: list[dict[str, object]]) -> dict[str, object]:
    n = len(rows)
    return {
        "model": model_id,
        "n": n,
        "clean_generation_rate": sum(bool(row["clean_generation"]) for row in rows) / max(n, 1),
        "math_ok_rate": sum(bool(row["ok"]) for row in rows) / max(n, 1),
        "mean_words": sum(float(row["word_count"]) for row in rows) / max(n, 1),
    }


def write_summary(path: Path, summary_rows: list[dict[str, object]], args) -> None:
    lines = [
        "# Math Reasoning Candidate Screen",
        "",
        "Focused cheap screen for whether a math merge gives a visible task gain.",
        "",
        f"- max new tokens: `{args.max_new_tokens}`",
        "",
        "| model | clean gen | math ok | mean words |",
        "|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['model']}` | {row['clean_generation_rate']:.3f} | "
            f"{row['math_ok_rate']:.3f} | {row['mean_words']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Current Decision",
            "",
            "- Prefer a merge only if it improves or preserves math while retaining safety in the chat screen.",
            "- This is a triage screen, not a benchmark.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument(
        "--result-dir",
        type=Path,
        default=ROOT / "stage0" / "results" / "candidate_screens" / "qwen1_5b_math_reasoning",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    models = parse_models(args.models)
    all_rows = []
    summary_rows = []
    for model_id in models:
        print(f"[load] {model_id}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(args.device)
        model.eval()
        rows = []
        for i, (prompt, expected_values) in enumerate(MATH_PROMPTS):
            generation = generate(model, tokenizer, prompt, device=args.device, max_new_tokens=args.max_new_tokens)
            quality = general_quality(generation)
            row = {
                "model": model_id,
                "prompt_id": i,
                "user": prompt,
                "expected": "|".join(expected_values),
                "generation": generation,
                **quality,
                "ok": bool(quality["clean_generation"] and answer_ok(generation, expected_values)),
            }
            rows.append(row)
            all_rows.append(row)
        summary = summarize(model_id, rows)
        summary_rows.append(summary)
        print(f"[done] {model_id} math_ok={summary['math_ok_rate']:.3f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_path = args.result_dir / "math_reasoning_metrics.csv"
    generations_path = args.result_dir / "math_reasoning_generations.jsonl"
    summary_path = args.result_dir / "MATH_REASONING_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, summary_rows)
    write_jsonl(generations_path, all_rows)
    write_summary(summary_path, summary_rows, args)
    manifest_path.write_text(
        json.dumps(
            {
                "models": models,
                "n_prompts": len(MATH_PROMPTS),
                "outputs": {
                    "metrics": str(metrics_path),
                    "generations": str(generations_path),
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
