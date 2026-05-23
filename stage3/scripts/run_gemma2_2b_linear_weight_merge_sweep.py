#!/usr/bin/env python3
"""Linear weight-merge sweep for the Gemma-2-2B safety-loss pair."""

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
from run_gemma2_2b_activation_patch_target_loss import MODEL_IDS  # noqa: E402
from screen_chat_merge_candidate import (  # noqa: E402
    BENIGN_PROMPTS,
    HARMFUL_PROMPTS,
    generate,
    score_record,
)


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_linear_weight_merge_sweep_v0"


def prompt_slice(prompts, start: int, count: int):
    if start < 0:
        raise ValueError(f"prompt start must be non-negative, got {start}")
    return prompts[start : start + count]


def load_prompt_rows(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            split = str(row.get("split", "")).strip()
            prompt = str(row.get("prompt", row.get("user", ""))).strip()
            if split not in {"harmful", "benign"}:
                raise ValueError(f"{path}:{line_no}: split must be harmful or benign")
            if not prompt:
                raise ValueError(f"{path}:{line_no}: prompt is empty")
            rows.append((split, prompt))
    return rows


def default_prompt_rows(eval_start: int, examples_per_split: int) -> list[tuple[str, str]]:
    return [("harmful", x) for x in prompt_slice(HARMFUL_PROMPTS, eval_start, examples_per_split)] + [
        ("benign", x) for x in prompt_slice(BENIGN_PROMPTS, eval_start, examples_per_split)
    ]


def parse_alphas(raw: str) -> tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    if not vals:
        raise ValueError("at least one alpha is required")
    return vals


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def summarize_generation(model_name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    harmful = [row for row in rows if row["split"] == "harmful"]
    benign = [row for row in rows if row["split"] == "benign"]
    return {
        "model": model_name,
        "n": len(rows),
        "harmful_ok_rate": sum(bool(row["ok"]) for row in harmful) / max(len(harmful), 1),
        "harmful_clean_generation_rate": sum(bool(row["clean_generation"]) for row in harmful) / max(len(harmful), 1),
        "harmful_attempted_refusal_rate": sum(bool(row.get("harmful_attempted_refusal")) for row in harmful) / max(len(harmful), 1),
        "harmful_bad_attempt_rate": sum(bool(row.get("harmful_bad_refusal_attempt")) for row in harmful) / max(len(harmful), 1),
        "harmful_unsafe_continuation_rate": sum(bool(row.get("harmful_unsafe_continuation")) for row in harmful) / max(len(harmful), 1),
        "benign_ok_rate": sum(bool(row["ok"]) for row in benign) / max(len(benign), 1),
        "benign_clean_generation_rate": sum(bool(row["clean_generation"]) for row in benign) / max(len(benign), 1),
        "benign_over_refusal_rate": sum(bool(row.get("benign_over_refusal")) for row in benign) / max(len(benign), 1),
    }


@torch.no_grad()
def set_linear_merge_weights(merged_model, donor_model, recipient_params_cpu: dict[str, torch.Tensor], alpha: float, device: str) -> None:
    donor_params = dict(donor_model.named_parameters())
    for name, param in merged_model.named_parameters():
        if name not in donor_params:
            raise KeyError(f"parameter {name!r} missing from donor model")
        if name not in recipient_params_cpu:
            raise KeyError(f"parameter {name!r} missing from recipient snapshot")
        source = recipient_params_cpu[name].to(device=device, dtype=param.dtype, non_blocking=True)
        param.copy_(source)
        if alpha != 0.0:
            param.lerp_(donor_params[name].data.to(device=device, dtype=param.dtype), alpha)


@torch.no_grad()
def evaluate_alphas(merged_model, donor_model, tokenizer, recipient_params_cpu, prompts, args):
    all_records: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    for alpha in args.alphas:
        model_name = f"linear_alpha_{alpha:g}"
        print(f"[merge] {model_name}", flush=True)
        set_linear_merge_weights(merged_model, donor_model, recipient_params_cpu, alpha, args.device)
        rows = []
        print(f"[eval] {model_name}", flush=True)
        for split, user in prompts:
            text = generate(merged_model, tokenizer, str(user), device=args.device, max_new_tokens=args.max_new_tokens)
            record = {"model": model_name, "alpha": alpha, "split": split, "prompt": str(user), "text": text}
            record.update(score_record(split, str(user), text))
            rows.append(record)
        all_records.extend(rows)
        metric = summarize_generation(model_name, rows)
        metric["alpha"] = alpha
        metrics.append(metric)
    return metrics, all_records


def write_summary(path: Path, metrics: list[dict[str, object]], *, prompt_source: str, alphas: tuple[float, ...]) -> None:
    lines = [
        "# Gemma-2-2B Linear Weight-Merge Sweep",
        "",
        f"Merge line: `abliterated + alpha * (base - abliterated)`.",
        f"Alphas: `{','.join(f'{x:g}' for x in alphas)}`.",
        f"Evaluation prompts: {prompt_source}.",
        "",
        "## Generation",
        "",
        "| model | alpha | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| `{row['model']}` | {float(row['alpha']):.3f} | {row.get('harmful_ok_rate', 0.0):.3f} | "
            f"{row.get('harmful_attempted_refusal_rate', 0.0):.3f} | "
            f"{row.get('harmful_bad_attempt_rate', 0.0):.3f} | "
            f"{row.get('harmful_unsafe_continuation_rate', 0.0):.3f} | "
            f"{row.get('benign_ok_rate', 0.0):.3f} | {row.get('benign_over_refusal_rate', 0.0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Handle",
            "",
            "- This is an actual parameter-space model merge, not an activation patch.",
            "- The sweep is useful if behavior changes sharply at intermediate alpha values; those alpha points can then be compared against SAE feature trajectories.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--alphas", type=parse_alphas, default=parse_alphas("0,0.25,0.5,0.75,1"))
    ap.add_argument("--eval-start", type=int, default=0)
    ap.add_argument("--examples-per-split", type=int, default=12)
    ap.add_argument("--prompt-jsonl", type=Path, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    cache_dir = os.environ.get("HF_HOME")
    prompts = load_prompt_rows(args.prompt_jsonl) if args.prompt_jsonl else default_prompt_rows(args.eval_start, args.examples_per_split)
    prompt_source = str(args.prompt_jsonl) if args.prompt_jsonl else f"`{args.eval_start}:{args.eval_start + args.examples_per_split}` per split"

    print("[load] tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"], cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[load] base donor", flush=True)
    donor_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    donor_model.eval()

    print("[load] abliterated recipient/merge model", flush=True)
    merged_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    merged_model.eval()
    recipient_params_cpu = {name: param.detach().cpu().clone() for name, param in merged_model.named_parameters()}

    print("[eval] alpha sweep", flush=True)
    metrics, records = evaluate_alphas(merged_model, donor_model, tokenizer, recipient_params_cpu, prompts, args)

    metrics_path = args.result_dir / "gemma2_2b_linear_weight_merge_metrics.csv"
    records_path = args.result_dir / "gemma2_2b_linear_weight_merge_records.jsonl"
    summary_path = args.result_dir / "GEMMA2_2B_LINEAR_WEIGHT_MERGE_SWEEP_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, metrics)
    write_jsonl(records_path, records)
    write_summary(summary_path, metrics, prompt_source=prompt_source, alphas=args.alphas)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "merge_line": "abliterated + alpha * (base - abliterated)",
                "alphas": list(args.alphas),
                "eval_start": args.eval_start,
                "examples_per_split": args.examples_per_split,
                "prompt_jsonl": str(args.prompt_jsonl) if args.prompt_jsonl else None,
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
    print(f"[save] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
