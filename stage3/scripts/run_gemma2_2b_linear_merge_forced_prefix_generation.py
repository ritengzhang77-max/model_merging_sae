#!/usr/bin/env python3
"""Generate from a linear-merge model after forcing assistant prefix text."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HOME", "/data/gavin/model_merging/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
sys.path.insert(0, str(ROOT / "stage2" / "scripts"))
sys.path.insert(0, str(ROOT / "stage3" / "scripts"))
from run_gemma2_2b_activation_patch_target_loss import MODEL_IDS  # noqa: E402
from run_gemma2_2b_gemmascope_mlp_sae_feature_subsets import (  # noqa: E402
    chat_prompt,
    clean_assistant_text,
    score_record,
    summarize_generation,
)
from run_gemma2_2b_linear_merge_sae_bundle_patch import load_prompt_rows, write_csv, write_jsonl  # noqa: E402
from run_gemma2_2b_linear_weight_merge_sweep import set_linear_merge_weights  # noqa: E402


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_linear_merge_forced_prefix_v0"


def parse_prefixes(raw: str) -> list[str]:
    return [item for item in raw.split("|||")]


def label_prefix(prefix: str) -> str:
    label = prefix.replace("\\n", "_").replace(" ", "_").replace("'", "")
    label = "".join(ch.lower() if ch.isalnum() else "_" for ch in label)
    return label.strip("_") or "empty"


@torch.no_grad()
def generate_forced(model, tokenizer, user: str, forced_prefix: str, args) -> str:
    prompt = chat_prompt(tokenizer, user)
    full_prompt = prompt + forced_prefix
    enc = tokenizer(full_prompt, return_tensors="pt").to(args.device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    for _ in range(args.max_new_tokens):
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_id)], dim=1)
        if int(next_id.item()) == tokenizer.eos_token_id:
            break
    decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    return clean_assistant_text(tokenizer, prompt, decoded)


def write_summary(path: Path, metrics, args) -> None:
    lines = [
        "# Gemma-2-2B Linear Merge Forced-Prefix Generation",
        "",
        f"Alpha: `{args.alpha:g}`.",
        f"Prompt file: `{args.prompt_jsonl}`.",
        f"Forced prefixes: `{args.forced_prefixes}`.",
        "",
        "| model | harmful clean | harmful attempt | unsafe | benign helpful | benign over-refusal |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| `{row['model']}` | {row.get('harmful_ok_rate', 0.0):.3f} | "
            f"{row.get('harmful_attempted_refusal_rate', 0.0):.3f} | "
            f"{row.get('harmful_unsafe_continuation_rate', 0.0):.3f} | "
            f"{row.get('benign_ok_rate', 0.0):.3f} | {row.get('benign_over_refusal_rate', 0.0):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--prompt-jsonl", type=Path, required=True)
    ap.add_argument("--forced-prefixes", default="I|||It")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    cache_dir = os.environ.get("HF_HOME")
    prompts = load_prompt_rows(args.prompt_jsonl)
    prefixes = parse_prefixes(args.forced_prefixes)

    print("[load] tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"], cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[load] base donor for weight construction", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    base_model.eval()
    print("[load] abliterated endpoint", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    model.eval()
    params_cpu = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
    set_linear_merge_weights(model, base_model, params_cpu, args.alpha, args.device)
    del base_model
    torch.cuda.empty_cache()

    by_model: dict[str, list[dict[str, object]]] = {}
    for forced_prefix in prefixes:
        model_name = f"alpha{args.alpha:g}_forced_{label_prefix(forced_prefix)}"
        rows = []
        print(f"[eval] {model_name}", flush=True)
        for row_idx, (split, user) in enumerate(prompts, start=1):
            print(f"[eval] {model_name} prompt {row_idx}/{len(prompts)} ({split})", flush=True)
            text = generate_forced(model, tokenizer, str(user), forced_prefix, args)
            record = {
                "model": model_name,
                "split": split,
                "prompt": str(user),
                "forced_prefix": forced_prefix,
                "text": text,
                "alpha": args.alpha,
                "patch_mode": "forced_assistant_prefix",
            }
            record.update(score_record(split, str(user), text))
            rows.append(record)
        by_model[model_name] = rows

    metrics = []
    for model_name, rows in by_model.items():
        row = summarize_generation(model_name, rows)
        row["alpha"] = args.alpha
        row["patch_mode"] = "forced_assistant_prefix"
        metrics.append(row)

    metrics_path = args.result_dir / "gemma2_2b_linear_merge_forced_prefix_metrics.csv"
    records_path = args.result_dir / "gemma2_2b_linear_merge_forced_prefix_records.jsonl"
    summary_path = args.result_dir / "GEMMA2_2B_LINEAR_MERGE_FORCED_PREFIX_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, metrics)
    write_jsonl(records_path, [row for rows in by_model.values() for row in rows])
    write_summary(summary_path, metrics, args)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "alpha": args.alpha,
                "prompt_jsonl": str(args.prompt_jsonl),
                "forced_prefixes": prefixes,
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
