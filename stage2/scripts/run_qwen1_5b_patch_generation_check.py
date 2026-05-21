#!/usr/bin/env python3
"""Generation check for top Qwen2.5-1.5B base-to-abliterated module patches."""

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
from analyze_qwen1_5b_refusal_rq1_rq2 import MODEL_IDS, copy_keys_to_model, module_keys  # noqa: E402
from screen_chat_merge_candidate import BENIGN_PROMPTS, HARMFUL_PROMPTS, generate, score_record  # noqa: E402


DEFAULT_PATCH_SPECS = ("14:mlp", "14:block", "16:mlp", "16:block")
RESULT_DIR = ROOT / "stage2" / "results" / "qwen1_5b_patch_generation_check"


def parse_specs(raw: str) -> list[tuple[tuple[int, ...], str, str]]:
    specs = []
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        layer_text, module = item.split(":", 1)
        layers = tuple(int(part) for part in layer_text.split("+") if part)
        specs.append((layers, module, item))
    return specs


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


def summarize(model_name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    out = {"model": model_name, "n": len(rows)}
    for split in ("harmful", "benign"):
        group = [row for row in rows if row["split"] == split]
        if not group:
            continue
        out[f"{split}_ok_rate"] = sum(bool(row["ok"]) for row in group) / len(group)
        out[f"{split}_clean_generation_rate"] = sum(bool(row["clean_generation"]) for row in group) / len(group)
    harmful = [row for row in rows if row["split"] == "harmful"]
    benign = [row for row in rows if row["split"] == "benign"]
    if harmful:
        out["harmful_attempted_refusal_rate"] = sum(bool(row.get("harmful_attempted_refusal")) for row in harmful) / len(harmful)
        out["harmful_bad_attempt_rate"] = sum(bool(row.get("harmful_bad_refusal_attempt")) for row in harmful) / len(harmful)
    if benign:
        out["benign_over_refusal_rate"] = sum(bool(row.get("benign_over_refusal")) for row in benign) / len(benign)
    return out


@torch.no_grad()
def evaluate_variant(model, tokenizer, variant_name: str, args) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = []
    prompts = []
    prompts.extend(("harmful", prompt, "") for prompt in HARMFUL_PROMPTS[: args.examples_per_split])
    prompts.extend(("benign", prompt, "") for prompt in BENIGN_PROMPTS[: args.examples_per_split])
    for split, user, expected in prompts:
        text = generate(model, tokenizer, user, device=args.device, max_new_tokens=args.max_new_tokens)
        scores = score_record(split, user, text, expected)
        rows.append(
            {
                "model": variant_name,
                "split": split,
                "user": user,
                "expected": expected,
                "generation": text,
                **scores,
            }
        )
    return summarize(variant_name, rows), rows


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Qwen2.5-1.5B Patch Generation Check",
        "",
        "Small generation audit for top base-to-abliterated static module patches.",
        "",
        "| model | harmful clean | harmful attempt | harmful bad | benign helpful | benign over-refusal |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['model']}` | {row.get('harmful_ok_rate', 0.0):.3f} | "
            f"{row.get('harmful_attempted_refusal_rate', 0.0):.3f} | "
            f"{row.get('harmful_bad_attempt_rate', 0.0):.3f} | "
            f"{row.get('benign_ok_rate', 0.0):.3f} | {row.get('benign_over_refusal_rate', 0.0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Current Decision",
            "",
            "- If target-loss repairs do not improve harmful generation, use them as localization signals only.",
            "- If a patch improves harmful refusal without benign over-refusal, promote it to a deeper causal patching pass.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--examples-per-split", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--patch-specs", default=",".join(DEFAULT_PATCH_SPECS))
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    ap.add_argument("--local-files-only", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    patch_specs = parse_specs(args.patch_specs)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"], local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[load] states", flush=True)
    base_cpu = AutoModelForCausalLM.from_pretrained(
        MODEL_IDS["base"],
        torch_dtype=torch.float16,
        local_files_only=args.local_files_only,
        device_map="cpu",
    )
    base_state = {k: v.detach().cpu().half() for k, v in base_cpu.state_dict().items()}
    del base_cpu
    recipient_cpu = AutoModelForCausalLM.from_pretrained(
        MODEL_IDS["abliterated"],
        torch_dtype=torch.float16,
        local_files_only=args.local_files_only,
        device_map="cpu",
    )
    recipient_state = {k: v.detach().cpu().half() for k, v in recipient_cpu.state_dict().items()}
    del recipient_cpu

    runtime = AutoModelForCausalLM.from_pretrained(
        MODEL_IDS["base"],
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        local_files_only=args.local_files_only,
    ).to(args.device)
    runtime.eval()

    summary_rows = []
    record_rows = []

    for variant, state in (("base", base_state), ("abliterated", recipient_state)):
        print(f"[eval] {variant}", flush=True)
        runtime.load_state_dict(state, strict=True)
        summary, records = evaluate_variant(runtime, tokenizer, variant, args)
        summary_rows.append(summary)
        record_rows.extend(records)

    for layers, module, spec in patch_specs:
        print(f"[eval] patch {spec}", flush=True)
        runtime.load_state_dict(recipient_state, strict=True)
        keys = []
        for layer in layers:
            keys.extend(module_keys(recipient_state, layer, module))
        copy_keys_to_model(runtime, base_state, keys)
        summary, records = evaluate_variant(runtime, tokenizer, f"patch_{spec}", args)
        summary_rows.append(summary)
        record_rows.extend(records)

    metrics_path = args.result_dir / "qwen1_5b_patch_generation_metrics.csv"
    records_path = args.result_dir / "qwen1_5b_patch_generation_records.jsonl"
    summary_path = args.result_dir / "QWEN1_5B_PATCH_GENERATION_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, summary_rows)
    write_jsonl(records_path, record_rows)
    write_summary(summary_path, summary_rows)
    manifest_path.write_text(
        json.dumps(
            {
                "patch_specs": [spec for _layers, _module, spec in patch_specs],
                "models": {"base": MODEL_IDS["base"], "recipient": MODEL_IDS["abliterated"]},
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
