#!/usr/bin/env python3
"""Small dynamic activation-patch generation check for Qwen2.5-1.5B."""

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
from run_qwen1_5b_activation_patch_target_loss import install_hooks, remove_hooks, parse_specs  # noqa: E402
from screen_chat_merge_candidate import (  # noqa: E402
    BENIGN_PROMPTS,
    HARMFUL_PROMPTS,
    clean_assistant_text,
    generate,
    score_record,
)


RESULT_DIR = ROOT / "stage2" / "results" / "qwen1_5b_dynamic_activation_patch_generation"


def chat_prompt(tokenizer, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def activation_patch_generate(
    donor,
    recipient,
    tokenizer,
    user: str,
    *,
    device: str,
    max_new_tokens: int,
    patch_points: tuple[tuple[int, str], ...],
    position: str,
) -> str:
    prompt = chat_prompt(tokenizer, user)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    cache, donor_handles, recipient_handles = install_hooks(donor, recipient, patch_points, position)
    try:
        for _ in range(max_new_tokens):
            cache.clear()
            _ = donor(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            out = recipient(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_id)], dim=1)
            if int(next_id.item()) == tokenizer.eos_token_id:
                break
    finally:
        remove_hooks(donor_handles)
        remove_hooks(recipient_handles)
    decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    return clean_assistant_text(tokenizer, prompt, decoded)


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
        out[f"{split}_ok_rate"] = sum(bool(row["ok"]) for row in group) / max(len(group), 1)
        out[f"{split}_clean_generation_rate"] = sum(bool(row["clean_generation"]) for row in group) / max(len(group), 1)
    harmful = [row for row in rows if row["split"] == "harmful"]
    benign = [row for row in rows if row["split"] == "benign"]
    out["harmful_attempted_refusal_rate"] = sum(bool(row.get("harmful_attempted_refusal")) for row in harmful) / max(len(harmful), 1)
    out["harmful_bad_attempt_rate"] = sum(bool(row.get("harmful_bad_refusal_attempt")) for row in harmful) / max(len(harmful), 1)
    out["benign_over_refusal_rate"] = sum(bool(row.get("benign_over_refusal")) for row in benign) / max(len(benign), 1)
    return out


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Qwen2.5-1.5B Dynamic Activation Patch Generation",
        "",
        "Small smoke test: run donor base on the current prefix, patch donor activations into abliterated recipient, then greedily decode one token.",
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
            "- This is a small smoke test because dynamic activation patching is much slower than static module replacement.",
            "- If dynamic activation patching fails while static replacement succeeds, the static repair likely depends on changed internal recurrence/propagation rather than simply copying donor activations on the current prefix.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--examples-per-split", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--patch-spec", default="12+13+14+15+16+17+18+19+20+21+22+23+24:mlp")
    ap.add_argument("--position", choices=("all", "target"), default="all")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    patch_specs = parse_specs(args.patch_spec)
    if len(patch_specs) != 1:
        raise ValueError("--patch-spec should contain exactly one patch spec")
    layers, module, spec = patch_specs[0]
    patch_points = tuple((layer, module) for layer in layers)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[load] base donor", flush=True)
    donor = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=torch.float16).to(args.device)
    donor.eval()
    print("[load] abliterated recipient", flush=True)
    recipient = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=torch.float16).to(args.device)
    recipient.eval()

    prompts = []
    prompts.extend(("harmful", prompt, "") for prompt in HARMFUL_PROMPTS[: args.examples_per_split])
    prompts.extend(("benign", prompt, "") for prompt in BENIGN_PROMPTS[: args.examples_per_split])

    all_records = []
    summary_rows = []
    for variant in ("base", "abliterated", f"activation_patch_{spec}"):
        print(f"[eval] {variant}", flush=True)
        records = []
        for split, user, expected in prompts:
            if variant == "base":
                text = generate(donor, tokenizer, user, device=args.device, max_new_tokens=args.max_new_tokens)
            elif variant == "abliterated":
                text = generate(recipient, tokenizer, user, device=args.device, max_new_tokens=args.max_new_tokens)
            else:
                text = activation_patch_generate(
                    donor,
                    recipient,
                    tokenizer,
                    user,
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                    patch_points=patch_points,
                    position=args.position,
                )
            scores = score_record(split, user, text, expected)
            record = {"model": variant, "split": split, "user": user, "expected": expected, "generation": text, **scores}
            records.append(record)
            all_records.append(record)
        summary_rows.append(summarize(variant, records))

    metrics_path = args.result_dir / "qwen1_5b_dynamic_activation_patch_generation_metrics.csv"
    records_path = args.result_dir / "qwen1_5b_dynamic_activation_patch_generation_records.jsonl"
    summary_path = args.result_dir / "QWEN1_5B_DYNAMIC_ACTIVATION_PATCH_GENERATION_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, summary_rows)
    write_jsonl(records_path, all_records)
    write_summary(summary_path, summary_rows)
    manifest_path.write_text(
        json.dumps(
            {
                "patch_spec": spec,
                "position": args.position,
                "examples_per_split": args.examples_per_split,
                "max_new_tokens": args.max_new_tokens,
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
