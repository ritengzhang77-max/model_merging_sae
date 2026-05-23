#!/usr/bin/env python3
"""Dynamic activation patching between Gemma linear-merge alpha points."""

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
sys.path.insert(0, str(ROOT / "stage3" / "scripts"))
from run_gemma2_2b_activation_patch_target_loss import MODEL_IDS, install_hooks, parse_specs, remove_hooks  # noqa: E402
from run_gemma2_2b_linear_weight_merge_sweep import set_linear_merge_weights  # noqa: E402
from screen_chat_merge_candidate import clean_assistant_text, generate, score_record  # noqa: E402


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_linear_merge_activation_patch_generation_v0"


def chat_prompt(tokenizer, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


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
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def summarize(model_name: str, rows: list[dict[str, object]], args) -> dict[str, object]:
    out: dict[str, object] = {
        "model": model_name,
        "n": len(rows),
        "donor_alpha": args.donor_alpha,
        "recipient_alpha": args.recipient_alpha,
    }
    harmful = [row for row in rows if row["split"] == "harmful"]
    benign = [row for row in rows if row["split"] == "benign"]
    out["harmful_ok_rate"] = sum(bool(row.get("ok")) for row in harmful) / max(len(harmful), 1)
    out["harmful_attempted_refusal_rate"] = sum(bool(row.get("harmful_attempted_refusal")) for row in harmful) / max(len(harmful), 1)
    out["harmful_unsafe_continuation_rate"] = sum(bool(row.get("harmful_unsafe_continuation")) for row in harmful) / max(len(harmful), 1)
    out["benign_ok_rate"] = sum(bool(row.get("ok")) for row in benign) / max(len(benign), 1)
    out["benign_over_refusal_rate"] = sum(bool(row.get("benign_over_refusal")) for row in benign) / max(len(benign), 1)
    return out


def write_summary(path: Path, metrics: list[dict[str, object]], args, prompt_source: str) -> None:
    lines = [
        "# Gemma-2-2B Linear Merge Activation Patch Generation",
        "",
        f"Donor alpha: `{args.donor_alpha:g}`.",
        f"Recipient alpha: `{args.recipient_alpha:g}`.",
        f"Patch position: `{args.position}`.",
        f"Prompts: {prompt_source}.",
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
    ap.add_argument("--donor-alpha", type=float, default=1.0)
    ap.add_argument("--recipient-alpha", type=float, default=0.75)
    ap.add_argument("--prompt-jsonl", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--patch-specs", default="16+17+18+19+20:mlp,12+13+14+15+16+17+18+19+20:mlp")
    ap.add_argument("--position", choices=("all", "target"), default="all")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    cache_dir = os.environ.get("HF_HOME")
    patch_specs = parse_specs(args.patch_specs)
    prompts = load_prompt_rows(args.prompt_jsonl)
    prompt_source = str(args.prompt_jsonl)

    print("[load] tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"], cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[load] base donor for weight construction", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    base_model.eval()
    print("[load] donor alpha model", flush=True)
    donor = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    donor.eval()
    recipient_params_cpu = {name: param.detach().cpu().clone() for name, param in donor.named_parameters()}
    set_linear_merge_weights(donor, base_model, recipient_params_cpu, args.donor_alpha, args.device)
    print("[load] recipient alpha model", flush=True)
    recipient = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    recipient.eval()
    set_linear_merge_weights(recipient, base_model, recipient_params_cpu, args.recipient_alpha, args.device)
    del base_model
    torch.cuda.empty_cache()

    variants: list[tuple[str, tuple[tuple[int, str], ...] | None, object | None]] = []
    if not args.skip_baselines:
        variants.extend(
            [
                (f"linear_alpha_{args.recipient_alpha:g}", None, recipient),
                (f"linear_alpha_{args.donor_alpha:g}", None, donor),
            ]
        )
    for layers, module, spec in patch_specs:
        variants.append((f"activation_patch_{spec}", tuple((layer, module) for layer in layers), None))

    all_records: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    for variant, patch_points, direct_model in variants:
        print(f"[eval] {variant}", flush=True)
        rows = []
        for row_idx, (split, user) in enumerate(prompts, start=1):
            print(f"[eval] {variant} prompt {row_idx}/{len(prompts)} ({split})", flush=True)
            if direct_model is not None:
                text = generate(direct_model, tokenizer, user, device=args.device, max_new_tokens=args.max_new_tokens)
            else:
                assert patch_points is not None
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
            record = {
                "model": variant,
                "split": split,
                "prompt": user,
                "text": text,
                "donor_alpha": args.donor_alpha,
                "recipient_alpha": args.recipient_alpha,
                "patch_position": args.position,
            }
            record.update(score_record(split, user, text))
            rows.append(record)
            all_records.append(record)
        metrics.append(summarize(variant, rows, args))

    metrics_path = args.result_dir / "gemma2_2b_linear_merge_activation_patch_generation_metrics.csv"
    records_path = args.result_dir / "gemma2_2b_linear_merge_activation_patch_generation_records.jsonl"
    summary_path = args.result_dir / "GEMMA2_2B_LINEAR_MERGE_ACTIVATION_PATCH_GENERATION_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, metrics)
    write_jsonl(records_path, all_records)
    write_summary(summary_path, metrics, args, prompt_source)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "donor_alpha": args.donor_alpha,
                "recipient_alpha": args.recipient_alpha,
                "patch_specs": [spec for _layers, _module, spec in patch_specs],
                "position": args.position,
                "prompt_source": prompt_source,
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
    print(f"[save] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
