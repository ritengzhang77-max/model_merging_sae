#!/usr/bin/env python3
"""Patch dense/SAE reconstruction formulas at a selected Gemma merge site."""

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
from run_gemma2_2b_activation_patch_target_loss import (  # noqa: E402
    MODEL_IDS,
    first_tensor,
    module_for,
    remove_hooks,
    replace_first_tensor,
)
from run_gemma2_2b_gemmascope_mlp_sae_feature_subsets import chat_prompt, patch_position_mask  # noqa: E402
from run_gemma2_2b_linear_merge_sae_bundle_patch import load_prompt_rows  # noqa: E402
from run_gemma2_2b_linear_weight_merge_sweep import set_linear_merge_weights  # noqa: E402
from screen_chat_merge_candidate import clean_assistant_text, score_record  # noqa: E402
from validate_gemma2_2b_gemmascope_mlp_sae import load_sae, select_sae_file  # noqa: E402


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_linear_merge_sae_reconstruction_formula_patch_v0"
DEFAULT_FORMULAS = "dense_donor,sae_donor_recon,recon_error_add,sae_delta_add,recipient_recon"


def parse_items(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


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


def patch_formula(sae, donor_out: torch.Tensor, recipient_out: torch.Tensor, formula: str) -> torch.Tensor:
    donor_f = sae.encode(donor_out)
    recipient_f = sae.encode(recipient_out)
    donor_recon = sae.decode(donor_f)
    recipient_recon = sae.decode(recipient_f)
    if formula == "dense_donor":
        return donor_out
    if formula == "sae_donor_recon":
        return donor_recon
    if formula == "recipient_recon":
        return recipient_recon
    if formula == "recon_error_add":
        return recipient_out + (donor_recon - donor_out)
    if formula == "sae_delta_add":
        return recipient_out + (donor_recon - recipient_recon)
    if formula == "dense_delta_half":
        return recipient_out + 0.5 * (donor_out - recipient_out)
    if formula == "recon_error_subtract":
        return recipient_out - (donor_recon - donor_out)
    raise ValueError(f"unknown formula: {formula}")


@torch.no_grad()
def formula_patch_generate(donor, recipient, tokenizer, sae, user: str, formula: str, args) -> str:
    prompt = chat_prompt(tokenizer, user)
    enc = tokenizer(prompt, return_tensors="pt").to(args.device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    prompt_length = int(input_ids.shape[1])
    donor_cache: dict[str, torch.Tensor] = {}
    current_mask = None
    donor_handles = []
    recipient_handles = []

    def donor_hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        donor_cache["x"] = tensor.detach()

    def recipient_hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        patched = patch_formula(sae, donor_cache["x"], tensor, formula)
        mask = current_mask.to(device=tensor.device).unsqueeze(-1)
        patched = torch.where(mask, patched, tensor)
        return replace_first_tensor(output, patched.to(device=tensor.device, dtype=tensor.dtype))

    donor_handles.append(module_for(donor, args.layer, args.module).register_forward_hook(donor_hook))
    recipient_handles.append(module_for(recipient, args.layer, args.module).register_forward_hook(recipient_hook))
    try:
        for _ in range(args.max_new_tokens):
            donor_cache.clear()
            donor(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            current_mask = patch_position_mask(
                tokenizer,
                input_ids,
                attention_mask,
                args.patch_token_filter,
                prompt_length=prompt_length,
            )
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


@torch.no_grad()
def formula_first_token_logits(donor, recipient, tokenizer, sae, user: str, formula: str, args) -> torch.Tensor:
    prompt = chat_prompt(tokenizer, user)
    enc = tokenizer(prompt, return_tensors="pt").to(args.device)
    prompt_length = int(enc["input_ids"].shape[1])
    donor_cache: dict[str, torch.Tensor] = {}
    current_mask = None
    donor_handles = []
    recipient_handles = []

    def donor_hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        donor_cache["x"] = tensor.detach()

    def recipient_hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        patched = patch_formula(sae, donor_cache["x"], tensor, formula)
        mask = current_mask.to(device=tensor.device).unsqueeze(-1)
        patched = torch.where(mask, patched, tensor)
        return replace_first_tensor(output, patched.to(device=tensor.device, dtype=tensor.dtype))

    donor_handles.append(module_for(donor, args.layer, args.module).register_forward_hook(donor_hook))
    recipient_handles.append(module_for(recipient, args.layer, args.module).register_forward_hook(recipient_hook))
    try:
        donor(**enc, use_cache=False)
        current_mask = patch_position_mask(
            tokenizer,
            enc["input_ids"],
            enc["attention_mask"],
            args.patch_token_filter,
            prompt_length=prompt_length,
        )
        out = recipient(**enc, use_cache=False)
        return out.logits[0, -1].detach().float().cpu()
    finally:
        remove_hooks(donor_handles)
        remove_hooks(recipient_handles)


def one_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not one token: {ids}")
    return int(ids[0])


def summarize_generation(model_name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    harmful = [row for row in rows if row["split"] == "harmful"]
    benign = [row for row in rows if row["split"] == "benign"]
    return {
        "model": model_name,
        "n": len(rows),
        "harmful_attempted_refusal_rate": sum(bool(row.get("harmful_attempted_refusal")) for row in harmful) / max(len(harmful), 1),
        "harmful_unsafe_continuation_rate": sum(bool(row.get("harmful_unsafe_continuation")) for row in harmful) / max(len(harmful), 1),
        "benign_over_refusal_rate": sum(bool(row.get("benign_over_refusal")) for row in benign) / max(len(benign), 1),
    }


def write_summary(path: Path, args, metrics, logit_rows) -> None:
    lines = [
        "# Gemma-2-2B SAE Reconstruction Formula Patch",
        "",
        f"Layer/module: `{args.layer}:{args.module}`.",
        f"Patch token filter: `{args.patch_token_filter}`.",
        f"Prompts: `{args.prompt_jsonl}`.",
        "",
        "## Generation",
        "",
        "| model | harmful attempted | unsafe | benign over-refusal |",
        "|---|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| `{row['model']}` | {float(row['harmful_attempted_refusal_rate']):.3f} | "
            f"{float(row['harmful_unsafe_continuation_rate']):.3f} | {float(row['benign_over_refusal_rate']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## First Token",
            "",
            "| model | split | prompt | top token | I-It |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in logit_rows:
        lines.append(
            f"| `{row['model']}` | `{row['split']}` | `{row['prompt']}` | `{row['top_token']}` | {float(row['i_minus_it']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--sae-dtype", choices=("float16", "bfloat16", "float32"), default="float32")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--module", choices=("post_ff", "mlp"), default="post_ff")
    ap.add_argument("--l0-target", type=int, default=80)
    ap.add_argument("--donor-alpha", type=float, default=1.0)
    ap.add_argument("--recipient-alpha", type=float, default=0.75)
    ap.add_argument("--prompt-jsonl", type=Path, required=True)
    ap.add_argument("--patch-token-filter", default="assistant_boundary_model_token")
    ap.add_argument("--formulas", default=DEFAULT_FORMULAS)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    sae_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.sae_dtype]
    cache_dir = os.environ.get("HF_HOME")
    formulas = parse_items(args.formulas)
    prompts = load_prompt_rows(args.prompt_jsonl)

    print("[load] tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"], cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_i = one_token_id(tokenizer, "I")
    token_it = one_token_id(tokenizer, "It")
    print("[load] SAE", flush=True)
    sae = load_sae(select_sae_file(args.layer, args.l0_target), device=args.device, dtype=sae_dtype, cache_dir=cache_dir)

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

    all_records = []
    metrics = []
    logit_rows = []
    for formula in formulas:
        model_name = f"formula_{formula}"
        rows = []
        print(f"[eval] {model_name}", flush=True)
        for row_idx, (split, user) in enumerate(prompts, start=1):
            print(f"[eval] {model_name} prompt {row_idx}/{len(prompts)} ({split})", flush=True)
            logits = formula_first_token_logits(donor, recipient, tokenizer, sae, user, formula, args)
            top_value, top_id = torch.max(logits, dim=0)
            logit_rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "prompt": user,
                    "top_token": tokenizer.decode([int(top_id)], skip_special_tokens=False),
                    "top_logit": float(top_value.item()),
                    "i_minus_it": float((logits[token_i] - logits[token_it]).item()),
                }
            )
            text = formula_patch_generate(donor, recipient, tokenizer, sae, user, formula, args)
            record = {
                "model": model_name,
                "split": split,
                "prompt": user,
                "text": text,
                "formula": formula,
                "patch_token_filter": args.patch_token_filter,
                "layer": args.layer,
                "module": args.module,
            }
            record.update(score_record(split, user, text))
            rows.append(record)
            all_records.append(record)
        metrics.append(summarize_generation(model_name, rows))

    metrics_path = args.result_dir / "formula_patch_metrics.csv"
    records_path = args.result_dir / "formula_patch_records.jsonl"
    logits_path = args.result_dir / "formula_first_token_logits.csv"
    summary_path = args.result_dir / "SAE_RECONSTRUCTION_FORMULA_PATCH_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, metrics)
    write_jsonl(records_path, all_records)
    write_csv(logits_path, logit_rows)
    write_summary(summary_path, args, metrics, logit_rows)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "donor_alpha": args.donor_alpha,
                "recipient_alpha": args.recipient_alpha,
                "sae_file": select_sae_file(args.layer, args.l0_target),
                "layer": args.layer,
                "module": args.module,
                "patch_token_filter": args.patch_token_filter,
                "formulas": formulas,
                "prompt_jsonl": str(args.prompt_jsonl),
                "outputs": {
                    "metrics": str(metrics_path),
                    "records": str(records_path),
                    "first_token_logits": str(logits_path),
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
