#!/usr/bin/env python3
"""Audit first-token logits under dense activation patches between merge alphas."""

from __future__ import annotations

import argparse
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
    parse_specs,
    remove_hooks,
    replace_first_tensor,
)
from run_gemma2_2b_gemmascope_mlp_sae_feature_subsets import chat_prompt, patch_position_mask  # noqa: E402
from run_gemma2_2b_linear_merge_sae_bundle_patch import load_prompt_rows, write_csv  # noqa: E402
from run_gemma2_2b_linear_weight_merge_sweep import set_linear_merge_weights  # noqa: E402


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_linear_merge_activation_patch_first_token_logits_v0"


def one_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not one token: {ids}")
    return int(ids[0])


@torch.no_grad()
def first_token_logits(model, tokenizer, user: str, device: str) -> torch.Tensor:
    prompt = chat_prompt(tokenizer, user)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**enc, use_cache=False)
    return out.logits[0, -1].detach().float().cpu()


@torch.no_grad()
def patched_first_token_logits(donor, recipient, tokenizer, user: str, patch_points, args) -> torch.Tensor:
    prompt = chat_prompt(tokenizer, user)
    enc = tokenizer(prompt, return_tensors="pt").to(args.device)
    prompt_length = int(enc["input_ids"].shape[1])
    cache: dict[tuple[int, str], torch.Tensor] = {}
    donor_handles = []
    recipient_handles = []
    current_mask = None

    def make_donor_hook(key):
        def hook(_module, _inputs, output):
            tensor, _rest = first_tensor(output)
            cache[key] = tensor.detach()

        return hook

    def make_recipient_hook(key):
        def hook(_module, _inputs, output):
            tensor, _rest = first_tensor(output)
            donor_tensor = cache[key].to(device=tensor.device, dtype=tensor.dtype)
            if args.position == "all":
                patched = donor_tensor
            else:
                assert current_mask is not None
                mask = current_mask.to(device=tensor.device).unsqueeze(-1)
                patched = torch.where(mask, donor_tensor, tensor)
            return replace_first_tensor(output, patched)

        return hook

    for key in patch_points:
        layer, kind = key
        donor_handles.append(module_for(donor, layer, kind).register_forward_hook(make_donor_hook(key)))
        recipient_handles.append(module_for(recipient, layer, kind).register_forward_hook(make_recipient_hook(key)))

    try:
        cache.clear()
        donor(**enc, use_cache=False)
        if args.position != "all":
            mask_mode = "last_token" if args.position == "target" else args.position
            current_mask = patch_position_mask(
                tokenizer,
                enc["input_ids"],
                enc["attention_mask"],
                mask_mode,
                prompt_length=prompt_length,
            )
        out = recipient(**enc, use_cache=False)
        return out.logits[0, -1].detach().float().cpu()
    finally:
        remove_hooks(donor_handles)
        remove_hooks(recipient_handles)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["condition"]), str(row["split"])), []).append(row)
    out = []
    for (condition, split), bucket in sorted(groups.items()):
        n = len(bucket)
        deltas_rec = [float(row["delta_i_minus_it_vs_recipient"]) for row in bucket if row["delta_i_minus_it_vs_recipient"] != ""]
        deltas_donor = [float(row["delta_i_minus_it_vs_donor"]) for row in bucket if row["delta_i_minus_it_vs_donor"] != ""]
        out.append(
            {
                "condition": condition,
                "split": split,
                "n": n,
                "top_i_rate": sum(1 for row in bucket if row["top_token"] == "I") / n,
                "top_it_rate": sum(1 for row in bucket if row["top_token"] == "It") / n,
                "mean_i_minus_it": sum(float(row["i_minus_it"]) for row in bucket) / n,
                "min_i_minus_it": min(float(row["i_minus_it"]) for row in bucket),
                "max_i_minus_it": max(float(row["i_minus_it"]) for row in bucket),
                "mean_delta_vs_recipient": "" if not deltas_rec else sum(deltas_rec) / len(deltas_rec),
                "mean_delta_vs_donor": "" if not deltas_donor else sum(deltas_donor) / len(deltas_donor),
            }
        )
    return out


def logit_row(row_base, condition: str, logits: torch.Tensor, token_i: int, token_it: int, tokenizer, baseline_lookup):
    prompt_index, split, prompt = row_base
    key = (prompt_index, split, prompt)
    rec_margin = baseline_lookup.get((key, "recipient_alpha"))
    donor_margin = baseline_lookup.get((key, "donor_alpha"))
    margin = float((logits[token_i] - logits[token_it]).item())
    top_value, top_id = torch.max(logits, dim=0)
    return {
        "prompt_index": prompt_index,
        "split": split,
        "prompt": prompt,
        "condition": condition,
        "token_i_id": token_i,
        "token_it_id": token_it,
        "logit_i": float(logits[token_i].item()),
        "logit_it": float(logits[token_it].item()),
        "i_minus_it": margin,
        "delta_i_minus_it_vs_recipient": "" if rec_margin is None else margin - rec_margin,
        "delta_i_minus_it_vs_donor": "" if donor_margin is None else margin - donor_margin,
        "top_token_id": int(top_id.item()),
        "top_token": tokenizer.decode([int(top_id)], skip_special_tokens=False),
        "top_logit": float(top_value.item()),
    }


def write_summary(path: Path, args, summary_rows) -> None:
    lines = [
        "# Gemma-2-2B Linear Merge Activation-Patch First-Token Logit Audit",
        "",
        f"Prompt file: `{args.prompt_jsonl}`.",
        f"Donor alpha: `{args.donor_alpha}`. Recipient alpha: `{args.recipient_alpha}`.",
        f"Patch position: `{args.position}`.",
        f"Patch specs: `{args.patch_specs}`.",
        "",
        "| condition | split | n | top I | top It | mean I-It | min I-It | max I-It | mean delta vs recipient | mean delta vs donor |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        rec = row["mean_delta_vs_recipient"]
        donor = row["mean_delta_vs_donor"]
        rec_s = "" if rec == "" else f"{float(rec):.4f}"
        donor_s = "" if donor == "" else f"{float(donor):.4f}"
        lines.append(
            f"| `{row['condition']}` | `{row['split']}` | {row['n']} | "
            f"{float(row['top_i_rate']):.3f} | {float(row['top_it_rate']):.3f} | "
            f"{float(row['mean_i_minus_it']):.4f} | {float(row['min_i_minus_it']):.4f} | "
            f"{float(row['max_i_minus_it']):.4f} | {rec_s} | {donor_s} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Dense activation patches identify which layers/sites can move the first",
            "assistant-token distribution toward the donor/refusal endpoint.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--donor-alpha", type=float, default=1.0)
    ap.add_argument("--recipient-alpha", type=float, default=0.75)
    ap.add_argument("--prompt-jsonl", type=Path, required=True)
    ap.add_argument("--patch-specs", required=True)
    ap.add_argument("--position", default="all")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    cache_dir = os.environ.get("HF_HOME")
    prompts = load_prompt_rows(args.prompt_jsonl)
    prompt_rows = [(idx, split, str(user)) for idx, (split, user) in enumerate(prompts, start=1)]
    patch_specs = parse_specs(args.patch_specs)

    print("[load] tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"], cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_i = one_token_id(tokenizer, "I")
    token_it = one_token_id(tokenizer, "It")

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

    rows = []
    baseline_lookup = {}
    for row_base in prompt_rows:
        prompt_index, split, user = row_base
        for condition, model in (("recipient_alpha", recipient), ("donor_alpha", donor)):
            print(f"[eval] {condition} prompt {prompt_index}/{len(prompt_rows)} ({split})", flush=True)
            logits = first_token_logits(model, tokenizer, user, args.device)
            margin = float((logits[token_i] - logits[token_it]).item())
            baseline_lookup[((prompt_index, split, user), condition)] = margin
            rows.append(logit_row(row_base, condition, logits, token_i, token_it, tokenizer, baseline_lookup))

    for layers, module, spec in patch_specs:
        condition = f"activation_patch_{spec}"
        patch_points = tuple((layer, module) for layer in layers)
        for row_base in prompt_rows:
            prompt_index, split, user = row_base
            print(f"[eval] {condition} prompt {prompt_index}/{len(prompt_rows)} ({split})", flush=True)
            logits = patched_first_token_logits(donor, recipient, tokenizer, user, patch_points, args)
            rows.append(logit_row(row_base, condition, logits, token_i, token_it, tokenizer, baseline_lookup))

    summary_rows = summarize(rows)
    write_csv(args.result_dir / "first_token_logits.csv", rows)
    write_csv(args.result_dir / "first_token_logit_summary.csv", summary_rows)
    write_summary(args.result_dir / "ACTIVATION_PATCH_FIRST_TOKEN_LOGIT_SUMMARY.md", args, summary_rows)
    (args.result_dir / "manifest.json").write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "donor_alpha": args.donor_alpha,
                "recipient_alpha": args.recipient_alpha,
                "prompt_jsonl": str(args.prompt_jsonl),
                "patch_specs": [spec for _layers, _module, spec in patch_specs],
                "position": args.position,
                "outputs": {
                    "detail": str(args.result_dir / "first_token_logits.csv"),
                    "summary": str(args.result_dir / "first_token_logit_summary.csv"),
                    "markdown": str(args.result_dir / "ACTIVATION_PATCH_FIRST_TOKEN_LOGIT_SUMMARY.md"),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {args.result_dir / 'ACTIVATION_PATCH_FIRST_TOKEN_LOGIT_SUMMARY.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
