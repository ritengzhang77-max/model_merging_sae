#!/usr/bin/env python3
"""Audit first assistant-token logits for selected SAE patch variants."""

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
from run_gemma2_2b_activation_patch_target_loss import MODEL_IDS  # noqa: E402
from run_gemma2_2b_gemmascope_mlp_sae_feature_subsets import chat_prompt, patch_position_mask  # noqa: E402
from run_gemma2_2b_linear_merge_sae_bundle_patch import load_prompt_rows, write_csv  # noqa: E402
from run_gemma2_2b_linear_merge_sae_mixed_timing_bundle_patch import (  # noqa: E402
    DEFAULT_RANK_CSV,
    apply_mixed_donor_subset_decode,
    merge_same_filter_groups,
    read_rank_features,
    tensor_groups,
    variant_groups,
)
from run_gemma2_2b_linear_weight_merge_sweep import set_linear_merge_weights  # noqa: E402
from validate_gemma2_2b_gemmascope_mlp_sae import (  # noqa: E402
    first_tensor,
    load_sae,
    module_mlp,
    parse_ints,
    remove_hooks,
    replace_first_tensor,
    select_sae_file,
)


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_linear_merge_sae_patch_first_token_logits_v0"
DEFAULT_VARIANT_LABELS = (
    "top3183_rank4000_abog_rank3201_boundary_partner_sweep,"
    "top3183_rank4000_abog_rank3202_boundary_partner_sweep"
)


def parse_labels(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def one_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not one token: {ids}")
    return int(ids[0])


@torch.no_grad()
def unpatched_logits(model, tokenizer, prompt: str, device: str) -> torch.Tensor:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**enc, use_cache=False)
    return out.logits[0, -1].detach().float().cpu()


@torch.no_grad()
def patched_logits(donor, recipient, tokenizer, sae, variant, user: str, args) -> torch.Tensor:
    donor_cache: dict[int, torch.Tensor] = {}
    current_masks: dict[str, torch.Tensor] = {}
    donor_handles = []
    recipient_handles = []

    def donor_hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        donor_cache[args.layer] = tensor.detach()

    def recipient_hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        groups = [group for group in variant["groups"] if int(group["layer"]) == args.layer]
        patched = apply_mixed_donor_subset_decode(sae, tensor, donor_cache[args.layer], groups, current_masks)
        return replace_first_tensor(output, patched.to(device=tensor.device, dtype=tensor.dtype))

    donor_handles.append(module_mlp(donor, args.layer, args.output_mode).register_forward_hook(donor_hook))
    recipient_handles.append(module_mlp(recipient, args.layer, args.output_mode).register_forward_hook(recipient_hook))
    prompt = chat_prompt(tokenizer, user)
    enc = tokenizer(prompt, return_tensors="pt").to(args.device)
    prompt_length = int(enc["input_ids"].shape[1])
    filters = sorted({str(group["filter"]) for group in variant["groups"]})
    try:
        donor(**enc, use_cache=False)
        for patch_filter in filters:
            current_masks[patch_filter] = patch_position_mask(
                tokenizer,
                enc["input_ids"],
                enc["attention_mask"],
                patch_filter,
                prompt_length=prompt_length,
            )
        out = recipient(**enc, use_cache=False)
        return out.logits[0, -1].detach().float().cpu()
    finally:
        remove_hooks(donor_handles)
        remove_hooks(recipient_handles)


def build_rows(rows, token_i: int, token_it: int, tokenizer, baseline_lookup):
    out = []
    for row in rows:
        logits = row.pop("logits")
        top_value, top_id = torch.max(logits, dim=0)
        key = (int(row["prompt_index"]), str(row["split"]), str(row["prompt"]))
        rec_margin = baseline_lookup.get((key, "recipient_alpha"))
        donor_margin = baseline_lookup.get((key, "donor_alpha"))
        margin = float((logits[token_i] - logits[token_it]).item())
        out.append(
            {
                **row,
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
        )
    return out


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


def write_summary(path: Path, args, summary_rows) -> None:
    lines = [
        "# Gemma-2-2B Linear Merge SAE Patch First-Token Logit Audit",
        "",
        f"Prompt file: `{args.prompt_jsonl}`.",
        f"Recipient alpha: `{args.recipient_alpha}`. Donor alpha: `{args.donor_alpha}`.",
        f"Variant labels: `{args.variant_labels}`.",
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
            "This audit tests whether selected sparse donor-subset patches change the first",
            "assistant-token distribution toward the donor/refusal endpoint before any",
            "generation history exists.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--sae-dtype", choices=("float16", "bfloat16", "float32"), default="float32")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--l0-target", type=int, default=80)
    ap.add_argument("--donor-alpha", type=float, default=1.0)
    ap.add_argument("--recipient-alpha", type=float, default=0.75)
    ap.add_argument("--rank-csv", type=Path, default=DEFAULT_RANK_CSV)
    ap.add_argument("--prompt-jsonl", type=Path, required=True)
    ap.add_argument("--prefix-top-k", type=int, default=3183)
    ap.add_argument("--extra-probe-ranks", default="")
    ap.add_argument("--timing-pair-ranks", default="")
    ap.add_argument("--fixed-abog-ranks", default="4000")
    ap.add_argument("--boundary-partner-ranks", default="3201,3202")
    ap.add_argument("--variant-labels", default=DEFAULT_VARIANT_LABELS)
    ap.add_argument("--merge-same-filter-groups", action="store_true")
    ap.add_argument("--output-mode", choices=("post_ff_norm", "raw_mlp"), default="post_ff_norm")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    sae_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.sae_dtype]
    cache_dir = os.environ.get("HF_HOME")
    prompts = load_prompt_rows(args.prompt_jsonl)
    wanted_labels = set(parse_labels(args.variant_labels))

    print("[load] tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"], cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_i = one_token_id(tokenizer, "I")
    token_it = one_token_id(tokenizer, "It")

    print("[load] SAE", flush=True)
    sae = load_sae(select_sae_file(args.layer, args.l0_target), device=args.device, dtype=sae_dtype, cache_dir=cache_dir)
    rank_to_feature = read_rank_features(args.rank_csv)
    raw_variants = variant_groups(
        rank_to_feature,
        args.prefix_top_k,
        tuple(parse_ints(args.extra_probe_ranks)),
        tuple(parse_ints(args.timing_pair_ranks)),
        tuple(parse_ints(args.fixed_abog_ranks)),
        tuple(parse_ints(args.boundary_partner_ranks)),
    )
    if args.merge_same_filter_groups:
        raw_variants = merge_same_filter_groups(raw_variants)
    raw_variants = [variant for variant in raw_variants if str(variant["label"]) in wanted_labels]
    found = {str(variant["label"]) for variant in raw_variants}
    missing = sorted(wanted_labels - found)
    if missing:
        raise ValueError(f"Missing requested variant labels: {missing}")
    variants, count_rows = tensor_groups(raw_variants, (args.layer,), args.device)

    print("[load] base donor for weight construction", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    base_model.eval()
    print("[load] donor endpoint model", flush=True)
    donor_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    donor_model.eval()
    endpoint_params_cpu = {name: param.detach().cpu().clone() for name, param in donor_model.named_parameters()}
    set_linear_merge_weights(donor_model, base_model, endpoint_params_cpu, args.donor_alpha, args.device)
    print("[load] recipient endpoint model", flush=True)
    recipient_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    recipient_model.eval()
    set_linear_merge_weights(recipient_model, base_model, endpoint_params_cpu, args.recipient_alpha, args.device)
    del base_model
    torch.cuda.empty_cache()

    raw_rows = []
    baseline_lookup = {}
    for prompt_index, (split, user) in enumerate(prompts, start=1):
        prompt = chat_prompt(tokenizer, str(user))
        for condition, model in (("recipient_alpha", recipient_model), ("donor_alpha", donor_model)):
            print(f"[eval] {condition} prompt {prompt_index}/{len(prompts)} ({split})", flush=True)
            logits = unpatched_logits(model, tokenizer, prompt, args.device)
            margin = float((logits[token_i] - logits[token_it]).item())
            key = (prompt_index, str(split), str(user))
            baseline_lookup[(key, condition)] = margin
            raw_rows.append(
                {
                    "prompt_index": prompt_index,
                    "split": str(split),
                    "prompt": str(user),
                    "condition": condition,
                    "variant": "",
                    "logits": logits,
                }
            )

    for variant in variants:
        label = str(variant["label"])
        for prompt_index, (split, user) in enumerate(prompts, start=1):
            print(f"[eval] {label} prompt {prompt_index}/{len(prompts)} ({split})", flush=True)
            raw_rows.append(
                {
                    "prompt_index": prompt_index,
                    "split": str(split),
                    "prompt": str(user),
                    "condition": label,
                    "variant": label,
                    "logits": patched_logits(donor_model, recipient_model, tokenizer, sae, variant, str(user), args),
                }
            )

    detail_rows = build_rows(raw_rows, token_i, token_it, tokenizer, baseline_lookup)
    summary_rows = summarize(detail_rows)
    write_csv(args.result_dir / "first_token_logits.csv", detail_rows)
    write_csv(args.result_dir / "first_token_logit_summary.csv", summary_rows)
    write_csv(args.result_dir / "selected_feature_counts.csv", count_rows)
    write_summary(args.result_dir / "PATCH_FIRST_TOKEN_LOGIT_SUMMARY.md", args, summary_rows)
    (args.result_dir / "manifest.json").write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "prompt_jsonl": str(args.prompt_jsonl),
                "rank_csv": str(args.rank_csv),
                "recipient_alpha": args.recipient_alpha,
                "donor_alpha": args.donor_alpha,
                "prefix_top_k": args.prefix_top_k,
                "variant_labels": sorted(wanted_labels),
                "merge_same_filter_groups": args.merge_same_filter_groups,
                "outputs": {
                    "detail": str(args.result_dir / "first_token_logits.csv"),
                    "summary": str(args.result_dir / "first_token_logit_summary.csv"),
                    "feature_counts": str(args.result_dir / "selected_feature_counts.csv"),
                    "markdown": str(args.result_dir / "PATCH_FIRST_TOKEN_LOGIT_SUMMARY.md"),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {args.result_dir / 'PATCH_FIRST_TOKEN_LOGIT_SUMMARY.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
