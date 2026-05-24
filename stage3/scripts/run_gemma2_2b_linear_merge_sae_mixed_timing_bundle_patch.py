#!/usr/bin/env python3
"""Evaluate layer-20 SAE donor-subset patches with per-feature timing masks."""

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
from run_gemma2_2b_activation_patch_target_loss import MODEL_IDS  # noqa: E402
from run_gemma2_2b_linear_merge_sae_bundle_patch import load_prompt_rows, write_csv, write_jsonl  # noqa: E402
from run_gemma2_2b_linear_weight_merge_sweep import set_linear_merge_weights  # noqa: E402
from run_gemma2_2b_gemmascope_mlp_sae_feature_subsets import (  # noqa: E402
    chat_prompt,
    clean_assistant_text,
    patch_position_mask,
    score_record,
    summarize_generation,
)
from validate_gemma2_2b_gemmascope_mlp_sae import (  # noqa: E402
    SAE_REPO,
    first_tensor,
    load_sae,
    module_mlp,
    parse_ints,
    remove_hooks,
    replace_first_tensor,
    select_sae_file,
)


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_linear_merge_sae_mixed_timing_bundle_patch_v0"
DEFAULT_RANK_CSV = (
    ROOT
    / "stage3"
    / "results"
    / "gemma2_2b_linear_merge_sae_decoder_contribution_rank_v0"
    / "hologram_success_l20_a075_to_a1_top5000"
    / "gemma2_2b_l20_decoder_contribution_features.csv"
)


def read_rank_features(path: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["rank"])] = int(row["feature_id"])
    return out


def variant_groups(rank_to_feature: dict[int, int], prefix_top_k: int, extra_probe_ranks: tuple[int, ...]) -> list[dict[str, object]]:
    prefix = [rank_to_feature[rank] for rank in range(1, prefix_top_k + 1)]
    boundary = [rank_to_feature[3308], rank_to_feature[3323]]
    generated_base = [rank_to_feature[4266]]
    edge_ranks = (3211, 3214)
    variants = []
    for edge_rank in edge_ranks:
        edge_feature = rank_to_feature[edge_rank]
        all_features = prefix + boundary + generated_base + [edge_feature]
        named_features = boundary + generated_base + [edge_feature]
        variants.append(
            {
                "label": f"all_abog_edge{edge_rank}",
                "groups": [
                    {"name": "all", "filter": "assistant_boundary_or_generated", "layer": 20, "features": all_features},
                ],
            }
        )
        variants.append(
            {
                "label": f"all_boundary_edge{edge_rank}",
                "groups": [
                    {"name": "all", "filter": "assistant_boundary", "layer": 20, "features": all_features},
                ],
            }
        )
        variants.append(
            {
                "label": f"all_generated_edge{edge_rank}",
                "groups": [
                    {"name": "all", "filter": "generated", "layer": 20, "features": all_features},
                ],
            }
        )
        variants.append(
            {
                "label": f"prefix_abog_edge{edge_rank}_split",
                "groups": [
                    {"name": "prefix", "filter": "assistant_boundary_or_generated", "layer": 20, "features": prefix},
                    {"name": "boundary", "filter": "assistant_boundary", "layer": 20, "features": boundary},
                    {"name": "generated", "filter": "generated", "layer": 20, "features": generated_base + [edge_feature]},
                ],
            }
        )
        variants.append(
            {
                "label": f"prefix_abog_named_boundary_edge{edge_rank}",
                "groups": [
                    {"name": "prefix", "filter": "assistant_boundary_or_generated", "layer": 20, "features": prefix},
                    {"name": "named", "filter": "assistant_boundary", "layer": 20, "features": named_features},
                ],
            }
        )
        variants.append(
            {
                "label": f"prefix_boundary_named_abog_edge{edge_rank}",
                "groups": [
                    {"name": "prefix", "filter": "assistant_boundary", "layer": 20, "features": prefix},
                    {"name": "named", "filter": "assistant_boundary_or_generated", "layer": 20, "features": named_features},
                ],
            }
        )
        variants.append(
            {
                "label": f"prefix_boundary_edge{edge_rank}_generated",
                "groups": [
                    {"name": "prefix", "filter": "assistant_boundary", "layer": 20, "features": prefix},
                    {"name": "boundary", "filter": "assistant_boundary", "layer": 20, "features": boundary},
                    {"name": "generated", "filter": "generated", "layer": 20, "features": generated_base + [edge_feature]},
                ],
            }
        )
        variants.append(
            {
                "label": f"prefix_generated_edge{edge_rank}_generated",
                "groups": [
                    {"name": "prefix", "filter": "generated", "layer": 20, "features": prefix},
                    {"name": "boundary", "filter": "assistant_boundary", "layer": 20, "features": boundary},
                    {"name": "generated", "filter": "generated", "layer": 20, "features": generated_base + [edge_feature]},
                ],
            }
        )
    edge_features = [rank_to_feature[rank] for rank in edge_ranks]
    edge3211_feature = rank_to_feature[3211]
    edge3214_feature = rank_to_feature[3214]
    all_features = prefix + boundary + generated_base + edge_features
    named_features = boundary + generated_base + edge_features
    variants.extend(
        [
            {
                "label": "prefix_abog_named_boundary_edges3211_3214",
                "groups": [
                    {"name": "prefix", "filter": "assistant_boundary_or_generated", "layer": 20, "features": prefix},
                    {"name": "named", "filter": "assistant_boundary", "layer": 20, "features": named_features},
                ],
            },
            {
                "label": "prefix_boundary_named_abog_edges3211_3214",
                "groups": [
                    {"name": "prefix", "filter": "assistant_boundary", "layer": 20, "features": prefix},
                    {"name": "named", "filter": "assistant_boundary_or_generated", "layer": 20, "features": named_features},
                ],
            },
            {
                "label": "prefix_abog_edges3211_3214_split",
                "groups": [
                    {"name": "prefix", "filter": "assistant_boundary_or_generated", "layer": 20, "features": prefix},
                    {"name": "boundary", "filter": "assistant_boundary", "layer": 20, "features": boundary},
                    {"name": "generated", "filter": "generated", "layer": 20, "features": generated_base + edge_features},
                ],
            },
            {
                "label": "prefix_boundary_edges3211_3214_generated",
                "groups": [
                    {"name": "prefix", "filter": "assistant_boundary", "layer": 20, "features": prefix},
                    {"name": "boundary", "filter": "assistant_boundary", "layer": 20, "features": boundary},
                    {"name": "generated", "filter": "generated", "layer": 20, "features": generated_base + edge_features},
                ],
            },
            {
                "label": "prefix_abog_boundary3211_edge3214_abog",
                "groups": [
                    {"name": "prefix", "filter": "assistant_boundary_or_generated", "layer": 20, "features": prefix},
                    {
                        "name": "boundary_base_edge3211",
                        "filter": "assistant_boundary",
                        "layer": 20,
                        "features": boundary + generated_base + [edge3211_feature],
                    },
                    {"name": "edge3214", "filter": "assistant_boundary_or_generated", "layer": 20, "features": [edge3214_feature]},
                ],
            },
            {
                "label": "prefix_abog_boundary3214_edge3211_abog",
                "groups": [
                    {"name": "prefix", "filter": "assistant_boundary_or_generated", "layer": 20, "features": prefix},
                    {
                        "name": "boundary_base_edge3214",
                        "filter": "assistant_boundary",
                        "layer": 20,
                        "features": boundary + generated_base + [edge3214_feature],
                    },
                    {"name": "edge3211", "filter": "assistant_boundary_or_generated", "layer": 20, "features": [edge3211_feature]},
                ],
            },
            {
                "label": "prefix_abog_boundarybase_edges_generated",
                "groups": [
                    {"name": "prefix", "filter": "assistant_boundary_or_generated", "layer": 20, "features": prefix},
                    {"name": "boundary_base", "filter": "assistant_boundary", "layer": 20, "features": boundary + generated_base},
                    {"name": "edges", "filter": "generated", "layer": 20, "features": edge_features},
                ],
            },
            {
                "label": "prefix_abog_boundarybase_edge3211_boundary_edge3214_generated",
                "groups": [
                    {"name": "prefix", "filter": "assistant_boundary_or_generated", "layer": 20, "features": prefix},
                    {
                        "name": "boundary_base_edge3211",
                        "filter": "assistant_boundary",
                        "layer": 20,
                        "features": boundary + generated_base + [edge3211_feature],
                    },
                    {"name": "edge3214", "filter": "generated", "layer": 20, "features": [edge3214_feature]},
                ],
            },
            {
                "label": "prefix_abog_boundarybase_edge3214_boundary_edge3211_generated",
                "groups": [
                    {"name": "prefix", "filter": "assistant_boundary_or_generated", "layer": 20, "features": prefix},
                    {
                        "name": "boundary_base_edge3214",
                        "filter": "assistant_boundary",
                        "layer": 20,
                        "features": boundary + generated_base + [edge3214_feature],
                    },
                    {"name": "edge3211", "filter": "generated", "layer": 20, "features": [edge3211_feature]},
                ],
            },
        ]
    )
    for extra_rank in range(3201, 3211):
        if extra_rank <= prefix_top_k:
            continue
        prefix_plus = prefix + [rank_to_feature[extra_rank]]
        variants.append(
            {
                "label": f"top{prefix_top_k}_plus_rank{extra_rank}_edge_cross_success",
                "groups": [
                    {"name": "prefix_plus", "filter": "assistant_boundary_or_generated", "layer": 20, "features": prefix_plus},
                    {
                        "name": "boundary_base_edge3211",
                        "filter": "assistant_boundary",
                        "layer": 20,
                        "features": boundary + generated_base + [edge3211_feature],
                    },
                    {"name": "edge3214", "filter": "generated", "layer": 20, "features": [edge3214_feature]},
                ],
            }
        )
    for extra_rank in extra_probe_ranks:
        if extra_rank <= prefix_top_k:
            continue
        prefix_plus = prefix + [rank_to_feature[extra_rank]]
        variants.append(
            {
                "label": f"top{prefix_top_k}_plus_rank{extra_rank}_edge_cross_extra_probe",
                "groups": [
                    {"name": "prefix_plus", "filter": "assistant_boundary_or_generated", "layer": 20, "features": prefix_plus},
                    {
                        "name": "boundary_base_edge3211",
                        "filter": "assistant_boundary",
                        "layer": 20,
                        "features": boundary + generated_base + [edge3211_feature],
                    },
                    {"name": "edge3214", "filter": "generated", "layer": 20, "features": [edge3214_feature]},
                ],
            }
        )
    return variants


def tensor_groups(variants: list[dict[str, object]], layers: tuple[int, ...], device: str):
    out = []
    count_rows = []
    for variant in variants:
        groups = []
        seen_by_layer: dict[tuple[int, str], set[int]] = {}
        for group in variant["groups"]:
            layer = int(group["layer"])
            if layer not in layers:
                continue
            key = (layer, str(group["filter"]))
            seen = seen_by_layer.setdefault(key, set())
            feature_ids = []
            for feature_id in group["features"]:
                feature_id = int(feature_id)
                if feature_id in seen:
                    continue
                seen.add(feature_id)
                feature_ids.append(feature_id)
            idx = torch.tensor(sorted(feature_ids), dtype=torch.long, device=device)
            groups.append({"name": group["name"], "filter": group["filter"], "layer": layer, "idx": idx})
            count_rows.append(
                {
                    "variant": variant["label"],
                    "group": group["name"],
                    "filter": group["filter"],
                    "layer": layer,
                    "selected_features": int(idx.numel()),
                    "feature_ids": ",".join(str(x) for x in idx.detach().cpu().tolist()),
                }
            )
        out.append({"label": variant["label"], "groups": groups})
    return out, count_rows


def apply_mixed_donor_subset_decode(sae, recipient_out, donor_out, groups, masks):
    donor_f = sae.encode(donor_out)
    patch_sum = torch.zeros_like(recipient_out)
    any_mask = torch.zeros(recipient_out.shape[:-1], dtype=torch.bool, device=recipient_out.device)
    for group in groups:
        idx = group["idx"]
        if idx.numel() == 0:
            continue
        mask = masks[str(group["filter"])].to(device=recipient_out.device)
        donor_sel = donor_f.index_select(-1, idx)
        contrib = donor_sel @ sae.W_dec.index_select(0, idx)
        patch_sum = patch_sum + torch.where(mask.unsqueeze(-1), contrib, torch.zeros_like(contrib))
        any_mask = any_mask | mask
    patched = patch_sum + sae.b_dec.to(device=recipient_out.device, dtype=recipient_out.dtype)
    return torch.where(any_mask.unsqueeze(-1), patched, recipient_out)


@torch.no_grad()
def mixed_timing_generate(donor, recipient, tokenizer, sae, variant, user: str, args) -> str:
    donor_cache = {}
    current_masks: dict[str, torch.Tensor] = {}
    donor_handles = []
    recipient_handles = []

    def donor_hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        donor_cache[20] = tensor.detach()

    def recipient_hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        groups = [group for group in variant["groups"] if int(group["layer"]) == 20]
        patched = apply_mixed_donor_subset_decode(sae, tensor, donor_cache[20], groups, current_masks)
        return replace_first_tensor(output, patched.to(device=tensor.device, dtype=tensor.dtype))

    donor_handles.append(module_mlp(donor, 20, args.output_mode).register_forward_hook(donor_hook))
    recipient_handles.append(module_mlp(recipient, 20, args.output_mode).register_forward_hook(recipient_hook))

    prompt = chat_prompt(tokenizer, user)
    enc = tokenizer(prompt, return_tensors="pt").to(args.device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    prompt_length = int(input_ids.shape[1])
    filters = sorted({str(group["filter"]) for group in variant["groups"]})
    try:
        for _ in range(args.max_new_tokens):
            donor_cache.clear()
            _ = donor(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            current_masks.clear()
            for patch_filter in filters:
                current_masks[patch_filter] = patch_position_mask(
                    tokenizer,
                    input_ids,
                    attention_mask,
                    patch_filter,
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
def evaluate(donor, recipient, tokenizer, sae, variants, prompts, args):
    by_model: dict[str, list[dict[str, object]]] = {}
    for variant in variants:
        model_name = f"mixed_timing_{variant['label']}"
        rows = []
        print(f"[eval] {model_name}", flush=True)
        for row_idx, (split, user) in enumerate(prompts, start=1):
            print(f"[eval] {model_name} prompt {row_idx}/{len(prompts)} ({split})", flush=True)
            text = mixed_timing_generate(donor, recipient, tokenizer, sae, variant, str(user), args)
            record = {
                "model": model_name,
                "split": split,
                "prompt": str(user),
                "text": text,
                "donor_alpha": args.donor_alpha,
                "recipient_alpha": args.recipient_alpha,
                "patch_mode": "mixed_timing_donor_subset_decode",
            }
            record.update(score_record(split, str(user), text))
            rows.append(record)
        by_model[model_name] = rows
    metrics = []
    for model_name, rows in by_model.items():
        row = summarize_generation(model_name, rows)
        row["donor_alpha"] = args.donor_alpha
        row["recipient_alpha"] = args.recipient_alpha
        row["patch_mode"] = "mixed_timing_donor_subset_decode"
        metrics.append(row)
    return metrics, [row for rows in by_model.values() for row in rows]


def write_summary(path: Path, metrics, count_rows, args) -> None:
    lines = [
        "# Gemma-2-2B Linear Merge SAE Mixed-Timing Bundle Patch",
        "",
        f"Donor alpha: `{args.donor_alpha:g}`.",
        f"Recipient alpha: `{args.recipient_alpha:g}`.",
        f"Prompt file: `{args.prompt_jsonl}`.",
        "",
        "## Generation",
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
    lines.extend(["", "## Bundle Groups", "", "| variant | group | filter | layer | selected features |", "|---|---|---|---:|---:|"])
    for row in count_rows:
        lines.append(
            f"| `{row['variant']}` | `{row['group']}` | `{row['filter']}` | "
            f"{row['layer']} | {row['selected_features']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--sae-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    ap.add_argument("--layers", default="20")
    ap.add_argument("--l0-target", type=int, default=80)
    ap.add_argument("--donor-alpha", type=float, default=1.0)
    ap.add_argument("--recipient-alpha", type=float, default=0.75)
    ap.add_argument("--prompt-jsonl", type=Path, required=True)
    ap.add_argument("--rank-csv", type=Path, default=DEFAULT_RANK_CSV)
    ap.add_argument("--prefix-top-k", type=int, default=3210)
    ap.add_argument("--extra-probe-ranks", default="3215,3250,3300,3400,3600,4000")
    ap.add_argument("--variant-labels", default="")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--output-mode", choices=("post_ff_norm", "raw_mlp"), default="post_ff_norm")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    args = ap.parse_args()
    args.layers = tuple(parse_ints(args.layers))
    args.extra_probe_ranks = tuple(parse_ints(args.extra_probe_ranks))
    if args.layers != (20,):
        raise ValueError("mixed timing runner currently supports layer 20 only")
    return args


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    sae_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.sae_dtype]
    cache_dir = os.environ.get("HF_HOME")
    prompts = load_prompt_rows(args.prompt_jsonl)
    rank_to_feature = read_rank_features(args.rank_csv)
    raw_variants = variant_groups(rank_to_feature, args.prefix_top_k, args.extra_probe_ranks)
    if args.variant_labels.strip():
        keep = {item.strip() for item in args.variant_labels.split(",") if item.strip()}
        raw_variants = [variant for variant in raw_variants if str(variant["label"]) in keep]
        missing = sorted(keep - {str(variant["label"]) for variant in raw_variants})
        if missing:
            raise ValueError(f"unknown variant labels: {', '.join(missing)}")
    variants, count_rows = tensor_groups(raw_variants, args.layers, args.device)

    print("[load] tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"], cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[files] selecting GemmaScope MLP SAE", flush=True)
    sae_file = select_sae_file(20, args.l0_target)
    print(f"[files] L20: {sae_file}", flush=True)
    print("[load] SAE", flush=True)
    sae = load_sae(sae_file, device=args.device, dtype=sae_dtype, cache_dir=cache_dir)

    print("[load] base donor for weight construction", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    base_model.eval()
    print("[load] high-alpha donor model", flush=True)
    donor_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    donor_model.eval()
    recipient_params_cpu = {name: param.detach().cpu().clone() for name, param in donor_model.named_parameters()}
    set_linear_merge_weights(donor_model, base_model, recipient_params_cpu, args.donor_alpha, args.device)
    print("[load] low-alpha recipient model", flush=True)
    recipient_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    recipient_model.eval()
    set_linear_merge_weights(recipient_model, base_model, recipient_params_cpu, args.recipient_alpha, args.device)
    del base_model
    torch.cuda.empty_cache()

    metrics, records = evaluate(donor_model, recipient_model, tokenizer, sae, variants, prompts, args)
    metrics_path = args.result_dir / "gemma2_2b_linear_merge_sae_mixed_timing_bundle_patch_metrics.csv"
    records_path = args.result_dir / "gemma2_2b_linear_merge_sae_mixed_timing_bundle_patch_records.jsonl"
    counts_path = args.result_dir / "gemma2_2b_linear_merge_sae_mixed_timing_bundle_patch_counts.csv"
    summary_path = args.result_dir / "GEMMA2_2B_LINEAR_MERGE_SAE_MIXED_TIMING_BUNDLE_PATCH_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, metrics)
    write_jsonl(records_path, records)
    write_csv(counts_path, count_rows)
    write_summary(summary_path, metrics, count_rows, args)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "sae_repo": SAE_REPO,
                "sae_file": sae_file,
                "rank_csv": str(args.rank_csv),
                "prefix_top_k": args.prefix_top_k,
                "extra_probe_ranks": args.extra_probe_ranks,
                "donor_alpha": args.donor_alpha,
                "recipient_alpha": args.recipient_alpha,
                "variants": raw_variants,
                "prompt_jsonl": str(args.prompt_jsonl),
                "outputs": {
                    "metrics": str(metrics_path),
                    "records": str(records_path),
                    "counts": str(counts_path),
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
