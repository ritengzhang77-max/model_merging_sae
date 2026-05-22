#!/usr/bin/env python3
"""Export top GemmaScope MLP-SAE feature IDs and activation snippets."""

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
sys.path.insert(0, str(ROOT / "stage2" / "scripts"))
sys.path.insert(0, str(ROOT / "stage3" / "scripts"))
from run_gemma2_2b_activation_patch_target_loss import MODEL_IDS  # noqa: E402
from run_gemma2_2b_gemmascope_mlp_sae_feature_subsets import (  # noqa: E402
    collect_feature_stats,
    make_mlp_cache_hooks,
    make_prompt_rows,
    parse_variant,
    select_indices,
    token_filter_mask,
    write_csv,
    write_jsonl,
)
from run_gemma2_2b_gemmascope_mlp_sae_layer_groups import parse_groups  # noqa: E402
from validate_gemma2_2b_gemmascope_mlp_sae import SAE_REPO, load_sae, remove_hooks, select_sae_file  # noqa: E402


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_gemmascope_mlp_sae_feature_audit"
DEFAULT_GROUPS = "all:12-20;mid_late:15-20"
DEFAULT_VARIANTS = "mix_decode_delta_abs_k1024,mix_decode_delta_abs_k2048"


def token_context(tokenizer, ids: list[int], pos: int, radius: int) -> str:
    left = max(0, pos - radius)
    right = min(len(ids), pos + radius + 1)
    return tokenizer.decode(ids[left:right], skip_special_tokens=False).replace("\n", "\\n")


def one_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")


def feature_score_row(group_name: str, layers: tuple[int, ...], layer: int, feature_id: int, rank: int, stats) -> dict[str, object]:
    row = stats[layer]
    harm_tokens = max(int(row["harm_tokens"]), 1)
    benign_tokens = max(int(row["benign_tokens"]), 1)
    harm_delta = float(row["harm_delta_abs"][feature_id].item())
    benign_delta = float(row["benign_delta_abs"][feature_id].item())
    return {
        "group": group_name,
        "layers": ",".join(str(x) for x in layers),
        "layer": layer,
        "feature_id": feature_id,
        "rank": rank,
        "harm_delta_abs_sum": harm_delta,
        "harm_delta_abs_mean": harm_delta / harm_tokens,
        "benign_delta_abs_sum": benign_delta,
        "benign_delta_abs_mean": benign_delta / benign_tokens,
        "harm_minus_benign_delta_mean": harm_delta / harm_tokens - benign_delta / benign_tokens,
        "harm_donor_mean": float(row["harm_donor"][feature_id].item()) / harm_tokens,
        "active_count": float(row["active"][feature_id].item()),
    }


def select_feature_rows(stats, groups, variants, *, random_seed: int, device: str, top_n_per_layer: int):
    feature_rows = []
    selected_by_layer: dict[int, set[int]] = {}
    selection_by_group: dict[str, dict[int, list[int]]] = {}
    for group_name, layers in groups:
        selected, _counts = select_indices(stats, layers, variants, random_seed, device)
        selection_by_group[group_name] = {}
        first_label = str(variants[0]["label"])
        for layer in layers:
            idx = selected[first_label][layer]
            if idx is None:
                continue
            ids = [int(x) for x in idx[:top_n_per_layer].detach().cpu().tolist()]
            selection_by_group[group_name][layer] = ids
            selected_by_layer.setdefault(layer, set()).update(ids)
            for rank, feature_id in enumerate(ids, start=1):
                feature_rows.append(feature_score_row(group_name, layers, layer, feature_id, rank, stats))
    return feature_rows, {layer: sorted(vals) for layer, vals in selected_by_layer.items()}, selection_by_group


def update_events(events: dict[tuple[int, int, str], list[dict[str, object]]], key, event, keep: int):
    bucket = events.setdefault(key, [])
    bucket.append(event)
    bucket.sort(key=lambda row: float(row["value"]), reverse=True)
    if len(bucket) > keep:
        del bucket[keep:]


@torch.no_grad()
def collect_feature_events(
    donor,
    recipient,
    tokenizer,
    saes,
    selected_by_layer: dict[int, list[int]],
    *,
    examples_per_split: int,
    prompt_start: int,
    max_length: int,
    device: str,
    output_mode: str,
    event_token_filter: str,
    event_k: int,
    context_radius: int,
    top_per_prompt: int,
):
    rows = make_prompt_rows(tokenizer, examples_per_split, max_length, prompt_start)
    layers = tuple(sorted(selected_by_layer))
    donor_cache, donor_handles = make_mlp_cache_hooks(donor, layers, output_mode)
    recipient_cache, recipient_handles = make_mlp_cache_hooks(recipient, layers, output_mode)
    events: dict[tuple[int, int, str], list[dict[str, object]]] = {}
    try:
        for prompt_idx, row in enumerate(rows):
            batch = {key: value.to(device) for key, value in row["enc"].items()}
            for layer in layers:
                donor_cache[layer].clear()
                recipient_cache[layer].clear()
            _ = donor(**batch, use_cache=False)
            _ = recipient(**batch, use_cache=False)
            ids = [int(x) for x in batch["input_ids"][0].detach().cpu().tolist()]
            valid_positions = token_filter_mask(tokenizer, batch["input_ids"], batch["attention_mask"], event_token_filter)[0].detach().cpu()
            for layer in layers:
                feature_ids = selected_by_layer[layer]
                if not feature_ids:
                    continue
                idx = torch.tensor(feature_ids, device=device, dtype=torch.long)
                donor_out = donor_cache[layer][0][0]
                recipient_out = recipient_cache[layer][0][0]
                donor_f = saes[layer].encode(donor_out).index_select(-1, idx).float().cpu()
                recipient_f = saes[layer].encode(recipient_out).index_select(-1, idx).float().cpu()
                delta_abs = (donor_f - recipient_f).abs()
                for col, feature_id in enumerate(feature_ids):
                    for metric, raw_values in (("donor_activation", donor_f[:, col]), ("abs_delta", delta_abs[:, col])):
                        values = raw_values.clone()
                        values[~valid_positions] = 0.0
                        k = min(top_per_prompt, int(values.numel()))
                        if k <= 0:
                            continue
                        vals, positions = torch.topk(values, k)
                        for value, pos_tensor in zip(vals.tolist(), positions.tolist()):
                            if value <= 0:
                                continue
                            pos = int(pos_tensor)
                            event = {
                                "layer": layer,
                                "feature_id": feature_id,
                                "metric": metric,
                                "value": float(value),
                                "split": row["split"],
                                "prompt_index": prompt_start + prompt_idx,
                                "prompt": row["user"],
                                "token_position": pos,
                                "token_id": ids[pos],
                                "token": one_token(tokenizer, ids[pos]),
                                "context": token_context(tokenizer, ids, pos, context_radius),
                            }
                            update_events(events, (layer, feature_id, metric), event, event_k)
    finally:
        remove_hooks(donor_handles)
        remove_hooks(recipient_handles)
    out = []
    for key in sorted(events):
        out.extend(events[key])
    return out


def write_summary(path: Path, feature_rows, event_rows, selection_by_group, args) -> None:
    by_group_layer: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in feature_rows:
        by_group_layer.setdefault((str(row["group"]), int(row["layer"])), []).append(row)
    lines = [
        "# Gemma-2-2B GemmaScope MLP SAE Feature Audit",
        "",
        f"Feature-selection prompts: `{args.basis_start}:{args.basis_start + args.basis_examples_per_split}` per split.",
        f"Feature-selection token filter: `{args.feature_token_filter}`.",
        f"Audit prompts: `{args.audit_start}:{args.audit_start + args.audit_examples_per_split}` per split.",
        f"Event token filter: `{args.event_token_filter}`.",
        f"Top exported features per layer: `{args.top_n_per_layer}`.",
        "",
        "## Top Features By Group And Layer",
        "",
        "| group | layer | top feature IDs | top harm-delta mean values |",
        "|---|---:|---|---|",
    ]
    for (group, layer), rows in sorted(by_group_layer.items()):
        top = sorted(rows, key=lambda row: int(row["rank"]))[:8]
        ids = ", ".join(str(row["feature_id"]) for row in top)
        vals = ", ".join(f"{float(row['harm_delta_abs_mean']):.3f}" for row in top)
        lines.append(f"| `{group}` | {layer} | `{ids}` | `{vals}` |")
    lines.extend(
        [
            "",
            "## Event Rows",
            "",
            f"- Exported `{len(feature_rows)}` feature score rows.",
            f"- Exported `{len(event_rows)}` top activation/delta event rows.",
            "- Event rows include exact feature IDs, token IDs, token text, prompt split, prompt text, and local token context.",
            "",
            "## Caveat",
            "",
            "- This file does not assign semantic feature labels. It is an audit substrate for manual or automated interpretation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--sae-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    ap.add_argument("--groups", default=DEFAULT_GROUPS)
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--top-n-per-layer", type=int, default=64)
    ap.add_argument("--l0-target", type=int, default=80)
    ap.add_argument("--basis-start", type=int, default=0)
    ap.add_argument("--basis-examples-per-split", type=int, default=4)
    ap.add_argument("--audit-start", type=int, default=0)
    ap.add_argument("--audit-examples-per-split", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--event-k", type=int, default=5)
    ap.add_argument("--top-per-prompt", type=int, default=2)
    ap.add_argument("--context-radius", type=int, default=8)
    ap.add_argument("--output-mode", choices=("post_ff_norm", "raw_mlp"), default="post_ff_norm")
    ap.add_argument("--feature-token-filter", choices=("all", "contentish"), default="all")
    ap.add_argument("--event-token-filter", choices=("all", "contentish"), default="all")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    groups = parse_groups(args.groups)
    variants = [parse_variant(item.strip()) for item in args.variants.split(",") if item.strip()]
    all_layers = tuple(sorted({layer for _name, layers in groups for layer in layers}))
    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    sae_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.sae_dtype]
    cache_dir = os.environ.get("HF_HOME")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[files] selecting GemmaScope MLP SAEs", flush=True)
    files = {layer: select_sae_file(layer, args.l0_target) for layer in all_layers}
    for layer, filename in files.items():
        print(f"[files] L{layer}: {filename}", flush=True)
    print("[load] SAEs", flush=True)
    saes = {layer: load_sae(filename, device=args.device, dtype=sae_dtype, cache_dir=cache_dir) for layer, filename in files.items()}
    print("[load] base donor", flush=True)
    donor = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=model_dtype).to(args.device)
    donor.eval()
    print("[load] abliterated recipient", flush=True)
    recipient = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype).to(args.device)
    recipient.eval()

    print("[features] collecting feature scores", flush=True)
    stats = collect_feature_stats(
        donor,
        recipient,
        tokenizer,
        saes,
        layers=all_layers,
        examples_per_split=args.basis_examples_per_split,
        batch_size=args.batch_size,
        max_length=args.max_length,
        prompt_start=args.basis_start,
        device=args.device,
        output_mode=args.output_mode,
        feature_token_filter=args.feature_token_filter,
    )
    feature_rows, selected_by_layer, selection_by_group = select_feature_rows(
        stats,
        groups,
        variants,
        random_seed=args.random_seed,
        device=args.device,
        top_n_per_layer=args.top_n_per_layer,
    )
    print("[events] collecting top token events", flush=True)
    event_rows = collect_feature_events(
        donor,
        recipient,
        tokenizer,
        saes,
        selected_by_layer,
        examples_per_split=args.audit_examples_per_split,
        prompt_start=args.audit_start,
        max_length=args.max_length,
        device=args.device,
        output_mode=args.output_mode,
        event_token_filter=args.event_token_filter,
        event_k=args.event_k,
        context_radius=args.context_radius,
        top_per_prompt=args.top_per_prompt,
    )

    feature_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_feature_scores.csv"
    event_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_feature_events.jsonl"
    summary_path = args.result_dir / "GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE_AUDIT_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(feature_path, feature_rows)
    write_jsonl(event_path, event_rows)
    write_summary(summary_path, feature_rows, event_rows, selection_by_group, args)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "sae_repo": SAE_REPO,
                "sae_files": {str(k): v for k, v in files.items()},
                "groups": {name: list(layers) for name, layers in groups},
                "variants": [str(v["label"]) for v in variants],
                "top_n_per_layer": args.top_n_per_layer,
                "output_mode": args.output_mode,
                "feature_token_filter": args.feature_token_filter,
                "event_token_filter": args.event_token_filter,
                "basis_start": args.basis_start,
                "basis_examples_per_split": args.basis_examples_per_split,
                "audit_start": args.audit_start,
                "audit_examples_per_split": args.audit_examples_per_split,
                "event_k": args.event_k,
                "outputs": {
                    "feature_scores": str(feature_path),
                    "feature_events": str(event_path),
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
