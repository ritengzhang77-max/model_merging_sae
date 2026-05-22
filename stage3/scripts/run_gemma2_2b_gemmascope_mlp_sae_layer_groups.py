#!/usr/bin/env python3
"""Layer-group localization for GemmaScope MLP SAE feature repairs."""

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
sys.path.insert(0, str(ROOT / "stage2" / "scripts"))
sys.path.insert(0, str(ROOT / "stage3" / "scripts"))
from run_gemma2_2b_activation_patch_target_loss import MODEL_IDS  # noqa: E402
from run_gemma2_2b_gemmascope_mlp_sae_feature_subsets import (  # noqa: E402
    PATCH_TOKEN_FILTERS,
    collect_feature_stats,
    parse_variant,
    run_generation,
    select_indices,
    write_csv,
    write_jsonl,
)
from validate_gemma2_2b_gemmascope_mlp_sae import SAE_REPO, load_sae, select_sae_file  # noqa: E402


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_gemmascope_mlp_sae_layer_groups"
DEFAULT_GROUPS = "all:12-20;early:12-14;mid:15-17;late:18-20"
DEFAULT_VARIANTS = (
    "full_decode",
    "delta_add_all",
    "mix_decode_delta_abs_k1024",
    "mix_decode_random_active_k1024",
    "mix_decode_random_active_k2048",
)


def parse_layer_spec(raw: str) -> tuple[int, ...]:
    layers: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start, end = int(left), int(right)
            if end < start:
                raise ValueError(f"invalid descending layer range: {item}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(item))
    if not layers:
        raise ValueError(f"empty layer spec: {raw!r}")
    return tuple(dict.fromkeys(layers))


def parse_groups(raw: str) -> list[tuple[str, tuple[int, ...]]]:
    groups = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"group must be name:layers, got {chunk!r}")
        name, layer_spec = chunk.split(":", 1)
        clean_name = name.strip().replace(" ", "_")
        if not clean_name:
            raise ValueError(f"empty group name in {chunk!r}")
        groups.append((clean_name, parse_layer_spec(layer_spec)))
    if not groups:
        raise ValueError("no layer groups specified")
    return groups


def prefixed_rows(group: str, layers: tuple[int, ...], metrics, records, include_baselines: bool):
    out_metrics = []
    out_records = []
    layer_spec = ",".join(str(x) for x in layers)
    for row in metrics:
        row = dict(row)
        model = str(row["model"])
        if model in {"base", "abliterated"}:
            if not include_baselines:
                continue
            row["layer_group"] = "baseline"
            row["layers"] = ""
        else:
            row["layer_group"] = group
            row["layers"] = layer_spec
            row["model"] = f"{group}__{model}"
        out_metrics.append(row)
    for row in records:
        row = dict(row)
        model = str(row["model"])
        if model in {"base", "abliterated"}:
            if not include_baselines:
                continue
            row["layer_group"] = "baseline"
            row["layers"] = ""
        else:
            row["layer_group"] = group
            row["layers"] = layer_spec
            row["model"] = f"{group}__{model}"
        out_records.append(row)
    return out_metrics, out_records


def write_summary(path: Path, metrics: list[dict[str, object]], groups, args) -> None:
    lines = [
        "# Gemma-2-2B GemmaScope MLP SAE Layer-Group Localization",
        "",
        f"Feature-selection prompts: `{args.basis_start}:{args.basis_start + args.basis_examples_per_split}` per split.",
        f"Feature-selection token filter: `{args.feature_token_filter}`.",
        f"Patch token filter: `{args.patch_token_filter}`.",
        f"Evaluation prompts: `{args.eval_start}:{args.eval_start + args.examples_per_split}` per split.",
        "",
        "Layer groups:",
    ]
    for name, layers in groups:
        lines.append(f"- `{name}`: `{','.join(str(x) for x in layers)}`")
    lines.extend(
        [
            "",
            "## Generation",
            "",
            "| layer group | variant | harmful clean | harmful attempt | unsafe continuation | benign helpful | benign over-refusal |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics:
        model = str(row["model"])
        if model in {"base", "abliterated"}:
            variant = model
        else:
            variant = model.split("__", 1)[1].replace("feature_subset_", "")
        lines.append(
            f"| `{row.get('layer_group', '')}` | `{variant}` | "
            f"{float(row.get('harmful_ok_rate', 0.0)):.3f} | "
            f"{float(row.get('harmful_attempted_refusal_rate', 0.0)):.3f} | "
            f"{float(row.get('harmful_unsafe_continuation_rate', 0.0)):.3f} | "
            f"{float(row.get('benign_ok_rate', 0.0)):.3f} | "
            f"{float(row.get('benign_over_refusal_rate', 0.0)):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- A layer group is independently sufficient only if its full decoded SAE and selected top-delta feature patch recover refusal while preserving benign helpfulness.",
            "- If only the all-layer group passes, the repair is distributed across the 12-20 range or requires cross-layer composition.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--sae-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    ap.add_argument("--groups", default=DEFAULT_GROUPS)
    ap.add_argument("--l0-target", type=int, default=80)
    ap.add_argument("--basis-start", type=int, default=0)
    ap.add_argument("--basis-examples-per-split", type=int, default=4)
    ap.add_argument("--eval-start", type=int, default=4)
    ap.add_argument("--examples-per-split", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    ap.add_argument("--output-mode", choices=("post_ff_norm", "raw_mlp"), default="post_ff_norm")
    ap.add_argument("--feature-token-filter", choices=("all", "contentish"), default="all")
    ap.add_argument("--patch-token-filter", choices=PATCH_TOKEN_FILTERS, default="all")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    groups = parse_groups(args.groups)
    all_layers = tuple(sorted({layer for _name, layers in groups for layer in layers}))
    variants = [parse_variant(item.strip()) for item in args.variants.split(",") if item.strip()]
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

    print("[features] collecting calibration feature stats", flush=True)
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

    all_metrics: list[dict[str, object]] = []
    all_records: list[dict[str, object]] = []
    all_counts: list[dict[str, object]] = []
    for group_idx, (group_name, layers) in enumerate(groups):
        print(f"[group] {group_name}: {','.join(str(x) for x in layers)}", flush=True)
        selected, counts = select_indices(stats, layers, variants, args.random_seed + group_idx, args.device)
        for row in counts:
            row["layer_group"] = group_name
            row["layers"] = ",".join(str(x) for x in layers)
        metrics, records = run_generation(
            donor,
            recipient,
            tokenizer,
            saes,
            selected,
            variants,
            layers=layers,
            examples_per_split=args.examples_per_split,
            prompt_start=args.eval_start,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            output_mode=args.output_mode,
            skip_baselines=group_idx > 0,
            patch_token_filter=args.patch_token_filter,
        )
        metrics, records = prefixed_rows(group_name, layers, metrics, records, include_baselines=group_idx == 0)
        all_metrics.extend(metrics)
        all_records.extend(records)
        all_counts.extend(counts)

    metrics_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_layer_group_metrics.csv"
    records_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_layer_group_records.jsonl"
    counts_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_layer_group_counts.csv"
    summary_path = args.result_dir / "GEMMA2_2B_GEMMASCOPE_MLP_SAE_LAYER_GROUP_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, all_metrics)
    write_jsonl(records_path, all_records)
    write_csv(counts_path, all_counts)
    write_summary(summary_path, all_metrics, groups, args)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "sae_repo": SAE_REPO,
                "sae_files": {str(k): v for k, v in files.items()},
                "groups": {name: list(layers) for name, layers in groups},
                "output_mode": args.output_mode,
                "feature_token_filter": args.feature_token_filter,
                "patch_token_filter": args.patch_token_filter,
                "basis_start": args.basis_start,
                "basis_examples_per_split": args.basis_examples_per_split,
                "eval_start": args.eval_start,
                "examples_per_split": args.examples_per_split,
                "variants": [str(v["label"]) for v in variants],
                "random_seed": args.random_seed,
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
