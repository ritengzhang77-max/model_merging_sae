#!/usr/bin/env python3
"""Random-active seed controls for GemmaScope MLP SAE feature patches."""

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
    parse_variant,
    run_generation,
    select_indices,
    write_csv,
    write_jsonl,
)
from run_gemma2_2b_gemmascope_mlp_sae_layer_groups import parse_groups  # noqa: E402
from validate_gemma2_2b_gemmascope_mlp_sae import SAE_REPO, load_sae, select_sae_file  # noqa: E402


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_gemmascope_mlp_sae_random_seed_controls"
DEFAULT_GROUPS = "all:12-20;mid_late:15-20;early_late:12-14,18-20"
DEFAULT_DETERMINISTIC_VARIANTS = "full_decode,delta_add_all,mix_decode_delta_abs_k1024,mix_decode_delta_abs_k2048"
DEFAULT_RANDOM_VARIANTS = "mix_decode_random_active_k1024,mix_decode_random_active_k2048"


def parse_ints(raw: str) -> tuple[int, ...]:
    vals = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not vals:
        raise ValueError(f"empty integer list: {raw!r}")
    return vals


def annotate_rows(rows, *, group_name: str, layers: tuple[int, ...], eval_start: int, random_seed: int | None, condition: str):
    layer_spec = ",".join(str(x) for x in layers)
    out = []
    for row in rows:
        row = dict(row)
        model = str(row["model"])
        row["eval_start"] = eval_start
        row["eval_slice"] = f"{eval_start}:{eval_start + row_eval_count(row)}"
        row["condition"] = condition
        row["random_seed"] = "" if random_seed is None else random_seed
        if model in {"base", "abliterated"}:
            row["layer_group"] = "baseline"
            row["layers"] = ""
        else:
            row["layer_group"] = group_name
            row["layers"] = layer_spec
            seed_tag = "det" if random_seed is None else f"seed{random_seed}"
            row["model"] = f"{group_name}__{seed_tag}__{model}"
        out.append(row)
    return out


def row_eval_count(row: dict[str, object]) -> int:
    # Records do not store the slice length; summaries are grouped by a fixed
    # examples-per-split value. This placeholder is overwritten in main.
    return int(row.get("_examples_per_split", 0))


def annotate_with_count(rows, examples_per_split: int):
    for row in rows:
        row["_examples_per_split"] = examples_per_split
    return rows


def aggregate_random(metrics: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str, str], list[float]] = {}
    unsafe: dict[tuple[str, str, str], list[float]] = {}
    benign: dict[tuple[str, str, str], list[float]] = {}
    for row in metrics:
        if row.get("condition") != "random":
            continue
        variant = str(row["model"]).split("__", 2)[-1].replace("feature_subset_", "")
        key = (str(row["eval_slice"]), str(row["layer_group"]), variant)
        buckets.setdefault(key, []).append(float(row["harmful_ok_rate"]))
        unsafe.setdefault(key, []).append(float(row["harmful_unsafe_continuation_rate"]))
        benign.setdefault(key, []).append(float(row["benign_ok_rate"]))
    rows = []
    for key, vals in sorted(buckets.items()):
        eval_slice, group, variant = key
        rows.append(
            {
                "eval_slice": eval_slice,
                "layer_group": group,
                "variant": variant,
                "n_seeds": len(vals),
                "harmful_ok_mean": sum(vals) / len(vals),
                "harmful_ok_min": min(vals),
                "harmful_ok_max": max(vals),
                "unsafe_mean": sum(unsafe[key]) / len(unsafe[key]),
                "benign_ok_mean": sum(benign[key]) / len(benign[key]),
            }
        )
    return rows


def write_summary(path: Path, metrics: list[dict[str, object]], random_agg: list[dict[str, object]], groups, eval_starts, args) -> None:
    deterministic = [row for row in metrics if row.get("condition") == "deterministic"]
    lines = [
        "# Gemma-2-2B GemmaScope MLP SAE Random-Seed Controls",
        "",
        f"Feature-selection prompts: `{args.basis_start}:{args.basis_start + args.basis_examples_per_split}` per split.",
        f"Feature-selection token filter: `{args.feature_token_filter}`.",
        f"Evaluation starts: `{','.join(str(x) for x in eval_starts)}` with `{args.examples_per_split}` prompts per split.",
        f"Random seeds: `{','.join(str(x) for x in parse_ints(args.random_seeds))}`.",
        "",
        "Layer groups:",
    ]
    for name, layers in groups:
        lines.append(f"- `{name}`: `{','.join(str(x) for x in layers)}`")
    lines.extend(
        [
            "",
            "## Deterministic Variants",
            "",
            "| eval slice | layer group | variant | harmful clean | unsafe continuation | benign helpful |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in deterministic:
        model = str(row["model"])
        if model in {"base", "abliterated"}:
            variant = model
        else:
            variant = model.split("__", 2)[-1].replace("feature_subset_", "")
        lines.append(
            f"| `{row['eval_slice']}` | `{row['layer_group']}` | `{variant}` | "
            f"{float(row['harmful_ok_rate']):.3f} | "
            f"{float(row['harmful_unsafe_continuation_rate']):.3f} | "
            f"{float(row['benign_ok_rate']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Random Active-Feature Controls",
            "",
            "| eval slice | layer group | variant | seeds | harmful mean | harmful min-max | unsafe mean | benign mean |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in random_agg:
        lines.append(
            f"| `{row['eval_slice']}` | `{row['layer_group']}` | `{row['variant']}` | "
            f"{int(row['n_seeds'])} | {float(row['harmful_ok_mean']):.3f} | "
            f"{float(row['harmful_ok_min']):.3f}-{float(row['harmful_ok_max']):.3f} | "
            f"{float(row['unsafe_mean']):.3f} | {float(row['benign_ok_mean']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- Top-delta feature patches are stronger evidence if they beat the mean and best random-active seed at the same feature budget.",
            "- If random controls match top-delta often, the sparse-feature claim should be weakened to a broad active-subspace effect.",
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
    ap.add_argument("--eval-starts", default="4,8")
    ap.add_argument("--examples-per-split", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--deterministic-variants", default=DEFAULT_DETERMINISTIC_VARIANTS)
    ap.add_argument("--random-variants", default=DEFAULT_RANDOM_VARIANTS)
    ap.add_argument("--random-seeds", default="0,1,2,3,4")
    ap.add_argument("--output-mode", choices=("post_ff_norm", "raw_mlp"), default="post_ff_norm")
    ap.add_argument("--feature-token-filter", choices=("all", "contentish"), default="all")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    groups = parse_groups(args.groups)
    eval_starts = parse_ints(args.eval_starts)
    random_seeds = parse_ints(args.random_seeds)
    all_layers = tuple(sorted({layer for _name, layers in groups for layer in layers}))
    deterministic_variants = [parse_variant(item.strip()) for item in args.deterministic_variants.split(",") if item.strip()]
    random_variants = [parse_variant(item.strip()) for item in args.random_variants.split(",") if item.strip()]
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
    for eval_idx, eval_start in enumerate(eval_starts):
        print(f"[eval-slice] {eval_start}:{eval_start + args.examples_per_split}", flush=True)
        for group_idx, (group_name, layers) in enumerate(groups):
            print(f"[group] {group_name}: {','.join(str(x) for x in layers)}", flush=True)
            selected, counts = select_indices(stats, layers, deterministic_variants, 0, args.device)
            metrics, records = run_generation(
                donor,
                recipient,
                tokenizer,
                saes,
                selected,
                deterministic_variants,
                layers=layers,
                examples_per_split=args.examples_per_split,
                prompt_start=eval_start,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                output_mode=args.output_mode,
                skip_baselines=not (eval_idx == 0 and group_idx == 0),
            )
            metrics = annotate_rows(
                annotate_with_count(metrics, args.examples_per_split),
                group_name=group_name,
                layers=layers,
                eval_start=eval_start,
                random_seed=None,
                condition="deterministic",
            )
            records = annotate_rows(
                annotate_with_count(records, args.examples_per_split),
                group_name=group_name,
                layers=layers,
                eval_start=eval_start,
                random_seed=None,
                condition="deterministic",
            )
            for row in counts:
                row["eval_start"] = eval_start
                row["eval_slice"] = f"{eval_start}:{eval_start + args.examples_per_split}"
                row["layer_group"] = group_name
                row["layers"] = ",".join(str(x) for x in layers)
                row["condition"] = "deterministic"
                row["random_seed"] = ""
            all_metrics.extend(metrics)
            all_records.extend(records)
            all_counts.extend(counts)

            for seed in random_seeds:
                selected, counts = select_indices(stats, layers, random_variants, seed, args.device)
                metrics, records = run_generation(
                    donor,
                    recipient,
                    tokenizer,
                    saes,
                    selected,
                    random_variants,
                    layers=layers,
                    examples_per_split=args.examples_per_split,
                    prompt_start=eval_start,
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                    output_mode=args.output_mode,
                    skip_baselines=True,
                )
                metrics = annotate_rows(
                    annotate_with_count(metrics, args.examples_per_split),
                    group_name=group_name,
                    layers=layers,
                    eval_start=eval_start,
                    random_seed=seed,
                    condition="random",
                )
                records = annotate_rows(
                    annotate_with_count(records, args.examples_per_split),
                    group_name=group_name,
                    layers=layers,
                    eval_start=eval_start,
                    random_seed=seed,
                    condition="random",
                )
                for row in counts:
                    row["eval_start"] = eval_start
                    row["eval_slice"] = f"{eval_start}:{eval_start + args.examples_per_split}"
                    row["layer_group"] = group_name
                    row["layers"] = ",".join(str(x) for x in layers)
                    row["condition"] = "random"
                    row["random_seed"] = seed
                all_metrics.extend(metrics)
                all_records.extend(records)
                all_counts.extend(counts)

    for row in all_metrics + all_records:
        row.pop("_examples_per_split", None)

    random_agg = aggregate_random(all_metrics)
    metrics_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_random_seed_metrics.csv"
    random_agg_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_random_seed_aggregate.csv"
    records_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_random_seed_records.jsonl"
    counts_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_random_seed_counts.csv"
    summary_path = args.result_dir / "GEMMA2_2B_GEMMASCOPE_MLP_SAE_RANDOM_SEED_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, all_metrics)
    write_csv(random_agg_path, random_agg)
    write_jsonl(records_path, all_records)
    write_csv(counts_path, all_counts)
    write_summary(summary_path, all_metrics, random_agg, groups, eval_starts, args)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "sae_repo": SAE_REPO,
                "sae_files": {str(k): v for k, v in files.items()},
                "groups": {name: list(layers) for name, layers in groups},
                "output_mode": args.output_mode,
                "feature_token_filter": args.feature_token_filter,
                "basis_start": args.basis_start,
                "basis_examples_per_split": args.basis_examples_per_split,
                "eval_starts": list(eval_starts),
                "examples_per_split": args.examples_per_split,
                "deterministic_variants": [str(v["label"]) for v in deterministic_variants],
                "random_variants": [str(v["label"]) for v in random_variants],
                "random_seeds": list(random_seeds),
                "outputs": {
                    "metrics": str(metrics_path),
                    "random_aggregate": str(random_agg_path),
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
