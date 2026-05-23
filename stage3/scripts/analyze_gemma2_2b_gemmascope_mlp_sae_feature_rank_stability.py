#!/usr/bin/env python3
"""Rank-stability audit for named GemmaScope MLP-SAE feature IDs."""

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
from screen_chat_merge_candidate import BENIGN_PROMPTS, HARMFUL_PROMPTS  # noqa: E402
from run_gemma2_2b_gemmascope_mlp_sae_feature_subsets import collect_feature_stats  # noqa: E402
from validate_gemma2_2b_gemmascope_mlp_sae import (  # noqa: E402
    SAE_REPO,
    load_sae,
    parse_ints,
    select_sae_file,
)


DEFAULT_RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_gemmascope_mlp_sae_feature_rank_stability_v0"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--sae-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    ap.add_argument("--layers", default="19")
    ap.add_argument("--feature-ids", default="16048")
    ap.add_argument("--basis-starts", default="0,4,8")
    ap.add_argument("--basis-examples-per-split", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--l0-target", type=int, default=80)
    ap.add_argument("--feature-token-filter", choices=("all", "contentish"), default="all")
    ap.add_argument("--output-mode", choices=("post_ff_norm", "raw_mlp"), default="post_ff_norm")
    ap.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    return ap.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, object]], *, feature_token_filter: str) -> None:
    lines = [
        "# GemmaScope MLP-SAE Feature Rank Stability",
        "",
        f"Feature-selection token filter: `{feature_token_filter}`.",
        "",
        "Ranks are 1-indexed within the layer by harmful donor-recipient absolute activation delta.",
        "",
        "| basis slice | layer | feature ID | harm-delta rank | harm delta | benign delta | harm - benign | active count |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['basis_start']}:{row['basis_end']}` | {row['layer']} | {row['feature_id']} | "
            f"{row['harm_delta_abs_rank']} | {row['harm_delta_abs']:.6f} | "
            f"{row['benign_delta_abs']:.6f} | {row['harm_minus_benign_delta']:.6f} | "
            f"{row['active_count']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "- A feature that is causal under one calibration slice is more convincing if it also ranks highly under neighboring slices.",
            "- A feature that only ranks highly under one slice should be treated as a prompt-family-specific lead, not a stable refusal mechanism.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rank_feature(score: torch.Tensor, feature_id: int) -> int:
    value = score[feature_id]
    return int((score > value).sum().item()) + 1


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_ints(args.layers)
    feature_ids = parse_ints(args.feature_ids)
    basis_starts = parse_ints(args.basis_starts)
    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    sae_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.sae_dtype]
    cache_dir = os.environ.get("HF_HOME")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    files = {layer: select_sae_file(layer, args.l0_target) for layer in layers}
    print("[load] SAEs", flush=True)
    saes = {layer: load_sae(filename, device=args.device, dtype=sae_dtype, cache_dir=cache_dir) for layer, filename in files.items()}
    print("[load] base donor", flush=True)
    donor = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=model_dtype).to(args.device)
    donor.eval()
    print("[load] abliterated recipient", flush=True)
    recipient = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype).to(args.device)
    recipient.eval()

    rows: list[dict[str, object]] = []
    for basis_start in basis_starts:
        print(f"[features] basis {basis_start}:{basis_start + args.basis_examples_per_split}", flush=True)
        stats = collect_feature_stats(
            donor,
            recipient,
            tokenizer,
            saes,
            layers=layers,
            examples_per_split=args.basis_examples_per_split,
            batch_size=args.batch_size,
            max_length=args.max_length,
            prompt_start=basis_start,
            feature_token_filter=args.feature_token_filter,
            device=args.device,
            output_mode=args.output_mode,
        )
        for layer in layers:
            row = stats[layer]
            harm_tokens = max(int(row["harm_tokens"]), 1)
            benign_tokens = max(int(row["benign_tokens"]), 1)
            harm_mean = row["harm_delta_abs"] / harm_tokens
            benign_mean = row["benign_delta_abs"] / benign_tokens
            for feature_id in feature_ids:
                if feature_id >= int(row["harm_delta_abs"].numel()):
                    raise ValueError(f"feature ID {feature_id} is outside layer {layer} feature dimension {row['harm_delta_abs'].numel()}")
                rows.append(
                    {
                        "basis_start": basis_start,
                        "basis_end": basis_start + args.basis_examples_per_split,
                        "layer": layer,
                        "feature_id": feature_id,
                        "harm_delta_abs": float(harm_mean[feature_id].item()),
                        "benign_delta_abs": float(benign_mean[feature_id].item()),
                        "harm_minus_benign_delta": float((harm_mean[feature_id] - benign_mean[feature_id]).item()),
                        "harm_delta_abs_rank": rank_feature(row["harm_delta_abs"], feature_id),
                        "active_count": float(row["active"][feature_id].item()),
                    }
                )

    csv_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_feature_rank_stability.csv"
    summary_path = args.result_dir / "GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE_RANK_STABILITY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(csv_path, rows)
    write_summary(summary_path, rows, feature_token_filter=args.feature_token_filter)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "sae_repo": SAE_REPO,
                "sae_files": {str(k): v for k, v in files.items()},
                "layers": list(layers),
                "feature_ids": feature_ids,
                "basis_starts": basis_starts,
                "basis_examples_per_split": args.basis_examples_per_split,
                "feature_token_filter": args.feature_token_filter,
                "output_mode": args.output_mode,
                "prompt_counts": {"harmful": len(HARMFUL_PROMPTS), "benign": len(BENIGN_PROMPTS)},
                "outputs": {"csv": str(csv_path), "summary": str(summary_path)},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[save] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
