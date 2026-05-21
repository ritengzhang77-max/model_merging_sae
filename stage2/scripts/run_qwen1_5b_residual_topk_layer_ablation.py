#!/usr/bin/env python3
"""Layer ablation for a family-specific residual top-coordinate patch."""

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
from run_qwen1_5b_lowdim_activation_patch_target_loss import (  # noqa: E402
    DEFAULT_LAYERS,
    build_bases,
    collect_basis_stats,
    make_batches as make_primary_basis_batches,
    parse_ints,
)
from run_qwen1_5b_pca_residual_patch_generation import parse_layer_spec  # noqa: E402
from run_qwen1_5b_pca_residual_target_loss import (  # noqa: E402
    gap_closed,
    losses_for_model,
    make_eval_batches,
)
from run_qwen1_5b_second_stage_residual_pca import (  # noqa: E402
    build_residual_bases,
    collect_residual_rows,
    make_harmful_batches,
    residual_basis_prompts,
    second_stage_losses,
)


RESULT_DIR = ROOT / "stage2" / "results" / "qwen1_5b_residual_topk_layer_ablation"


def label_layers(layers: tuple[int, ...]) -> str:
    if not layers:
        return "none"
    return "+".join(str(layer) for layer in layers)


def contiguous_specs(layers: tuple[int, ...]) -> list[tuple[str, tuple[int, ...]]]:
    specs: list[tuple[str, tuple[int, ...]]] = []
    specs.append(("full_" + label_layers(layers), layers))
    for layer in layers:
        specs.append((f"only_{layer}", (layer,)))
    for layer in layers:
        remaining = tuple(x for x in layers if x != layer)
        specs.append((f"drop_{layer}", remaining))
    windows = (
        (16, 18),
        (19, 21),
        (22, 23),
        (16, 19),
        (20, 23),
        (16, 21),
        (18, 23),
        (16, 22),
        (17, 23),
    )
    layer_set = set(layers)
    for lo, hi in windows:
        win = tuple(x for x in range(lo, hi + 1) if x in layer_set)
        if win and win != layers:
            specs.append((f"window_{lo}-{hi}", win))
    dedup: list[tuple[str, tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()
    for label, spec in specs:
        if spec not in seen:
            seen.add(spec)
            dedup.append((label, spec))
    return dedup


def filter_residual_bases(residual_bases, active_layers: tuple[int, ...]):
    active = set(active_layers)
    out = {}
    for kind, mapping in residual_bases.items():
        out[kind] = {}
        for key, value in mapping.items():
            if isinstance(key, tuple):
                layer = int(key[0])
            else:
                layer = int(key)
            if layer in active:
                out[kind][key] = value
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, object]], *, basis_mode: str, top_k: int) -> None:
    lines = [
        "# Qwen2.5-1.5B Residual Top-K Layer Ablation",
        "",
        f"Residual basis mode: `{basis_mode}`",
        f"Patch: `PCA64 + residual topk{top_k}`",
        "",
    ]
    for eval_mode in sorted({str(row["eval_mode"]) for row in rows}):
        lines.extend(
            [
                f"## `{eval_mode}`",
                "",
                "| variant | active layers | harmful loss | gap closed vs PCA64 | harmful gap closed |",
                "|---|---|---:|---:|---:|",
            ]
        )
        subset = [row for row in rows if str(row["eval_mode"]) == eval_mode]
        subset = sorted(subset, key=lambda row: float(row.get("harmful_refusal_loss", 999.0)))
        for row in subset:
            lines.append(
                f"| `{row['variant']}` | `{row['active_layers']}` | "
                f"{float(row.get('harmful_refusal_loss', 0.0)):.3f} | "
                f"{float(row.get('harmful_gap_closed_vs_pca64', 0.0)):.3f} | "
                f"{float(row.get('harmful_gap_closed', 0.0)):.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--residual-basis-mode", default="tracking_original")
    ap.add_argument("--eval-modes", default="heldout_failures_harmful_only,stress_permission_harmful_only")
    ap.add_argument("--examples-per-split", type=int, default=12)
    ap.add_argument("--basis-examples-per-split", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--primary-layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    ap.add_argument("--residual-layers", default="16-23")
    ap.add_argument("--pca-rank", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=1024)
    ap.add_argument("--max-pca-rows-per-layer", type=int, default=512)
    ap.add_argument("--max-residual-rows-per-layer", type=int, default=512)
    ap.add_argument("--pca-oversample", type=int, default=8)
    ap.add_argument("--pca-n-iter", type=int, default=1)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    primary_layers = parse_ints(args.primary_layers)
    residual_layers = parse_layer_spec(args.residual_layers)
    eval_modes = tuple(item.strip() for item in args.eval_modes.split(",") if item.strip())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    primary_batches = make_primary_basis_batches(
        tokenizer,
        args.basis_examples_per_split,
        args.max_length,
        args.batch_size,
    )
    residual_prompts = residual_basis_prompts(args.residual_basis_mode, args.examples_per_split)
    residual_batches = make_harmful_batches(tokenizer, residual_prompts, args.max_length, args.batch_size)
    eval_batches_by_mode = {
        mode: make_eval_batches(tokenizer, mode, args.examples_per_split, args.max_length, args.batch_size)
        for mode in eval_modes
    }

    print("[load] base donor", flush=True)
    donor = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=torch.float16).to(args.device)
    donor.eval()
    print("[load] abliterated recipient", flush=True)
    recipient = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=torch.float16).to(args.device)
    recipient.eval()

    print("[basis] collect primary PCA64 calibration activations", flush=True)
    primary_stats = collect_basis_stats(
        donor,
        recipient,
        primary_batches,
        primary_layers,
        device=args.device,
        max_rows_per_layer=args.max_pca_rows_per_layer,
    )
    primary_bases = build_bases(
        primary_stats,
        primary_layers,
        (),
        (args.pca_rank,),
        (),
        random_seed=args.random_seed,
        pca_oversample=args.pca_oversample,
        pca_n_iter=args.pca_n_iter,
    )

    print("[basis] collect residual rows", flush=True)
    residual_rows, _energy, _n_rows = collect_residual_rows(
        donor,
        recipient,
        residual_batches,
        residual_layers,
        primary_bases,
        args.pca_rank,
        device=args.device,
        max_rows_per_layer=args.max_residual_rows_per_layer,
        basis_mask="all",
    )
    residual_bases, explained_rows = build_residual_bases(
        residual_rows,
        residual_layers,
        (32,),
        (args.top_k,),
        random_seed=args.random_seed + 1009,
        pca_oversample=args.pca_oversample,
        pca_n_iter=args.pca_n_iter,
    )

    variants = [("pca64", ())]
    variants.extend(contiguous_specs(residual_layers))
    rows: list[dict[str, object]] = []
    for eval_mode, eval_batches in eval_batches_by_mode.items():
        print(f"[eval:{eval_mode}] base", flush=True)
        base_losses = losses_for_model(donor, eval_batches, args.device)
        print(f"[eval:{eval_mode}] abliterated", flush=True)
        recipient_losses = losses_for_model(recipient, eval_batches, args.device)
        print(f"[eval:{eval_mode}] pca64", flush=True)
        pca_losses = second_stage_losses(
            donor,
            recipient,
            eval_batches,
            primary_layers,
            (),
            primary_bases,
            args.pca_rank,
            residual_bases,
            None,
            None,
            args.device,
        )
        pca_harm = pca_losses.get("harmful_refusal_loss", float("nan"))
        base_harm = base_losses.get("harmful_refusal_loss", float("nan"))
        for variant, active_layers in variants:
            if variant == "pca64":
                losses = pca_losses
            else:
                print(f"[eval:{eval_mode}] {variant}", flush=True)
                filtered = filter_residual_bases(residual_bases, active_layers)
                losses = second_stage_losses(
                    donor,
                    recipient,
                    eval_batches,
                    primary_layers,
                    (),
                    primary_bases,
                    args.pca_rank,
                    filtered,
                    "top_neuron",
                    args.top_k,
                    args.device,
                )
            harmful = losses.get("harmful_refusal_loss", float("nan"))
            denom = pca_harm - base_harm
            rows.append(
                {
                    "eval_mode": eval_mode,
                    "variant": variant,
                    "active_layers": label_layers(active_layers),
                    "harmful_refusal_loss": harmful,
                    "harmful_gap_closed": gap_closed(
                        recipient_losses.get("harmful_refusal_loss", float("nan")),
                        base_harm,
                        harmful,
                    ),
                    "harmful_gap_closed_vs_pca64": float("nan") if abs(denom) < 1e-9 else (pca_harm - harmful) / denom,
                }
            )

    metrics_path = args.result_dir / "qwen1_5b_residual_topk_layer_ablation_losses.csv"
    explained_path = args.result_dir / "qwen1_5b_residual_topk_layer_ablation_explained.csv"
    summary_path = args.result_dir / "QWEN1_5B_RESIDUAL_TOPK_LAYER_ABLATION_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, rows)
    write_csv(explained_path, explained_rows)
    write_summary(summary_path, rows, basis_mode=args.residual_basis_mode, top_k=args.top_k)
    manifest_path.write_text(
        json.dumps(
            {
                "models": {"donor": MODEL_IDS["base"], "recipient": MODEL_IDS["abliterated"]},
                "residual_basis_mode": args.residual_basis_mode,
                "residual_basis_prompts": residual_prompts,
                "eval_modes": eval_modes,
                "primary_layers": primary_layers,
                "residual_layers": residual_layers,
                "pca_rank": args.pca_rank,
                "top_k": args.top_k,
                "outputs": {
                    "metrics": str(metrics_path),
                    "explained": str(explained_path),
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
