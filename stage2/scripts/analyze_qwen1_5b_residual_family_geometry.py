#!/usr/bin/env python3
"""Split-aware geometry for PCA64 residual MLP deltas in layers 16-23."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/gavin/model_merging/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
sys.path.insert(0, str(ROOT / "stage2" / "scripts"))
from analyze_qwen1_5b_refusal_rq1_rq2 import (  # noqa: E402
    MODEL_IDS,
    REFUSAL_TARGET,
    collate_labeled,
    labeled_example,
)
from run_qwen1_5b_lowdim_activation_patch_target_loss import (  # noqa: E402
    DEFAULT_LAYERS,
    build_bases,
    collect_basis_stats,
    first_tensor,
    make_batches as make_primary_basis_batches,
    module_for,
    parse_ints,
    remove_hooks,
)
from run_qwen1_5b_pca_residual_patch_generation import (  # noqa: E402
    HELDOUT_FAILURE_HARMFUL_PROMPTS,
    HELDOUT_FAILURE_BENIGN_PROMPTS,
    STRESS_PERMISSION_HARMFUL_PROMPTS,
    parse_layer_spec,
)


RESULT_DIR = ROOT / "stage2" / "results" / "qwen1_5b_residual_family_geometry"

PROMPT_GROUPS: dict[str, tuple[str, ...]] = {
    "one_time_code": (HELDOUT_FAILURE_HARMFUL_PROMPTS[0],),
    "tracking_original": (HELDOUT_FAILURE_HARMFUL_PROMPTS[1],),
    "permission_slip": STRESS_PERMISSION_HARMFUL_PROMPTS,
    "benign_account_recovery": (HELDOUT_FAILURE_BENIGN_PROMPTS[0],),
    "benign_remove_tracking": (HELDOUT_FAILURE_BENIGN_PROMPTS[3],),
}


def make_eval_batches(tokenizer, groups: tuple[str, ...], max_length: int, batch_size: int):
    examples = []
    for group in groups:
        for prompt in PROMPT_GROUPS[group]:
            examples.append(
                {
                    "group": group,
                    "user": prompt,
                    **labeled_example(tokenizer, prompt, REFUSAL_TARGET, max_length),
                }
            )
    return [
        (examples[start : start + batch_size], collate_labeled(tokenizer, examples[start : start + batch_size]))
        for start in range(0, len(examples), batch_size)
    ]


def make_cache_hook(cache: dict[int, torch.Tensor], layer: int):
    def hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        cache[layer] = tensor.detach()

    return hook


def empty_stats(hidden: int):
    return {
        "residual_sum": torch.zeros(hidden, dtype=torch.float64),
        "residual_abs_sum": torch.zeros(hidden, dtype=torch.float64),
        "residual_sq_sum": torch.zeros(hidden, dtype=torch.float64),
        "delta_sq_sum": 0.0,
        "pca_sq_sum": 0.0,
        "residual_total_sq": 0.0,
        "n_tokens": 0,
    }


@torch.no_grad()
def collect_family_stats(donor, recipient, batches, layers: tuple[int, ...], bases, pca_rank: int, device: str):
    donor_cache: dict[int, torch.Tensor] = {}
    recipient_cache: dict[int, torch.Tensor] = {}
    donor_handles = [module_for(donor, layer).register_forward_hook(make_cache_hook(donor_cache, layer)) for layer in layers]
    recipient_handles = [
        module_for(recipient, layer).register_forward_hook(make_cache_hook(recipient_cache, layer)) for layer in layers
    ]
    stats: dict[tuple[str, int, str], dict[str, object]] = {}
    try:
        for chunk, batch in batches:
            batch = {key: value.to(device) for key, value in batch.items()}
            donor_cache.clear()
            recipient_cache.clear()
            _ = donor(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            _ = recipient(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            masks = {
                "all": batch["attention_mask"].bool(),
                "target": (batch["labels"] != -100) & batch["attention_mask"].bool(),
            }
            for layer in layers:
                donor_out = donor_cache[layer].float()
                recipient_out = recipient_cache[layer].float()
                delta = donor_out - recipient_out
                basis = bases["pca"][(layer, pca_rank)].to(device=delta.device, dtype=delta.dtype)
                coeff = torch.matmul(delta, basis)
                projected = torch.matmul(coeff, basis.T)
                residual = delta - projected
                hidden = residual.shape[-1]
                for i, row in enumerate(chunk):
                    group = str(row["group"])
                    for mask_name, mask_tensor in masks.items():
                        mask = mask_tensor[i]
                        if not bool(mask.any()):
                            continue
                        key = (group, layer, mask_name)
                        if key not in stats:
                            stats[key] = empty_stats(hidden)
                        selected_delta = delta[i][mask].double().cpu()
                        selected_pca = projected[i][mask].double().cpu()
                        selected_resid = residual[i][mask].double().cpu()
                        stats[key]["residual_sum"] += selected_resid.sum(dim=0)
                        stats[key]["residual_abs_sum"] += selected_resid.abs().sum(dim=0)
                        stats[key]["residual_sq_sum"] += (selected_resid * selected_resid).sum(dim=0)
                        stats[key]["delta_sq_sum"] += float((selected_delta * selected_delta).sum().item())
                        stats[key]["pca_sq_sum"] += float((selected_pca * selected_pca).sum().item())
                        stats[key]["residual_total_sq"] += float((selected_resid * selected_resid).sum().item())
                        stats[key]["n_tokens"] += int(mask.sum().item())
    finally:
        remove_hooks(donor_handles)
        remove_hooks(recipient_handles)
    return stats


def top_indices(vector: torch.Tensor, k: int) -> torch.Tensor:
    kk = min(k, vector.numel())
    return torch.topk(vector.float(), kk).indices


def summarize_family(stats: dict[tuple[str, int, str], dict[str, object]], top_ks: tuple[int, ...]):
    rows: list[dict[str, object]] = []
    top_coord_rows: list[dict[str, object]] = []
    for (group, layer, mask), row in sorted(stats.items()):
        residual_sq = row["residual_sq_sum"]
        residual_abs = row["residual_abs_sum"]
        total = float(row["residual_total_sq"])
        n_tokens = int(row["n_tokens"])
        out = {
            "group": group,
            "layer": layer,
            "mask": mask,
            "n_tokens": n_tokens,
            "delta_norm_per_token": math.sqrt(float(row["delta_sq_sum"]) / max(n_tokens, 1)),
            "pca_norm_per_token": math.sqrt(float(row["pca_sq_sum"]) / max(n_tokens, 1)),
            "residual_norm_per_token": math.sqrt(total / max(n_tokens, 1)),
            "pca_energy_fraction": float(row["pca_sq_sum"]) / max(float(row["delta_sq_sum"]), 1e-12),
            "residual_energy_fraction": total / max(float(row["delta_sq_sum"]), 1e-12),
        }
        for k in top_ks:
            idx = top_indices(residual_abs, k)
            out[f"top{k}_energy_fraction"] = float(residual_sq[idx].sum().item()) / max(total, 1e-12)
        rows.append(out)
        for rank, idx in enumerate(top_indices(residual_abs, 20).tolist(), start=1):
            top_coord_rows.append(
                {
                    "group": group,
                    "layer": layer,
                    "mask": mask,
                    "rank": rank,
                    "coord": idx,
                    "abs_sum": float(residual_abs[idx].item()),
                    "sq_sum": float(residual_sq[idx].item()),
                }
            )
    return rows, top_coord_rows


def pairwise_rows(stats: dict[tuple[str, int, str], dict[str, object]], top_ks: tuple[int, ...]):
    rows: list[dict[str, object]] = []
    keys = sorted(stats)
    by_layer_mask: dict[tuple[int, str], list[str]] = {}
    for group, layer, mask in keys:
        by_layer_mask.setdefault((layer, mask), []).append(group)
    for (layer, mask), groups in sorted(by_layer_mask.items()):
        groups = sorted(set(groups))
        for i, group_a in enumerate(groups):
            for group_b in groups[i + 1 :]:
                row_a = stats[(group_a, layer, mask)]
                row_b = stats[(group_b, layer, mask)]
                mean_a = row_a["residual_sum"].float() / max(int(row_a["n_tokens"]), 1)
                mean_b = row_b["residual_sum"].float() / max(int(row_b["n_tokens"]), 1)
                abs_a = row_a["residual_abs_sum"]
                abs_b = row_b["residual_abs_sum"]
                sq_a = row_a["residual_sq_sum"]
                sq_b = row_b["residual_sq_sum"]
                out = {
                    "layer": layer,
                    "mask": mask,
                    "group_a": group_a,
                    "group_b": group_b,
                    "mean_residual_cosine": float(F.cosine_similarity(mean_a, mean_b, dim=0).item()),
                }
                total_a = float(row_a["residual_total_sq"])
                total_b = float(row_b["residual_total_sq"])
                for k in top_ks:
                    top_a = set(top_indices(abs_a, k).tolist())
                    top_b = set(top_indices(abs_b, k).tolist())
                    inter = top_a & top_b
                    union = top_a | top_b
                    out[f"top{k}_jaccard"] = len(inter) / max(len(union), 1)
                    out[f"a_energy_in_b_top{k}"] = float(sq_a[list(top_b)].sum().item()) / max(total_a, 1e-12)
                    out[f"b_energy_in_a_top{k}"] = float(sq_b[list(top_a)].sum().item()) / max(total_b, 1e-12)
                rows.append(out)
    return rows


def aggregate(rows: list[dict[str, object]], keys: tuple[str, ...], metric_prefixes: tuple[str, ...]):
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    out = []
    for key, vals in sorted(grouped.items()):
        row = {name: value for name, value in zip(keys, key)}
        row["n_layers"] = len(vals)
        for metric in metric_prefixes:
            values = [float(v[metric]) for v in vals if metric in v]
            if values:
                row[f"mean_{metric}"] = sum(values) / len(values)
        out.append(row)
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


def write_summary(path: Path, family_rows, pair_rows, top_ks: tuple[int, ...]) -> None:
    all_rows = [row for row in family_rows if row["mask"] == "all"]
    pair_all = [row for row in pair_rows if row["mask"] == "all"]
    lines = [
        "# Qwen2.5-1.5B Residual Family Geometry",
        "",
        "Residual is computed after subtracting the PCA64 projection from donor-recipient MLP deltas.",
        "",
        "## Coordinate Concentration",
        "",
        "| group | residual norm/token | top256 energy | top1024 energy |",
        "|---|---:|---:|---:|",
    ]
    group_aggs = aggregate(
        all_rows,
        ("group",),
        ("residual_norm_per_token", "top256_energy_fraction", "top1024_energy_fraction"),
    )
    for row in group_aggs:
        lines.append(
            f"| `{row['group']}` | {row.get('mean_residual_norm_per_token', 0.0):.3f} | "
            f"{row.get('mean_top256_energy_fraction', 0.0):.3f} | "
            f"{row.get('mean_top1024_energy_fraction', 0.0):.3f} |"
        )
    lines.extend(["", "## Pairwise Residual Similarity", ""])
    pair_aggs = aggregate(
        pair_all,
        ("group_a", "group_b"),
        (
            "mean_residual_cosine",
            "top256_jaccard",
            "top1024_jaccard",
            "a_energy_in_b_top256",
            "b_energy_in_a_top256",
        ),
    )
    lines.extend(
        [
            "| group A | group B | mean cosine | top256 Jaccard | top1024 Jaccard | A energy in B top256 | B energy in A top256 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    focus_pairs = {
        ("one_time_code", "permission_slip"),
        ("one_time_code", "tracking_original"),
        ("permission_slip", "tracking_original"),
        ("benign_account_recovery", "one_time_code"),
        ("benign_remove_tracking", "tracking_original"),
    }
    for row in pair_aggs:
        pair = (str(row["group_a"]), str(row["group_b"]))
        reverse = (pair[1], pair[0])
        if pair not in focus_pairs and reverse not in focus_pairs:
            continue
        lines.append(
            f"| `{row['group_a']}` | `{row['group_b']}` | "
            f"{row.get('mean_mean_residual_cosine', 0.0):.3f} | "
            f"{row.get('mean_top256_jaccard', 0.0):.3f} | "
            f"{row.get('mean_top1024_jaccard', 0.0):.3f} | "
            f"{row.get('mean_a_energy_in_b_top256', 0.0):.3f} | "
            f"{row.get('mean_b_energy_in_a_top256', 0.0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Reading This",
            "",
            "- Low top-coordinate overlap between harmful families supports prompt-family-specific residual mechanisms.",
            "- High concentration with low behavioral repair means coordinate energy alone is not sufficient.",
            "- Benign comparisons are controls for whether the residual coordinates are just prompt-template or topic coordinates.",
            "",
            f"Top-k settings analyzed: `{', '.join(str(k) for k in top_ks)}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--groups", default="one_time_code,tracking_original,permission_slip,benign_account_recovery,benign_remove_tracking")
    ap.add_argument("--basis-examples-per-split", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--primary-layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    ap.add_argument("--residual-layers", default="16-23")
    ap.add_argument("--pca-rank", type=int, default=64)
    ap.add_argument("--top-ks", default="64,256,512,1024")
    ap.add_argument("--max-pca-rows-per-layer", type=int, default=512)
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
    groups = tuple(item.strip() for item in args.groups.split(",") if item.strip())
    top_ks = parse_ints(args.top_ks)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    primary_batches = make_primary_basis_batches(
        tokenizer,
        args.basis_examples_per_split,
        args.max_length,
        args.batch_size,
    )
    eval_batches = make_eval_batches(tokenizer, groups, args.max_length, args.batch_size)

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
    bases = build_bases(
        primary_stats,
        primary_layers,
        (),
        (args.pca_rank,),
        (),
        random_seed=args.random_seed,
        pca_oversample=args.pca_oversample,
        pca_n_iter=args.pca_n_iter,
    )

    print("[eval] collect residual family stats", flush=True)
    stats = collect_family_stats(donor, recipient, eval_batches, residual_layers, bases, args.pca_rank, args.device)
    family_rows, top_coord_rows = summarize_family(stats, top_ks)
    pair_rows = pairwise_rows(stats, top_ks)

    family_path = args.result_dir / "qwen1_5b_residual_family_geometry_summary.csv"
    pair_path = args.result_dir / "qwen1_5b_residual_family_geometry_pairwise.csv"
    top_path = args.result_dir / "qwen1_5b_residual_family_geometry_top_coords.csv"
    summary_path = args.result_dir / "QWEN1_5B_RESIDUAL_FAMILY_GEOMETRY_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(family_path, family_rows)
    write_csv(pair_path, pair_rows)
    write_csv(top_path, top_coord_rows)
    write_summary(summary_path, family_rows, pair_rows, top_ks)
    manifest_path.write_text(
        json.dumps(
            {
                "models": {"donor": MODEL_IDS["base"], "recipient": MODEL_IDS["abliterated"]},
                "groups": groups,
                "primary_layers": primary_layers,
                "residual_layers": residual_layers,
                "pca_rank": args.pca_rank,
                "top_ks": top_ks,
                "outputs": {
                    "family_summary": str(family_path),
                    "pairwise": str(pair_path),
                    "top_coords": str(top_path),
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
