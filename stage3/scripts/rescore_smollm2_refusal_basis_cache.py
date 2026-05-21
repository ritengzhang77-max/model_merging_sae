#!/usr/bin/env python3
"""Recompute Stage 3 refusal-basis labels and metrics from the activation cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage1" / "scripts"))
sys.path.insert(0, str(ROOT / "stage3" / "scripts"))

from analyze_smollm2_rq0 import write_csv  # noqa: E402
from analyze_smollm2_refusal_basis import (  # noqa: E402
    CACHE_STAGE3,
    RESULT_DIR,
    build_summary_md,
    balanced_accuracy,
    centroid_predict,
    labels_from_records,
    score_stage3_refusal_quality,
    summarize_generation,
    write_json,
)
from evaluate_smollm2_refusal_quality import write_jsonl  # noqa: E402


def precompute_transforms(
    x: torch.Tensor,
    prompt_ids: torch.Tensor,
    bases: tuple[tuple[str, int], ...],
    *,
    folds: int,
    seed: int,
) -> dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Precompute basis transforms once per representation/fold.

    The labels do not affect PCA, random projection, or top-variance selection,
    so recomputing them for every target wastes most of the runtime.
    """
    out = {}
    x = x.float()
    for fold in range(folds):
        test_mask = (prompt_ids % folds) == fold
        train_mask = ~test_mask
        train_raw = x[train_mask]
        test_raw = x[test_mask]
        mean = train_raw.mean(dim=0, keepdim=True)
        train_centered = train_raw - mean
        test_centered = test_raw - mean

        pca_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for basis, dim in bases:
            if basis == "raw":
                train_x, test_x = train_raw, test_raw
            elif basis == "topvar":
                q = min(dim, train_centered.shape[1])
                idx = torch.topk(train_centered.var(dim=0), k=q).indices
                train_x, test_x = train_centered[:, idx], test_centered[:, idx]
            elif basis == "random":
                gen = torch.Generator()
                gen.manual_seed(seed + fold)
                q = min(dim, train_centered.shape[1])
                proj = torch.randn(train_centered.shape[1], q, generator=gen)
                proj = F.normalize(proj, dim=0)
                train_x, test_x = train_centered @ proj, test_centered @ proj
            elif basis == "pca":
                q = min(dim, train_centered.shape[0] - 1, train_centered.shape[1])
                if q not in pca_cache:
                    arr = train_centered.numpy()
                    _u, _s, vh = np.linalg.svd(arr, full_matrices=False)
                    v = torch.from_numpy(vh[:q].T).float()
                    pca_cache[q] = (train_centered @ v, test_centered @ v)
                train_x, test_x = pca_cache[q]
            else:
                raise ValueError(f"unknown basis {basis}")
            out[(basis, dim, fold)] = (train_x, test_x, train_mask, test_mask)
    return out


def score_from_transforms(
    transforms: dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    y: torch.Tensor,
    *,
    basis: str,
    dim: int,
    folds: int,
):
    if len(set(int(v) for v in y.tolist())) < 2:
        return None
    preds = torch.full_like(y, fill_value=-1)
    used = torch.zeros_like(y, dtype=torch.bool)
    for fold in range(folds):
        train_x, test_x, train_mask, test_mask = transforms[(basis, dim, fold)]
        if int(test_mask.sum()) == 0 or int(train_mask.sum()) == 0:
            continue
        if len(set(int(v) for v in y[train_mask].tolist())) < 2:
            continue
        preds[test_mask] = centroid_predict(train_x, y[train_mask], test_x)
        used[test_mask] = True
    if int(used.sum()) == 0:
        return None
    y_used = y[used]
    p_used = preds[used]
    return {
        "n_eval": int(used.sum().item()),
        "accuracy": float((p_used == y_used).float().mean().item()),
        "balanced_accuracy": balanced_accuracy(y_used, p_used),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE_STAGE3 / "smollm2_refusal_basis_activations.pt")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    features = cache["features"]
    records = []
    for row in cache["records"]:
        new = dict(row)
        scores = score_stage3_refusal_quality(str(new["generation"]))
        new.update(scores)
        records.append(new)
    prompt_ids = cache["prompt_ids"].long()
    layers = tuple(cache.get("layers", (15, 20, 25, 29)))

    records_path = args.result_dir / "smollm2_refusal_basis_records.jsonl"
    generation_summary_path = args.result_dir / "smollm2_refusal_basis_generation_summary.csv"
    metrics_path = args.result_dir / "smollm2_refusal_basis_metrics.csv"
    summary_path = args.result_dir / "SMOLLM2_REFUSAL_BASIS_SUMMARY.md"
    meta_path = args.result_dir / "smollm2_refusal_basis_summary.json"

    write_jsonl(records_path, records)
    generation_summary = summarize_generation(records)
    write_csv(generation_summary_path, generation_summary)

    targets = (
        "strict_clean_refusal",
        "strict_messy_refusal",
        "stage3_keyword_refusal",
        "clean_refusal",
        "keyword_refusal",
        "messy_refusal",
        "full_merge_like",
    )
    bases = (("raw", 0), ("pca", 16), ("pca", 32), ("random", 16), ("random", 32), ("topvar", 16), ("topvar", 32))
    metric_rows: list[dict[str, object]] = []
    for rep_name, x in sorted(features.items()):
        print(f"[metrics] {rep_name}", flush=True)
        transforms = precompute_transforms(x, prompt_ids, bases, folds=args.folds, seed=args.seed)
        for target in targets:
            y = labels_from_records(records, target)
            for basis, dim in bases:
                score = score_from_transforms(transforms, y, basis=basis, dim=dim, folds=args.folds)
                if score is None:
                    continue
                metric_rows.append(
                    {
                        "representation": rep_name,
                        "target": target,
                        "basis": basis,
                        "dim": "full" if basis == "raw" else dim,
                        "n_total": len(y),
                        "n_positive": int(y.sum().item()),
                        "n_negative": int((1 - y).sum().item()),
                        **score,
                    }
                )
    write_csv(metrics_path, metric_rows)

    summary_args = SimpleNamespace(examples=len({int(r["prompt_id"]) for r in records}), layers=",".join(str(x) for x in layers))
    build_summary_md(summary_path, metric_rows, generation_summary, summary_args)
    write_json(
        meta_path,
        {
            "rescore_from_cache": str(args.cache),
            "examples": summary_args.examples,
            "layers": layers,
            "outputs": {
                "records": str(records_path),
                "generation_summary": str(generation_summary_path),
                "metrics": str(metrics_path),
                "summary": str(summary_path),
            },
        },
    )
    print(f"[save] {records_path}")
    print(f"[save] {generation_summary_path}")
    print(f"[save] {metrics_path}")
    print(f"[save] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
