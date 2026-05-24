#!/usr/bin/env python3
"""Build SAE bundle specs from signed prompt-token delta detail rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_labels(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def read_rows(path: Path, split: str, top_k: int) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            rank = int(row["rank"])
            if top_k > 0 and rank > top_k:
                continue
            rows.append(row)
    rows.sort(key=lambda row: int(row["rank"]))
    if not rows:
        raise SystemExit(f"no rows selected from {path} for split={split!r} top_k={top_k}")
    return rows


def feature_specs(rows: list[dict[str, str]], sign: str, eps: float, layer: int) -> list[str]:
    selected = []
    for row in rows:
        delta = float(row["delta_value"])
        if sign == "positive" and delta <= eps:
            continue
        if sign == "negative" and delta >= -eps:
            continue
        if sign == "nonzero" and abs(delta) <= eps:
            continue
        selected.append(f"{layer}:{int(row['feature_id'])}")
    return selected


def signed_rows(rows: list[dict[str, str]], sign: str, eps: float) -> list[dict[str, str]]:
    if sign == "positive":
        return [row for row in rows if float(row["delta_value"]) > eps]
    if sign == "negative":
        return [row for row in rows if float(row["delta_value"]) < -eps]
    if sign == "nonzero":
        return [row for row in rows if abs(float(row["delta_value"])) > eps]
    raise ValueError(f"bad sign: {sign}")


def rows_to_specs(rows: list[dict[str, str]], layer: int) -> list[str]:
    return [f"{layer}:{int(row['feature_id'])}" for row in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail-csv", type=Path, required=True)
    ap.add_argument("--split", choices=("harmful", "benign"), default="harmful")
    ap.add_argument("--top-k", type=int, default=33)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--eps", type=float, default=1e-9)
    ap.add_argument(
        "--labels",
        type=parse_labels,
        default=parse_labels("positive,negative,nonzero"),
        help="Comma-separated subset labels to emit: positive, negative, nonzero.",
    )
    ap.add_argument("--prefix", default="top{top_k}_{sign}_{split}_delta")
    ap.add_argument(
        "--combo-sweeps",
        action="store_true",
        help="Also emit positive-all plus negative prefixes, and negative-all plus positive prefixes.",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    args = ap.parse_args()

    rows = read_rows(args.detail_csv, args.split, args.top_k)
    parts = []
    counts = {}
    for label in args.labels:
        if label not in {"positive", "negative", "nonzero"}:
            raise SystemExit(f"unsupported label: {label}")
        specs = feature_specs(rows, label, args.eps, args.layer)
        name = args.prefix.format(top_k=args.top_k, sign=label, split=args.split)
        parts.append(f"{name}={','.join(specs)}")
        counts[name] = len(specs)

    if args.combo_sweeps:
        positive = signed_rows(rows, "positive", args.eps)
        negative = signed_rows(rows, "negative", args.eps)
        positive_specs = rows_to_specs(positive, args.layer)
        negative_specs = rows_to_specs(negative, args.layer)
        for k in range(0, len(negative) + 1):
            name = f"top{args.top_k}_pos_all_plus_neg_prefix{k}_{args.split}_delta"
            specs = positive_specs + rows_to_specs(negative[:k], args.layer)
            parts.append(f"{name}={','.join(specs)}")
            counts[name] = len(specs)
        for k in range(0, len(positive) + 1):
            name = f"top{args.top_k}_neg_all_plus_pos_prefix{k}_{args.split}_delta"
            specs = negative_specs + rows_to_specs(positive[:k], args.layer)
            parts.append(f"{name}={','.join(specs)}")
            counts[name] = len(specs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(";".join(parts) + "\n", encoding="utf-8")
    print(f"[save] {args.output}")
    for name, count in counts.items():
        print(f"[bundle] {name}: {count}")

    manifest_path = args.manifest
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "detail_csv": str(args.detail_csv),
                    "split": args.split,
                    "top_k": args.top_k,
                    "layer": args.layer,
                    "eps": args.eps,
                    "labels": list(args.labels),
                    "combo_sweeps": args.combo_sweeps,
                    "output": str(args.output),
                    "counts": counts,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[save] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
