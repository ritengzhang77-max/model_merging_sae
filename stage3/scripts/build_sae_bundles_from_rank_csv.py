#!/usr/bin/env python3
"""Build SAE bundle specs from a feature-ranking CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_cutoffs(raw: str) -> tuple[int, ...]:
    return tuple(int(x) for x in raw.split(",") if x.strip())


def parse_rank_items(raw: str) -> list[int]:
    ranks: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start = int(left)
            stop = int(right)
            step = 1 if stop >= start else -1
            ranks.extend(range(start, stop + step, step))
        else:
            ranks.append(int(item))
    return ranks


def parse_rank_sets(raw: str) -> list[tuple[str, list[int]]]:
    out: list[tuple[str, list[int]]] = []
    for spec in raw.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        label, _, ranks = spec.partition("=")
        if not label or not ranks:
            raise ValueError(f"Bad rank-set spec: {spec!r}")
        out.append((label.strip(), parse_rank_items(ranks)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking-csv", type=Path, required=True)
    ap.add_argument("--cutoffs", type=parse_cutoffs, default=())
    ap.add_argument(
        "--rank-sets",
        type=parse_rank_sets,
        default=(),
        help="Optional semicolon-separated specs like label=1-32,34;single33=33.",
    )
    ap.add_argument(
        "--leave-one-out-top-k",
        type=int,
        default=0,
        help="Emit full_topK and topK_minus_rankNNN bundles for ranks 1..K.",
    )
    ap.add_argument("--prefix", default="prompt_delta_top")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not args.cutoffs and not args.rank_sets and args.leave_one_out_top_k <= 0:
        raise SystemExit("provide --cutoffs or --rank-sets")

    with args.ranking_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty ranking CSV: {args.ranking_csv}")

    layer = int(rows[0]["layer"])
    feature_ids = [int(row["feature_id"]) for row in rows]
    parts = []
    for cutoff in args.cutoffs:
        selected = feature_ids[: min(cutoff, len(feature_ids))]
        spec = ",".join(f"{layer}:{feature_id}" for feature_id in selected)
        parts.append(f"{args.prefix}{cutoff}={spec}")
    if args.leave_one_out_top_k > 0:
        k = min(args.leave_one_out_top_k, len(feature_ids))
        top = feature_ids[:k]
        full_spec = ",".join(f"{layer}:{feature_id}" for feature_id in top)
        parts.append(f"top{k}_full={full_spec}")
        for rank in range(1, k + 1):
            selected = top[: rank - 1] + top[rank:]
            spec = ",".join(f"{layer}:{feature_id}" for feature_id in selected)
            parts.append(f"top{k}_minus_rank{rank:03d}={spec}")
    for label, ranks in args.rank_sets:
        selected = []
        for rank in ranks:
            if rank < 1 or rank > len(feature_ids):
                raise ValueError(f"Rank {rank} out of range for {args.ranking_csv}")
            selected.append(feature_ids[rank - 1])
        spec = ",".join(f"{layer}:{feature_id}" for feature_id in selected)
        parts.append(f"{label}={spec}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(";".join(parts) + "\n", encoding="utf-8")
    print(f"[save] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
