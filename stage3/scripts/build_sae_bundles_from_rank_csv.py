#!/usr/bin/env python3
"""Build SAE bundle specs from a feature-ranking CSV."""

from __future__ import annotations

import argparse
import csv
import itertools
import random
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
    ap.add_argument(
        "--base-ranks",
        type=parse_rank_items,
        default=(),
        help="Optional base rank set for base-plus-singleton bundles.",
    )
    ap.add_argument(
        "--extra-singleton-ranks",
        type=parse_rank_items,
        default=(),
        help="Optional extra ranks to add one at a time to --base-ranks.",
    )
    ap.add_argument(
        "--extra-pairs",
        action="store_true",
        help="With --base-ranks and --extra-singleton-ranks, also emit all base-plus-two-extra-rank bundles.",
    )
    ap.add_argument("--base-label", default="base")
    ap.add_argument("--random-subsets-top-k", type=int, default=0)
    ap.add_argument("--random-subset-size", type=int, default=0)
    ap.add_argument("--random-subset-count", type=int, default=0)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--prefix", default="prompt_delta_top")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if (
        not args.cutoffs
        and not args.rank_sets
        and args.leave_one_out_top_k <= 0
        and not (args.base_ranks and args.extra_singleton_ranks)
        and args.random_subset_count <= 0
    ):
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
    if args.base_ranks and args.extra_singleton_ranks:
        for rank in [*args.base_ranks, *args.extra_singleton_ranks]:
            if rank < 1 or rank > len(feature_ids):
                raise ValueError(f"Rank {rank} out of range for {args.ranking_csv}")
        base_features = [feature_ids[rank - 1] for rank in args.base_ranks]
        base_spec = ",".join(f"{layer}:{feature_id}" for feature_id in base_features)
        parts.append(f"{args.base_label}= {base_spec}".replace("= ", "="))
        for rank in args.extra_singleton_ranks:
            selected = base_features + [feature_ids[rank - 1]]
            spec = ",".join(f"{layer}:{feature_id}" for feature_id in selected)
            parts.append(f"{args.base_label}_plus_rank{rank:03d}={spec}")
        if args.extra_pairs:
            for left, right in itertools.combinations(args.extra_singleton_ranks, 2):
                selected = base_features + [feature_ids[left - 1], feature_ids[right - 1]]
                spec = ",".join(f"{layer}:{feature_id}" for feature_id in selected)
                parts.append(f"{args.base_label}_plus_rank{left:03d}_rank{right:03d}={spec}")
    if args.random_subset_count > 0:
        if args.random_subsets_top_k <= 0 or args.random_subset_size <= 0:
            raise SystemExit("--random-subsets-top-k and --random-subset-size are required for random subsets")
        if args.random_subsets_top_k > len(feature_ids):
            raise ValueError(f"--random-subsets-top-k exceeds ranking length: {args.random_subsets_top_k}")
        if args.random_subset_size > args.random_subsets_top_k:
            raise ValueError("--random-subset-size must be <= --random-subsets-top-k")
        rng = random.Random(args.random_seed)
        population = list(range(1, args.random_subsets_top_k + 1))
        seen: set[tuple[int, ...]] = set()
        attempts = 0
        while len(seen) < args.random_subset_count:
            attempts += 1
            if attempts > args.random_subset_count * 100:
                raise RuntimeError("could not sample enough unique random subsets")
            ranks = tuple(sorted(rng.sample(population, args.random_subset_size)))
            if ranks in seen:
                continue
            seen.add(ranks)
            selected = [feature_ids[rank - 1] for rank in ranks]
            spec = ",".join(f"{layer}:{feature_id}" for feature_id in selected)
            label = f"random_seed{args.random_seed}_top{args.random_subsets_top_k}_k{args.random_subset_size}_{len(seen):03d}"
            parts.append(f"{label}={spec}")
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
