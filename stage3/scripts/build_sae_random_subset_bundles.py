#!/usr/bin/env python3
"""Build deterministic random SAE feature-subset bundle specs."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def parse_bundle_text(text: str) -> dict[str, list[tuple[int, int]]]:
    bundles: dict[str, list[tuple[int, int]]] = {}
    for raw_entry in text.strip().split(";"):
        entry = raw_entry.strip()
        if not entry:
            continue
        label, payload = entry.split("=", 1)
        features: list[tuple[int, int]] = []
        for raw_feature in payload.split(","):
            if not raw_feature.strip():
                continue
            layer_text, feature_text = raw_feature.split(":", 1)
            features.append((int(layer_text), int(feature_text)))
        bundles[label] = features
    if not bundles:
        raise ValueError("no bundles found")
    return bundles


def format_bundle(label: str, features: list[tuple[int, int]]) -> str:
    return f"{label}=" + ",".join(f"{layer}:{feature}" for layer, feature in features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-bundles-file", type=Path, required=True)
    parser.add_argument("--pool-label", default=None)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=240524)
    parser.add_argument("--label-prefix", default="random")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="random_bundles.txt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    bundles = parse_bundle_text(args.pool_bundles_file.read_text(encoding="utf-8"))
    if args.pool_label is None:
        if len(bundles) != 1:
            raise ValueError("--pool-label is required when the pool file has multiple bundles")
        pool_label = next(iter(bundles))
    else:
        pool_label = args.pool_label
    pool = bundles[pool_label]
    if args.sample_size > len(pool):
        raise ValueError(f"sample size {args.sample_size} exceeds pool size {len(pool)}")

    rng = random.Random(args.seed)
    entries: list[str] = []
    rows: list[dict[str, object]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    attempts = 0
    while len(entries) < args.num_samples:
        attempts += 1
        if attempts > args.num_samples * 100:
            raise RuntimeError("too many duplicate random subsets")
        sample = tuple(sorted(rng.sample(pool, args.sample_size)))
        if sample in seen:
            continue
        seen.add(sample)
        index = len(entries) + 1
        label = f"{args.label_prefix}_seed{args.seed}_k{args.sample_size}_{index:03d}"
        entries.append(format_bundle(label, list(sample)))
        rows.append(
            {
                "label": label,
                "sample_index": index,
                "seed": args.seed,
                "sample_size": args.sample_size,
                "pool_label": pool_label,
                "pool_size": len(pool),
                "features": ",".join(f"{layer}:{feature}" for layer, feature in sample),
            }
        )

    bundles_path = args.result_dir / args.output_name
    manifest_csv_path = args.result_dir / f"{Path(args.output_name).stem}_manifest.csv"
    manifest_json_path = args.result_dir / f"{Path(args.output_name).stem}_build_manifest.json"
    bundles_path.write_text(";".join(entries) + "\n", encoding="utf-8")
    with manifest_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    manifest_json_path.write_text(
        json.dumps(
            {
                "pool_bundles_file": str(args.pool_bundles_file),
                "pool_label": pool_label,
                "pool_size": len(pool),
                "sample_size": args.sample_size,
                "num_samples": args.num_samples,
                "seed": args.seed,
                "outputs": {
                    "bundles": str(bundles_path),
                    "manifest_csv": str(manifest_csv_path),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[save] {bundles_path} ({len(entries)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
