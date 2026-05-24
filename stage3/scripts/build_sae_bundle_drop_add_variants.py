#!/usr/bin/env python3
"""Build SAE bundle variants by dropping k source features and adding m pool features."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path


Feature = tuple[int, int]


def parse_bundle_text(text: str) -> dict[str, list[Feature]]:
    bundles: dict[str, list[Feature]] = {}
    for raw_entry in text.strip().split(";"):
        entry = raw_entry.strip()
        if not entry:
            continue
        label, payload = entry.split("=", 1)
        features: list[Feature] = []
        for raw_feature in payload.split(","):
            if not raw_feature.strip():
                continue
            layer_text, feature_text = raw_feature.split(":", 1)
            features.append((int(layer_text), int(feature_text)))
        bundles[label] = features
    if not bundles:
        raise ValueError("no bundles found")
    return bundles


def select_pool_bundle(bundles: dict[str, list[Feature]], label: str | None) -> tuple[str, list[Feature]]:
    if label is None:
        if len(bundles) != 1:
            raise ValueError("--pool-label is required when the pool file has multiple bundles")
        selected = next(iter(bundles))
    else:
        selected = label
    return selected, bundles[selected]


def format_bundle(label: str, features: list[Feature]) -> str:
    return f"{label}=" + ",".join(f"{layer}:{feature}" for layer, feature in features)


def feature_suffix(features: tuple[Feature, ...]) -> str:
    return "_".join(f"f{feature}" for _layer, feature in features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundles-file", type=Path, required=True)
    parser.add_argument("--pool-bundles-file", type=Path, required=True)
    parser.add_argument("--pool-label", default=None)
    parser.add_argument("--drop-count", type=int, required=True)
    parser.add_argument("--add-count", type=int, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="drop_add_bundles.txt")
    parser.add_argument("--label-prefix", default="dropadd")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.drop_count < 1:
        raise ValueError("--drop-count must be positive")
    if args.add_count < 1:
        raise ValueError("--add-count must be positive")
    args.result_dir.mkdir(parents=True, exist_ok=True)

    source_bundles = parse_bundle_text(args.source_bundles_file.read_text(encoding="utf-8"))
    pool_label, pool_features = select_pool_bundle(
        parse_bundle_text(args.pool_bundles_file.read_text(encoding="utf-8")),
        args.pool_label,
    )

    entries: list[str] = []
    rows: list[dict[str, object]] = []
    seen: set[tuple[Feature, ...]] = set()
    skipped_duplicates = 0
    for source_label, source_features in source_bundles.items():
        source_set = set(source_features)
        extras = [feature for feature in pool_features if feature not in source_set]
        if args.drop_count > len(source_features):
            raise ValueError(f"drop count exceeds source feature count for {source_label}")
        if args.add_count > len(extras):
            raise ValueError(f"add count exceeds available pool extras for {source_label}")

        for drop_features in itertools.combinations(source_features, args.drop_count):
            reduced = [feature for feature in source_features if feature not in set(drop_features)]
            for add_features in itertools.combinations(extras, args.add_count):
                selected = tuple(sorted(reduced + list(add_features)))
                if selected in seen:
                    skipped_duplicates += 1
                    continue
                seen.add(selected)
                label = (
                    f"{args.label_prefix}_{source_label}"
                    f"_drop_{feature_suffix(drop_features)}"
                    f"_add_{feature_suffix(add_features)}"
                )
                entries.append(format_bundle(label, list(selected)))
                rows.append(
                    {
                        "source_bundle": source_label,
                        "pool_bundle": pool_label,
                        "variant": label,
                        "drop_count": args.drop_count,
                        "add_count": args.add_count,
                        "dropped_features": ",".join(f"{layer}:{feature}" for layer, feature in drop_features),
                        "added_features": ",".join(f"{layer}:{feature}" for layer, feature in add_features),
                        "source_feature_count": len(source_features),
                        "pool_extra_count": len(extras),
                        "selected_features": len(selected),
                        "features": ",".join(f"{layer}:{feature}" for layer, feature in selected),
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
                "source_bundles_file": str(args.source_bundles_file),
                "pool_bundles_file": str(args.pool_bundles_file),
                "pool_label": pool_label,
                "drop_count": args.drop_count,
                "add_count": args.add_count,
                "variant_count": len(entries),
                "skipped_duplicate_feature_sets": skipped_duplicates,
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
    print(f"[save] {bundles_path} ({len(entries)} variants, {skipped_duplicates} duplicate feature sets skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
