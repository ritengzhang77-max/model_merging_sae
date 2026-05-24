#!/usr/bin/env python3
"""Build drop-one/add-one SAE bundle variants from a source and feature pool."""

from __future__ import annotations

import argparse
import csv
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


def select_bundle(bundles: dict[str, list[Feature]], label: str | None, flag_name: str) -> tuple[str, list[Feature]]:
    if label is None:
        if len(bundles) != 1:
            raise ValueError(f"{flag_name} is required when the file has multiple bundles")
        selected_label = next(iter(bundles))
    else:
        selected_label = label
    return selected_label, bundles[selected_label]


def format_bundle(label: str, features: list[Feature]) -> str:
    return f"{label}=" + ",".join(f"{layer}:{feature}" for layer, feature in features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundles-file", type=Path, required=True)
    parser.add_argument("--source-label", default=None)
    parser.add_argument("--pool-bundles-file", type=Path, required=True)
    parser.add_argument("--pool-label", default=None)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="swap_one_bundles.txt")
    parser.add_argument("--label-prefix", default="swap")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    source_label, source_features = select_bundle(
        parse_bundle_text(args.source_bundles_file.read_text(encoding="utf-8")),
        args.source_label,
        "--source-label",
    )
    pool_label, pool_features = select_bundle(
        parse_bundle_text(args.pool_bundles_file.read_text(encoding="utf-8")),
        args.pool_label,
        "--pool-label",
    )

    source_set = set(source_features)
    extras = [feature for feature in pool_features if feature not in source_set]
    if not extras:
        raise ValueError("pool has no features outside the source bundle")

    entries: list[str] = []
    rows: list[dict[str, object]] = []
    seen: set[tuple[Feature, ...]] = set()
    for drop_index, drop_feature in enumerate(source_features, start=1):
        reduced = [feature for feature in source_features if feature != drop_feature]
        for add_index, add_feature in enumerate(extras, start=1):
            swapped = tuple(sorted(reduced + [add_feature]))
            if swapped in seen:
                continue
            seen.add(swapped)
            label = (
                f"{args.label_prefix}_{source_label}"
                f"_drop_f{drop_feature[1]}_add_f{add_feature[1]}"
            )
            entries.append(format_bundle(label, list(swapped)))
            rows.append(
                {
                    "source_bundle": source_label,
                    "pool_bundle": pool_label,
                    "variant": label,
                    "drop_index": drop_index,
                    "dropped_layer": drop_feature[0],
                    "dropped_feature_id": drop_feature[1],
                    "add_index": add_index,
                    "added_layer": add_feature[0],
                    "added_feature_id": add_feature[1],
                    "source_feature_count": len(source_features),
                    "pool_extra_count": len(extras),
                    "selected_features": len(swapped),
                    "features": ",".join(f"{layer}:{feature}" for layer, feature in swapped),
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
                "source_label": source_label,
                "pool_bundles_file": str(args.pool_bundles_file),
                "pool_label": pool_label,
                "source_feature_count": len(source_features),
                "pool_feature_count": len(pool_features),
                "pool_extra_count": len(extras),
                "variant_count": len(entries),
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
    print(f"[save] {bundles_path} ({len(entries)} variants)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
