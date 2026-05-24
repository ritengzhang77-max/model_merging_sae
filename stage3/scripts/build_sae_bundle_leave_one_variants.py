#!/usr/bin/env python3
"""Build leave-one-feature variants from SAE bundle specs."""

from __future__ import annotations

import argparse
import csv
import json
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
    parser.add_argument("--bundles-file", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="leave_one_bundles.txt")
    parser.add_argument("--label-prefix", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    bundles = parse_bundle_text(args.bundles_file.read_text(encoding="utf-8"))

    entries: list[str] = []
    rows: list[dict[str, object]] = []
    for source_label, features in bundles.items():
        for index, (drop_layer, drop_feature) in enumerate(features, start=1):
            reduced = [(layer, feature) for layer, feature in features if not (layer == drop_layer and feature == drop_feature)]
            out_label = f"{args.label_prefix}{source_label}_drop_f{drop_feature}"
            entries.append(format_bundle(out_label, reduced))
            rows.append(
                {
                    "source_bundle": source_label,
                    "variant": out_label,
                    "source_feature_index": index,
                    "dropped_layer": drop_layer,
                    "dropped_feature_id": drop_feature,
                    "source_feature_count": len(features),
                    "selected_features": len(reduced),
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
                "bundles_file": str(args.bundles_file),
                "output_bundles": str(bundles_path),
                "manifest_csv": str(manifest_csv_path),
                "bundle_count": len(entries),
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
