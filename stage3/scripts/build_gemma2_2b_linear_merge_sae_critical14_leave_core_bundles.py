#!/usr/bin/env python3
"""Build leave-one-core variants for validated critical14 bundles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE_BUNDLES = (
    ROOT
    / "stage3"
    / "results"
    / "gemma2_2b_linear_merge_sae_bundle_patch_v0"
    / "fake_id_hologram_l20_final_newline_delta_add_prompt_delta_critical14_candidates_float32_max160"
    / "critical14_candidate_bundles.txt"
)
DEFAULT_FEATURE_TABLE = (
    ROOT
    / "stage3"
    / "results"
    / "gemma2_2b_linear_merge_sae_prompt_token_delta_rank_v0"
    / "fake_id_hologram_l20_final_newline_delta_abs_float32"
    / "critical14_feature_table"
    / "critical14_feature_table.csv"
)
DEFAULT_RESULT_DIR = (
    ROOT
    / "stage3"
    / "results"
    / "gemma2_2b_linear_merge_sae_bundle_first_token_logits_v0"
    / "fake_id_hologram_l20_final_newline_delta_add_prompt_delta_critical14_leave_core_float32"
)


def parse_bundle_text(text: str) -> dict[str, list[tuple[int, int]]]:
    bundles: dict[str, list[tuple[int, int]]] = {}
    for raw_entry in text.strip().split(";"):
        entry = raw_entry.strip()
        if not entry:
            continue
        label, payload = entry.split("=", 1)
        items: list[tuple[int, int]] = []
        for raw_item in payload.split(","):
            layer_text, feature_text = raw_item.split(":", 1)
            items.append((int(layer_text), int(feature_text)))
        bundles[label] = items
    return bundles


def format_bundle(label: str, items: list[tuple[int, int]]) -> str:
    return f"{label}=" + ",".join(f"{layer}:{feature}" for layer, feature in items)


def read_feature_table(path: Path) -> tuple[set[int], dict[int, int]]:
    critical_features: set[int] = set()
    feature_to_rank: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            feature_id = int(row["feature_id"])
            rank = int(row["rank"])
            feature_to_rank[feature_id] = rank
            if row["role"] == "critical_core":
                critical_features.add(feature_id)
    return critical_features, feature_to_rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-bundles", type=Path, default=DEFAULT_CANDIDATE_BUNDLES)
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    critical_features, feature_to_rank = read_feature_table(args.feature_table)
    candidates = parse_bundle_text(args.candidate_bundles.read_text(encoding="utf-8"))

    entries: list[str] = []
    rows: list[dict[str, object]] = []
    for label, items in candidates.items():
        for layer, feature_id in items:
            if feature_id not in critical_features:
                continue
            rank = feature_to_rank[feature_id]
            reduced = [(item_layer, item_feature) for item_layer, item_feature in items if item_feature != feature_id]
            out_label = f"{label}_drop_rank{rank:03d}"
            entries.append(format_bundle(out_label, reduced))
            rows.append(
                {
                    "source_bundle": label,
                    "variant": out_label,
                    "dropped_rank": rank,
                    "dropped_feature_id": feature_id,
                    "selected_features": len(reduced),
                    "layer": layer,
                }
            )

    bundles_path = args.result_dir / "critical14_leave_core_bundles.txt"
    csv_path = args.result_dir / "critical14_leave_core_manifest.csv"
    json_path = args.result_dir / "build_manifest.json"
    bundles_path.write_text(";".join(entries) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {
                "candidate_bundles": str(args.candidate_bundles),
                "feature_table": str(args.feature_table),
                "outputs": {
                    "bundles": str(bundles_path),
                    "manifest_csv": str(csv_path),
                },
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
