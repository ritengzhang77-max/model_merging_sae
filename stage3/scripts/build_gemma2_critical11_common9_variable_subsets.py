#!/usr/bin/env python3
"""Build common9-plus-variable subset bundles for the k=11 pass class."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_linear_merge_sae_11feature_identity_v0"
OUT_BUNDLES = OUT_DIR / "common9_variable_subsets_bundles.txt"
OUT_MANIFEST = OUT_DIR / "common9_variable_subsets_manifest.csv"
OUT_JSON = OUT_DIR / "common9_variable_subsets_build_manifest.json"

LAYER = 20
COMMON9 = [1813, 8754, 9135, 9149, 12652, 12704, 13622, 14991, 15169]
VARIABLES = [1338, 6289, 7531, 8775, 9407]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    bundle_parts = []
    for k in range(0, len(VARIABLES) + 1):
        for combo in itertools.combinations(VARIABLES, k):
            features = sorted(set(COMMON9).union(combo))
            combo_name = "none" if not combo else "_".join(str(x) for x in combo)
            name = f"common9_plus_{combo_name}"
            spec = ",".join(f"{LAYER}:{feature}" for feature in features)
            bundle_parts.append(f"{name}={spec}")
            rows.append(
                {
                    "bundle": name,
                    "variable_count": k,
                    "variables_added": ",".join(str(x) for x in combo),
                    "selected_features": ",".join(str(x) for x in features),
                    "feature_count": len(features),
                }
            )

    OUT_BUNDLES.write_text(";".join(bundle_parts) + "\n", encoding="utf-8")
    with OUT_MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    OUT_JSON.write_text(
        json.dumps(
            {
                "layer": LAYER,
                "common9": COMMON9,
                "variables": VARIABLES,
                "bundle_count": len(rows),
                "outputs": {
                    "bundles": str(OUT_BUNDLES),
                    "manifest": str(OUT_MANIFEST),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[save] {OUT_BUNDLES}")
    print(f"[save] {OUT_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
