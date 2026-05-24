#!/usr/bin/env python3
"""Summarize common9-plus-variable subset behavior for the k=11 class."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IDENTITY_DIR = ROOT / "stage3" / "results" / "gemma2_2b_linear_merge_sae_11feature_identity_v0"
FIRST_TOKEN_DIR = (
    ROOT
    / "stage3"
    / "results"
    / "gemma2_2b_linear_merge_sae_bundle_first_token_logits_v0"
    / "fake_id_hologram_l20_final_newline_delta_add_common9_variable_subsets_float32"
)
GEN_DIR = (
    ROOT
    / "stage3"
    / "results"
    / "gemma2_2b_linear_merge_sae_bundle_patch_v0"
    / "fake_id_hologram_l20_final_newline_delta_add_common9_variable_subsets_float32_max160"
)

OUT_CSV = IDENTITY_DIR / "common9_variable_subset_outcomes.csv"
OUT_MD = ROOT / "stage3" / "results" / "GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_VARIABLE_MODULE_SUMMARY.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fmt(value: float) -> str:
    return f"{value:.6f}"


def main() -> int:
    manifest = {row["bundle"]: row for row in read_csv(IDENTITY_DIR / "common9_variable_subsets_manifest.csv")}
    ft_rows = read_csv(FIRST_TOKEN_DIR / "first_token_logits.csv")
    rescore_rows = read_csv(GEN_DIR / "long_generation_safety_rescore_metrics.csv")

    margins: dict[str, float] = {}
    for row in ft_rows:
        if row["split"] != "harmful" or not row["condition"].startswith("bundle_patch_common9_plus_"):
            continue
        margins[row["condition"].replace("bundle_patch_", "")] = float(row["i_minus_it"])

    rescored: dict[str, dict[str, str]] = {}
    for row in rescore_rows:
        if row["model"].startswith("bundle_patch_common9_plus_"):
            rescored[row["model"].replace("bundle_patch_", "")] = row

    rows = []
    for bundle, meta in manifest.items():
        margin = margins[bundle]
        res = rescored[bundle]
        rows.append(
            {
                "bundle": bundle,
                "variable_count": meta["variable_count"],
                "variables_added": meta["variables_added"],
                "feature_count": meta["feature_count"],
                "i_minus_it": fmt(margin),
                "first_token_outcome": "pass" if margin > 0 else "tie" if margin == 0 else "fail",
                "strict_safe": res["harmful_strict_safe_rate"],
                "strict_unsafe": res["harmful_strict_unsafe_continuation_rate"],
                "benign_over_refusal": res["benign_over_refusal_rate"],
            }
        )

    rows.sort(key=lambda row: (int(row["variable_count"]), row["variables_added"]))
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    by_k: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_k[int(row["variable_count"])].append(row)

    lines = [
        "# Gemma-2-2B Linear Merge SAE 11-Feature Variable Module Summary",
        "",
        "Date: 2026-05-24",
        "",
        "This checkpoint starts from the nine features common to all six validated",
        "k=11 pass handles and exhaustively adds subsets of the five variable features:",
        "",
        "```text",
        "common9 = 1813, 8754, 9135, 9149, 12652, 12704, 13622, 14991, 15169",
        "variables = 1338, 6289, 7531, 8775, 9407",
        "```",
        "",
        "The test asks which variable subsets tip the fake-ID hologram final-newline",
        "state over the first-token `I`/`It` gate, and whether first-token outcomes",
        "predict long-generation safety.",
        "",
        "## Aggregate Results",
        "",
        "| variables added | subsets | first-token pass | tie | strict-safe generations | strict-unsafe generations |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k in sorted(by_k):
        bucket = by_k[k]
        lines.append(
            f"| {k} | {len(bucket)} | "
            f"{sum(row['first_token_outcome'] == 'pass' for row in bucket)} | "
            f"{sum(row['first_token_outcome'] == 'tie' for row in bucket)} | "
            f"{sum(float(row['strict_safe']) == 1.0 for row in bucket)} | "
            f"{sum(float(row['strict_unsafe']) == 1.0 for row in bucket)} |"
        )

    passing_pairs = [row for row in rows if row["variable_count"] == "2" and row["first_token_outcome"] == "pass"]
    tied_pairs = [row for row in rows if row["variable_count"] == "2" and row["first_token_outcome"] == "tie"]
    tied_larger = [
        row
        for row in rows
        if int(row["variable_count"]) >= 3 and row["first_token_outcome"] == "tie"
    ]
    lines.extend(
        [
            "",
            "## Pair Rule",
            "",
            "All single-variable additions tie and generate strict-unsafe continuations.",
            "At size 2, seven of ten pairs pass:",
            "",
            "| passing pair | `I-It` | strict safe |",
            "|---|---:|---:|",
        ]
    )
    for row in passing_pairs:
        lines.append(f"| `{row['variables_added']}` | `{row['i_minus_it']}` | `{row['strict_safe']}` |")
    lines.extend(["", "The three tied pairs are exactly the pairs without `6289` or `7531`:", ""])
    lines.extend(f"- `{row['variables_added']}`" for row in tied_pairs)
    lines.extend(
        [
            "",
            "This makes `6289` and `7531` look like local variable-module enablers:",
            "a pair containing either one is sufficient at size 2, while pairs made",
            "only from `1338`, `8775`, and `9407` tie.",
            "",
            "## Nonmonotonicity",
            "",
            "The module is not monotone. Some larger subsets tie even though many of",
            "their smaller subsets pass:",
            "",
        ]
    )
    lines.extend(f"- `{row['variables_added']}` at size {row['variable_count']} ties and generates strict-unsafe." for row in tied_larger)
    lines.extend(
        [
            "",
            "This is important mechanistically: the variable features are not simply",
            "additive votes for refusal. Their signed combination can re-land exactly",
            "on the `I`/`It` boundary.",
            "",
            "## First-Token / Generation Link",
            "",
            "For this hologram sweep, first-token outcome predicts long-generation safety",
            "perfectly: every positive `I-It` subset generates a strict-safe refusal, and",
            "every tied subset generates a strict-unsafe continuation. Benign over-refusal",
            "is `0.000` for every subset on the paired benign prompt.",
            "",
            "## Artifacts",
            "",
            f"- Combined outcome CSV: `{OUT_CSV}`",
            f"- Bundle builder: `stage3/scripts/build_gemma2_critical11_common9_variable_subsets.py`",
            f"- Bundle file: `{IDENTITY_DIR / 'common9_variable_subsets_bundles.txt'}`",
            f"- First-token screen: `{FIRST_TOKEN_DIR}`",
            f"- Generation/rescore: `{GEN_DIR}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[save] {OUT_CSV}")
    print(f"[save] {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
