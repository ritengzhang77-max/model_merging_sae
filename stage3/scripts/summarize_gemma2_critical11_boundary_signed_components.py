#!/usr/bin/env python3
"""Summarize signed boundary-component causal tests for the k=11 handle class."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "stage3" / "results"
FIRST_TOKEN_DIR = (
    RESULTS
    / "gemma2_2b_linear_merge_sae_bundle_first_token_logits_v0"
    / "fake_id_hologram_l20_final_newline_delta_add_boundary_signed_components_float32"
)
OUT_DIR = RESULTS / "gemma2_2b_linear_merge_sae_11feature_identity_v0"
OUT_CSV = OUT_DIR / "boundary_signed_component_outcomes.csv"
OUT_MD = RESULTS / "GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_BOUNDARY_SIGNED_COMPONENT_SUMMARY.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.6f}"


def outcome(margin: float) -> str:
    if margin > 0:
        return "pass"
    if margin == 0:
        return "tie"
    return "fail"


def main() -> int:
    summary = read_csv(FIRST_TOKEN_DIR / "first_token_logit_summary.csv")
    by_condition_split = {(row["condition"], row["split"]): row for row in summary}
    rows = []
    conditions = sorted({row["condition"] for row in summary})
    for condition in conditions:
        harmful = by_condition_split.get((condition, "harmful"))
        benign = by_condition_split.get((condition, "benign"))
        if harmful is None:
            continue
        harmful_margin = float(harmful["mean_i_minus_it"])
        benign_margin = float(benign["mean_i_minus_it"]) if benign else 0.0
        rows.append(
            {
                "condition": condition,
                "harmful_i_minus_it": fmt(harmful_margin),
                "harmful_outcome": outcome(harmful_margin),
                "benign_i_minus_it": fmt(benign_margin),
            }
        )
    write_csv(OUT_CSV, rows)
    row_by_condition = {row["condition"]: row for row in rows}

    key_order = [
        "linear_alpha_0.75",
        "linear_alpha_1",
        "bundle_patch_boundary_donor_higher9",
        "bundle_patch_boundary_recipient_higher5",
        "bundle_patch_boundary_signed_all14",
        "bundle_patch_common9_donor_higher6",
        "bundle_patch_common9_recipient_higher3",
        "bundle_patch_common9_signed9",
        "bundle_patch_variable_donor_higher3",
        "bundle_patch_variable_recipient_higher2",
        "bundle_patch_common9_signed9_plus_variable_donor3",
        "bundle_patch_common9_signed9_plus_variable_recipient2",
        "bundle_patch_boundary_donor_higher9_plus_12704",
        "bundle_patch_boundary_donor_higher9_plus_6289",
        "bundle_patch_boundary_donor_higher9_plus_9149",
        "bundle_patch_boundary_donor_higher9_plus_13622",
        "bundle_patch_boundary_donor_higher9_plus_9407",
    ]

    def margin(condition: str) -> str:
        return row_by_condition[condition]["harmful_i_minus_it"]

    lines = [
        "# Gemma-2-2B Linear Merge SAE 11-Feature Boundary Signed-Component Summary",
        "",
        "Date: 2026-05-24",
        "",
        "This checkpoint causally tests the signed interpretation from the",
        "prompt-boundary event audit. Features are split by whether their layer-20",
        "SAE value is donor-higher or recipient-higher at the assistant final",
        "newline.",
        "",
        "## First-Token Results",
        "",
        "| condition | harmful `I-It` | outcome | benign `I-It` |",
        "|---|---:|---|---:|",
    ]
    for condition in key_order:
        row = row_by_condition[condition]
        lines.append(
            f"| `{condition}` | `{row['harmful_i_minus_it']}` | "
            f"{row['harmful_outcome']} | `{row['benign_i_minus_it']}` |"
        )

    lines.extend(
        [
            "",
            "## Mechanistic Reading",
            "",
            "Neither sign-side works by itself. The donor-higher boundary features improve",
            f"the harmful margin from alpha0.75 `{margin('linear_alpha_0.75')}` to",
            f"`{margin('bundle_patch_boundary_donor_higher9')}`, but still fail. The",
            "recipient-higher features alone barely move the baseline.",
            "",
            "The common signed backbone is necessary but still only reaches the exact",
            f"boundary: `common9_signed9` has `{margin('bundle_patch_common9_signed9')}`.",
            "Adding either donor-side variable features or recipient-side variable",
            "suppression to that signed common backbone crosses the gate:",
            "",
            f"- `common9_signed9_plus_variable_donor3`: `{margin('bundle_patch_common9_signed9_plus_variable_donor3')}`",
            f"- `common9_signed9_plus_variable_recipient2`: `{margin('bundle_patch_common9_signed9_plus_variable_recipient2')}`",
            "",
            "Single recipient-higher additions to the donor-higher bundle do not cross;",
            "the best tested single addition, `12704`, remains at",
            f"`{margin('bundle_patch_boundary_donor_higher9_plus_12704')}`. This argues",
            "against a single negative-feature explanation. The signed effect is",
            "cooperative and threshold-like.",
            "",
            "## Interpretation",
            "",
            "This is the clearest causal support so far for the prompt-boundary account:",
            "the merge-created state is tipped by a signed mixture of donor additions and",
            "recipient suppressions. That is exactly the kind of mechanism that would be",
            "hard to read from feature labels alone.",
            "",
            "## Artifacts",
            "",
            f"- Boundary signed bundles: `{OUT_DIR / 'boundary_signed_component_bundles.txt'}`",
            f"- First-token audit: `{FIRST_TOKEN_DIR}`",
            f"- Outcome CSV: `{OUT_CSV}`",
            "- Prompt-boundary event summary: `stage3/results/GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_PROMPT_BOUNDARY_EVENT_SUMMARY.md`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[save] {OUT_CSV}")
    print(f"[save] {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
