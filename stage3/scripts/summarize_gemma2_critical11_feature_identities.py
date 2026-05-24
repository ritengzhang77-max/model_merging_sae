#!/usr/bin/env python3
"""Summarize local identities and roles for the critical k=11 SAE handles."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "stage3" / "results"

HOLOGRAM_DIR = (
    RESULTS
    / "gemma2_2b_linear_merge_sae_prompt_token_delta_rank_v0"
    / "fake_id_hologram_l20_final_newline_delta_abs_float32"
)
FAMILY_DIR = (
    RESULTS
    / "gemma2_2b_linear_merge_sae_prompt_token_delta_rank_v0"
    / "fake_id_family_v1_harmful_l20_final_newline_delta_abs_float32"
)
BROAD_DIR = (
    RESULTS
    / "gemma2_2b_linear_merge_sae_prompt_token_delta_rank_v0"
    / "default_paraphrase_guard_harmful_l20_final_newline_delta_abs_float32"
)
SWAP_DIR = (
    RESULTS
    / "gemma2_2b_linear_merge_sae_bundle_first_token_logits_v0"
    / "fake_id_hologram_l20_final_newline_delta_add_prompt_delta_critical11_swap_one_top33_float32"
)

OUT_DIR = RESULTS / "gemma2_2b_linear_merge_sae_11feature_identity_v0"
OUT_CSV = OUT_DIR / "critical11_feature_identity_table.csv"
OUT_MD = RESULTS / "GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_IDENTITY_SUMMARY.md"


def read_csv_by_int(path: Path, key: str) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {int(row[key]): row for row in csv.DictReader(f)}


def read_csv_by_key(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {str(row["key"]): row for row in csv.DictReader(f)}


def parse_bundle_features(raw: str) -> list[set[int]]:
    bundles: list[set[int]] = []
    for item in raw.strip().split(";"):
        if not item.strip():
            continue
        _, features = item.split("=", 1)
        bundle = set()
        for entry in features.split(","):
            layer, feature = entry.split(":")
            if int(layer) != 20:
                raise ValueError(f"expected layer 20 feature, got {entry}")
            bundle.add(int(feature))
        bundles.append(bundle)
    return bundles


def fmt_float(value: str | float | None, digits: int = 3) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_int(value: str | int | None) -> str:
    if value in (None, ""):
        return ""
    return str(int(value))


def short_logits(raw: str, limit: int = 4) -> str:
    if not raw:
        return ""
    parts = [part.strip() for part in raw.split("|") if part.strip()]
    return " | ".join(parts[:limit])


def infer_role(
    feature: int,
    handle_count: int,
    pass_bundle_count: int,
    drop_row: dict[str, str] | None,
    add_row: dict[str, str] | None,
) -> str:
    if handle_count == pass_bundle_count:
        if drop_row is None:
            return "invariant in pass handles; no one-swap drop row"
        passes = int(drop_row["pass_count"])
        ties = int(drop_row["tie_count"])
        max_margin = float(drop_row["max_margin"])
        mean_margin = float(drop_row["mean_margin"])
        if passes > 0:
            return "common but locally replaceable"
        if ties == 0:
            return "core-like; no one-swap replacement ties"
        if max_margin == 0.0 and mean_margin < -0.02:
            return "near-core; replacements mostly fail"
        return "common support; removals can tie but not pass"
    if drop_row and int(drop_row["pass_count"]) > 0:
        return "source feature with passing one-swap replacements"
    if add_row and int(add_row["pass_count"]) > 0:
        return "contextual substitute in pass handles"
    return "variable pass-handle member"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = read_csv_by_int(HOLOGRAM_DIR / "neuronpedia_top33" / "neuronpedia_feature_metadata.csv", "feature_id")
    hologram = read_csv_by_int(HOLOGRAM_DIR / "prompt_token_delta_features.csv", "feature_id")
    family = read_csv_by_int(FAMILY_DIR / "prompt_token_delta_features.csv", "feature_id")
    broad = read_csv_by_int(BROAD_DIR / "prompt_token_delta_features.csv", "feature_id")
    drop_rows = read_csv_by_key(SWAP_DIR / "critical11_swap_one_top33_effect_by_drop.csv")
    add_rows = read_csv_by_key(SWAP_DIR / "critical11_swap_one_top33_effect_by_add.csv")
    pass_bundles = parse_bundle_features((SWAP_DIR / "critical11_swap_one_top33_pass_bundles.txt").read_text(encoding="utf-8"))

    feature_union = sorted(set().union(*pass_bundles))
    feature_counts = {feature: sum(feature in bundle for bundle in pass_bundles) for feature in feature_union}
    pass_bundle_count = len(pass_bundles)
    feature_intersection = sorted(feature for feature, count in feature_counts.items() if count == pass_bundle_count)

    rows: list[dict[str, str]] = []
    for feature in feature_union:
        meta = metadata.get(feature, {})
        holo = hologram.get(feature, {})
        fam = family.get(feature, {})
        broad_row = broad.get(feature, {})
        drop = drop_rows.get(str(feature))
        add = add_rows.get(str(feature))
        rows.append(
            {
                "feature_id": str(feature),
                "pass_handle_count": str(feature_counts[feature]),
                "hologram_rank": fmt_int(holo.get("rank")),
                "hologram_signed_delta": fmt_float(holo.get("signed_delta_sum")),
                "hologram_donor_mean": fmt_float(holo.get("donor_mean")),
                "hologram_recipient_mean": fmt_float(holo.get("recipient_mean")),
                "family_rank": fmt_int(fam.get("rank")),
                "family_signed_delta": fmt_float(fam.get("signed_delta_sum")),
                "family_donor_mean": fmt_float(fam.get("donor_mean")),
                "family_recipient_mean": fmt_float(fam.get("recipient_mean")),
                "broad_harmful_rank": fmt_int(broad_row.get("rank")),
                "broad_harmful_signed_delta": fmt_float(broad_row.get("signed_delta_sum")),
                "broad_harmful_donor_mean": fmt_float(broad_row.get("donor_mean")),
                "broad_harmful_recipient_mean": fmt_float(broad_row.get("recipient_mean")),
                "drop_passes": drop["pass_count"] if drop else "",
                "drop_ties": drop["tie_count"] if drop else "",
                "drop_mean_margin": fmt_float(drop.get("mean_margin") if drop else None, 4),
                "drop_max_margin": fmt_float(drop.get("max_margin") if drop else None, 4),
                "add_passes": add["pass_count"] if add else "",
                "add_ties": add["tie_count"] if add else "",
                "add_mean_margin": fmt_float(add.get("mean_margin") if add else None, 4),
                "description": meta.get("description", ""),
                "positive_logits": short_logits(meta.get("positive_logits", "")),
                "negative_logits": short_logits(meta.get("negative_logits", "")),
                "role": infer_role(feature, feature_counts[feature], pass_bundle_count, drop, add),
            }
        )

    fieldnames = [
        "feature_id",
        "pass_handle_count",
        "hologram_rank",
        "hologram_signed_delta",
        "hologram_donor_mean",
        "hologram_recipient_mean",
        "family_rank",
        "family_signed_delta",
        "family_donor_mean",
        "family_recipient_mean",
        "broad_harmful_rank",
        "broad_harmful_signed_delta",
        "broad_harmful_donor_mean",
        "broad_harmful_recipient_mean",
        "drop_passes",
        "drop_ties",
        "drop_mean_margin",
        "drop_max_margin",
        "add_passes",
        "add_ties",
        "add_mean_margin",
        "description",
        "positive_logits",
        "negative_logits",
        "role",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Gemma-2-2B Linear Merge SAE 11-Feature Identity Summary",
        "",
        "Date: 2026-05-24",
        "",
        "This memo links the six validated k=11 first-token-passing handles to",
        "local feature identities. The goal is not to assign final semantic labels;",
        "it is to separate three things that were previously mixed together:",
        "",
        "- which SAE features are present in the local pass class;",
        "- how those features move between donor alpha `1.0` and recipient alpha `0.75`;",
        "- which features behave as local backbone, exchangeable members, or substitutes.",
        "",
        "Neuronpedia descriptions are used as weak labels only. Local activation and",
        "causal-swap evidence takes priority over autointerp wording.",
        "",
        "## Pass-Class Feature Set",
        "",
        f"The one-swap screen has `{pass_bundle_count}` first-token-passing k=11 handles.",
        f"Their feature union has `{len(feature_union)}` features, and their intersection has `{len(feature_intersection)}` features:",
        "",
        "```text",
        ", ".join(str(feature) for feature in feature_intersection),
        "```",
        "",
        "The intersection is not the same thing as causal necessity. The one-swap",
        "drop screen is stricter: features such as `15169` and `14991` never tie when",
        "dropped, while several other intersection features can tie but not pass.",
        "",
        "## Feature Table",
        "",
        "| feature | in passes | hologram rank | fake-ID family rank | broad harmful rank | broad delta | drop pass/tie | add pass/tie | local role | weak Neuronpedia label |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['feature_id']}` | {row['pass_handle_count']}/6 | "
            f"{row['hologram_rank'] or '-'} | {row['family_rank'] or '-'} | "
            f"{row['broad_harmful_rank'] or '-'} | {row['broad_harmful_signed_delta'] or '-'} | "
            f"{row['drop_passes'] or '0'}/{row['drop_ties'] or '0'} | "
            f"{row['add_passes'] or '0'}/{row['add_ties'] or '0'} | "
            f"{row['role']} | {row['description'] or '-'} |"
        )

    lines.extend(
        [
            "",
            "## What This Says Mechanistically",
            "",
            "The current pass class looks less like a clean semantic refusal circuit and",
            "more like a signed first-token control bundle. The strongest local backbone",
            "evidence is feature `15169`: it is the largest hologram-prompt delta, appears",
            "in all six pass handles, and every one-swap replacement after dropping it",
            "fails far below the gate. Feature `14991` is also core-like: it appears in",
            "all pass handles, is the top fake-ID-family harmful delta, and no one-swap",
            "replacement even ties after dropping it.",
            "",
            "Several other common features are better described as support features, not",
            "individually necessary semantic atoms. Dropping them often creates ties near",
            "the `I`/`It` boundary, which means the residual state remains close to the",
            "decision surface but loses enough signed composition to stop crossing it.",
            "",
            "The variable features give the clearest equivalence-class evidence. Features",
            "`6289`, `8775`, and `1338` can be exchanged in narrow contexts, while `7531`",
            "and `9407` are contextual substitutes that produce new passing handles only",
            "for specific source/drop combinations.",
            "",
            "The broad harmful-prompt ranking adds an important generalization check.",
            "Feature `14991` remains rank `1`, feature `15169` remains high at rank `15`,",
            "and feature `1813` is rank `19`. Those three look like a general refusal-boundary",
            "backbone rather than fake-ID-only features. By contrast, several",
            "other pass-class members fall far down the broad ranking or have zero broad",
            "delta: `6289`, `9149`, `12704`, and `13622` are examples. This supports a",
            "two-part account: the handle combines a general donor-refusal backbone with",
            "domain/local support features needed to tip the fake-ID final-newline state.",
            "",
            "A key caution is that the autointerp labels are not safety-specific. Some",
            "important features have labels about code, organization, or self-reflection;",
            "their local logits and causal role matter more here than the surface label.",
            "This supports the project's emerging thesis: merging moves a model across a",
            "behavioral boundary through a small signed feature class, but the class is",
            "not a simple human-readable refusal feature set.",
            "",
            "## Output Artifact",
            "",
            f"- CSV table: `{OUT_CSV}`",
            f"- Broad harmful ranking: `{BROAD_DIR / 'PROMPT_TOKEN_DELTA_RANKING_SUMMARY.md'}`",
            "- Decomposition summary: `stage3/results/GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_BACKBONE_DECOMP_SUMMARY.md`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[save] {OUT_CSV}")
    print(f"[save] {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
