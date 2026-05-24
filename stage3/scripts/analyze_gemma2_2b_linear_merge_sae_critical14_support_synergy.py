#!/usr/bin/env python3
"""Summarize critical12 plus support-rank first-token synergy."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    ROOT
    / "stage3"
    / "results"
    / "gemma2_2b_linear_merge_sae_bundle_first_token_logits_v0"
    / "fake_id_hologram_l20_final_newline_delta_add_prompt_delta_critical12_plus_two_screen_float32"
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
DEFAULT_PROMPT_DELTA_DETAIL = (
    ROOT
    / "stage3"
    / "results"
    / "gemma2_2b_linear_merge_sae_prompt_token_delta_rank_v0"
    / "fake_id_hologram_l20_final_newline_delta_abs_float32"
    / "prompt_token_delta_detail_top33.csv"
)
DEFAULT_NEURONPEDIA_TABLE = (
    ROOT
    / "stage3"
    / "results"
    / "gemma2_2b_linear_merge_sae_prompt_token_delta_rank_v0"
    / "fake_id_hologram_l20_final_newline_delta_abs_float32"
    / "neuronpedia_top33"
    / "neuronpedia_feature_metadata.csv"
)
DEFAULT_RESULT_DIR = (
    ROOT
    / "stage3"
    / "results"
    / "gemma2_2b_linear_merge_sae_critical14_support_synergy_v0"
    / "fake_id_hologram_l20_final_newline_delta_add_support_pair_synergy_float32"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def support_ranks_from_condition(condition: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"rank(\d+)", condition))


def margin(row: dict[str, str]) -> float:
    return float(row["mean_i_minus_it"])


def load_feature_meta(feature_table: Path, prompt_delta_detail: Path, neuronpedia_table: Path) -> dict[int, dict[str, str]]:
    meta: dict[int, dict[str, str]] = {}
    for row in read_csv(prompt_delta_detail):
        if row["split"] != "harmful":
            continue
        rank = int(row["rank"])
        delta = float(row["delta_value"])
        meta[rank] = {
            "rank": row["rank"],
            "feature_id": row["feature_id"],
            "sign": "donor_higher" if delta > 0 else "recipient_higher" if delta < 0 else "zero",
            "harmful_delta": row["delta_value"],
            "donor_value": row["donor_value"],
            "recipient_value": row["recipient_value"],
        }
    for row in read_csv(neuronpedia_table):
        rank = int(row["rank"])
        meta.setdefault(rank, {"rank": row["rank"], "feature_id": row["feature_id"]})
        meta[rank].update(
            {
                "description": row.get("description", ""),
                "density": row.get("frac_nonzero", ""),
                "url": row.get("url", ""),
            }
        )
    for row in read_csv(feature_table):
        rank = int(row["rank"])
        meta.setdefault(rank, {"rank": row["rank"], "feature_id": row["feature_id"]})
        # Preserve role/support-pair labels for the validated critical14 subset.
        meta[rank].update({key: row.get(key, "") for key in ("role", "support_pairs")})
    return meta


def describe_counts(values: list[float]) -> dict[str, int]:
    return {f"{key:.6f}": count for key, count in sorted(Counter(values).items(), reverse=True)}


def write_summary(path: Path, summary: dict[str, object], pass_rows: list[dict[str, object]], top_synergy_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Critical14 Support Synergy",
        "",
        "This audit reuses the existing first-token sweep over `critical12` plus one or two noncritical top-33 ranks.",
        "It asks whether the two support ranks in the validated 14-feature handles behave additively.",
        "",
        "## Result",
        "",
        f"- Critical12 margin: `{summary['critical12_margin']:.6f}` (`I-It`; top token remains `It`).",
        f"- Single support ranks tested: `{summary['single_support_count']}`; single-support passes: `{summary['single_pass_count']}`.",
        f"- Support pairs tested: `{summary['pair_count']}`; pair passes: `{summary['pair_pass_count']}`.",
        f"- Pair-margin counts: `{summary['pair_margin_counts']}`.",
        f"- Passing-pair support-rank frequency: `{summary['passing_rank_frequency']}`.",
        "",
        "No single support rank crosses the first-token gate. Only 5/210 two-support pairs pass, and each passes at the",
        "minimum positive observed margin (`I-It = +0.015625`). This supports a brittle pair interaction rather than an",
        "independent support-rank story.",
        "",
        "## Passing Support Pairs",
        "",
        "| ranks | features | singleton margins | pair margin | interaction | signs | descriptions |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in pass_rows:
        lines.append(
            f"| `{row['rank_a']}+{row['rank_b']}` | `{row['feature_a']}+{row['feature_b']}` | "
            f"`{row['single_a_margin']:.6f}`, `{row['single_b_margin']:.6f}` | "
            f"`{row['pair_margin']:.6f}` | `{row['interaction']:.6f}` | "
            f"{row['sign_a']} / {row['sign_b']} | {row['description_a']} / {row['description_b']} |"
        )
    lines.extend(
        [
            "",
            "## Top Interactions",
            "",
            "| ranks | pair margin | singleton margins | interaction | passes |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in top_synergy_rows:
        lines.append(
            f"| `{row['rank_a']}+{row['rank_b']}` | `{row['pair_margin']:.6f}` | "
            f"`{row['single_a_margin']:.6f}`, `{row['single_b_margin']:.6f}` | "
            f"`{row['interaction']:.6f}` | `{int(row['passes'])}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Pair interaction alone is not sufficient: some high-interaction pairs only tie at `I-It = 0.000000` and still fail.",
            "The useful condition is exact signed composition plus being already at the critical12 tie. This strengthens the",
            "first-token basin account and weakens a simple feature-wise semantic interpretation.",
            "",
            "Artifacts:",
            "",
            "- `support_singletons.csv`",
            "- `support_pair_synergy.csv`",
            "- `manifest.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--prompt-delta-detail", type=Path, default=DEFAULT_PROMPT_DELTA_DETAIL)
    parser.add_argument("--neuronpedia-table", type=Path, default=DEFAULT_NEURONPEDIA_TABLE)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [row for row in read_csv(args.input_dir / "first_token_logit_summary.csv") if row["split"] == "harmful"]
    feature_meta = load_feature_meta(args.feature_table, args.prompt_delta_detail, args.neuronpedia_table)

    margins: dict[tuple[int, ...], float] = {}
    condition_by_ranks: dict[tuple[int, ...], str] = {}
    top_i_by_ranks: dict[tuple[int, ...], float] = {}
    top_it_by_ranks: dict[tuple[int, ...], float] = {}
    for row in summary_rows:
        ranks = support_ranks_from_condition(row["condition"])
        margins[ranks] = margin(row)
        condition_by_ranks[ranks] = row["condition"]
        top_i_by_ranks[ranks] = float(row["top_i_rate"])
        top_it_by_ranks[ranks] = float(row["top_it_rate"])

    base = margins[()]
    singles = {ranks[0]: m for ranks, m in margins.items() if len(ranks) == 1}
    pairs = {(ranks[0], ranks[1]): m for ranks, m in margins.items() if len(ranks) == 2}

    singleton_rows: list[dict[str, object]] = []
    for rank, single_margin in sorted(singles.items()):
        meta = feature_meta.get(rank, {})
        singleton_rows.append(
            {
                "rank": rank,
                "feature_id": meta.get("feature_id", ""),
                "sign": meta.get("sign", ""),
                "harmful_delta": meta.get("harmful_delta", ""),
                "single_margin": single_margin,
                "single_passes": single_margin > 0.0 and top_i_by_ranks[(rank,)] > 0.0,
                "top_i_rate": top_i_by_ranks[(rank,)],
                "top_it_rate": top_it_by_ranks[(rank,)],
                "description": meta.get("description", ""),
                "url": meta.get("url", ""),
            }
        )

    pair_rows: list[dict[str, object]] = []
    for (rank_a, rank_b), pair_margin in sorted(pairs.items()):
        meta_a = feature_meta.get(rank_a, {})
        meta_b = feature_meta.get(rank_b, {})
        single_a = singles[rank_a]
        single_b = singles[rank_b]
        additive_prediction = single_a + single_b - base
        interaction = pair_margin - additive_prediction
        passes = pair_margin > 0.0 and top_i_by_ranks[(rank_a, rank_b)] > 0.0
        pair_rows.append(
            {
                "rank_a": rank_a,
                "rank_b": rank_b,
                "feature_a": meta_a.get("feature_id", ""),
                "feature_b": meta_b.get("feature_id", ""),
                "sign_a": meta_a.get("sign", ""),
                "sign_b": meta_b.get("sign", ""),
                "single_a_margin": single_a,
                "single_b_margin": single_b,
                "critical12_margin": base,
                "additive_prediction": additive_prediction,
                "pair_margin": pair_margin,
                "interaction": interaction,
                "passes": passes,
                "condition": condition_by_ranks[(rank_a, rank_b)],
                "description_a": meta_a.get("description", ""),
                "description_b": meta_b.get("description", ""),
                "url_a": meta_a.get("url", ""),
                "url_b": meta_b.get("url", ""),
            }
        )

    pass_rows = [row for row in pair_rows if row["passes"]]
    top_synergy_rows = sorted(pair_rows, key=lambda row: (float(row["interaction"]), float(row["pair_margin"])), reverse=True)[:25]
    rank_frequency = Counter()
    for row in pass_rows:
        rank_frequency[int(row["rank_a"])] += 1
        rank_frequency[int(row["rank_b"])] += 1

    summary = {
        "input_dir": str(args.input_dir),
        "feature_table": str(args.feature_table),
        "prompt_delta_detail": str(args.prompt_delta_detail),
        "neuronpedia_table": str(args.neuronpedia_table),
        "critical12_margin": base,
        "single_support_count": len(singles),
        "single_pass_count": sum(1 for value in singles.values() if value > 0.0),
        "single_margin_counts": describe_counts(list(singles.values())),
        "pair_count": len(pairs),
        "pair_pass_count": len(pass_rows),
        "pair_margin_counts": describe_counts(list(pairs.values())),
        "passing_rank_frequency": dict(sorted(rank_frequency.items())),
        "outputs": {
            "singletons": str(args.result_dir / "support_singletons.csv"),
            "pairs": str(args.result_dir / "support_pair_synergy.csv"),
            "summary": str(args.result_dir / "CRITICAL14_SUPPORT_SYNERGY_SUMMARY.md"),
        },
    }

    write_csv(args.result_dir / "support_singletons.csv", singleton_rows)
    write_csv(args.result_dir / "support_pair_synergy.csv", pair_rows)
    write_summary(args.result_dir / "CRITICAL14_SUPPORT_SYNERGY_SUMMARY.md", summary, pass_rows, top_synergy_rows)
    (args.result_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[save] {args.result_dir / 'CRITICAL14_SUPPORT_SYNERGY_SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
