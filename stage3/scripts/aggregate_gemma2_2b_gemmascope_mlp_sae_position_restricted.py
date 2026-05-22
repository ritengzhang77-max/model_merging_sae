#!/usr/bin/env python3
"""Aggregate atomic GemmaScope MLP-SAE position-restricted patch runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "stage3" / "results" / "gemma2_2b_gemmascope_mlp_sae_position_restricted_atomic_v0"
METRICS_NAME = "gemma2_2b_gemmascope_mlp_sae_feature_subset_metrics.csv"


def parse_dir_name(name: str) -> dict[str, object]:
    group_raw, rest = name.split("_eval_", 1)
    eval_raw, patch_raw = rest.split("_patch_", 1)
    patch_filter, budget_raw = patch_raw.rsplit("_k", 1)
    eval_start, eval_end = (int(x) for x in eval_raw.split("_", 1))
    group = {"all_12_20": "all", "mid_late_15_20": "mid_late"}.get(group_raw, group_raw)
    layers = {"all": "12-20", "mid_late": "15-20"}.get(group, "")
    return {
        "layer_group": group,
        "layers": layers,
        "eval_slice": f"{eval_start}:{eval_end}",
        "eval_start": eval_start,
        "examples_per_split": eval_end - eval_start,
        "patch_token_filter": patch_filter,
        "budget": f"k{budget_raw}",
    }


def collect_rows(result_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(result_root.glob(f"*/{METRICS_NAME}")):
        meta = parse_dir_name(metrics_path.parent.name)
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                out = dict(meta)
                out.update(
                    {
                        "variant": str(row["model"]).replace("feature_subset_", ""),
                        "harmful_clean": float(row["harmful_ok_rate"]),
                        "unsafe": float(row["harmful_unsafe_continuation_rate"]),
                        "benign_helpful": float(row["benign_ok_rate"]),
                        "source_dir": str(metrics_path.parent),
                    }
                )
                rows.append(out)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = [
        "layer_group",
        "layers",
        "eval_slice",
        "eval_start",
        "examples_per_split",
        "budget",
        "patch_token_filter",
        "variant",
        "harmful_clean",
        "unsafe",
        "benign_helpful",
        "source_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def metric(rows: list[dict[str, object]], group: str, eval_slice: str, patch_filter: str) -> dict[str, object] | None:
    for row in rows:
        if (
            row["layer_group"] == group
            and row["eval_slice"] == eval_slice
            and row["patch_token_filter"] == patch_filter
            and row["budget"] == "k1024"
        ):
            return row
    return None


def fmt_cell(row: dict[str, object] | None) -> str:
    if row is None:
        return ""
    return f"{float(row['harmful_clean']):.3f} / {float(row['unsafe']):.3f}"


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Gemma-2-2B GemmaScope MLP-SAE Position-Restricted Patch Findings",
        "",
        "Date: 2026-05-22",
        "",
        "This checkpoint tests whether successful all-token selected SAE features repair refusal because of prompt content, static assistant-boundary tokens, generated-token trajectory state, or a combination.",
        "",
        "All runs use the same all-token top-delta `mix_decode_delta_abs_k1024` feature selection from prompt slice `0:4` per split.",
        "",
        "## Main Result",
        "",
        "The strongest current mechanism is not content-token semantics and not a static assistant-boundary-only patch.",
        "",
        "The best reduced patch is `assistant_boundary_or_generated`: patch the assistant response boundary in the prompt, then patch generated-token history during autoregressive rollout.",
        "",
        "This nearly matches all-position repair while `contentish_or_generated` remains weak or zero.",
        "",
        "## Key Aggregate Table",
        "",
        "Cells are `harmful clean / unsafe continuation`; benign helpfulness is `1.000` for all listed rows.",
        "",
        "| layer group | eval slice | all positions | assistant boundary only | content only | generated only | prompt all | prompt+last | boundary+generated | content+generated | template+generated |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("all", "mid_late"):
        for eval_slice in ("4:8", "8:12"):
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{group}`",
                        f"`{eval_slice}`",
                        fmt_cell(metric(rows, group, eval_slice, "all")),
                        fmt_cell(metric(rows, group, eval_slice, "assistant_boundary")),
                        fmt_cell(metric(rows, group, eval_slice, "contentish")),
                        fmt_cell(metric(rows, group, eval_slice, "generated")),
                        fmt_cell(metric(rows, group, eval_slice, "prompt_all")),
                        fmt_cell(metric(rows, group, eval_slice, "prompt_or_last")),
                        fmt_cell(metric(rows, group, eval_slice, "assistant_boundary_or_generated")),
                        fmt_cell(metric(rows, group, eval_slice, "contentish_or_generated")),
                        fmt_cell(metric(rows, group, eval_slice, "prompt_template_or_generated")),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `assistant_boundary` alone fails on `12-20` `4:8` despite the audit showing top feature deltas at the assistant boundary.",
            "- `contentish` and `contentish_or_generated` do not reproduce the repair, so harmful-content token features are not the main causal path in these runs.",
            "- `generated` alone fails, so generated-history patching needs a prompt-side state seed.",
            "- `assistant_boundary_or_generated` matches all-position repair on `12-20` `8:12` and reaches `0.750` on `12-20` `4:8`; it also tracks the late-layer `15-20` positive control on `8:12`.",
            "- `prompt_template_or_generated` matches the same reduced performance, so the useful prompt-side seed appears to be template/boundary state rather than ordinary content words.",
            "",
            "## Current Mechanistic Hypothesis",
            "",
            "The all-token GemmaScope sparse repair works by restoring an autoregressive refusal-state trajectory: a donor-like assistant-start/template state plus continued generated-token state maintenance. It is not just a static harmful-content feature patch.",
            "",
            "## Caveats",
            "",
            "- These are small heldout prompt slices with local heuristic scoring.",
            "- The position masks are token-level heuristics over the Gemma chat template.",
            "- The result uses k1024 top-delta features; k2048 and direct feature-ID ablations remain useful follow-ups.",
            "",
            "## Artifacts",
            "",
            f"- Aggregate CSV: `{path.parent / 'gemma2_2b_gemmascope_mlp_sae_position_restricted_metrics.csv'}`",
            f"- Atomic result root: `{RESULT_ROOT}`",
            "- Script support: `stage3/scripts/run_gemma2_2b_gemmascope_mlp_sae_feature_subsets.py`",
            "- Aggregator: `stage3/scripts/aggregate_gemma2_2b_gemmascope_mlp_sae_position_restricted.py`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = collect_rows(RESULT_ROOT)
    if not rows:
        raise SystemExit(f"no atomic metrics found under {RESULT_ROOT}")
    rows.sort(key=lambda row: (str(row["layer_group"]), str(row["eval_slice"]), str(row["patch_token_filter"])))
    csv_path = RESULT_ROOT / "gemma2_2b_gemmascope_mlp_sae_position_restricted_metrics.csv"
    summary_path = RESULT_ROOT / "GEMMA2_2B_GEMMASCOPE_MLP_SAE_POSITION_RESTRICTED_SUMMARY.md"
    manifest_path = RESULT_ROOT / "manifest.json"
    write_csv(csv_path, rows)
    write_summary(summary_path, rows)
    manifest_path.write_text(
        json.dumps(
            {
                "result_root": str(RESULT_ROOT),
                "aggregate_metrics": str(csv_path),
                "summary": str(summary_path),
                "n_atomic_runs": len(rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[csv] {csv_path}")
    print(f"[summary] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
