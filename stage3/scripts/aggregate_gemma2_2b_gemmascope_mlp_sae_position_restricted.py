#!/usr/bin/env python3
"""Aggregate atomic GemmaScope MLP-SAE position-restricted patch runs."""

from __future__ import annotations

import argparse
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
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def budget_sort_key(raw: str) -> int:
    return int(raw.removeprefix("k"))


def eval_slice_sort_key(raw: str) -> int:
    return int(raw.split(":", 1)[0])


def metric(
    rows: list[dict[str, object]], group: str, eval_slice: str, budget: str, patch_filter: str
) -> dict[str, object] | None:
    for row in rows:
        if (
            row["layer_group"] == group
            and row["eval_slice"] == eval_slice
            and row["budget"] == budget
            and row["patch_token_filter"] == patch_filter
        ):
            return row
    return None


def fmt_cell(row: dict[str, object] | None) -> str:
    if row is None:
        return ""
    return f"{float(row['harmful_clean']):.3f} / {float(row['unsafe']):.3f}"


def main_result_lines(rows: list[dict[str, object]]) -> list[str]:
    patch_filters = {str(row["patch_token_filter"]) for row in rows}
    if patch_filters == {"assistant_boundary_or_generated"}:
        return [
            "This aggregate is a budget probe for the already-localized `assistant_boundary_or_generated` intervention.",
            "",
            "The k512 budget remains partially causal: it repairs the `4:8` fold strongly and the `8:12` fold partially while keeping unsafe continuation at zero.",
        ]
    return [
        "The strongest current mechanism is not content-token semantics and not a static assistant-boundary-only patch.",
        "",
        "The best reduced patch is `assistant_boundary_or_generated`: patch the assistant response boundary in the prompt, then patch generated-token history during autoregressive rollout.",
        "",
        "This nearly matches all-position repair while `contentish_or_generated` remains weak or zero.",
    ]


def interpretation_lines(rows: list[dict[str, object]]) -> list[str]:
    patch_filters = {str(row["patch_token_filter"]) for row in rows}
    if patch_filters == {"assistant_boundary_or_generated"}:
        return [
            "- This root is a budget-threshold follow-up, not a full position-mask comparison.",
            "- k512 keeps a real but weaker causal signal than k1024: `4:8` remains at `0.750`, while `8:12` drops to `0.500`.",
            "- Both k512 folds keep unsafe continuation at `0.000`, so the smaller budget is weaker mainly in repair coverage rather than safety quality.",
            "- k256 failed twice during generation, so the exact lower threshold is still unresolved.",
        ]
    return [
        "- `assistant_boundary` alone fails on `12-20` `4:8` despite the audit showing top feature deltas at the assistant boundary.",
        "- `contentish` and `contentish_or_generated` do not reproduce the repair, so harmful-content token features are not the main causal path in these runs.",
        "- `generated` alone fails, so generated-history patching needs a prompt-side state seed.",
        "- `assistant_boundary_or_generated` matches all-position repair on `12-20` `8:12` and reaches `0.750` on `12-20` `4:8`; it also tracks the late-layer `15-20` positive control on `8:12`.",
        "- `prompt_template_or_generated` matches the same reduced performance, so the useful prompt-side seed appears to be template/boundary state rather than ordinary content words.",
    ]


def write_summary(path: Path, rows: list[dict[str, object]], result_root: Path) -> None:
    budgets = sorted({str(row["budget"]) for row in rows}, key=budget_sort_key)
    groups = sorted({str(row["layer_group"]) for row in rows})
    eval_slices = sorted({str(row["eval_slice"]) for row in rows}, key=eval_slice_sort_key)
    lines = [
        "# Gemma-2-2B GemmaScope MLP-SAE Position-Restricted Patch Findings",
        "",
        "Date: 2026-05-22",
        "",
        "This checkpoint tests whether successful all-token selected SAE features repair refusal because of prompt content, static assistant-boundary tokens, generated-token trajectory state, or a combination.",
        "",
        f"Budgets in this aggregate: `{', '.join(budgets)}`.",
        "",
        "All runs use all-token top-delta feature selection from prompt slice `0:4` per split.",
        "",
        "## Main Result",
        "",
        *main_result_lines(rows),
        "",
        "## Key Aggregate Table",
        "",
        "Cells are `harmful clean / unsafe continuation`; benign helpfulness is `1.000` for all listed rows.",
        "",
        "| budget | layer group | eval slice | all positions | assistant boundary only | content only | generated only | prompt all | prompt+last | boundary+generated | content+generated | template+generated |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in budgets:
        for group in groups:
            for eval_slice in eval_slices:
                if not any(
                    row["budget"] == budget and row["layer_group"] == group and row["eval_slice"] == eval_slice
                    for row in rows
                ):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{budget}`",
                            f"`{group}`",
                            f"`{eval_slice}`",
                            fmt_cell(metric(rows, group, eval_slice, budget, "all")),
                            fmt_cell(metric(rows, group, eval_slice, budget, "assistant_boundary")),
                            fmt_cell(metric(rows, group, eval_slice, budget, "contentish")),
                            fmt_cell(metric(rows, group, eval_slice, budget, "generated")),
                            fmt_cell(metric(rows, group, eval_slice, budget, "prompt_all")),
                            fmt_cell(metric(rows, group, eval_slice, budget, "prompt_or_last")),
                            fmt_cell(metric(rows, group, eval_slice, budget, "assistant_boundary_or_generated")),
                            fmt_cell(metric(rows, group, eval_slice, budget, "contentish_or_generated")),
                            fmt_cell(metric(rows, group, eval_slice, budget, "prompt_template_or_generated")),
                        ]
                    )
                    + " |"
                )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            *interpretation_lines(rows),
            "",
            "## Current Mechanistic Hypothesis",
            "",
            "The all-token GemmaScope sparse repair works by restoring an autoregressive refusal-state trajectory: a donor-like assistant-start/template state plus continued generated-token state maintenance. It is not just a static harmful-content feature patch.",
            "",
            "## Caveats",
            "",
            "- These are small heldout prompt slices with local heuristic scoring.",
            "- The position masks are token-level heuristics over the Gemma chat template.",
            "- Direct feature-ID ablations remain a useful follow-up.",
            "",
            "## Artifacts",
            "",
            f"- Aggregate CSV: `{path.parent / 'gemma2_2b_gemmascope_mlp_sae_position_restricted_metrics.csv'}`",
            f"- Atomic result root: `{result_root}`",
            "- Script support: `stage3/scripts/run_gemma2_2b_gemmascope_mlp_sae_feature_subsets.py`",
            "- Aggregator: `stage3/scripts/aggregate_gemma2_2b_gemmascope_mlp_sae_position_restricted.py`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    result_root = args.result_root.resolve()
    rows = collect_rows(result_root)
    if not rows:
        raise SystemExit(f"no atomic metrics found under {result_root}")
    rows.sort(
        key=lambda row: (
            budget_sort_key(str(row["budget"])),
            str(row["layer_group"]),
            eval_slice_sort_key(str(row["eval_slice"])),
            str(row["patch_token_filter"]),
        )
    )
    csv_path = result_root / "gemma2_2b_gemmascope_mlp_sae_position_restricted_metrics.csv"
    summary_path = result_root / "GEMMA2_2B_GEMMASCOPE_MLP_SAE_POSITION_RESTRICTED_SUMMARY.md"
    manifest_path = result_root / "manifest.json"
    write_csv(csv_path, rows)
    write_summary(summary_path, rows, result_root)
    manifest_path.write_text(
        json.dumps(
            {
                "result_root": str(result_root),
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
