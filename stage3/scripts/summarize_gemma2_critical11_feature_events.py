#!/usr/bin/env python3
"""Summarize generated-token feature events for the k=11 variable-module audit."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "stage3" / "results"
AUDIT_DIR = (
    RESULTS
    / "gemma2_2b_linear_merge_sae_feature_event_audit_v0"
    / "common9_variable_subsets_harmful_l20_core_variable_float32"
)
RECORDS = (
    RESULTS
    / "gemma2_2b_linear_merge_sae_bundle_patch_v0"
    / "fake_id_hologram_l20_final_newline_delta_add_common9_variable_subsets_float32_max160"
    / "gemma2_2b_linear_merge_sae_bundle_patch_records.jsonl"
)
OUT_DIR = RESULTS / "gemma2_2b_linear_merge_sae_11feature_identity_v0"
OUT_CSV = OUT_DIR / "common9_variable_feature_event_outcome_aggregate.csv"
OUT_EVENT_CSV = OUT_DIR / "common9_variable_feature_event_examples.csv"
OUT_MD = RESULTS / "GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_EVENT_AUDIT_SUMMARY.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_records(path: Path) -> dict[str, dict[str, object]]:
    records = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == "harmful":
                records[str(row["model"])] = row
    return records


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def fmt(value: float) -> str:
    return f"{value:.6f}"


def clipped_context(raw: str, limit: int = 120) -> str:
    text = " ".join(str(raw).replace("\\n", " ").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def outcome_for_record(row: dict[str, object]) -> str:
    return "unsafe" if row.get("harmful_unsafe_continuation") else "safe"


def summarize_detail(records: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    detail = read_csv(AUDIT_DIR / "gemma2_2b_linear_merge_sae_feature_event_detail.csv")
    grouped: dict[int, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in detail:
        model = row["source_model"]
        grouped[int(row["feature_id"])][outcome_for_record(records[model])].append(row)

    rows = []
    for feature_id in sorted(grouped):
        safe_rows = grouped[feature_id].get("safe", [])
        unsafe_rows = grouped[feature_id].get("unsafe", [])
        safe_delta = [float(row["donor_minus_recipient_gen_mean"]) for row in safe_rows]
        unsafe_delta = [float(row["donor_minus_recipient_gen_mean"]) for row in unsafe_rows]
        safe_abs = [float(row["abs_delta_gen_mean"]) for row in safe_rows]
        unsafe_abs = [float(row["abs_delta_gen_mean"]) for row in unsafe_rows]
        safe_donor = [float(row["donor_gen_mean"]) for row in safe_rows]
        unsafe_donor = [float(row["donor_gen_mean"]) for row in unsafe_rows]
        safe_recipient = [float(row["recipient_gen_mean"]) for row in safe_rows]
        unsafe_recipient = [float(row["recipient_gen_mean"]) for row in unsafe_rows]
        rows.append(
            {
                "feature_id": feature_id,
                "safe_n": len(safe_rows),
                "unsafe_n": len(unsafe_rows),
                "safe_donor_minus_recipient_gen_mean": fmt(mean(safe_delta)),
                "unsafe_donor_minus_recipient_gen_mean": fmt(mean(unsafe_delta)),
                "safe_minus_unsafe_delta": fmt(mean(safe_delta) - mean(unsafe_delta)),
                "safe_abs_delta_gen_mean": fmt(mean(safe_abs)),
                "unsafe_abs_delta_gen_mean": fmt(mean(unsafe_abs)),
                "safe_donor_gen_mean": fmt(mean(safe_donor)),
                "unsafe_donor_gen_mean": fmt(mean(unsafe_donor)),
                "safe_recipient_gen_mean": fmt(mean(safe_recipient)),
                "unsafe_recipient_gen_mean": fmt(mean(unsafe_recipient)),
            }
        )
    return rows


def summarize_events(records: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    events = read_jsonl(AUDIT_DIR / "gemma2_2b_linear_merge_sae_feature_event_top_events.jsonl")
    interesting_metrics = {"donor_minus_recipient", "recipient_minus_donor", "donor_activation"}
    buckets: dict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in events:
        metric = str(row["metric"])
        if metric not in interesting_metrics:
            continue
        model = str(row["source_model"])
        outcome = outcome_for_record(records[model])
        buckets[(int(row["feature_id"]), outcome, metric)].append(row)

    rows = []
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda row: float(row["value"]), reverse=True)[:5]
        for row in bucket:
            rows.append(
                {
                    "feature_id": key[0],
                    "outcome": key[1],
                    "metric": key[2],
                    "value": fmt(float(row["value"])),
                    "source_model": row["source_model"],
                    "token": row["token"],
                    "context": clipped_context(str(row["context"])),
                }
            )
    return rows


def write_summary(aggregate_rows: list[dict[str, object]], event_rows: list[dict[str, object]]) -> None:
    by_feature = {int(row["feature_id"]): row for row in aggregate_rows}

    lines = [
        "# Gemma-2-2B Linear Merge SAE 11-Feature Event Audit Summary",
        "",
        "Date: 2026-05-24",
        "",
        "This memo audits generated-token activations for the 14 features in the",
        "`common9 + variable` k=11 handle class, using the saved harmful hologram",
        "generations from the exhaustive variable-subset sweep.",
        "",
        "Important caveat: this is a descriptive event audit on generated text. It",
        "does not replace the causal first-token patch evidence, because it observes",
        "donor/recipient feature values on continuations that have already been",
        "generated.",
        "",
        "## Outcome Aggregate",
        "",
        "| feature | safe n | unsafe n | safe donor-recipient | unsafe donor-recipient | safe-unsafe delta | safe abs delta | unsafe abs delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| `{row['feature_id']}` | {row['safe_n']} | {row['unsafe_n']} | "
            f"`{row['safe_donor_minus_recipient_gen_mean']}` | "
            f"`{row['unsafe_donor_minus_recipient_gen_mean']}` | "
            f"`{row['safe_minus_unsafe_delta']}` | "
            f"`{row['safe_abs_delta_gen_mean']}` | `{row['unsafe_abs_delta_gen_mean']}` |"
        )

    def field(feature_id: int, name: str) -> str:
        return str(by_feature[feature_id][name])

    lines.extend(
        [
            "",
            "## Mechanistic Reading",
            "",
            "- Feature `14991` is the cleanest safe-output generated-token separator in",
            "  this audit: its mean donor-recipient generation delta is",
            f"  `{field(14991, 'safe_donor_minus_recipient_gen_mean')}` on strict-safe",
            f"  continuations and `{field(14991, 'unsafe_donor_minus_recipient_gen_mean')}`",
            "  on unsafe warning-plus-compliance continuations. Its top events occur in",
            "  later legal-consequence / reminder portions of strict-safe outputs, not",
            "  in the procedural continuation region.",
            "- Features `1813` and `15169` are not simple safe-output markers here. Their",
            "  donor-recipient generation deltas are larger on unsafe continuations",
            f"  (`1813`: `{field(1813, 'unsafe_donor_minus_recipient_gen_mean')}` unsafe",
            f"  vs. `{field(1813, 'safe_donor_minus_recipient_gen_mean')}` safe;",
            f"  `15169`: `{field(15169, 'unsafe_donor_minus_recipient_gen_mean')}` unsafe",
            f"  vs. `{field(15169, 'safe_donor_minus_recipient_gen_mean')}` safe).",
            "  Their events mostly sit on the initial warning/refusal preamble. That",
            "  preamble can still be followed by unsafe continuation, so these features",
            "  are better read as boundary/preamble features than complete safety",
            "  features.",
            "- Variable/enabler features such as `6289` and `8775` are more positive on",
            "  strict-safe continuations than unsafe continuations in generated-token",
            "  donor-recipient means. Their event contexts are tied to redirection and",
            "  legal-consequence segments, consistent with the pair-rule enabler role.",
            "- Feature `7531` has zero generated-token activation in this audit despite",
            "  being causally useful at the patched final newline. Its role is therefore",
            "  likely in the prompt-boundary state that chooses the continuation, not in",
            "  recurring generated-token content.",
            "",
            "## Interpretation",
            "",
            "This strengthens the current thesis without overclaiming a semantic circuit.",
            "The causal handle works at the first-token boundary, while generated-token",
            "events show a mixture of warning preamble, legal-consequence redirection,",
            "and domain/procedural text features. The project should keep the distinction",
            "between a compact causal boundary handle and a human-readable refusal",
            "module.",
            "",
            "## Artifacts",
            "",
            f"- Event audit directory: `{AUDIT_DIR}`",
            f"- Outcome aggregate CSV: `{OUT_CSV}`",
            f"- Event examples CSV: `{OUT_EVENT_CSV}`",
            f"- Source records: `{RECORDS}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    records = read_records(RECORDS)
    aggregate_rows = summarize_detail(records)
    event_rows = summarize_events(records)
    write_csv(OUT_CSV, aggregate_rows)
    write_csv(OUT_EVENT_CSV, event_rows)
    write_summary(aggregate_rows, event_rows)
    print(f"[save] {OUT_CSV}")
    print(f"[save] {OUT_EVENT_CSV}")
    print(f"[save] {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
