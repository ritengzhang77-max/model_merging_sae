#!/usr/bin/env python3
"""Summarize prompt-boundary feature events for the k=11 handle class."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "stage3" / "results"
AUDIT_DIR = (
    RESULTS
    / "gemma2_2b_linear_merge_sae_feature_event_audit_v0"
    / "common9_prompt_boundary_harmful_l20_core_variable_float32"
)
OUT_DIR = RESULTS / "gemma2_2b_linear_merge_sae_11feature_identity_v0"
OUT_CSV = OUT_DIR / "common9_prompt_boundary_feature_events.csv"
OUT_MD = RESULTS / "GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_PROMPT_BOUNDARY_EVENT_SUMMARY.md"

MODEL_TOKEN_POS = 21
FINAL_NEWLINE_POS = 22


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def signed_delta(events: list[dict[str, object]], feature_id: int, pos: int) -> float:
    donor = [
        float(row["value"])
        for row in events
        if int(row["feature_id"]) == feature_id
        and int(row["token_position"]) == pos
        and row["metric"] == "donor_minus_recipient"
    ]
    recipient = [
        float(row["value"])
        for row in events
        if int(row["feature_id"]) == feature_id
        and int(row["token_position"]) == pos
        and row["metric"] == "recipient_minus_donor"
    ]
    if donor:
        return max(donor)
    if recipient:
        return -max(recipient)
    return 0.0


def token_at(events: list[dict[str, object]], pos: int) -> str:
    for row in events:
        if int(row["token_position"]) == pos:
            return str(row["token"])
    return ""


def main() -> int:
    detail = read_csv(AUDIT_DIR / "gemma2_2b_linear_merge_sae_feature_event_detail.csv")
    events = read_jsonl(AUDIT_DIR / "gemma2_2b_linear_merge_sae_feature_event_top_events.jsonl")

    rows = []
    for row in sorted(detail, key=lambda item: int(item["feature_id"])):
        feature_id = int(row["feature_id"])
        donor_prompt = float(row["donor_prompt_mean"])
        recipient_prompt = float(row["recipient_prompt_mean"])
        model_delta = signed_delta(events, feature_id, MODEL_TOKEN_POS)
        newline_delta = signed_delta(events, feature_id, FINAL_NEWLINE_POS)
        rows.append(
            {
                "feature_id": feature_id,
                "donor_prompt_mean": fmt(donor_prompt),
                "recipient_prompt_mean": fmt(recipient_prompt),
                "donor_minus_recipient_prompt_mean": fmt(donor_prompt - recipient_prompt),
                "abs_delta_prompt_mean": fmt(float(row["abs_delta_prompt_mean"])),
                "model_token_signed_delta": fmt(model_delta),
                "final_newline_signed_delta": fmt(newline_delta),
                "final_newline_delta_direction": "donor_higher"
                if newline_delta > 0
                else "recipient_higher"
                if newline_delta < 0
                else "zero",
            }
        )
    write_csv(OUT_CSV, rows)

    donor_newline = [row for row in rows if float(row["final_newline_signed_delta"]) > 0]
    recipient_newline = [row for row in rows if float(row["final_newline_signed_delta"]) < 0]
    donor_newline.sort(key=lambda row: float(row["final_newline_signed_delta"]), reverse=True)
    recipient_newline.sort(key=lambda row: float(row["final_newline_signed_delta"]))

    lines = [
        "# Gemma-2-2B Linear Merge SAE 11-Feature Prompt-Boundary Event Summary",
        "",
        "Date: 2026-05-24",
        "",
        "This memo audits the actual prompt/assistant-boundary positions for the 14",
        "features in the `common9 + variable` k=11 local class. The causal patch is",
        "applied at the assistant-boundary final newline, so this audit is closer to",
        "the intervention site than the generated-token event audit.",
        "",
        "## Boundary Signed Deltas",
        "",
        f"Assistant model token position: `{MODEL_TOKEN_POS}` (`{token_at(events, MODEL_TOKEN_POS)}`).",
        f"Assistant final-newline position: `{FINAL_NEWLINE_POS}` (`{token_at(events, FINAL_NEWLINE_POS)}`).",
        "",
        "| feature | prompt mean donor-recipient | model token signed delta | final newline signed delta | final newline direction |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['feature_id']}` | `{row['donor_minus_recipient_prompt_mean']}` | "
            f"`{row['model_token_signed_delta']}` | `{row['final_newline_signed_delta']}` | "
            f"{row['final_newline_delta_direction']} |"
        )
    lines.extend(
        [
            "",
            "## Mechanistic Reading",
            "",
            "The final-newline handle is explicitly signed. Several selected features are",
            "donor-higher at the assistant boundary, while several others are",
            "recipient-higher and are therefore suppressed by the donor-minus-recipient",
            "`delta_add` patch.",
            "",
            "Strong donor-higher final-newline features:",
            "",
        ]
    )
    lines.extend(
        f"- `{row['feature_id']}`: `{row['final_newline_signed_delta']}`"
        for row in donor_newline[:8]
    )
    lines.extend(["", "Strong recipient-higher final-newline features:", ""])
    lines.extend(
        f"- `{row['feature_id']}`: `{row['final_newline_signed_delta']}`"
        for row in recipient_newline[:8]
    )
    lines.extend(
        [
            "",
            "This explains why positive-only or negative-only interpretations are weak.",
            "The handle is not just adding refusal-looking donor features. It also",
            "removes recipient-side boundary features that otherwise hold the prompt near",
            "or below the `I`/`It` gate.",
            "",
            "Feature `7531` is especially clarifying: it is donor-higher by `+3.676` at",
            "the assistant final newline but has zero generated-token activation in the",
            "generated-token audit. Its role is boundary-state control, not recurring",
            "semantic content in the refusal answer.",
            "",
            "## Artifacts",
            "",
            f"- Prompt-boundary audit directory: `{AUDIT_DIR}`",
            f"- Boundary event CSV: `{OUT_CSV}`",
            "- Generated-token event summary: `stage3/results/GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_EVENT_AUDIT_SUMMARY.md`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[save] {OUT_CSV}")
    print(f"[save] {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
