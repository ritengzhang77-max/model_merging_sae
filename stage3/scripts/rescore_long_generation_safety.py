#!/usr/bin/env python3
"""Length-robust safety summaries for long generated records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    models = sorted({str(row.get("model", "")) for row in rows})
    for model in models:
        model_rows = [row for row in rows if str(row.get("model", "")) == model]
        harmful = [row for row in model_rows if row.get("split") == "harmful"]
        benign = [row for row in model_rows if row.get("split") == "benign"]

        harmful_attempt = sum(bool(row.get("harmful_attempted_refusal")) for row in harmful)
        harmful_unsafe = sum(bool(row.get("harmful_unsafe_continuation")) for row in harmful)
        harmful_attempt_no_unsafe = sum(
            bool(row.get("harmful_attempted_refusal")) and not bool(row.get("harmful_unsafe_continuation"))
            for row in harmful
        )
        harmful_no_attempt = sum(not bool(row.get("harmful_attempted_refusal")) for row in harmful)
        benign_over = sum(bool(row.get("benign_over_refusal")) for row in benign)
        benign_not_over = sum(not bool(row.get("benign_over_refusal")) for row in benign)

        out.append(
            {
                "model": model,
                "n_harmful": len(harmful),
                "harmful_attempted_refusal_rate": rate(harmful_attempt, len(harmful)),
                "harmful_attempted_no_unsafe_rate": rate(harmful_attempt_no_unsafe, len(harmful)),
                "harmful_unsafe_continuation_rate": rate(harmful_unsafe, len(harmful)),
                "harmful_no_attempt_rate": rate(harmful_no_attempt, len(harmful)),
                "n_benign": len(benign),
                "benign_not_over_refusal_rate": rate(benign_not_over, len(benign)),
                "benign_over_refusal_rate": rate(benign_over, len(benign)),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, metrics: list[dict[str, object]], source: Path) -> None:
    lines = [
        "# Long-Generation Safety Rescore",
        "",
        f"Source records: `{source}`.",
        "",
        "This summary ignores the short-generation `too_long` clean/helpful penalty and reports separate long-generation signals.",
        "",
        "| model | harmful attempted | attempted and no unsafe | unsafe continuation | no attempt | benign not over-refusal | benign over-refusal |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| `{row['model']}` | "
            f"{row['harmful_attempted_refusal_rate']:.3f} | "
            f"{row['harmful_attempted_no_unsafe_rate']:.3f} | "
            f"{row['harmful_unsafe_continuation_rate']:.3f} | "
            f"{row['harmful_no_attempt_rate']:.3f} | "
            f"{row['benign_not_over_refusal_rate']:.3f} | "
            f"{row['benign_over_refusal_rate']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--result-dir", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    result_dir = args.result_dir or args.records.parent
    result_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.records)
    metrics = summarize(rows)
    write_csv(result_dir / "long_generation_safety_rescore_metrics.csv", metrics)
    write_summary(result_dir / "LONG_GENERATION_SAFETY_RESCORE_SUMMARY.md", metrics, args.records)
    print(f"[save] {result_dir / 'LONG_GENERATION_SAFETY_RESCORE_SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
