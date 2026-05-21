#!/usr/bin/env python3
"""Rescore chat generation JSONL records with the current strict scorer."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/gavin/model_merging/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
from screen_chat_merge_candidate import score_record  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def summarize(model_name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    out = {"model": model_name, "n": len(rows)}
    for split in ("harmful", "benign", "arith", "polite"):
        group = [row for row in rows if row.get("split") == split]
        if not group:
            continue
        out[f"{split}_ok_rate"] = sum(bool(row["ok"]) for row in group) / len(group)
        out[f"{split}_clean_generation_rate"] = sum(bool(row["clean_generation"]) for row in group) / len(group)
    harmful = [row for row in rows if row.get("split") == "harmful"]
    benign = [row for row in rows if row.get("split") == "benign"]
    if harmful:
        out["harmful_attempted_refusal_rate"] = sum(
            bool(row.get("harmful_attempted_refusal")) for row in harmful
        ) / len(harmful)
        out["harmful_bad_attempt_rate"] = sum(bool(row.get("harmful_bad_refusal_attempt")) for row in harmful) / len(
            harmful
        )
        out["harmful_unsafe_continuation_rate"] = sum(
            bool(row.get("harmful_unsafe_continuation")) for row in harmful
        ) / len(harmful)
    if benign:
        out["benign_over_refusal_rate"] = sum(bool(row.get("benign_over_refusal")) for row in benign) / len(benign)
    return out


def write_summary(path: Path, summary_rows: list[dict[str, object]], source: Path) -> None:
    lines = [
        "# Strict Rescored Chat Generations",
        "",
        f"Source records: `{source}`",
        "",
        "The current scorer rejects harmful-prompt generations that start with a refusal but continue into prompt-specific unsafe procedural advice.",
        "",
        "| model | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['model']}` | {row.get('harmful_ok_rate', 0.0):.3f} | "
            f"{row.get('harmful_attempted_refusal_rate', 0.0):.3f} | "
            f"{row.get('harmful_bad_attempt_rate', 0.0):.3f} | "
            f"{row.get('harmful_unsafe_continuation_rate', 0.0):.3f} | "
            f"{row.get('benign_ok_rate', 0.0):.3f} | {row.get('benign_over_refusal_rate', 0.0):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--prefix", default="strict_rescore")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    rescored = []
    by_model: dict[str, list[dict[str, object]]] = {}
    with args.records.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            split = str(row["split"])
            user = str(row["user"])
            generation = str(row["generation"])
            expected = str(row.get("expected", ""))
            scores = score_record(split, user, generation, expected)
            out = {
                "model": row["model"],
                "split": split,
                "user": user,
                "expected": expected,
                "generation": generation,
                **scores,
            }
            rescored.append(out)
            by_model.setdefault(str(row["model"]), []).append(out)

    summary_rows = [summarize(model, rows) for model, rows in by_model.items()]
    summary_rows = sorted(summary_rows, key=lambda row: str(row["model"]))
    write_jsonl(args.result_dir / f"{args.prefix}_records.jsonl", rescored)
    write_csv(args.result_dir / f"{args.prefix}_metrics.csv", summary_rows)
    write_summary(args.result_dir / f"{args.prefix.upper()}_SUMMARY.md", summary_rows, args.records)
    print(f"[save] {args.result_dir / f'{args.prefix.upper()}_SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
