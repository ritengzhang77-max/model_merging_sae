#!/usr/bin/env python3
"""Fetch Neuronpedia feature metadata for ranked SAE features."""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_MODEL = "gemma-2-2b"
DEFAULT_SOURCE = "20-gemmascope-mlp-16k"


def read_feature_ids(path: Path, top_k: int) -> list[int]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    if top_k > 0:
        rows = rows[:top_k]
    return [int(row["feature_id"]) for row in rows]


def fetch_json(url: str, timeout: int, retries: int) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def first_explanation(data: dict[str, object]) -> tuple[str, str, str]:
    explanations = data.get("explanations")
    if not isinstance(explanations, list) or not explanations:
        return "", "", ""
    item = explanations[0]
    if not isinstance(item, dict):
        return "", "", ""
    return (
        str(item.get("description") or "").strip(),
        str(item.get("explanationModelName") or ""),
        str(item.get("typeName") or ""),
    )


def compact_logits(data: dict[str, object], key_str: str, key_values: str, limit: int) -> str:
    toks = data.get(key_str)
    vals = data.get(key_values)
    if not isinstance(toks, list) or not isinstance(vals, list):
        return ""
    parts = []
    for token, value in zip(toks[:limit], vals[:limit]):
        parts.append(f"{str(token)}:{float(value):.3g}")
    return " | ".join(parts)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Neuronpedia Feature Metadata",
        "",
        f"Model/source: `{args.model_id}/{args.source_id}`.",
        f"Ranking CSV: `{args.ranking_csv}`.",
        f"Top K: `{args.top_k}`.",
        "",
        "Descriptions are Neuronpedia autointerp labels. Treat them as hypotheses, not causal proof.",
        "",
        "| rank | feature | description | density | explanation model | URL |",
        "|---:|---:|---|---:|---|---|",
    ]
    for row in rows:
        desc = str(row["description"]).replace("|", "\\|")
        lines.append(
            f"| {row['rank']} | {row['feature_id']} | {desc} | "
            f"{float(row['frac_nonzero']):.5f} | `{row['explanation_model']}` | "
            f"[link]({row['url']}) |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking-csv", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=33)
    ap.add_argument("--model-id", default=DEFAULT_MODEL)
    ap.add_argument("--source-id", default=DEFAULT_SOURCE)
    ap.add_argument("--result-dir", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--logit-limit", type=int, default=8)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    feature_ids = read_feature_ids(args.ranking_csv, args.top_k)
    rows = []
    for rank, feature_id in enumerate(feature_ids, start=1):
        url = f"https://www.neuronpedia.org/api/feature/{args.model_id}/{args.source_id}/{feature_id}"
        print(f"[fetch] rank {rank} feature {feature_id}", flush=True)
        data = fetch_json(url, timeout=args.timeout, retries=args.retries)
        description, explanation_model, explanation_type = first_explanation(data)
        rows.append(
            {
                "rank": rank,
                "feature_id": feature_id,
                "model_id": args.model_id,
                "source_id": args.source_id,
                "url": url.replace("/api/feature", ""),
                "description": description,
                "explanation_model": explanation_model,
                "explanation_type": explanation_type,
                "frac_nonzero": float(data.get("frac_nonzero") or 0.0),
                "max_act_approx": float(data.get("maxActApprox") or 0.0),
                "positive_logits": compact_logits(data, "pos_str", "pos_values", args.logit_limit),
                "negative_logits": compact_logits(data, "neg_str", "neg_values", args.logit_limit),
            }
        )
        if args.sleep > 0:
            time.sleep(args.sleep)

    csv_path = args.result_dir / "neuronpedia_feature_metadata.csv"
    summary_path = args.result_dir / "NEURONPEDIA_FEATURE_METADATA_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(csv_path, rows)
    write_summary(summary_path, args, rows)
    manifest_path.write_text(
        json.dumps(
            {
                "ranking_csv": str(args.ranking_csv),
                "model_id": args.model_id,
                "source_id": args.source_id,
                "top_k": args.top_k,
                "outputs": {
                    "csv": str(csv_path),
                    "summary": str(summary_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
