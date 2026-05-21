#!/usr/bin/env python3
"""Full Stage 3 failure-mode analysis for SmolLM2 refusal generations.

This applies the assistant-audit labeling rules to all 336 Stage 3 records and
then asks which simple activation bases separate each failure mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage1" / "scripts"))
sys.path.insert(0, str(ROOT / "stage3" / "scripts"))

from analyze_smollm2_rq0 import write_csv  # noqa: E402
from analyze_smollm2_refusal_basis import CACHE_STAGE3, RESULT_DIR, score_stage3_refusal_quality  # noqa: E402
from apply_smollm2_refusal_assistant_audit import assistant_label, boolish, has_refusal_attempt  # noqa: E402
from evaluate_smollm2_refusal_quality import write_jsonl  # noqa: E402
from rescore_smollm2_refusal_basis_cache import precompute_transforms, score_from_transforms  # noqa: E402


PROBLEM_LABELS = {
    "messy_refusal_repetition",
    "messy_refusal_artifact",
    "contradictory_or_unsafe",
}
TARGETS = (
    "failure_attempted_refusal",
    "failure_clean_refusal",
    "failure_problem_response",
    "failure_bad_attempt",
    "failure_repetition",
    "failure_artifact",
    "failure_unsafe_or_contradictory",
    "failure_no_refusal",
)


def write_csv_local(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def label_record(row: dict[str, object]) -> dict[str, object]:
    new = dict(row)
    new.update(score_stage3_refusal_quality(str(new["generation"])))
    label, note = assistant_label(new)
    new["failure_label"] = label
    new["failure_notes"] = note
    new["failure_refusal_attempt"] = has_refusal_attempt(new)
    new["failure_problem_response"] = label in PROBLEM_LABELS
    new["failure_bad_attempt"] = bool(new["failure_refusal_attempt"]) and label != "clean_refusal"
    return new


def target_value(row: dict[str, object], target: str) -> bool:
    label = str(row["failure_label"])
    if target == "failure_attempted_refusal":
        return boolish(row["failure_refusal_attempt"])
    if target == "failure_clean_refusal":
        return label == "clean_refusal"
    if target == "failure_problem_response":
        return boolish(row["failure_problem_response"])
    if target == "failure_bad_attempt":
        return boolish(row["failure_bad_attempt"])
    if target == "failure_repetition":
        return label == "messy_refusal_repetition"
    if target == "failure_artifact":
        return label == "messy_refusal_artifact"
    if target == "failure_unsafe_or_contradictory":
        return label == "contradictory_or_unsafe"
    if target == "failure_no_refusal":
        return label == "no_refusal"
    raise ValueError(f"unknown target {target}")


def summarize_by_model(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_model = defaultdict(list)
    for row in rows:
        by_model[str(row["model"])].append(row)

    out = []
    for model, group in sorted(by_model.items()):
        n = len(group)
        counts = Counter(str(row["failure_label"]) for row in group)
        refusal_attempts = sum(boolish(row["failure_refusal_attempt"]) for row in group)
        bad_attempts = sum(boolish(row["failure_bad_attempt"]) for row in group)
        out.append(
            {
                "model": model,
                "n": n,
                "clean_refusal_rate": counts["clean_refusal"] / n,
                "attempted_refusal_rate": refusal_attempts / n,
                "bad_attempt_rate": bad_attempts / n,
                "problem_response_rate": sum(counts[label] for label in PROBLEM_LABELS) / n,
                "repetition_rate": counts["messy_refusal_repetition"] / n,
                "artifact_rate": counts["messy_refusal_artifact"] / n,
                "unsafe_or_contradictory_rate": counts["contradictory_or_unsafe"] / n,
                "no_refusal_rate": counts["no_refusal"] / n,
                "invalid_generation_rate": counts["invalid_generation"] / n,
            }
        )
    return out


def summarize_by_prompt(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[int(row["prompt_id"])].append(row)

    out = []
    for prompt_id, group in sorted(by_prompt.items()):
        n = len(group)
        counts = Counter(str(row["failure_label"]) for row in group)
        out.append(
            {
                "prompt_id": prompt_id,
                "prompt_type": group[0]["prompt_type"],
                "user": group[0]["user"],
                "n": n,
                "attempted_refusal_rate": sum(boolish(row["failure_refusal_attempt"]) for row in group) / n,
                "problem_response_rate": sum(counts[label] for label in PROBLEM_LABELS) / n,
                "clean_refusal_rate": counts["clean_refusal"] / n,
                "no_refusal_rate": counts["no_refusal"] / n,
            }
        )
    return out


def compute_basis_metrics(
    features: dict[str, torch.Tensor],
    rows: list[dict[str, object]],
    prompt_ids: torch.Tensor,
    *,
    folds: int,
    seed: int,
) -> list[dict[str, object]]:
    bases = (("raw", 0), ("pca", 16), ("pca", 32), ("random", 16), ("random", 32), ("topvar", 16), ("topvar", 32))
    target_tensors = {
        target: torch.tensor([target_value(row, target) for row in rows], dtype=torch.long)
        for target in TARGETS
    }

    metric_rows: list[dict[str, object]] = []
    for rep_name, x in sorted(features.items()):
        print(f"[metrics] {rep_name}", flush=True)
        transforms = precompute_transforms(x, prompt_ids, bases, folds=folds, seed=seed)
        for target, y in target_tensors.items():
            for basis, dim in bases:
                score = score_from_transforms(transforms, y, basis=basis, dim=dim, folds=folds)
                if score is None:
                    continue
                metric_rows.append(
                    {
                        "representation": rep_name,
                        "target": target,
                        "basis": basis,
                        "dim": "full" if basis == "raw" else dim,
                        "n_total": len(y),
                        "n_positive": int(y.sum().item()),
                        "n_negative": int((1 - y).sum().item()),
                        **score,
                    }
                )
    return metric_rows


def top_rows(metric_rows: list[dict[str, object]], target: str, k: int = 8) -> list[dict[str, object]]:
    return sorted(
        [row for row in metric_rows if row["target"] == target],
        key=lambda row: (-float(row["balanced_accuracy"]), str(row["representation"]), str(row["basis"])),
    )[:k]


def write_summary_md(
    path: Path,
    model_summary: list[dict[str, object]],
    prompt_summary: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# SmolLM2 Refusal Failure-Mode Analysis",
        "",
        "This applies the assistant-audit labeling rules to all 336 Stage 3 generations. It should be treated as a heuristic audit, not a substitute for independent human labels.",
        "",
        "## Model-Level Failure Modes",
        "",
        "| model | n | clean | attempt | bad attempt | problem | repetition | artifact | unsafe/contradictory | no refusal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_summary:
        lines.append(
            f"| {row['model']} | {row['n']} | {row['clean_refusal_rate']:.3f} | "
            f"{row['attempted_refusal_rate']:.3f} | {row['bad_attempt_rate']:.3f} | "
            f"{row['problem_response_rate']:.3f} | {row['repetition_rate']:.3f} | "
            f"{row['artifact_rate']:.3f} | {row['unsafe_or_contradictory_rate']:.3f} | "
            f"{row['no_refusal_rate']:.3f} |"
        )

    lines.extend(["", "## Top Basis Separators", ""])
    for target in TARGETS:
        rows = top_rows(metric_rows, target)
        if not rows:
            continue
        positives = rows[0]["n_positive"]
        lines.extend(
            [
                f"### {target}",
                "",
                f"Positive rows: {positives}",
                "",
                "| representation | basis | dim | balanced_acc | acc | n |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['representation']} | {row['basis']} | {row['dim']} | "
                f"{row['balanced_accuracy']:.3f} | {row['accuracy']:.3f} | {row['n_eval']} |"
            )
        lines.append("")

    hardest_prompts = sorted(prompt_summary, key=lambda row: (-float(row["problem_response_rate"]), -float(row["attempted_refusal_rate"])))[:10]
    lines.extend(
        [
            "## Highest Problem-Response Prompts",
            "",
            "| prompt_id | type | attempt | problem | clean | no refusal | user |",
            "|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in hardest_prompts:
        user = str(row["user"]).replace("|", "/")
        lines.append(
            f"| {row['prompt_id']} | {row['prompt_type']} | {row['attempted_refusal_rate']:.3f} | "
            f"{row['problem_response_rate']:.3f} | {row['clean_refusal_rate']:.3f} | "
            f"{row['no_refusal_rate']:.3f} | {user} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Treat attempted refusal as the stable mechanism target.",
            "- Treat clean refusal as unavailable in this toy setup; there are too few clean positives for reliable mechanistic claims.",
            "- Quality failures split into repetition, artifacts, and unsafe/contradictory continuations; these should be separate targets before SAE/transcoder work.",
            "- If sparse features are introduced next, they must explain one of these failure modes better than the raw/PCA baselines above.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE_STAGE3 / "smollm2_refusal_basis_activations.pt")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    rows = [label_record(row) for row in cache["records"]]
    model_summary = summarize_by_model(rows)
    prompt_summary = summarize_by_prompt(rows)
    metric_rows = compute_basis_metrics(
        cache["features"],
        rows,
        cache["prompt_ids"].long(),
        folds=args.folds,
        seed=args.seed,
    )

    labeled_jsonl = args.result_dir / "smollm2_refusal_failure_modes_labeled.jsonl"
    labeled_csv = args.result_dir / "smollm2_refusal_failure_modes_labeled.csv"
    model_summary_path = args.result_dir / "smollm2_refusal_failure_modes_by_model.csv"
    prompt_summary_path = args.result_dir / "smollm2_refusal_failure_modes_by_prompt.csv"
    metrics_path = args.result_dir / "smollm2_refusal_failure_mode_basis_metrics.csv"
    summary_path = args.result_dir / "SMOLLM2_REFUSAL_FAILURE_MODE_SUMMARY.md"
    meta_path = args.result_dir / "smollm2_refusal_failure_mode_summary.json"

    write_jsonl(labeled_jsonl, rows)
    write_csv_local(labeled_csv, rows)
    write_csv(model_summary_path, model_summary)
    write_csv(prompt_summary_path, prompt_summary)
    write_csv(metrics_path, metric_rows)
    write_summary_md(summary_path, model_summary, prompt_summary, metric_rows)
    meta_path.write_text(
        json.dumps(
            {
                "activation_cache": str(args.cache),
                "rows": len(rows),
                "targets": TARGETS,
                "outputs": {
                    "labeled_jsonl": str(labeled_jsonl),
                    "labeled_csv": str(labeled_csv),
                    "model_summary": str(model_summary_path),
                    "prompt_summary": str(prompt_summary_path),
                    "basis_metrics": str(metrics_path),
                    "summary": str(summary_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[save] {labeled_jsonl}")
    print(f"[save] {labeled_csv}")
    print(f"[save] {model_summary_path}")
    print(f"[save] {prompt_summary_path}")
    print(f"[save] {metrics_path}")
    print(f"[save] {summary_path}")
    print("[labels]")
    for label, count in sorted(Counter(row["failure_label"] for row in rows).items()):
        print(f"  {label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

