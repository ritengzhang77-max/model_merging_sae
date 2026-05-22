#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def plot_qwen() -> None:
    rows = read_csv(DATA / "qwen_generated_trace_metrics.csv")
    labels = []
    values = []
    for row in rows:
        model = row["model"]
        if model == "residual_sae_d512_l1_0.0001_16-23":
            label = "SAE d512\ngenerated"
        elif model == "residual_sae_d1024_l1_0.0001_16-23":
            label = "SAE d1024\ngenerated"
        elif model == "residual_topk1344_16-23":
            label = "TopK1344\ngenerated"
        elif model == "full_16-23":
            label = "Full 16-23"
        elif model == "pca64":
            label = "PCA64"
        else:
            continue
        labels.append(label)
        values.append(float(row["harmful_ok_rate"]))
    allpos = read_csv(DATA / "qwen_generated_allpos_metrics.csv")
    for row in allpos:
        if row["model"] == "residual_sae_d512_l1_0.0001_16-23":
            labels.append("SAE d512\nall-pos")
            values.append(float(row["harmful_ok_rate"]))
    order = [0, 2, 3, 1, 5, 4] if len(labels) == 6 else range(len(labels))
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(labels, values, color=["#6b7280", "#9ca3af", "#111827", "#2f80ed", "#60a5fa", "#16a34a"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Harmful clean refusal")
    ax.set_title("Qwen v0: Generated-trace SAE improves family split but does not pass")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "qwen_generated_trace_harmful_clean.pdf")
    fig.savefig(FIGURES / "qwen_generated_trace_harmful_clean.png", dpi=180)


def plot_gemma() -> None:
    rows = read_csv(DATA / "gemma_activation_similarity.csv")
    by_split: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        by_split.setdefault(row["split"], []).append((int(row["layer"]), float(row["row_cosine"])))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for split, color in [("harmful", "#dc2626"), ("benign", "#16a34a"), ("arith", "#2563eb"), ("polite", "#7c3aed")]:
        points = sorted(by_split[split])
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=split, color=color)
    ax.set_ylim(0.65, 1.01)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Base vs abliterated cosine")
    ax.set_title("Gemma-2-2B: harmful-specific activation drift")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "gemma_activation_similarity.pdf")
    fig.savefig(FIGURES / "gemma_activation_similarity.png", dpi=180)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_qwen()
    plot_gemma()


if __name__ == "__main__":
    main()
