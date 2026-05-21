#!/usr/bin/env python3
"""Build figures for the model-merging full overview presentation."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.pdf")
    plt.savefig(FIG / f"{name}.png", dpi=180)
    plt.close()


def box(ax, xy, text, *, fc="#F2F4F8", ec="#2F3A4A", w=1.9, h=0.7, fs=10):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    return patch


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, linewidth=1.2, color="#2F3A4A"))


def plot_merge_concept() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.set_axis_off()
    box(ax, (0.2, 2.6), "Base model\nweights", fc="#EAF2FF")
    box(ax, (2.8, 3.2), "Math expert\nbase + math delta", fc="#E8F6EF")
    box(ax, (2.8, 2.1), "Code/chat expert\nbase + skill delta", fc="#E8F6EF")
    box(ax, (2.8, 1.0), "Safety/refusal expert\nbase + safety delta", fc="#E8F6EF")
    box(ax, (6.2, 2.1), "Merged model\none set of weights", fc="#FFF4DE", w=2.1)
    ax.text(
        4.85,
        3.85,
        r"$\theta_{merge} = \theta_{base} + \alpha\Delta_{math} + \beta\Delta_{code} + \gamma\Delta_{safety}$",
        ha="center",
        fontsize=12,
    )
    for y in (3.55, 2.45, 1.35):
        arrow(ax, (4.72, y), (6.15, 2.45))
    arrow(ax, (2.15, 2.95), (2.75, 3.55))
    arrow(ax, (2.15, 2.95), (2.75, 2.45))
    arrow(ax, (2.15, 2.95), (2.75, 1.35))
    ax.text(
        4.75,
        0.35,
        "Merging changes the weights before inference. It is not concatenating models or voting between experts.",
        ha="center",
        fontsize=10,
        color="#4A5568",
    )
    ax.set_xlim(0, 8.7)
    ax.set_ylim(0, 4.3)
    savefig("merge_concept")


def plot_project_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10.0, 3.2))
    ax.set_axis_off()
    stages = [
        ("Stage 0", "Find real\nmerge case"),
        ("RQ1/RQ2", "Localize drift\nand loss"),
        ("RQ3", "Causal module\npatches"),
        ("RQ0", "Baseline basis:\nneurons/PCA/random"),
        ("Next", "SAE only if it\nbeats baselines"),
    ]
    xs = [0.2, 2.2, 4.2, 6.2, 8.2]
    for i, ((title, body), x) in enumerate(zip(stages, xs)):
        box(ax, (x, 1.25), f"{title}\n{body}", fc=["#EAF2FF", "#EEF7EA", "#FFF4DE", "#FDECEC", "#EFEAFB"][i], w=1.55, h=1.0, fs=9)
        if i < len(xs) - 1:
            arrow(ax, (x + 1.6, 1.75), (xs[i + 1] - 0.1, 1.75))
    ax.text(5.0, 0.35, "Principle imported from value_action: do not trust a sparse basis until it beats honest baselines.", ha="center", fontsize=10)
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 3.0)
    savefig("project_pipeline")


def plot_activation_drift() -> None:
    rows = []
    with (DATA / "qwen1_5b_refusal_activation_similarity.csv").open() as f:
        for row in csv.DictReader(f):
            if row["model_a"] == "base" and row["model_b"] == "abliterated" and row["split"] in {"harmful", "benign"}:
                rows.append(row)
    by_split = {"harmful": [], "benign": []}
    for row in rows:
        by_split[row["split"]].append((int(row["layer"]), float(row["row_cosine"])))
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for split, color in [("harmful", "#C03221"), ("benign", "#3066BE")]:
        vals = sorted(by_split[split])
        ax.plot([x for x, _ in vals], [y for _, y in vals], marker="o", linewidth=1.8, label=split, color=color)
    ax.axvspan(20, 24, color="#FFE8A3", alpha=0.35, label="largest drift region")
    ax.set_xlabel("layer")
    ax.set_ylabel("base vs abliterated activation cosine")
    ax.set_ylim(0.65, 1.02)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    ax.set_title("Prompt-selective activation drift")
    savefig("activation_drift")


def plot_localization_range() -> None:
    rows = []
    with (DATA / "qwen1_5b_patch_generation_metrics.csv").open() as f:
        for row in csv.DictReader(f):
            rows.append(row)
    keep = [
        ("base", "base"),
        ("abliterated", "ablit"),
        ("patch_12+13:mlp", "12-13"),
        ("patch_12+13+14+15+16:mlp", "12-16"),
        ("patch_12+13+14+15+16+17+18+19:mlp", "12-19"),
        ("patch_12+13+14+15+16+17+18+19+20+21:mlp", "12-21"),
        ("patch_12+13+14+15+16+17+18+19+20+21+22+23+24:mlp", "12-24"),
    ]
    lookup = {row["model"]: float(row["harmful_ok_rate"]) for row in rows}
    labels = [label for _model, label in keep]
    vals = [lookup[model] for model, _label in keep]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.bar(labels, vals, color=["#4C566A", "#B7B7B7"] + ["#3066BE"] * (len(labels) - 2))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("harmful clean refusal rate")
    ax.set_title("Static MLP patch localization: broad range needed")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)
    savefig("localization_range")


def plot_pca_target_loss() -> None:
    rows = []
    with (DATA / "qwen1_5b_lowdim_activation_patch_target_losses.csv").open() as f:
        for row in csv.DictReader(f):
            rows.append(row)
    keep = [
        ("full_12_24_mlp", "full MLP"),
        ("mean_delta_rank1", "mean delta"),
        ("pca_rank16", "PCA 16"),
        ("pca_rank64", "PCA 64"),
        ("random_rank64", "random 64"),
    ]
    lookup = {row["variant"]: float(row["harmful_gap_closed"]) for row in rows}
    labels = [label for _variant, label in keep]
    vals = [lookup[variant] for variant, _label in keep]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.bar(labels, vals, color=["#4C566A", "#3066BE", "#5E81AC", "#00A676", "#E07A5F"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("harmful refusal loss gap closed")
    ax.set_title("Target-loss repair: PCA beats random projection")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.025, f"{v:.3f}", ha="center", fontsize=9)
    savefig("pca_target_loss")


def plot_pca_generation_audit() -> None:
    rows = []
    with (DATA / "qwen1_5b_lowdim_activation_patch_generation_metrics.csv").open() as f:
        for row in csv.DictReader(f):
            rows.append(row)
    keep = ["base", "abliterated", "mean_delta_rank1", "pca_rank16", "pca_rank64", "random_rank64"]
    label_map = {
        "base": "base",
        "abliterated": "ablit",
        "mean_delta_rank1": "mean delta",
        "pca_rank16": "PCA 16",
        "pca_rank64": "PCA 64",
        "random_rank64": "random 64",
    }
    strict = {
        "base": 1.0,
        "abliterated": 0.0,
        "mean_delta_rank1": 0.75,
        "pca_rank16": 0.625,
        "pca_rank64": 1.0,
        "random_rank64": 0.0,
    }
    by_model = {row["model"]: row for row in rows}
    labels = [label_map[m] for m in keep]
    auto = [float(by_model[m]["harmful_ok_rate"]) for m in keep]
    manual = [strict[m] for m in keep]
    benign = [float(by_model[m]["benign_ok_rate"]) for m in keep]
    fig, ax = plt.subplots(figsize=(9.4, 3.8))
    x = list(range(len(keep)))
    width = 0.27
    ax.bar([i - width for i in x], auto, width, label="auto harmful clean", color="#3066BE")
    ax.bar(x, manual, width, label="strict manual harmful clean", color="#E07A5F")
    ax.bar([i + width for i in x], benign, width, label="benign helpful", color="#00A676")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("rate")
    ax.set_title("Behavioral repair: strict audit matters")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.34))
    savefig("pca_generation_audit")


def main() -> int:
    plot_merge_concept()
    plot_project_pipeline()
    plot_activation_drift()
    plot_localization_range()
    plot_pca_target_loss()
    plot_pca_generation_audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
