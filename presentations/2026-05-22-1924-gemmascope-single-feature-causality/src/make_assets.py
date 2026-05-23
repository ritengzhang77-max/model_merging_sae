#!/usr/bin/env python3
"""Build figures and PowerPoint for the single-feature causality checkpoint."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"


def read_metrics() -> list[dict[str, str]]:
    with (DATA_DIR / "l19_tail_block_metrics.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric(rows: list[dict[str, str]], condition: str) -> float:
    for row in rows:
        if row["condition"] == condition:
            return float(row["harmful_clean"])
    raise KeyError(condition)


def make_causality_plot(rows: list[dict[str, str]]) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "feature_16048_causality.png"
    conditions = [
        ("k896", "mix_decode_delta_abs_k896_plus_l19_rank1001_1001"),
        ("k896 +\nrank1001", "mix_decode_delta_abs_k896_plus_l19_rank1001_1001"),
        ("k896 +\nrank1006", "mix_decode_delta_abs_k896_plus_l19_rank1006_1006"),
        ("k1024 -\nrank1006", "mix_decode_delta_abs_k1024_minus_l19_rank1006_1006"),
        ("k1024 -\nother singleton", "mix_decode_delta_abs_k1024_minus_l19_rank1007_1007"),
    ]
    vals = [0.5, metric(rows, conditions[1][1]), metric(rows, conditions[2][1]), metric(rows, conditions[3][1]), metric(rows, conditions[4][1])]
    colors = ["#4a5568", "#718096", "#2f855a", "#c05621", "#2b6cb0"]
    fig, ax = plt.subplots(figsize=(8.7, 4.3))
    bars = ax.bar(range(len(vals)), vals, color=colors)
    for bar, val in zip(bars, vals, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03, f"{val:.2f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Harmful clean refusal rate")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([label for label, _ in conditions])
    ax.set_title("Layer 19 feature 16048 is add-on sufficient and necessary")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_event_plot() -> Path:
    path = FIG_DIR / "feature_16048_event_tokens.png"
    counts: Counter[str] = Counter()
    with (DATA_DIR / "l19_feature_16048_events.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            counts[json.loads(line)["token"]] += 1
    top = counts.most_common()
    labels = [token if token != "\\n" else "newline" for token, _ in top]
    vals = [count for _, count in top]
    fig, ax = plt.subplots(figsize=(8.7, 3.8))
    bars = ax.bar(range(len(vals)), vals, color="#805ad5")
    for bar, val in zip(bars, vals, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05, str(val), ha="center", fontsize=9)
    ax.set_ylim(0, max(vals) + 1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Top audit event count")
    ax.set_title("Feature 16048 events are prompt-end/state-like, not a direct fake-ID label")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def add_title(slide, title: str, subtitle: str = "") -> None:
    box = slide.shapes.add_textbox(Inches(0.45), Inches(0.25), Inches(12.3), Inches(0.55))
    frame = box.text_frame
    frame.clear()
    p = frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(27)
    p.font.bold = True
    if subtitle:
        sub = slide.shapes.add_textbox(Inches(0.48), Inches(0.88), Inches(12.2), Inches(0.35))
        frame = sub.text_frame
        frame.clear()
        p = frame.paragraphs[0]
        p.text = subtitle
        p.font.size = Pt(13)


def add_bullets(slide, bullets: list[str], top: float = 1.35, size: int = 20) -> None:
    box = slide.shapes.add_textbox(Inches(0.75), Inches(top), Inches(11.9), Inches(5.5))
    frame = box.text_frame
    frame.word_wrap = True
    frame.clear()
    for idx, item in enumerate(bullets):
        p = frame.paragraphs[0] if idx == 0 else frame.add_paragraph()
        p.text = item
        p.font.size = Pt(size)
        p.space_after = Pt(8)


def add_table(slide, rows: list[list[str]], left: float, top: float, width: float, height: float) -> None:
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(left), Inches(top), Inches(width), Inches(height))
    table = shape.table
    for r_idx, row in enumerate(rows):
        for c_idx, text in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = text
            for para in cell.text_frame.paragraphs:
                para.alignment = PP_ALIGN.CENTER
                for run in para.runs:
                    run.font.size = Pt(10 if r_idx else 11)
                    run.font.bold = r_idx == 0


def make_pptx(causality_plot: Path, event_plot: Path) -> Path:
    pptx_path = ROOT / "gemmascope_single_feature_causality_checkpoint.pptx"
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    slide = prs.slides.add_slide(blank)
    add_title(slide, "GemmaScope Single-Feature Causality", "Model merging SAE project - 2026-05-22")
    add_bullets(
        slide,
        [
            "Checkpoint reason: the L19 tail-band effect now localizes to one SAE feature.",
            "Feature: layer 19, feature ID 16048, global delta rank 1006.",
            "Scope: fake-ID prompt recovery on heldout 8:12, conditional on the k896 prefix.",
        ],
    )

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Causal Evidence")
    add_table(
        slide,
        [
            ["Condition", "Clean", "Unsafe", "Fake-ID"],
            ["k896 baseline", "0.50", "0.00", "fail"],
            ["k896 + feature 16048", "0.75", "0.25", "pass"],
            ["k1024 baseline", "0.75", "0.00", "pass"],
            ["k1024 - feature 16048", "0.50", "0.00", "fail"],
            ["k1024 - other singleton", "0.75", "0.00", "pass"],
        ],
        0.75,
        1.35,
        11.8,
        3.1,
    )
    add_bullets(slide, ["Clean = harmful clean refusal rate over four harmful prompts."], top=5.05, size=17)

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Plot View")
    slide.shapes.add_picture(str(causality_plot), Inches(1.0), Inches(1.25), width=Inches(11.2))

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Feature Event Audit")
    slide.shapes.add_picture(str(event_plot), Inches(1.0), Inches(1.25), width=Inches(11.2))

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Interpretation")
    add_bullets(
        slide,
        [
            "This is not a standalone refusal module; feature 16048 only works with the broader k896 prefix.",
            "The feature is not a simple fake-ID semantic detector; its top events are prompt-end/state-like.",
            "The strongest current claim is response-state refinement at L19, not harmful-content semantics.",
            "Next: replicate feature 16048 on more prompt slices and test generated-token timing.",
        ],
    )

    prs.save(pptx_path)
    return pptx_path


def main() -> int:
    rows = read_metrics()
    causality_plot = make_causality_plot(rows)
    event_plot = make_event_plot()
    pptx_path = make_pptx(causality_plot, event_plot)
    print(causality_plot)
    print(event_plot)
    print(pptx_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
