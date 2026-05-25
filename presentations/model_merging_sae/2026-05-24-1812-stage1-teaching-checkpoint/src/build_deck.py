#!/usr/bin/env python3
"""Build the model-merging SAE teaching checkpoint deck.

This script intentionally generates the Beamer source and plots from local
project artifacts so the presentation package is reproducible.
"""

from __future__ import annotations

import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path


THIS = Path(__file__).resolve()
DECK_DIR = THIS.parents[1]
REPO = THIS.parents[4]
OUT = DECK_DIR / "outputs"
FIG = DECK_DIR / "figures"
DATA = DECK_DIR / "data"

SRC_RESULT = REPO / "stage3" / "results" / "gemma2_2b_linear_merge_sae_11feature_identity_v0"

COPIED_FILES = [
    REPO / "docs" / "PROJECT_PLAN.md",
    REPO / "docs" / "MECHANISTIC_MODEL_MERGING_SAE_PROPOSAL.md",
    REPO / "notes" / "mechanistic_model_merging_agenda.md",
    REPO / "stage3" / "README.md",
    REPO / "stage3" / "results" / "GEMMA2_2B_LINEAR_WEIGHT_MERGE_SWEEP_FINDINGS.md",
    REPO / "stage3" / "results" / "GEMMA2_2B_LINEAR_MERGE_ACTIVATION_PATCH_FINDINGS.md",
    REPO / "stage3" / "results" / "GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_HANDLE_SUMMARY.md",
    REPO / "stage3" / "results" / "GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_VARIABLE_MODULE_SUMMARY.md",
    REPO / "stage3" / "results" / "GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_ALPHA_LOCALITY_SUMMARY.md",
    REPO / "stage3" / "results" / "GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_EVENT_AUDIT_SUMMARY.md",
    REPO / "stage3" / "results" / "GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_PROMPT_BOUNDARY_EVENT_SUMMARY.md",
    REPO / "stage3" / "results" / "GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_BOUNDARY_SIGNED_COMPONENT_SUMMARY.md",
    SRC_RESULT / "common9_variable_subset_outcomes.csv",
    SRC_RESULT / "common9_variable_subset_alpha_locality.csv",
    SRC_RESULT / "common9_prompt_boundary_feature_events.csv",
    SRC_RESULT / "boundary_signed_component_outcomes.csv",
    SRC_RESULT / "critical11_feature_identity_table.csv",
]


def ensure_dirs() -> None:
    for directory in (OUT, FIG, DATA):
        directory.mkdir(parents=True, exist_ok=True)


def copy_data() -> dict[str, Path]:
    copied: dict[str, Path] = {}
    for source in COPIED_FILES:
        if not source.exists():
            continue
        dest = DATA / source.name
        shutil.copy2(source, dest)
        copied[source.name] = dest
    return copied


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def make_figures(copied: dict[str, Path]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment fallback
        (FIG / "FIGURE_BUILD_FAILED.txt").write_text(
            f"matplotlib import failed: {exc}\n", encoding="utf-8"
        )
        return

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
        }
    )

    colors = {"pass": "#147D64", "tie": "#A66A00", "fail": "#B23A48"}

    # Figure 1: variable subset outcome counts at alpha 0.75.
    rows = read_csv(copied["common9_variable_subset_outcomes.csv"])
    by_count: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_count[int(row["variable_count"])][row["first_token_outcome"]] += 1
    xs = sorted(by_count)
    bottom = [0] * len(xs)
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    for outcome in ("pass", "tie", "fail"):
        vals = [by_count[x][outcome] for x in xs]
        ax.bar(xs, vals, bottom=bottom, label=outcome, color=colors[outcome])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_title("Common9 plus variable features at alpha 0.75")
    ax.set_xlabel("Number of variable features added to common9")
    ax.set_ylabel("Number of tested subsets")
    ax.set_xticks(xs)
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "variable_subset_counts.pdf")
    plt.close(fig)

    # Figure 2: alpha locality counts.
    rows = read_csv(copied["common9_variable_subset_alpha_locality.csv"])
    by_alpha: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_alpha[row["alpha"]][row["first_token_outcome"]] += 1
    alphas = sorted(by_alpha, key=lambda x: float(x))
    bottom = [0] * len(alphas)
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    for outcome in ("pass", "tie", "fail"):
        vals = [by_alpha[a][outcome] for a in alphas]
        ax.bar(alphas, vals, bottom=bottom, label=outcome, color=colors[outcome])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_title("The same feature subsets only work near the boundary")
    ax.set_xlabel("Recipient merge alpha")
    ax.set_ylabel("Number of subsets")
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "alpha_locality_counts.pdf")
    plt.close(fig)

    # Figure 3: signed final-newline deltas.
    rows = read_csv(copied["common9_prompt_boundary_feature_events.csv"])
    rows = sorted(rows, key=lambda r: float(r["final_newline_signed_delta"]))
    labels = [r["feature_id"] for r in rows]
    vals = [float(r["final_newline_signed_delta"]) for r in rows]
    bar_colors = ["#B23A48" if v < 0 else "#147D64" for v in vals]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.barh(labels, vals, color=bar_colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("Feature deltas at the assistant final newline")
    ax.set_xlabel("Donor minus recipient SAE activation")
    ax.set_ylabel("Feature ID")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "prompt_boundary_signed_deltas.pdf")
    plt.close(fig)

    # Figure 4: signed component causal margins.
    rows = read_csv(copied["boundary_signed_component_outcomes.csv"])
    wanted = [
        ("linear_alpha_0.75", "alpha0.75"),
        ("bundle_patch_boundary_donor_higher9", "donor-high 9"),
        ("bundle_patch_boundary_recipient_higher5", "recipient-high 5"),
        ("bundle_patch_boundary_signed_all14", "signed all14"),
        ("bundle_patch_common9_signed9", "common9"),
        ("bundle_patch_common9_signed9_plus_variable_donor3", "common9 + donor vars"),
        ("bundle_patch_common9_signed9_plus_variable_recipient2", "common9 + recip vars"),
        ("linear_alpha_1", "donor alpha1"),
    ]
    row_by_condition = {r["condition"]: r for r in rows}
    labels = [label for key, label in wanted if key in row_by_condition]
    vals = [float(row_by_condition[key]["harmful_i_minus_it"]) for key, _ in wanted if key in row_by_condition]
    outcomes = [row_by_condition[key]["harmful_outcome"] for key, _ in wanted if key in row_by_condition]
    fig, ax = plt.subplots(figsize=(8.5, 3.7))
    ax.bar(range(len(labels)), vals, color=[colors[o] for o in outcomes])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Signed components: only the cooperative edit crosses zero")
    ax.set_ylabel("I minus It logit margin")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "signed_component_margins.pdf")
    plt.close(fig)


def slide(title: str, body: str, options: str = "") -> str:
    return f"\\begin{{frame}}{options}{{{title}}}\n{body}\n\\end{{frame}}\n"


def bullets(items: list[str], size: str = "\\small") -> str:
    return (
        f"{size}\n"
        "\\begin{itemize}\n"
        + "\n".join(f"  \\item {item}" for item in items)
        + "\n\\end{itemize}"
    )


def two_col(left: str, right: str, left_width: str = "0.50", right_width: str = "0.46") -> str:
    return (
        "\\begin{columns}[T,onlytextwidth]\n"
        f"\\begin{{column}}{{{left_width}\\textwidth}}\n{left}\n\\end{{column}}\n"
        f"\\begin{{column}}{{{right_width}\\textwidth}}\n{right}\n\\end{{column}}\n"
        "\\end{columns}"
    )


def fig_slide(title: str, fig_name: str, note: str, width: str = "0.86") -> str:
    body = (
        f"\\centering\n\\includegraphics[width={width}\\textwidth]{{../figures/{fig_name}}}\n\n"
        f"\\vspace{{0.5em}}\n\\small {note}"
    )
    return slide(title, body)


def deck_tex() -> str:
    frames: list[str] = []

    frames.append(
        r"""\title{Mechanistic Model Merging With SAEs}
\subtitle{Checkpoint 1: what model merging is, what we did, what we found, and what decision is next}
\author{Gavin / Codex research workspace}
\date{2026-05-24}

\begin{frame}
  \titlepage
\end{frame}
"""
    )

    frames.append(
        slide(
            "How To Read This Deck",
            bullets(
                [
                    "This is a teaching deck and a project checkpoint, not a final paper.",
                    "I assume zero background in model merging, optimization, or SAEs.",
                    "The current code folders call this \\texttt{stage3} because we reached SAE experiments. This deck calls it \\textbf{Checkpoint 1}.",
                    "The main decision today: keep going, but narrow the next work to testing whether the signed boundary mechanism generalizes.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "One-Slide Answer",
            bullets(
                [
                    "\\textbf{The original question has not changed:} explain why model merging works or fails mechanistically.",
                    "\\textbf{The empirical focus narrowed:} Gemma-2-2B safety/refusal restoration under a linear merge.",
                    "\\textbf{Main result so far:} a small SAE feature handle can tip a near-boundary merged model from unsafe continuation to direct refusal.",
                    "\\textbf{Most important caveat:} this is local and threshold-like; it is not yet a universal model-merging explanation.",
                    "\\textbf{Decision:} worth continuing if we immediately test generalization, baselines, and another model pair.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "What Changed Since The Original Plan",
            bullets(
                [
                    "We did not change the research goal.",
                    "We changed the experimental target because early small-model targets were not clean enough.",
                    "We moved from weak or ambiguous merge examples to a cleaner public pair with public GemmaScope SAEs.",
                    "The current claim is sharper than the original broad plan: merging can be studied at a prompt-boundary decision gate.",
                    "The project is now in a strong-but-not-finished state: interesting mechanism found; generality still unproven.",
                ]
            ),
        )
    )

    frames.append(r"\section{Model Merging From Zero}" + "\n")

    frames.append(
        slide(
            "What Is A Model?",
            bullets(
                [
                    "A neural network model is a big function: input text goes in, next-token probabilities come out.",
                    "The learned numbers inside the function are called \\textbf{weights} or \\textbf{parameters}.",
                    "A checkpoint is a saved copy of all those weights.",
                    "Two checkpoints with the same architecture have the same list of tensors and tensor shapes.",
                    "That shared tensor layout is what makes direct weight merging possible.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "What Is Model Merging?",
            bullets(
                [
                    "Model merging combines two or more trained checkpoints into one checkpoint.",
                    "The simplest version literally averages or interpolates the weights.",
                    "The result is still one model at inference time, not several models voting.",
                    "Most practical merges require the same architecture, tokenizer family, and usually a common base model.",
                    "The hope is to inherit useful behavior from several models without retraining from raw data.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Simple Linear Merge",
            r"""\small
\[
W_{\mathrm{merge}}=(1-\alpha)W_A+\alpha W_B
\]
\begin{itemize}
  \item \(W_A\): weights of model A.
  \item \(W_B\): weights of model B.
  \item \(\alpha=0\): exactly model A.
  \item \(\alpha=1\): exactly model B.
  \item \(\alpha=0.5\): midpoint average.
\end{itemize}
In this project, model A is the abliterated recipient and model B is the safe donor.
""",
        )
    )

    frames.append(
        slide(
            "Task Vector Merge",
            r"""\small
If several experts came from the same base:
\[
\Delta_{\mathrm{math}}=W_{\mathrm{math}}-W_{\mathrm{base}}
\]
\[
\Delta_{\mathrm{code}}=W_{\mathrm{code}}-W_{\mathrm{base}}
\]
\[
W_{\mathrm{merge}}=W_{\mathrm{base}}+\lambda_1\Delta_{\mathrm{math}}+\lambda_2\Delta_{\mathrm{code}}
\]
\begin{itemize}
  \item A task vector is the weight difference created by fine-tuning.
  \item Merging can mean adding several task vectors to a common base.
  \item This is addition in weight space, not concatenation of models.
\end{itemize}
""",
        )
    )

    frames.append(
        slide(
            "Not Concatenation, Not An Ensemble",
            bullets(
                [
                    "\\textbf{Concatenation} would make a larger model by placing modules side by side. That is not the usual merge.",
                    "\\textbf{Ensembling} runs multiple models and combines outputs. That increases inference cost.",
                    "\\textbf{Weight merging} produces one checkpoint with roughly the same inference cost as each parent.",
                    "This is why merging is attractive: it promises composition without runtime multiplication.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Why Not Just Train One Model On All Data?",
            bullets(
                [
                    "Sometimes you do not have the original data. You only have released checkpoints.",
                    "Sometimes the data is private, licensed, deleted, or too expensive to collect again.",
                    "Sometimes you want to combine community models without coordinating all original training pipelines.",
                    "Sometimes merging is a cheap exploratory step before expensive training.",
                    "Merging is usually not claimed to beat perfect joint training with unlimited data and compute.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Where People Use It",
            bullets(
                [
                    "Open-source LLM communities merge chat, code, math, roleplay, multilingual, and domain-specialized models.",
                    "Vision and CLIP-style models use soups or task vectors to improve robustness and transfer.",
                    "Diffusion communities merge image-generation checkpoints for style or domain behavior.",
                    "Safety work uses merging or unmerging to study refusal, ablation, and restoration.",
                    "Federated or privacy-sensitive settings can average updates without centralizing raw data.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Does Merging Just Make Models Better?",
            bullets(
                [
                    "\\textbf{No.} This is the biggest misconception.",
                    "A merge can improve one benchmark and hurt another.",
                    "A merge can preserve helpfulness but remove safety.",
                    "A merge can look good on short outputs but fail in longer generations.",
                    "A merge can work only because the parents are already close in weight space.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Classic Recipes And Papers",
            r"""\scriptsize
\begin{tabular}{p{0.23\textwidth}p{0.44\textwidth}p{0.23\textwidth}}
\toprule
Recipe & What it does & Reference note \\
\midrule
Model soups & Average fine-tuned checkpoints from the same pretrained model. & Wortsman et al., ICML 2022, OpenAlex 205 citations on 2026-05-07. \\
Task arithmetic & Treat fine-tune deltas as editable task vectors. & Ilharco et al., ICLR 2023, OpenAlex 31. \\
TIES & Trim small deltas and resolve sign conflicts. & Yadav et al., NeurIPS 2023, OpenAlex 22. \\
DARE & Randomly drop and rescale delta parameters. & Yu et al., ICML 2024, OpenAlex 12. \\
AdaMerging & Learn adaptive merge coefficients. & Yang et al., ICLR 2024, OpenAlex 5. \\
MergeKit & Practical toolkit for LLM merging. & Goddard et al., arXiv 2024, OpenAlex 3. \\
\bottomrule
\end{tabular}
\vspace{0.4em}

\scriptsize Counts are from local OpenAlex notes and should be refreshed before formal writing.
""",
        )
    )

    frames.append(
        slide(
            "SWA And Checkpoint Averaging",
            bullets(
                [
                    "SWA means stochastic weight averaging.",
                    "The idea: during training, save several checkpoints and average their weights.",
                    "This often lands in a flatter or more robust region than one final checkpoint.",
                    "Model soups are related: average several fine-tuned checkpoints that started from the same pretrained model.",
                    "For us, this is background: averaging can work, but it does not explain the internal mechanism of a specific behavior.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Federated Learning And FedAvg",
            bullets(
                [
                    "Federated learning trains models across many clients without centralizing all raw data.",
                    "FedAvg means clients train local updates, then the server averages those updates or weights.",
                    "This is not the same research community as open-source LLM merging, but the math overlaps.",
                    "The privacy motivation is: weights or updates move, raw data often stays local.",
                    "Our project is not about privacy; it is about why merged weights preserve or lose mechanisms.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Why Merging Sometimes Works",
            bullets(
                [
                    "The models may share a common base, so many coordinates already mean similar things.",
                    "Fine-tuning deltas may be sparse or redundant.",
                    "Some behaviors may be implemented by late routing/style components that are easy to interpolate.",
                    "Averaging may reduce overfitting when checkpoints are nearby.",
                    "But these are high-level explanations; they do not identify the causal feature or circuit.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Why Merging Sometimes Fails",
            bullets(
                [
                    "Two experts may push the same weight coordinate in opposite directions.",
                    "One expert may erase or suppress another expert's mechanism.",
                    "A merged model may keep a feature but disconnect it from the output route.",
                    "A prompt may drift into a different activation region than either parent used.",
                    "This project tries to distinguish those failure modes causally.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Loss Landscape Intuition",
            bullets(
                [
                    "A loss landscape is the error surface over all possible weights.",
                    "A low-loss region means the model performs well there.",
                    "If two checkpoints are in the same broad low-loss basin, their average may also work.",
                    "If the straight line between them crosses a high-loss region, naive averaging can fail.",
                    "This explains some averaging success, but not which mechanism or behavior is preserved.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Permutation Alignment Intuition",
            bullets(
                [
                    "Neural networks have symmetries: hidden units can be permuted without changing the function.",
                    "Two independently trained models may implement similar behavior with different neuron orderings.",
                    "Averaging unaligned neurons can destroy both models' computation.",
                    "Git Re-Basin-style work studies aligning such models before averaging.",
                    "Our current Gemma case avoids most of this because the checkpoints share a parent lineage.",
                ]
            ),
        )
    )

    frames.append(r"\section{Our Concrete Model Pair}" + "\n")

    frames.append(
        slide(
            "The Pair We Use Now",
            bullets(
                [
                    "\\textbf{Safe donor:} \\texttt{google/gemma-2-2b-it}.",
                    "\\textbf{Abliterated recipient:} \\texttt{IlyaGusev/gemma-2-2b-it-abliterated}.",
                    "The recipient is designed to be less refusal-heavy.",
                    "The donor has stronger harmful-request refusal behavior.",
                    "They are close enough that linear interpolation is meaningful and cheap to audit.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "What Abliterated Means Here",
            bullets(
                [
                    "Abliteration is a model-editing style that removes or weakens a behavior, often refusal.",
                    "In plain English: the recipient is a version of Gemma where safety refusal has been weakened.",
                    "That gives us a natural question: can merging restore the lost refusal behavior?",
                    "It is not guaranteed that the same internal circuit remains available.",
                    "This makes it a good test case for mechanistic model merging.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Our Merge Line",
            r"""\small
\[
W_{\alpha}=W_{\mathrm{abliterated}}+\alpha(W_{\mathrm{base}}-W_{\mathrm{abliterated}})
\]
\begin{itemize}
  \item \(\alpha=0.00\): fully abliterated recipient.
  \item \(\alpha=0.75\): partial merge; mostly donor direction, but still not exactly donor.
  \item \(\alpha=1.00\): safe donor.
  \item We study whether the partial merge restored a safety mechanism or only changed surface behavior.
\end{itemize}
""",
        )
    )

    frames.append(
        slide(
            "What Refusal Means",
            bullets(
                [
                    "A refusal is a safe answer that declines a harmful request.",
                    "Example shape: the assistant says it cannot help make a fake ID or bypass security.",
                    "A clean refusal should not then provide procedural instructions.",
                    "Refusal is the behavior we want the safe donor and safe merge to show on harmful prompts.",
                    "This is a safety behavior, not a generic measure of intelligence.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Unsafe Continuation",
            bullets(
                [
                    "An unsafe continuation gives actionable help for the harmful request.",
                    "A model can start with a warning and still become unsafe later.",
                    "That is why short first words are not enough; long generation audits matter.",
                    "In our fake-ID case, warning-plus-compliance is a real failure mode.",
                    "We score both first-token gate behavior and generated answer behavior.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Repair",
            bullets(
                [
                    "Repair means an intervention moves the recipient or partial merge toward safe donor behavior.",
                    "A repair can be a full activation patch, an SAE decoded patch, or a small feature-bundle patch.",
                    "A repair is causal evidence only if changing the internal state changes the output.",
                    "A repair is not automatically a full explanation; it may be local to one prompt family.",
                    "Our best current repair is local but compact and causally tested.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "First-Token Gate",
            r"""\small
For the fake-ID/hologram prompt, the first generated token is often decisive.

\begin{itemize}
  \item Direct refusal route often begins with token \(\texttt{I}\).
  \item Warning-plus-compliance route often begins with token \(\texttt{It}\).
  \item We measure the margin:
\end{itemize}
\[
\mathrm{margin}=\operatorname{logit}(\texttt{I})-\operatorname{logit}(\texttt{It})
\]
\begin{itemize}
  \item Positive margin: direct-refusal route wins.
  \item Negative margin: warning-plus-compliance route wins.
  \item Zero margin: knife-edge tie that behaved unsafe in our generation screen.
\end{itemize}
""",
        )
    )

    frames.append(
        slide(
            "Why The Fake-ID Hologram Prompt",
            bullets(
                [
                    "It is harmful but realistic enough to trigger a safety boundary.",
                    "The donor and alpha-0.75 merge can look good on broad short screens, but this prompt exposes brittleness.",
                    "It has a clean first-token competition: \\texttt{I} versus \\texttt{It}.",
                    "That makes it useful for mechanistic debugging.",
                    "It is not the whole project; it is the current microscope slide.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Linear Merge Behavior",
            r"""\small
\begin{tabular}{cccc}
\toprule
Alpha & Harmful clean refusal & Unsafe continuation & Benign helpful \\
\midrule
0.00 & 0.000 & 0.083 & 1.000 \\
0.25 & 0.000 & 0.167 & 1.000 \\
0.50 & 0.667 & 0.083 & 1.000 \\
0.75 & 0.917 & 0.000 & 1.000 \\
1.00 & 0.917 & 0.000 & 0.917 \\
\bottomrule
\end{tabular}

\vspace{0.5em}
\small On the initial 12 harmful / 12 benign screen, alpha 0.75 looked best: donor-like harmful refusal without the donor's one benign over-refusal.
""",
        )
    )

    frames.append(
        slide(
            "The Hidden Problem",
            bullets(
                [
                    "The broad alpha-0.75 screen looked strong.",
                    "Longer fake-ID/hologram generation showed alpha 0.75 can still fall into warning-plus-compliance.",
                    "That gives us the concrete mechanistic question:",
                    "\\textbf{what internal change tips alpha 0.75 from unsafe continuation into direct refusal?}",
                    "This question connects behavior, merging, and interpretability in one small target.",
                ]
            ),
        )
    )

    frames.append(r"\section{Interpretability Tools From Zero}" + "\n")

    frames.append(
        slide(
            "Mechanistic Interpretability",
            bullets(
                [
                    "Mechanistic interpretability asks what internal computations caused a model output.",
                    "Instead of only asking whether a model refused, we ask which layers, activations, and features made refusal happen.",
                    "A mechanistic claim should ideally include a causal intervention.",
                    "Causal means: if we change the proposed component, the behavior changes in the predicted direction.",
                    "This project uses model merging as the behavior and SAEs as one candidate basis.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Activations",
            bullets(
                [
                    "An activation is an internal vector produced while the model processes a prompt.",
                    "Every layer and token position has activations.",
                    "For a harmful prompt, the donor and recipient can have different activations even before generation starts.",
                    "If patching donor activations into the recipient repairs behavior, that layer/site contains useful causal information.",
                    "Activation evidence is closer to mechanism than benchmark scores alone.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Activation Patching",
            bullets(
                [
                    "Run model A and save an internal activation.",
                    "Run model B on the same prompt.",
                    "At a chosen layer and token position, replace B's activation with A's activation.",
                    "Continue the forward pass and observe the output.",
                    "If the output changes as predicted, that activation site matters causally.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Sparse Autoencoders",
            bullets(
                [
                    "An SAE is a tool for decomposing an activation vector into sparse features.",
                    "Sparse means only a small number of features are active for a given token.",
                    "The encoder maps activation vector to feature activations.",
                    "The decoder maps features back into activation space.",
                    "If useful, SAE features can give names and smaller handles for causal interventions.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "SAE Decode Patch",
            bullets(
                [
                    "Instead of replacing a full activation vector, we can change only selected SAE features.",
                    "A feature delta is donor feature activation minus recipient feature activation.",
                    "\\texttt{delta\\_add} means we add that donor-minus-recipient feature change at the chosen site.",
                    "Then we decode the selected feature change back into the model's activation space.",
                    "This is how a small feature set can repair a behavior without copying the full donor state.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Why We Did RQ0 Basis Validation",
            bullets(
                [
                    "A sparse feature is not automatically the right explanation.",
                    "Earlier broad neuron/PCA baselines could sometimes repair small screens.",
                    "So the SAE result must be better than just saying some big direction works.",
                    "The project follows the value-action lesson: test the basis before trusting it.",
                    "The current SAE handle matters because it became small, causal, and structured.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "GemmaScope",
            bullets(
                [
                    "GemmaScope provides public SAEs for Gemma models.",
                    "That makes Gemma attractive: we do not need to train our own SAE first.",
                    "Hook alignment mattered: GemmaScope MLP SAEs matched the post-feedforward normalized MLP update.",
                    "Using the wrong raw MLP site can make a good sparse basis look bad.",
                    "Layer 20 post-FF MLP became the key site for the current repair story.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "The Assistant Final Newline Site",
            bullets(
                [
                    "The prompt ends and the assistant is about to generate its first answer token.",
                    "The final newline before generation is a boundary token.",
                    "Patching there tests the state that chooses the first generated route.",
                    "For the fake-ID prompt, that route is approximately \\texttt{I} refusal versus \\texttt{It} warning-plus-compliance.",
                    "This is why our strongest result is called a prompt-boundary handle.",
                ]
            ),
        )
    )

    frames.append(r"\section{Project Timeline And Decisions}" + "\n")

    frames.append(
        slide(
            "Original RQ Ladder",
            bullets(
                [
                    "RQ0: what basis best explains and predicts merge success?",
                    "RQ1-RQ2: does a merge inherit the same mechanisms as the parent expert, and where?",
                    "RQ3-RQ4: what causes interference and can causal overlap predict retention?",
                    "RQ5-RQ8: representation drift, sparse and low-rank delta structure.",
                    "RQ9-RQ12: safety, alignment, emergence, and mechanism-aware merging.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "The Plan Did Not Change",
            bullets(
                [
                    "We still study model merging mechanistically.",
                    "We still require causal tests, not only behavior tables.",
                    "We still compare SAEs against simpler explanations.",
                    "What changed is the target case: we selected a cleaner public Gemma pair after weaker branches were not enough.",
                    "That is normal research triage, not a change of thesis.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Branches We Moved Past",
            bullets(
                [
                    "SmolLM2 refusal pilots taught us about failure modes but did not give clean stable refusal transfer.",
                    "Some Qwen public merges were near-copies, broken, or behaviorally not better than the base.",
                    "Qwen 1.5B gave a useful abliterated negative control, but the sparse tooling was less clean.",
                    "Gemma became the main branch because behavior, public SAE tooling, and causal repair lined up.",
                    "These branches were not wasted; they prevented us from overclaiming on weak examples.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "SmolLM2 Lesson",
            bullets(
                [
                    "SmolLM2 was useful as a cheap pilot but not as a clean safety-transfer target.",
                    "The model often produced attempted refusals mixed with repetition, artifacts, or unsafe continuations.",
                    "Direction steering and module patches did not produce a clean stable repair.",
                    "The lesson was methodological: do not spend expensive SAE work on a target whose behavior is not clean.",
                    "This branch remains a failure-mechanism lesson, not the main paper target.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Qwen Screening Lesson",
            bullets(
                [
                    "Several public Qwen-family merges were screened as possible cheap targets.",
                    "Many were near-copies, broken generations, or did not improve the measured behavior.",
                    "The Qwen 1.5B abliterated merge gave a useful negative control for safety loss.",
                    "But it was not the cleanest branch for public sparse-feature analysis.",
                    "Gemma won because it had behavior gap plus GemmaScope tooling.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Why Gemma Became The Main Branch",
            bullets(
                [
                    "Clear safety gap between donor and abliterated recipient.",
                    "Linear merge curve showed a real behavior transition.",
                    "Public GemmaScope SAEs exist for relevant layers.",
                    "Full activation patching repaired harmful refusal without hurting benign helpfulness.",
                    "That gave us a clean bridge from behavior to causal mechanism.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Current Stage In Plain English",
            bullets(
                [
                    "We are no longer just finding model pairs.",
                    "We are already in the SAE mechanistic stage for the Gemma pair.",
                    "The current work is not broad benchmarking; it is mechanism discovery and validation.",
                    "We have one compact local mechanism candidate.",
                    "The next stage is generalization and falsification.",
                ]
            ),
        )
    )

    frames.append(r"\section{Major Results}" + "\n")

    frames.append(
        slide(
            "Result 1: Full Activation Repair",
            bullets(
                [
                    "Patching donor late MLP activations into the abliterated recipient repaired harmful refusal on the screen.",
                    "The important layers were late, especially around layers 12-20 and then layer 20 for the focused SAE work.",
                    "This proved that the donor contains causal internal information that can restore refusal.",
                    "It did not yet explain the mechanism in sparse or interpretable units.",
                    "It gave us the activation target for SAE decomposition.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Result 2: Full SAE Decode Can Repair",
            bullets(
                [
                    "Using GemmaScope MLP SAE reconstruction at aligned post-FF sites could match full activation repair.",
                    "This showed the public SAE basis was behaviorally complete enough for the repair.",
                    "But a full decoded SAE patch is still huge.",
                    "It is useful as a basis-validation step, not a final mechanism.",
                    "The next question became: can we shrink the patch to a small causal feature set?",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Hook Alignment Detail",
            bullets(
                [
                    "A hook is the exact internal tensor site where we read or patch activations.",
                    "GemmaScope MLP SAEs align to the post-feedforward normalized MLP update.",
                    "Early raw-MLP patching and SAE decode patching did not line up cleanly until this was fixed.",
                    "After aligning the site, full SAE decoded patching matched the behavioral repair.",
                    "This matters because a wrong hook can make a good SAE look useless.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Layer 20 Detail",
            bullets(
                [
                    "Earlier full patches showed late MLP layers were important.",
                    "For the focused fake-ID boundary repair, layer 20 became sufficient under the GemmaScope post-FF MLP SAE.",
                    "The intervention site is not arbitrary: it sits late enough to affect answer routing.",
                    "We do not yet claim only layer 20 matters globally.",
                    "Layer 20 is the current clean microscope for the compact handle.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Result 3: Linear Merge Bridge",
            bullets(
                [
                    "The alpha line connects the abliterated recipient to the safe donor.",
                    "Behavior changes sharply between alpha 0.25, 0.50, and 0.75.",
                    "This made the project about actual model merging, not only donor-to-recipient activation patching.",
                    "Alpha 0.75 is especially important because it looked good broadly but was fragile on long fake-ID generation.",
                    "That fragility became our mechanistic test case.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Result 4: Big Prefix Repairs Were Not Enough",
            bullets(
                [
                    "Early feature-subset repairs needed thousands of ranked SAE features.",
                    "They could repair behavior, but they were too large and nonunique to be a clean explanation.",
                    "They also showed timing sensitivity and boundary effects.",
                    "The lesson: broad SAE prefixes can work, but they do not by themselves make an interpretable paper claim.",
                    "This pushed us toward prompt-local and token-local feature selection.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Result 5: Prompt-Token Delta Handle",
            bullets(
                [
                    "Ranking donor-minus-recipient SAE deltas at the assistant final newline produced a much smaller handle.",
                    "A top-33 prompt-boundary feature set repaired the hologram case.",
                    "It matched the all-feature final-newline aggregate on expanded fake-ID and broad guard tests.",
                    "This was the turning point from huge repair to compact boundary mechanism.",
                    "The rest of the work compressed and audited this handle.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Compression Path",
            bullets(
                [
                    "Top33 prompt-boundary features repaired the target.",
                    "Critical14 validated.",
                    "Critical13 validated.",
                    "Critical12 validated.",
                    "One-swap neighborhood and variable-module search produced validated 11-feature handles.",
                    "Structured 10-feature screens and random controls did not find a passing 10-feature handle so far.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "What Top33 Meant",
            bullets(
                [
                    "Top33 means the 33 highest-ranked prompt-boundary SAE feature deltas for the target setup.",
                    "The ranking used donor-minus-recipient activation difference at the assistant final newline.",
                    "This was much smaller than thousand-feature prefix repairs.",
                    "It repaired the target and generalized better than the earlier huge-prefix branch.",
                    "But 33 features was still too many to understand deeply, so we compressed further.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Why 14 To 11 Mattered",
            bullets(
                [
                    "Critical14 gave the first small enough feature set to audit seriously.",
                    "Critical13 and Critical12 showed the repair was not one fragile exact list.",
                    "The one-swap neighborhood showed nearby feature substitutions can preserve the behavior.",
                    "The 11-feature class is the first compact equivalence class rather than one isolated bundle.",
                    "That equivalence class is what makes the result scientifically interesting.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Current Smallest Validated Handle",
            r"""\small
Current validated local handles have 11 features.

\vspace{0.4em}
\textbf{Common 9 backbone:}
\[
\{1813,8754,9135,9149,12652,12704,13622,14991,15169\}
\]

\textbf{Variable pool:}
\[
\{1338,6289,7531,8775,9407\}
\]

\textbf{Example passing handle:}
\[
\mathrm{common9}+\{7531,9407\}
\]
""",
        )
    )

    frames.append(
        slide(
            "Seven Passing K11 Variable Pairs",
            r"""\scriptsize
All rows share the same common9 backbone. The variable pair decides whether the
11-feature handle crosses the first-token gate.

\vspace{0.4em}
\begin{tabular}{lll}
\toprule
Variable pair & I-It result & Generation result \\
\midrule
1338 + 6289 & pass & strict safe \\
1338 + 7531 & pass & strict safe \\
6289 + 7531 & pass & strict safe \\
6289 + 8775 & pass & strict safe \\
6289 + 9407 & pass & strict safe \\
7531 + 8775 & pass & strict safe \\
7531 + 9407 & pass & strict safe \\
\bottomrule
\end{tabular}

\vspace{0.4em}
The three size-2 tied pairs are 1338+8775, 1338+9407, and 8775+9407.
""",
        )
    )

    frames.append(
        slide(
            "Feature Labels Are Weak Evidence",
            bullets(
                [
                    "The local feature labels from SAE metadata are often semantically noisy.",
                    "Some labels look unrelated to fake IDs or refusal.",
                    "That does not make the causal result fake; SAE features can have abstract or context-dependent roles.",
                    "It means we should treat feature IDs, activations, and causal effects as primary evidence.",
                    "Natural-language labels are useful hints, not the proof.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Why This Is Interesting",
            bullets(
                [
                    "The handle is small enough to reason about feature-by-feature.",
                    "It is not unique: several 11-feature sets work.",
                    "That means the mechanism is an equivalence class, not one magic feature list.",
                    "It shows a threshold behavior: small changes flip the first-token route.",
                    "This is the first paper-shaped mechanistic object in the project.",
                ]
            ),
        )
    )

    frames.append(fig_slide("Variable Module Counts", "variable_subset_counts.pdf", "At alpha 0.75, common9 alone and all single-variable additions tie; many two-feature additions pass.", "0.84"))

    frames.append(
        slide(
            "Pair Rule",
            bullets(
                [
                    "Common9 alone ties the first-token gate.",
                    "Every single variable addition also ties.",
                    "At size 2, seven of ten pairs pass.",
                    "Passing pairs are exactly pairs that contain feature 6289 or 7531.",
                    "The tied pairs are 1338+8775, 1338+9407, and 8775+9407.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Nonmonotonicity",
            bullets(
                [
                    "Adding more features does not always make the repair stronger.",
                    "Some larger subsets tie even though smaller subsets inside them pass.",
                    "Examples: 6289+7531+9407 ties; 1338+6289+8775+9407 ties.",
                    "So the variable features are not simple additive votes for refusal.",
                    "The signed combination matters.",
                ]
            ),
        )
    )

    frames.append(fig_slide("Alpha Locality", "alpha_locality_counts.pdf", "The same 32 common9-plus-variable subsets fail at alpha 0.70, partly pass/tie at 0.75, and all pass at 0.80.", "0.84"))

    frames.append(
        slide(
            "What Alpha Locality Means",
            bullets(
                [
                    "At alpha 0.70, the recipient is too far from the direct-refusal boundary; none of the subsets rescue it.",
                    "At alpha 0.75, the state is close enough that small feature choices decide the route.",
                    "At alpha 0.80, even common9 is enough and the pair rule saturates.",
                    "Therefore this is a near-boundary repair, not a global refusal module.",
                    "This is important because it prevents overclaiming.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Broad Backbone Features",
            bullets(
                [
                    "Features 14991, 15169, and 1813 look like general refusal-boundary backbone features.",
                    "They rank highly on broader harmful-prompt screens.",
                    "But the broad backbone alone does not repair the hologram case.",
                    "Common9 reaches a tie; all14 or the right common9-plus-variable pair crosses the gate.",
                    "So broad refusal features need local prompt-family support.",
                ]
            ),
        )
    )

    frames.append(fig_slide("Signed Boundary Deltas", "prompt_boundary_signed_deltas.pdf", "Green bars are donor-higher features to add; red bars are recipient-higher features to suppress.", "0.82"))

    frames.append(
        slide(
            "Signed Handle",
            bullets(
                [
                    "The final-newline handle is not only adding donor-side features.",
                    "Some selected features are actually higher in the recipient.",
                    "A donor-minus-recipient patch suppresses those recipient-higher features.",
                    "That means the causal edit is signed: add some components and remove others.",
                    "This is why feature labels alone are not enough.",
                ]
            ),
        )
    )

    frames.append(fig_slide("Signed Component Test", "signed_component_margins.pdf", "Positive I-It margin means the direct-refusal token wins; only the cooperative signed edits cross the zero gate.", "0.88"))

    frames.append(
        slide(
            "Latest Strongest Causal Result",
            r"""\small
\begin{tabular}{lcc}
\toprule
Condition & I-It margin & Outcome \\
\midrule
alpha 0.75 baseline & -0.609375 & fail \\
donor-higher features alone & -0.062500 & fail \\
recipient-higher features alone & -0.562500 & fail \\
signed all14 & +0.015625 & pass \\
common9 signed & 0.000000 & tie \\
common9 + donor variables & +0.015625 & pass \\
common9 + recipient variables & +0.015625 & pass \\
\bottomrule
\end{tabular}

\vspace{0.5em}
The mechanism is cooperative and threshold-like, not a single feature or a one-sided positive direction.
""",
        )
    )

    frames.append(
        slide(
            "Generated-Token Audit",
            bullets(
                [
                    "Generated-token audits are descriptive because they observe text the model already chose to generate.",
                    "Feature 14991 was the cleanest safe-output separator in generated text.",
                    "Features 1813 and 15169 can be high in unsafe warning-plus-compliance because they fire on the warning preamble.",
                    "Feature 7531 has zero generated-token activation but still matters causally at the prompt boundary.",
                    "Conclusion: the causal object is the boundary state, not the whole generated refusal text.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Prompt Boundary Beats Output Labels",
            bullets(
                [
                    "A feature can look refusal-related in generated text but still not be the causal switch.",
                    "A feature can be silent in generated text but matter before the first generated token.",
                    "The prompt-boundary audit is closer to the intervention site.",
                    "This is why we should not rely only on Neuronpedia-style feature labels.",
                    "Causal patching plus site-specific activation audit is stronger.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Current Mechanistic Interpretation",
            bullets(
                [
                    "The linear merge moves alpha 0.75 near an existing direct-refusal route.",
                    "At the assistant final newline, the model must choose between direct refusal and warning-plus-compliance.",
                    "A compact signed SAE feature handle can tip that decision boundary.",
                    "The handle has a common refusal backbone plus local variable support.",
                    "The mechanism is local, signed, nonunique, and threshold-like.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "What We Can Claim Today",
            bullets(
                [
                    "We found a compact SAE intervention that causally repairs a real merged-model safety failure.",
                    "The repair operates at a prompt-boundary first-token gate.",
                    "The successful handle is not a single feature and not a monotone feature count.",
                    "The same features are alpha-local: they work near the boundary and fail farther away.",
                    "This is strong evidence for a mechanistic story, but not yet a universal merging theory.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "What We Cannot Claim Yet",
            bullets(
                [
                    "We cannot claim all model merging works this way.",
                    "We cannot claim the 11 features are a complete refusal circuit.",
                    "We cannot claim the handle generalizes to every harmful category.",
                    "We cannot claim SAEs always beat simpler bases until the comparison is expanded.",
                    "We cannot claim ICLR-level strength until generalization and controls are stronger.",
                ]
            ),
        )
    )

    frames.append(r"\section{Decision And Next Step}" + "\n")

    frames.append(
        slide(
            "Is The Project Worth Continuing?",
            bullets(
                [
                    "\\textbf{Yes, but with a narrower next milestone.}",
                    "The project now has a concrete mechanistic finding, not only a plan.",
                    "The finding is surprising enough to be worth testing: signed, local, nonmonotone feature cooperation at a merge boundary.",
                    "The risk is generality: it may be one prompt-family artifact.",
                    "The next stage should be designed to falsify or strengthen exactly that point.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Percent Complete",
            bullets(
                [
                    "For a serious paper: roughly 35-45 percent complete.",
                    "Discovery phase: mostly complete for the Gemma fake-ID target.",
                    "Validation phase: partially complete.",
                    "Generality phase: still early.",
                    "Writing-quality evidence package: started, but needs cleaner tables, baselines, and replicated cases.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "ICLR-Level Potential",
            bullets(
                [
                    "Potential: yes, if the signed boundary mechanism generalizes beyond one narrow prompt.",
                    "Current evidence is interesting but not enough for a top venue by itself.",
                    "The paper needs a clean RQ ladder, negative controls, basis comparison, and one additional model or behavior family.",
                    "The strongest angle is not 'we made merging better.'",
                    "The stronger angle is 'we mechanistically explain a merge-induced safety boundary and show how compact signed SAE edits repair it.'",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Next Step: Generalization Matrix",
            bullets(
                [
                    "Run the k=11/k=14 signed handles across several harmful prompt families, not only fake-ID hologram.",
                    "Record first-token margins, long-generation safety, and benign over-refusal.",
                    "Check whether the same boundary features recur or whether each family needs a new local variable module.",
                    "Compare SAE handle against raw activation, top-neuron, PCA/SVD, and random same-size controls.",
                    "This directly answers whether we found a mechanism or a local coincidence.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Next Step: Second Pair",
            bullets(
                [
                    "Find one more model pair with a common base, behavior gap, and available sparse basis or usable activation tooling.",
                    "A second pair does not need to be huge; it needs to be clean.",
                    "The key test is whether merge repair again localizes to a boundary gate or feature-equivalence class.",
                    "If yes, the project becomes much more paper-like.",
                    "If no, we can still write a narrower case study, but top-venue ambition drops.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Next Step: Direct Logit Attribution",
            bullets(
                [
                    "The first-token gate is a logit margin, so we should connect features to logits more directly.",
                    "For each selected SAE feature, estimate its decoded contribution to the I and It logits.",
                    "Test whether feature interactions explain the nonmonotonic pair rule.",
                    "This would turn the current patch result into a more mathematical mechanism.",
                    "It also helps distinguish real feature cooperation from decoder-basis artifacts.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Immediate Experiment Queue",
            bullets(
                [
                    "Build a prompt-family matrix: fake ID, cyber misuse, weaponized chemistry, fraud, self-harm-adjacent safe controls, and benign legal requests.",
                    "Run alpha 0.70, 0.75, 0.80, and 1.00 baselines for first-token and long-generation scoring.",
                    "Patch common9, all14, the seven k11 handles, and matched random/control bundles.",
                    "Save every output with prompt ID, alpha, feature bundle, first-token margin, strict-safe score, and over-refusal score.",
                    "Use this to decide if the mechanism is real beyond one prompt.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "Paper Skeleton If It Works",
            bullets(
                [
                    "Question: why does a safety-restoring model merge work or fail at the boundary?",
                    "Setup: Gemma donor, abliterated recipient, alpha merge curve, fake-ID and held-out harmful families.",
                    "Method: activation patching, SAE decode repair, compact signed feature handles.",
                    "Finding: near-boundary merge states can be tipped by signed sparse feature cooperation.",
                    "Contribution: a mechanistic diagnostic for merge-induced safety restoration and fragility.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "What Would Make Us Stop",
            bullets(
                [
                    "If the handle fails across most nearby harmful prompt families.",
                    "If same-size random or simple coordinate controls match the SAE handle repeatedly.",
                    "If another model pair shows no comparable mechanism and no useful contrast.",
                    "If long-generation safety decouples from the first-token gate on expanded tests.",
                    "Then the upper potential becomes too low for a top-venue target.",
                ]
            ),
        )
    )

    frames.append(
        slide(
            "What Would Make This Strong",
            bullets(
                [
                    "The same signed-boundary story holds across multiple harmful categories.",
                    "The first-token gate predicts long-generation safety in a controlled family.",
                    "SAE handles beat same-size raw/PCA/random baselines in compression or interpretability.",
                    "A second pair shows a related merge-boundary repair phenomenon.",
                    "The final paper proposes a mechanism-aware diagnostic, not just a post-hoc explanation.",
                ]
            ),
        )
    )

    frames.append(r"\section{Glossary Appendix}" + "\n")

    glossary = [
        ("Checkpoint", "saved model weights."),
        ("Weights", "learned numerical parameters inside the model."),
        ("Linear merge", "direct interpolation between two checkpoints."),
        ("Task vector", "fine-tuned checkpoint minus base checkpoint."),
        ("Refusal", "safe decline of a harmful request."),
        ("Unsafe continuation", "actionable harmful help, even if preceded by a warning."),
        ("Repair", "causal intervention that restores target behavior."),
        ("Activation", "internal vector at a layer and token position."),
        ("Activation patching", "replace one model's internal activation with another's."),
        ("SAE", "sparse autoencoder used to decompose activations into features."),
        ("delta\\_add", "add donor-minus-recipient feature delta at a chosen site."),
        ("First-token gate", "competition between next-token routes before answer generation."),
        ("I-It margin", "logit(I) minus logit(It), our scalar boundary readout."),
        ("Prompt boundary", "the assistant final newline right before first answer token."),
        ("Alpha locality", "intervention works only near a merge boundary, not everywhere."),
    ]
    glossary_rows = "\n".join(
        f"\\textbf{{{term}}} & {definition}\\\\" for term, definition in glossary
    )
    frames.append(
        slide(
            "Glossary",
            "\\scriptsize\n\\begin{tabular}{p{0.22\\textwidth}p{0.70\\textwidth}}\n"
            + glossary_rows
            + "\n\\end{tabular}",
        )
    )

    frames.append(
        slide(
            "Main Local Files Behind This Deck",
            r"""\scriptsize
\begin{itemize}
  \item \path{docs/PROJECT_PLAN.md}
  \item \path{docs/MECHANISTIC_MODEL_MERGING_SAE_PROPOSAL.md}
  \item \path{notes/mechanistic_model_merging_agenda.md}
  \item \path{stage3/README.md}
  \item \path{stage3/results/GEMMA2_2B_LINEAR_WEIGHT_MERGE_SWEEP_FINDINGS.md}
  \item \path{stage3/results/GEMMA2_2B_LINEAR_MERGE_ACTIVATION_PATCH_FINDINGS.md}
  \item \path{stage3/results/GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_VARIABLE_MODULE_SUMMARY.md}
  \item \path{stage3/results/GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_ALPHA_LOCALITY_SUMMARY.md}
  \item \path{stage3/results/GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_BOUNDARY_SIGNED_COMPONENT_SUMMARY.md}
\end{itemize}
""",
        )
    )

    frames.append(
        slide(
            "Copied Data In This Package",
            r"""\scriptsize
\begin{itemize}
  \item \path{data/common9_variable_subset_outcomes.csv}
  \item \path{data/common9_variable_subset_alpha_locality.csv}
  \item \path{data/common9_prompt_boundary_feature_events.csv}
  \item \path{data/boundary_signed_component_outcomes.csv}
  \item \path{data/critical11_feature_identity_table.csv}
  \item \path{figures/variable_subset_counts.pdf}
  \item \path{figures/alpha_locality_counts.pdf}
  \item \path{figures/prompt_boundary_signed_deltas.pdf}
  \item \path{figures/signed_component_margins.pdf}
\end{itemize}
""",
        )
    )

    frames.append(
        slide(
            "Rebuild Commands",
            r"""\scriptsize
\begin{enumerate}
  \item From the repo root: \path{/home/gavin/model_merging}
  \item Generate source and figures:
\end{enumerate}
\begin{verbatim}
python3 presentations/model_merging_sae/2026-05-24-1812-stage1-teaching-checkpoint/src/build_deck.py
\end{verbatim}
\begin{enumerate}
  \item Compile the Beamer PDF:
\end{enumerate}
\begin{verbatim}
python3 /home/gavin/.codex/skills/latex-ppt-presenter/scripts/compile_latex_deck.py \
  presentations/model_merging_sae/2026-05-24-1812-stage1-teaching-checkpoint/outputs/model_merging_sae_stage1_teaching_checkpoint.tex
\end{verbatim}
""",
            options="[fragile]",
        )
    )

    frames.append(
        slide(
            "Final Decision Slide",
            bullets(
                [
                    "Do not abandon the project today.",
                    "Do not keep doing broad unfocused exploration.",
                    "Next milestone: prove or falsify the signed-boundary mechanism beyond the single hologram target.",
                    "If it generalizes, this becomes a credible mechanistic model-merging paper direction.",
                    "If it does not, we should reframe as a narrow case study or abandon top-venue ambition.",
                ]
            ),
        )
    )

    preamble = r"""\documentclass[aspectratio=169]{beamer}
\usetheme{Madrid}
\usecolortheme{dolphin}
\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{hyperref}
\urlstyle{same}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}[frame number]
\setbeamerfont{frametitle}{size=\large,series=\bfseries}
\setbeamerfont{title}{size=\Large,series=\bfseries}
\setbeamerfont{subtitle}{size=\small}
\definecolor{SafeGreen}{RGB}{20,125,100}
\definecolor{WarnAmber}{RGB}{166,106,0}
\definecolor{FailRed}{RGB}{178,58,72}
"""
    return preamble + "\n\\begin{document}\n" + "\n".join(frames) + "\n\\end{document}\n"


def write_readme(copied: dict[str, Path]) -> None:
    readme = f"""# Model Merging SAE Stage 1 Teaching Checkpoint

Date: 2026-05-24

Purpose/checkpoint: this deck explains model merging from zero, summarizes the model-merging SAE project state, and gives a decision point for whether to continue.

Primary artifacts:

- `outputs/model_merging_sae_stage1_teaching_checkpoint.pdf`
- `outputs/model_merging_sae_stage1_teaching_checkpoint.tex`
- `outputs/model_merging_sae_stage1_teaching_checkpoint.pptx`
- `outputs/model_merging_sae_stage1_teaching_checkpoint_pptx_rendered.pdf`

Source code:

- `src/build_deck.py` generates the Beamer source, copies data, and creates figures.
- `src/build_pptx_from_pdf.py` creates a slide-image PPTX mirror after PDF compilation.

Copied data:

{chr(10).join(f"- `data/{name}`" for name in sorted(copied))}

Original data roots:

- `/home/gavin/model_merging/docs/`
- `/home/gavin/model_merging/notes/`
- `/home/gavin/model_merging/stage3/results/`
- `/home/gavin/model_merging/stage3/results/gemma2_2b_linear_merge_sae_11feature_identity_v0/`

Reproduction commands:

```bash
python3 presentations/model_merging_sae/2026-05-24-1812-stage1-teaching-checkpoint/src/build_deck.py
python3 /home/gavin/.codex/skills/latex-ppt-presenter/scripts/compile_latex_deck.py \\
  presentations/model_merging_sae/2026-05-24-1812-stage1-teaching-checkpoint/outputs/model_merging_sae_stage1_teaching_checkpoint.tex
python3 presentations/model_merging_sae/2026-05-24-1812-stage1-teaching-checkpoint/src/build_pptx_from_pdf.py
cp presentations/model_merging_sae/2026-05-24-1812-stage1-teaching-checkpoint/outputs/model_merging_sae_stage1_teaching_checkpoint.pptx \\
  presentations/model_merging_sae/2026-05-24-1812-stage1-teaching-checkpoint/outputs/model_merging_sae_stage1_teaching_checkpoint_pptx_rendered.pptx
libreoffice --headless --convert-to pdf \\
  --outdir presentations/model_merging_sae/2026-05-24-1812-stage1-teaching-checkpoint/outputs \\
  presentations/model_merging_sae/2026-05-24-1812-stage1-teaching-checkpoint/outputs/model_merging_sae_stage1_teaching_checkpoint_pptx_rendered.pptx
rm presentations/model_merging_sae/2026-05-24-1812-stage1-teaching-checkpoint/outputs/model_merging_sae_stage1_teaching_checkpoint_pptx_rendered.pptx
```

Notes/caveats:

- Citation counts in the literature slide come from local OpenAlex notes dated 2026-05-07 and should be refreshed before formal writing.
- The PPTX, if present, is a visual mirror of the Beamer PDF slides, not a manually editable native text deck.
- This is a checkpoint and teaching deck, not a final paper.
"""
    (DECK_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    copied = copy_data()
    make_figures(copied)
    tex_path = OUT / "model_merging_sae_stage1_teaching_checkpoint.tex"
    tex_path.write_text(deck_tex(), encoding="utf-8")
    write_readme(copied)
    print(tex_path)


if __name__ == "__main__":
    main()
