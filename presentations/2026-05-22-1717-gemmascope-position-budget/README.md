# GemmaScope Position/Budget Checkpoint Deck

Purpose: checkpoint the Stage 3 position-restricted GemmaScope MLP-SAE repair
after the k2048 follow-up and k512 boundary/generated budget probe.

Primary artifacts:

- `gemmascope_position_budget_checkpoint.pdf`
- `gemmascope_position_budget_checkpoint.pptx`
- `gemmascope_position_budget_checkpoint.tex`

Source code:

- `src/make_assets.py` builds the plot and native PowerPoint deck.

Local copied data:

- `data/k1024_position_restricted_metrics.csv`
- `data/k2048_position_restricted_metrics.csv`
- `data/k512_boundary_generated_budget_metrics.csv`

Original data roots:

- `stage3/results/gemma2_2b_gemmascope_mlp_sae_position_restricted_atomic_v0/`
- `stage3/results/gemma2_2b_gemmascope_mlp_sae_position_restricted_k2048_v0/`
- `stage3/results/gemma2_2b_gemmascope_mlp_sae_boundary_generated_budget_v0/`

Reproduction commands:

```bash
python3 presentations/2026-05-22-1717-gemmascope-position-budget/src/make_assets.py
python3 /home/gavin/.codex/skills/latex-ppt-presenter/scripts/compile_latex_deck.py \
  presentations/2026-05-22-1717-gemmascope-position-budget/gemmascope_position_budget_checkpoint.tex
```

Notes:

- The native PPTX is generated directly with `python-pptx`.
- The Beamer PDF uses the same copied CSV evidence and generated plot.
