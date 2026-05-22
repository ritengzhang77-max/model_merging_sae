# GemmaScope Position-Restricted Checkpoint

Date: 2026-05-22

This package records the Stage 3 position-restricted causal checkpoint for
GemmaScope MLP-SAE feature patches.

## Primary Files

- `gemmascope_position_restricted_checkpoint.tex`: Beamer source.
- `gemmascope_position_restricted_checkpoint.pdf`: compiled PDF deck.
- `gemmascope_position_restricted_checkpoint.pptx`: native PowerPoint deck.
- `src/make_position_restricted_artifacts.py`: regenerates plots and PPTX from
  copied aggregate metrics.

## Evidence

- `data/GEMMA2_2B_GEMMASCOPE_MLP_SAE_POSITION_RESTRICTED_SUMMARY.md`
- `data/position_restricted_metrics.csv`
- `figures/all_12_20_4_8_position_modes.png`
- `figures/all_12_20_8_12_position_modes.png`

## Rebuild

```bash
python3 presentations/2026-05-22-1644-gemmascope-position-restricted/src/make_position_restricted_artifacts.py
python3 /home/gavin/.codex/skills/latex-ppt-presenter/scripts/compile_latex_deck.py \
  presentations/2026-05-22-1644-gemmascope-position-restricted/gemmascope_position_restricted_checkpoint.tex
```
