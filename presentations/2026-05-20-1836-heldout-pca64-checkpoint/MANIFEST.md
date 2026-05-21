# Held-out PCA64 Checkpoint Presentation Package

Created: 2026-05-20

## Main Files

- `heldout_pca64_checkpoint.tex` - LaTeX Beamer source.
- `heldout_pca64_checkpoint.pdf` - compiled PDF.
- `heldout_pca64_checkpoint.pptx` - native PowerPoint version.

## Supporting Files

- `data/` - copied metrics for held-out PCA64, calibration-12 PCA64, and full
  activation patch.
- `figures/heldout_patch_comparison.*` - generated comparison plot.
- `src/build_checkpoint_assets.py` - regenerates the plot and PowerPoint.

## Build Commands

```bash
python3 presentations/2026-05-20-1836-heldout-pca64-checkpoint/src/build_checkpoint_assets.py
python3 /home/gavin/.codex/skills/latex-ppt-presenter/scripts/compile_latex_deck.py presentations/2026-05-20-1836-heldout-pca64-checkpoint/heldout_pca64_checkpoint.tex
```
