# Model Merging Full Overview Presentation Package

Created: 2026-05-20

## Main Files

- `model_merging_full_overview.tex` - LaTeX Beamer source.
- `model_merging_full_overview.pdf` - compiled PDF.
- `model_merging_full_overview.pptx` - native PowerPoint version.

## Supporting Files

- `figures/` - generated plot assets used in the PDF and PPTX.
- `data/` - copied metrics, manual audit, proposal, and provenance files used by
  the slides.
- `src/build_figures.py` - regenerates the figures from `data/`.
- `src/build_pptx.py` - regenerates the PowerPoint file.

## Build Commands

```bash
python3 presentations/2026-05-20-1618-model-merging-full-overview/src/build_figures.py
python3 /home/gavin/.codex/skills/latex-ppt-presenter/scripts/compile_latex_deck.py presentations/2026-05-20-1618-model-merging-full-overview/model_merging_full_overview.tex
python3 presentations/2026-05-20-1618-model-merging-full-overview/src/build_pptx.py
```
