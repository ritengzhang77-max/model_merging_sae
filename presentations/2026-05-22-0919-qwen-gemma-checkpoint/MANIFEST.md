# Qwen/Gemma Checkpoint Presentation

Created: 2026-05-22 09:19

## Files

- `qwen_gemma_checkpoint.tex` - Beamer source
- `qwen_gemma_checkpoint.pdf` - compiled PDF deck
- `qwen_gemma_checkpoint.pptx` - native PowerPoint deck
- `figures/` - generated plots used in the deck
- `data/` - copied CSV evidence used by the plots and tables
- `src/make_figures.py` - plot generation script
- `src/make_pptx.py` - PowerPoint generation script

## Checkpoint Reason

This deck records a research decision point:

- Qwen generated-trace residual SAE training improved the repaired prompt
  family but still failed the full sparse-basis gate.
- Gemma-2-2B abliterated passed behavior, architecture, and harmful-specific
  activation-drift screens, making it the next candidate branch before public
  GemmaScope SAE/transcoder work.
