# GemmaScope Single-Feature Causality Checkpoint Deck

Purpose: present the refined feature-ID result that localizes the L19 tail-band
effect to layer 19 feature ID `16048`.

Primary artifacts:

- `gemmascope_single_feature_causality_checkpoint.pdf`
- `gemmascope_single_feature_causality_checkpoint.pptx`
- `gemmascope_single_feature_causality_checkpoint.tex`

Source code:

- `src/make_assets.py` builds the figures and native PowerPoint deck.

Local copied data:

- `data/l19_tail_block_metrics.csv`
- `data/l19_rank1006_feature.csv`
- `data/l19_feature_16048_events.jsonl`

Original data roots:

- `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature_id_l19_tail_blocks_v0/`
- `stage3/results/gemma2_2b_gemmascope_mlp_sae_l19_tail_feature_audit_v0/`

Reproduction commands:

```bash
python3 presentations/2026-05-22-1924-gemmascope-single-feature-causality/src/make_assets.py
python3 /home/gavin/.codex/skills/latex-ppt-presenter/scripts/compile_latex_deck.py \
  presentations/2026-05-22-1924-gemmascope-single-feature-causality/gemmascope_single_feature_causality_checkpoint.tex
```

Notes:

- The native PPTX is generated directly with `python-pptx`.
- The Beamer PDF and PPTX use copied CSV/JSONL evidence under `data/`.
