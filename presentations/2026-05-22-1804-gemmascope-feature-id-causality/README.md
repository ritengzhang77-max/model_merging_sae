# GemmaScope Feature-ID Causality Checkpoint Deck

Purpose: present the first layer-local feature-band causal lead inside the
GemmaScope `assistant_boundary_or_generated` sparse repair path.

Primary artifacts:

- `gemmascope_feature_id_causality_checkpoint.pdf`
- `gemmascope_feature_id_causality_checkpoint.pptx`
- `gemmascope_feature_id_causality_checkpoint.tex`

Source code:

- `src/make_assets.py` builds the figures and native PowerPoint deck.

Local copied data:

- `data/feature_id_threshold_metrics.csv`
- `data/l19_tail_feature_scores.csv`
- `data/l19_tail_feature_events.jsonl`

Original data roots:

- `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature_id_threshold_v0/`
- `stage3/results/gemma2_2b_gemmascope_mlp_sae_l19_tail_feature_audit_v0/`

Reproduction commands:

```bash
python3 presentations/2026-05-22-1804-gemmascope-feature-id-causality/src/make_assets.py
python3 /home/gavin/.codex/skills/latex-ppt-presenter/scripts/compile_latex_deck.py \
  presentations/2026-05-22-1804-gemmascope-feature-id-causality/gemmascope_feature_id_causality_checkpoint.tex
```

Notes:

- The native PPTX is generated directly with `python-pptx`.
- The Beamer PDF and PPTX use copied CSV/JSONL evidence under `data/`.
