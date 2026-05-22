# GemmaScope Boundary-vs-Content Checkpoint

Date: 2026-05-22

This package records the Stage 3 checkpoint where the GemmaScope MLP-SAE
feature-subset result reframed from harmful-content semantics to
assistant-boundary refusal-state transfer.

## Primary Files

- `gemmascope_boundary_content_checkpoint.tex`: Beamer source.
- `gemmascope_boundary_content_checkpoint.pdf`: compiled PDF deck.
- `gemmascope_boundary_content_checkpoint.pptx`: native PowerPoint deck.
- `src/make_boundary_content_artifacts.py`: regenerates CSV summaries, figures,
  and the PPTX from copied evidence.

## Generated Evidence

- `figures/all_vs_content_harmful_clean.png`: all-token vs content-token
  harmful clean refusal comparison.
- `figures/event_category_share.png`: assistant-boundary/template share of top
  absolute-delta event rows.
- `data/boundary_content_comparison.csv`: plotted harmful-clean comparison.
- `data/all_token_abs_delta_event_categories.csv`: event-category counts.
- `data/all_token_abs_delta_top_tokens.csv`: most frequent top absolute-delta
  tokens.

## Copied Source Evidence

- `data/GEMMA2_2B_GEMMASCOPE_MLP_SAE_BOUNDARY_VS_CONTENT_FINDINGS.md`
- `data/all_token_random_seed_metrics.csv`
- `data/content_token_random_seed_metrics.csv`
- `data/all_token_feature_events.jsonl`
- `data/content_token_feature_events.jsonl`
- `data/all_token_feature_audit_summary.md`
- `data/content_token_feature_audit_summary.md`

## Rebuild

From the repository root:

```bash
python3 presentations/2026-05-22-1525-gemmascope-boundary-content/src/make_boundary_content_artifacts.py
python3 /home/gavin/.codex/skills/latex-ppt-presenter/scripts/compile_latex_deck.py \
  presentations/2026-05-22-1525-gemmascope-boundary-content/gemmascope_boundary_content_checkpoint.tex
```
