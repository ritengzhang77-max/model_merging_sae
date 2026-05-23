# GemmaScope Feature 16048 Validation Checkpoint

Purpose: checkpoint the validation/reframing of layer-19 GemmaScope MLP-SAE
feature `16048`.

Primary artifacts:

- `gemmascope_feature16048_validation_checkpoint.pdf`
- `gemmascope_feature16048_validation_checkpoint.pptx`
- `gemmascope_feature16048_validation_checkpoint.tex`

Main conclusion:

- Feature `16048` is causal for fake-ID recovery under calibration bases `0:4`
  and `0:8`, but it is not a standalone refusal feature.
- Calibration basis `4:8` selects feature `16048` inside k896 but still fails
  fake-ID recovery, so the cooperating top-delta prefix matters.

Local copied data:

- `data/full_0_12_metrics.csv`
- `data/full_0_12_records.jsonl`
- `data/basis_4_8_eval_8_12_metrics.csv`
- `data/basis_4_8_eval_8_12_records.jsonl`
- `data/basis_0_8_eval_8_12_metrics.csv`
- `data/basis_0_8_eval_8_12_records.jsonl`
- `data/rank_stability_4_prompt_slices.csv`
- `data/rank_stability_basis0_8.csv`
- `data/fake_id_validation_summary.csv`

Original data roots:

- `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature_id_l19_feature16048_validation_v0/`
- `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature_rank_stability_v0/`

Reproduction:

```bash
python3 presentations/2026-05-22-2001-gemmascope-feature16048-validation/src/make_assets.py
python3 /home/gavin/.codex/skills/latex-ppt-presenter/scripts/compile_latex_deck.py \
  presentations/2026-05-22-2001-gemmascope-feature16048-validation/gemmascope_feature16048_validation_checkpoint.tex
```

Notes:

- The Beamer PDF and native PPTX are generated independently from the same
  copied data.
- The result narrows the mechanistic claim rather than abandoning the branch.
