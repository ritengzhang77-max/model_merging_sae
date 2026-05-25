# Model Merging SAE Stage 1 Teaching Checkpoint

Date: 2026-05-24

Purpose/checkpoint: this 88-slide deck explains model merging from zero, summarizes the model-merging SAE project state, and gives a decision point for whether to continue.

Primary artifacts:

- `outputs/model_merging_sae_stage1_teaching_checkpoint.pdf`
- `outputs/model_merging_sae_stage1_teaching_checkpoint.tex`
- `outputs/model_merging_sae_stage1_teaching_checkpoint.pptx`
- `outputs/model_merging_sae_stage1_teaching_checkpoint_pptx_rendered.pdf`

Source code:

- `src/build_deck.py` generates the Beamer source, copies data, and creates figures.
- `src/build_pptx_from_pdf.py` creates a slide-image PPTX mirror after PDF compilation.

Copied data:

- `data/GEMMA2_2B_LINEAR_MERGE_ACTIVATION_PATCH_FINDINGS.md`
- `data/GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_ALPHA_LOCALITY_SUMMARY.md`
- `data/GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_BOUNDARY_SIGNED_COMPONENT_SUMMARY.md`
- `data/GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_EVENT_AUDIT_SUMMARY.md`
- `data/GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_HANDLE_SUMMARY.md`
- `data/GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_PROMPT_BOUNDARY_EVENT_SUMMARY.md`
- `data/GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_VARIABLE_MODULE_SUMMARY.md`
- `data/GEMMA2_2B_LINEAR_WEIGHT_MERGE_SWEEP_FINDINGS.md`
- `data/MECHANISTIC_MODEL_MERGING_SAE_PROPOSAL.md`
- `data/PROJECT_PLAN.md`
- `data/README.md`
- `data/boundary_signed_component_outcomes.csv`
- `data/common9_prompt_boundary_feature_events.csv`
- `data/common9_variable_subset_alpha_locality.csv`
- `data/common9_variable_subset_outcomes.csv`
- `data/critical11_feature_identity_table.csv`
- `data/mechanistic_model_merging_agenda.md`

Original data roots:

- `/home/gavin/model_merging/docs/`
- `/home/gavin/model_merging/notes/`
- `/home/gavin/model_merging/stage3/results/`
- `/home/gavin/model_merging/stage3/results/gemma2_2b_linear_merge_sae_11feature_identity_v0/`

Reproduction commands:

```bash
python3 presentations/model_merging_sae/2026-05-24-1812-stage1-teaching-checkpoint/src/build_deck.py
python3 /home/gavin/.codex/skills/latex-ppt-presenter/scripts/compile_latex_deck.py \
  presentations/model_merging_sae/2026-05-24-1812-stage1-teaching-checkpoint/outputs/model_merging_sae_stage1_teaching_checkpoint.tex
python3 presentations/model_merging_sae/2026-05-24-1812-stage1-teaching-checkpoint/src/build_pptx_from_pdf.py
cp presentations/model_merging_sae/2026-05-24-1812-stage1-teaching-checkpoint/outputs/model_merging_sae_stage1_teaching_checkpoint.pptx \
  presentations/model_merging_sae/2026-05-24-1812-stage1-teaching-checkpoint/outputs/model_merging_sae_stage1_teaching_checkpoint_pptx_rendered.pptx
libreoffice --headless --convert-to pdf \
  --outdir presentations/model_merging_sae/2026-05-24-1812-stage1-teaching-checkpoint/outputs \
  presentations/model_merging_sae/2026-05-24-1812-stage1-teaching-checkpoint/outputs/model_merging_sae_stage1_teaching_checkpoint_pptx_rendered.pptx
rm presentations/model_merging_sae/2026-05-24-1812-stage1-teaching-checkpoint/outputs/model_merging_sae_stage1_teaching_checkpoint_pptx_rendered.pptx
```

Notes/caveats:

- Citation counts in the literature slide come from local OpenAlex notes dated 2026-05-07 and should be refreshed before formal writing.
- The PPTX, if present, is a visual mirror of the Beamer PDF slides, not a manually editable native text deck.
- This is a checkpoint and teaching deck, not a final paper.
