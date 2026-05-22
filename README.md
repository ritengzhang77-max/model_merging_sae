# Model Merging SAE

Mechanistic model-merging research workspace focused on explaining why a public
Qwen2.5-1.5B abliterated merge loses refusal behavior and how donor activations
or sparse residual patches repair it.

Current core finding:

- `PCA64` over donor-recipient MLP deltas gives a strong broad repair.
- A tracking-basis residual `topk1024` patch in layers `16-18 + 20-23` repairs
  the original one-time-code/tracking residual pair while preserving local
  benign controls.
- The permission-slip residual still requires fuller donor MLP activations in
  `16-23`, making it the current hard target for SAE/transcoder analysis.

Key docs:

- `docs/MECHANISTIC_MODEL_MERGING_SAE_PROPOSAL.md`
- `docs/VALUE_ACTION_LESSONS_FOR_MODEL_MERGING.md`
- `stage2/results/PROVENANCE.md`
- `stage2/results/qwen1_5b_residual_prompt_catalog/RESIDUAL_PROMPT_CATALOG.md`
- `stage2/results/qwen1_5b_residual_family_geometry/FAMILY_SPLIT_INTERPRETATION.md`

Large local checkpoints, caches, and downloaded paper PDFs are intentionally
ignored by Git. Rebuild or redownload them from the scripts and recorded
provenance when needed.
