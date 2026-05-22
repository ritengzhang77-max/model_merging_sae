# Model Merging SAE

Mechanistic model-merging research workspace focused on explaining why public
abliterated model variants lose refusal behavior and whether raw activations,
low-rank geometry, sparse residual bases, or public SAE/transcoder ecosystems
can explain the repair.

Current core finding:

- `PCA64` over donor-recipient MLP deltas gives a strong broad repair.
- A tracking-basis residual `topk1024` patch in layers `16-18 + 20-23` repairs
  the original one-time-code/tracking residual pair while preserving local
  benign controls.
- The permission-slip residual still requires fuller donor MLP activations in
  `16-23`, making it the current hard target for SAE/transcoder analysis.
- Vanilla Qwen residual SAEs with high reconstruction EV do not pass the full
  behavioral gate. Generated-token training shifts the family repaired
  one-time-code plus permission-slip, but still misses tracking.
- A new Gemma-2-2B abliterated branch looks promising: it has a clean safety
  behavior gap, harmful-specific activation drift, and public GemmaScope sparse
  bases. It still needs module/activation patching before feature work.

Key docs:

- `docs/MECHANISTIC_MODEL_MERGING_SAE_PROPOSAL.md`
- `docs/VALUE_ACTION_LESSONS_FOR_MODEL_MERGING.md`
- `stage2/results/PROVENANCE.md`
- `stage2/results/qwen1_5b_residual_prompt_catalog/RESIDUAL_PROMPT_CATALOG.md`
- `stage2/results/qwen1_5b_residual_family_geometry/FAMILY_SPLIT_INTERPRETATION.md`

Large local checkpoints, caches, and downloaded paper PDFs are intentionally
ignored by Git. Rebuild or redownload them from the scripts and recorded
provenance when needed.
