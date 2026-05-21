# Stage 3 Results Provenance

This ledger records result artifacts that are likely to feed later paper tables,
figures, or decision memos.

## Refusal Direction Ablation, HF Generation

- Date appended: 2026-05-16
- Artifact status: canonical Stage 3 decision-gate evidence
- Generating script: `stage3/scripts/run_smollm2_refusal_direction_ablation.py`
- Source activation cache: `stage3/cache/smollm2_refusal_basis_activations.pt`
- Source labels: `stage3/results/smollm2_refusal_failure_modes_labeled.csv`
- Summary memo: `stage3/results/STAGE3_REFUSAL_FAILURE_FINDINGS.md`

Artifacts:

- `stage3/results/direction_ablation_repetition_hf_last/`
  - positive-projection removal on `merge_arith_refusal`
  - targets: `mlp_out_l20:failure_repetition`,
    `mlp_out_l20:failure_bad_attempt`
  - key result: no measurable reduction in attempted refusal, problem response,
    or repetition.
- `stage3/results/direction_ablation_repetition_hf_constant/`
  - constant subtraction on `merge_arith_refusal`
  - target: `mlp_out_l20:failure_repetition`
  - key result: strong subtraction reduces literal repetition but shifts the
    output into artifact/corruption or no-refusal; it does not produce clean
    refusal.
- `stage3/results/direction_ablation_mixed_hf_constant/`
  - constant subtraction on `merge_all_linear`
  - targets: `mlp_out_l20:failure_repetition`,
    `resid_l20:failure_artifact`,
    `mlp_out_l25:failure_unsafe_or_contradictory`
  - key result: artifact and unsafe directions preserve attempted refusal but
    do not reduce problem rate; repetition direction gives the only reduction,
    partly by weakening refusal.
- `stage3/results/direction_ablation_repetition_hf_fine/`
  - fine alpha sweep on `merge_all_linear`
  - target: `mlp_out_l20:failure_repetition`
  - command:

```bash
HF_HOME=/data/gavin/model_merging/hf_cache python3 stage3/scripts/run_smollm2_refusal_direction_ablation.py \
  --recipient merge_all_linear \
  --examples 16 \
  --max-new-tokens 32 \
  --direction-specs 'mlp_out_l20:failure_repetition' \
  --alphas 0.01,0.02,0.05,0.075,0.1,0.15,0.2,0.25 \
  --position last \
  --mode constant \
  --generation-mode hf \
  --result-dir stage3/results/direction_ablation_repetition_hf_fine
```

Key numeric values:

- `merge_arith_refusal` baseline: attempted refusal 1.000, problem response
  1.000, repetition 1.000.
- `merge_arith_refusal` positive-projection removal: no measurable effect at
  tested alphas.
- `merge_arith_refusal` constant subtraction at alpha 1.0: repetition 0.250,
  artifact 0.625, problem response 0.938.
- `merge_all_linear` baseline: attempted refusal 0.750, problem response 0.812,
  repetition 0.250, artifact 0.125, unsafe/contradictory 0.438, no-refusal
  0.188.
- `merge_all_linear` fine sweep at alpha 0.05: attempted refusal 0.688, problem
  response 0.750, repetition 0.188, artifact 0.188, unsafe/contradictory 0.375,
  no-refusal 0.250.
- `merge_all_linear` fine sweep at alpha 0.25: attempted refusal 0.562, problem
  response 0.688, repetition 0.125, artifact 0.125, unsafe/contradictory 0.438,
  no-refusal 0.250.

Caveats:

- These runs use heuristic assistant-audit labels, not a human audit.
- Prompt count is 16 for the direction-ablation screens and fine sweep.
- Hugging Face generation is used for the main failure comparison because the
  custom no-cache loop can hide the repetition failure.
- The result supports a negative control claim: predictive failure directions
  are not clean causal repair handles in this setup.
