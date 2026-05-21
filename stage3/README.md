# Stage 3: Basis Validation For Clean vs Messy Refusal

Stage 3 tests whether a richer representational basis helps explain the key
Stage 2 finding:

> Late MLP module insertion restores a refusal signal, but the full merge gives
> cleaner refusal behavior.

Stage 3 has since corrected that framing. The behavior is better described as
attempted-refusal transfer with frequent quality failure, not as reliable clean
refusal transfer. The main decision memo is
`results/STAGE3_REFUSAL_FAILURE_FINDINGS.md`.

The first target is deliberately conservative. Before training or trusting
SAEs/transcoders, compare simple bases:

- residual stream activations;
- MLP output activations;
- raw coordinates;
- PCA projections;
- random projections;
- top-variance neuron subsets.

Only if sparse features beat or complement these baselines should we spend more
compute on SAE/transcoder interpretation.

Primary script:

- `scripts/analyze_smollm2_refusal_basis.py`

Main outputs:

- `results/smollm2_refusal_basis_records.jsonl`
- `results/smollm2_refusal_basis_metrics.csv`
- `results/SMOLLM2_REFUSAL_BASIS_SUMMARY.md`
- `cache/smollm2_refusal_basis_activations.pt`

Audit helper:

- `scripts/prepare_smollm2_refusal_audit.py`
- `scripts/rescore_smollm2_refusal_basis_cache.py`
- `scripts/apply_smollm2_refusal_assistant_audit.py`
- `scripts/analyze_smollm2_refusal_failure_modes.py`
- `scripts/run_smollm2_refusal_direction_steering.py`
- `scripts/run_smollm2_refusal_quality_module_patches.py`
- `scripts/run_smollm2_refusal_activation_patches.py`
- `scripts/run_smollm2_refusal_direction_ablation.py`
- `results/smollm2_refusal_manual_audit_sample.csv`
- `results/SMOLLM2_REFUSAL_MANUAL_AUDIT_GUIDE.md`
- `results/SMOLLM2_REFUSAL_ASSISTANT_AUDIT_SUMMARY.md`
- `results/SMOLLM2_REFUSAL_FAILURE_MODE_SUMMARY.md`
- `results/SMOLLM2_REFUSAL_DIRECTION_STEERING_SUMMARY.md`
- `results/SMOLLM2_REFUSAL_QUALITY_MODULE_PATCHES_*.md`
- `results/SMOLLM2_REFUSAL_ACTIVATION_PATCHES_*.md`
- `results/SMOLLM2_REFUSAL_DIRECTION_ABLATION_*.md`
