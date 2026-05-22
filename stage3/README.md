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

## Active Qwen Residual Track

The active clean target has moved to the public Qwen2.5-1.5B safety-loss case:

- donor/base: `Qwen/Qwen2.5-1.5B-Instruct`;
- recipient: `nbeerbower/EVA-abliterated-TIES-Qwen2.5-1.5B`;
- shared baseline: PCA64 donor-recipient MLP delta on layers `12-24`;
- hard residual repair target: full donor MLP activations in layers `16-23`.

The first frozen benchmark file is:

- `data/qwen1_5b_residual_benchmark/qwen1_5b_residual_benchmark_v0.jsonl`

This benchmark separates donor-solved residual harmful prompts from paired
benign controls. It is intentionally small; it exists to make subsequent
SAE/transcoder validation reproducible before expanding prompt coverage.

Current v0 baseline:

- full donor MLP `16-23` passes all harmful and benign v0 prompts;
- residual PCA/mean baselines do not;
- native MLP-output coordinate top-k only passes the full v0 benchmark at
  `topk1344`, which is `87.5%` of the `1536`-dimensional MLP output space.

This means the first SAE/transcoder target is not merely "beat PCA"; it must
beat or explain a broad coordinate-residual baseline.

First learned SAE smoke:

- per-layer residual SAEs with high train EV (`0.993-0.996`) do not preserve
  the full v0 behavior;
- all tested SAE variants repair one-time-code but fail tracking-script and
  permission-slip;
- this holds even for weakly sparse `d1536`/`d2048` dictionaries.

The immediate lesson is that reconstruction quality is not causal completeness.
The next sparse attempt should change the training distribution or model class,
not just increase vanilla SAE size.

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
