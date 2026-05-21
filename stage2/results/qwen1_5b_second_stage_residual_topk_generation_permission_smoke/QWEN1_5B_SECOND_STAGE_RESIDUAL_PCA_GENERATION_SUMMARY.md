# Qwen2.5-1.5B Second-Stage Residual PCA Generation

Prompt modes: `heldout_failures_harmful_only, stress_permission_harmful_only`

| prompt mode | model | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |
|---|---|---:|---:|---:|---:|---:|---:|
| `heldout_failures_harmful_only` | `pca64` | 0.000 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 |
| `heldout_failures_harmful_only` | `pca64_plus_residual_topk256_16-23` | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 |
| `heldout_failures_harmful_only` | `pca64_plus_residual_topk1024_16-23` | 0.500 | 0.500 | 0.000 | 0.500 | 0.000 | 0.000 |
| `heldout_failures_harmful_only` | `pca64_plus_full_16-23` | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `stress_permission_harmful_only` | `pca64` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `stress_permission_harmful_only` | `pca64_plus_residual_topk256_16-23` | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| `stress_permission_harmful_only` | `pca64_plus_residual_topk1024_16-23` | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| `stress_permission_harmful_only` | `pca64_plus_full_16-23` | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
