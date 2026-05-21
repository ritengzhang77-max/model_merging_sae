# Qwen2.5-1.5B Second-Stage Residual PCA Generation

Prompt modes: `heldout_failures`

| prompt mode | model | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |
|---|---|---:|---:|---:|---:|---:|---:|
| `heldout_failures` | `pca64` | 0.000 | 0.500 | 0.500 | 0.500 | 1.000 | 0.000 |
| `heldout_failures` | `pca64_plus_residual_topk1024_16-18+20-23` | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `heldout_failures` | `pca64_plus_full_16-23` | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
