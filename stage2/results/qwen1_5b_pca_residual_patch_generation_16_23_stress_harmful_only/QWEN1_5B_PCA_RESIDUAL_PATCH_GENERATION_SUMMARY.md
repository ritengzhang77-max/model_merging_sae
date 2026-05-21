# Qwen2.5-1.5B PCA64 Residual Patch Generation

PCA64 everywhere, with selected layer ranges upgraded to full donor MLP activations.

| model | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|---:|
| `pca64_plus_full_16+17+18+19+20+21+22+23` | 0.833 | 0.833 | 0.000 | 0.167 | 0.000 | 0.000 |
