# Qwen2.5-1.5B PCA64 Residual Patch Generation

PCA64 everywhere, with selected layer ranges upgraded to full donor MLP activations.

| model | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|---:|
| `pca64` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pca64_plus_full_16+17+18+19+20+21+22+23_all` | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pca64_plus_full_16+17+18+19+20+21+22+23_prompt` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pca64_plus_full_16+17+18+19+20+21+22+23_generated` | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `pca64_plus_full_16+17+18+19+20+21+22+23_last` | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
