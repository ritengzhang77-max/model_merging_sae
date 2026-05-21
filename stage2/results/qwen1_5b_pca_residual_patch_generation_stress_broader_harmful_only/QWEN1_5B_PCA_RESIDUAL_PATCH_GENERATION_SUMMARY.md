# Qwen2.5-1.5B PCA64 Residual Patch Generation

PCA64 everywhere, with selected layer ranges upgraded to full donor MLP activations.

| model | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|---:|
| `pca64_plus_full_20+21+22+23+24` | 0.750 | 0.750 | 0.000 | 0.250 | 0.000 | 0.000 |
| `full_12_24` | 0.750 | 0.750 | 0.000 | 0.083 | 0.000 | 0.000 |
