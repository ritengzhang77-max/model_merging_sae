# Qwen2.5-1.5B PCA64 Residual Target-Loss Diagnostic

Prompt mode: `stress`

Teacher-forced refusal-target loss with PCA64 on all layers and selected layers upgraded to full donor MLP activations.

| variant | harmful loss | harmful gap closed vs ablated | gap closed vs PCA64 | benign loss | benign gap closed |
|---|---:|---:|---:|---:|---:|
| `base` | 0.126 | 1.000 | 0.000 | 0.409 | 1.000 |
| `abliterated` | 2.685 | 0.000 | 0.000 | 3.147 | 0.000 |
| `pca64` | 0.267 | 0.945 | 0.000 | 0.651 | 0.911 |
| `pca64_plus_full_20+21+23` | 0.265 | 0.945 | 0.009 | 0.616 | 0.924 |
| `pca64_plus_full_20+21+22+23+24` | 0.238 | 0.956 | 0.205 | 0.518 | 0.960 |
| `full_12_24` | 0.278 | 0.941 | -0.077 | 0.540 | 0.952 |
