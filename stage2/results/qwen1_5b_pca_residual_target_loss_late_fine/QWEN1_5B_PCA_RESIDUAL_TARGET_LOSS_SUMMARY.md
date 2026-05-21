# Qwen2.5-1.5B PCA64 Residual Target-Loss Diagnostic

Prompt mode: `heldout_failures`

Teacher-forced refusal-target loss with PCA64 on all layers and selected layers upgraded to full donor MLP activations.

| variant | harmful loss | harmful gap closed vs ablated | gap closed vs PCA64 | benign loss | benign gap closed |
|---|---:|---:|---:|---:|---:|
| `base` | 0.176 | 1.000 | 0.000 | 0.340 | 1.000 |
| `abliterated` | 2.834 | 0.000 | 0.000 | 3.049 | 0.000 |
| `pca64` | 0.392 | 0.919 | 0.000 | 0.575 | 0.913 |
| `pca64_plus_full_20` | 0.393 | 0.918 | -0.003 | 0.527 | 0.931 |
| `pca64_plus_full_21` | 0.355 | 0.933 | 0.172 | 0.541 | 0.926 |
| `pca64_plus_full_22` | 0.365 | 0.929 | 0.126 | 0.536 | 0.928 |
| `pca64_plus_full_23` | 0.378 | 0.924 | 0.065 | 0.593 | 0.907 |
| `pca64_plus_full_24` | 0.362 | 0.930 | 0.140 | 0.535 | 0.928 |
| `pca64_plus_full_20+21` | 0.371 | 0.927 | 0.101 | 0.518 | 0.934 |
| `pca64_plus_full_22+23+24` | 0.330 | 0.942 | 0.290 | 0.500 | 0.941 |
| `pca64_plus_full_20+21+22` | 0.329 | 0.943 | 0.295 | 0.453 | 0.958 |
| `pca64_plus_full_23+24` | 0.350 | 0.935 | 0.197 | 0.549 | 0.923 |
| `pca64_plus_full_20+21+22+23+24` | 0.303 | 0.952 | 0.412 | 0.428 | 0.967 |
| `pca64_plus_full_17+18+19+20+21+22+23+24` | 0.322 | 0.945 | 0.328 | 0.436 | 0.965 |
| `full_12_24` | 0.347 | 0.936 | 0.211 | 0.445 | 0.961 |
