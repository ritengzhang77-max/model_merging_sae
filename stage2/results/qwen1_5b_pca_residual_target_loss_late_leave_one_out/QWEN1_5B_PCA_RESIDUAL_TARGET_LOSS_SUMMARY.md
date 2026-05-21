# Qwen2.5-1.5B PCA64 Residual Target-Loss Diagnostic

Prompt mode: `heldout_failures`

Teacher-forced refusal-target loss with PCA64 on all layers and selected layers upgraded to full donor MLP activations.

| variant | harmful loss | harmful gap closed vs ablated | gap closed vs PCA64 | benign loss | benign gap closed |
|---|---:|---:|---:|---:|---:|
| `base` | 0.176 | 1.000 | 0.000 | 0.340 | 1.000 |
| `abliterated` | 2.834 | 0.000 | 0.000 | 3.049 | 0.000 |
| `pca64` | 0.392 | 0.919 | 0.000 | 0.575 | 0.913 |
| `pca64_plus_full_20+21+22+23+24` | 0.303 | 0.952 | 0.412 | 0.428 | 0.967 |
| `pca64_plus_full_21+22+23+24` | 0.306 | 0.951 | 0.399 | 0.464 | 0.954 |
| `pca64_plus_full_20+22+23+24` | 0.321 | 0.946 | 0.332 | 0.451 | 0.959 |
| `pca64_plus_full_20+21+23+24` | 0.329 | 0.943 | 0.296 | 0.486 | 0.946 |
| `pca64_plus_full_20+21+22+24` | 0.316 | 0.947 | 0.353 | 0.424 | 0.969 |
| `pca64_plus_full_20+21+22+23` | 0.307 | 0.951 | 0.395 | 0.449 | 0.960 |
| `full_12_24` | 0.347 | 0.936 | 0.211 | 0.445 | 0.961 |
