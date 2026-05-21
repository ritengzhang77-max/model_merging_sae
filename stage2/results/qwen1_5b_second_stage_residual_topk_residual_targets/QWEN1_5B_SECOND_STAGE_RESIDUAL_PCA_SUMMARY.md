# Qwen2.5-1.5B Second-Stage Residual PCA

Residual basis mode: `residual_targets`
Residual layers: `16+17+18+19+20+21+22+23`
Residual basis mask: `all`

Patch definition: PCA64 on layers 12-24, plus a second PCA projection of the remaining donor-recipient MLP delta on selected layers.

## Target-Loss Results

### `heldout_failures_harmful_only`

| variant | residual kind | residual rank | harmful loss | harmful gap closed | gap closed vs PCA64 | benign loss | benign gap closed |
|---|---|---:|---:|---:|---:|---:|---:|
| `base` | `base` | base | 0.176 | 1.000 | 0.000 | - | - |
| `abliterated` | `none` | 0 | 2.834 | 0.000 | 0.000 | - | - |
| `pca64` | `none` | - | 0.392 | 0.919 | 0.000 | - | - |
| `pca64_plus_residual_mean_dir_16-23` | `mean_dir` | - | 0.391 | 0.919 | 0.008 | - | - |
| `pca64_plus_residual_mean_vec_16-23` | `mean_vec` | - | 0.394 | 0.918 | -0.009 | - | - |
| `pca64_plus_residual_topk64_16-23` | `top_neuron` | 64 | 0.387 | 0.921 | 0.023 | - | - |
| `pca64_plus_residual_topk256_16-23` | `top_neuron` | 256 | 0.381 | 0.923 | 0.054 | - | - |
| `pca64_plus_residual_topk512_16-23` | `top_neuron` | 512 | 0.368 | 0.928 | 0.111 | - | - |
| `pca64_plus_residual_topk1024_16-23` | `top_neuron` | 1024 | 0.302 | 0.953 | 0.417 | - | - |
| `pca64_plus_residual_centered_pca32_16-23` | `centered_pca` | 32 | 0.390 | 0.919 | 0.009 | - | - |
| `pca64_plus_residual_centered_pca64_16-23` | `centered_pca` | 64 | 0.396 | 0.917 | -0.016 | - | - |
| `pca64_plus_residual_raw_pca32_16-23` | `raw_pca` | 32 | 0.387 | 0.921 | 0.024 | - | - |
| `pca64_plus_residual_raw_pca64_16-23` | `raw_pca` | 64 | 0.396 | 0.917 | -0.015 | - | - |
| `pca64_plus_full_16-23` | `none` | - | 0.328 | 0.943 | 0.297 | - | - |
| `pca64_plus_full_12-24` | `none` | - | 0.347 | 0.936 | 0.211 | - | - |

### `stress_harmful_only`

| variant | residual kind | residual rank | harmful loss | harmful gap closed | gap closed vs PCA64 | benign loss | benign gap closed |
|---|---|---:|---:|---:|---:|---:|---:|
| `base` | `base` | base | 0.126 | 1.000 | 0.000 | - | - |
| `abliterated` | `none` | 0 | 2.685 | 0.000 | 0.000 | - | - |
| `pca64` | `none` | - | 0.267 | 0.945 | 0.000 | - | - |
| `pca64_plus_residual_mean_dir_16-23` | `mean_dir` | - | 0.260 | 0.947 | 0.045 | - | - |
| `pca64_plus_residual_mean_vec_16-23` | `mean_vec` | - | 0.258 | 0.948 | 0.061 | - | - |
| `pca64_plus_residual_topk64_16-23` | `top_neuron` | 64 | 0.270 | 0.943 | -0.027 | - | - |
| `pca64_plus_residual_topk256_16-23` | `top_neuron` | 256 | 0.263 | 0.946 | 0.029 | - | - |
| `pca64_plus_residual_topk512_16-23` | `top_neuron` | 512 | 0.267 | 0.945 | -0.004 | - | - |
| `pca64_plus_residual_topk1024_16-23` | `top_neuron` | 1024 | 0.249 | 0.952 | 0.127 | - | - |
| `pca64_plus_residual_centered_pca32_16-23` | `centered_pca` | 32 | 0.262 | 0.947 | 0.032 | - | - |
| `pca64_plus_residual_centered_pca64_16-23` | `centered_pca` | 64 | 0.275 | 0.941 | -0.061 | - | - |
| `pca64_plus_residual_raw_pca32_16-23` | `raw_pca` | 32 | 0.256 | 0.949 | 0.075 | - | - |
| `pca64_plus_residual_raw_pca64_16-23` | `raw_pca` | 64 | 0.265 | 0.945 | 0.009 | - | - |
| `pca64_plus_full_16-23` | `none` | - | 0.261 | 0.947 | 0.044 | - | - |
| `pca64_plus_full_12-24` | `none` | - | 0.278 | 0.941 | -0.077 | - | - |

### `stress_permission_harmful_only`

| variant | residual kind | residual rank | harmful loss | harmful gap closed | gap closed vs PCA64 | benign loss | benign gap closed |
|---|---|---:|---:|---:|---:|---:|---:|
| `base` | `base` | base | 0.101 | 1.000 | 0.000 | - | - |
| `abliterated` | `none` | 0 | 2.567 | 0.000 | 0.000 | - | - |
| `pca64` | `none` | - | 0.367 | 0.892 | 0.000 | - | - |
| `pca64_plus_residual_mean_dir_16-23` | `mean_dir` | - | 0.357 | 0.896 | 0.041 | - | - |
| `pca64_plus_residual_mean_vec_16-23` | `mean_vec` | - | 0.359 | 0.895 | 0.032 | - | - |
| `pca64_plus_residual_topk64_16-23` | `top_neuron` | 64 | 0.344 | 0.901 | 0.088 | - | - |
| `pca64_plus_residual_topk256_16-23` | `top_neuron` | 256 | 0.292 | 0.923 | 0.283 | - | - |
| `pca64_plus_residual_topk512_16-23` | `top_neuron` | 512 | 0.334 | 0.906 | 0.127 | - | - |
| `pca64_plus_residual_topk1024_16-23` | `top_neuron` | 1024 | 0.334 | 0.906 | 0.125 | - | - |
| `pca64_plus_residual_centered_pca32_16-23` | `centered_pca` | 32 | 0.342 | 0.902 | 0.097 | - | - |
| `pca64_plus_residual_centered_pca64_16-23` | `centered_pca` | 64 | 0.339 | 0.903 | 0.106 | - | - |
| `pca64_plus_residual_raw_pca32_16-23` | `raw_pca` | 32 | 0.335 | 0.905 | 0.121 | - | - |
| `pca64_plus_residual_raw_pca64_16-23` | `raw_pca` | 64 | 0.339 | 0.904 | 0.107 | - | - |
| `pca64_plus_full_16-23` | `none` | - | 0.309 | 0.916 | 0.221 | - | - |
| `pca64_plus_full_12-24` | `none` | - | 0.286 | 0.925 | 0.307 | - | - |

## Residual Basis Explained Energy

- `centered_pca` rank `32`: mean residual-basis energy explained across layers = `0.726`
- `centered_pca` rank `64`: mean residual-basis energy explained across layers = `0.934`
- `raw_pca` rank `32`: mean residual-basis energy explained across layers = `0.753`
- `raw_pca` rank `64`: mean residual-basis energy explained across layers = `0.940`
