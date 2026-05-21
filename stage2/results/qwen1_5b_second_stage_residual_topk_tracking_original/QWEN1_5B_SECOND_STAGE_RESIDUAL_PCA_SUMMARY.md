# Qwen2.5-1.5B Second-Stage Residual PCA

Residual basis mode: `tracking_original`
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
| `pca64_plus_residual_mean_dir_16-23` | `mean_dir` | - | 0.392 | 0.919 | 0.002 | - | - |
| `pca64_plus_residual_mean_vec_16-23` | `mean_vec` | - | 0.395 | 0.918 | -0.011 | - | - |
| `pca64_plus_residual_topk64_16-23` | `top_neuron` | 64 | 0.374 | 0.926 | 0.087 | - | - |
| `pca64_plus_residual_topk256_16-23` | `top_neuron` | 256 | 0.386 | 0.921 | 0.028 | - | - |
| `pca64_plus_residual_topk512_16-23` | `top_neuron` | 512 | 0.363 | 0.930 | 0.134 | - | - |
| `pca64_plus_residual_topk1024_16-23` | `top_neuron` | 1024 | 0.339 | 0.939 | 0.247 | - | - |
| `pca64_plus_residual_centered_pca32_16-23` | `centered_pca` | 32 | 0.384 | 0.922 | 0.040 | - | - |
| `pca64_plus_residual_raw_pca32_16-23` | `raw_pca` | 32 | 0.380 | 0.923 | 0.057 | - | - |
| `pca64_plus_full_16-23` | `none` | - | 0.328 | 0.943 | 0.297 | - | - |
| `pca64_plus_full_12-24` | `none` | - | 0.347 | 0.936 | 0.211 | - | - |

### `stress_harmful_only`

| variant | residual kind | residual rank | harmful loss | harmful gap closed | gap closed vs PCA64 | benign loss | benign gap closed |
|---|---|---:|---:|---:|---:|---:|---:|
| `base` | `base` | base | 0.126 | 1.000 | 0.000 | - | - |
| `abliterated` | `none` | 0 | 2.685 | 0.000 | 0.000 | - | - |
| `pca64` | `none` | - | 0.267 | 0.945 | 0.000 | - | - |
| `pca64_plus_residual_mean_dir_16-23` | `mean_dir` | - | 0.265 | 0.945 | 0.010 | - | - |
| `pca64_plus_residual_mean_vec_16-23` | `mean_vec` | - | 0.261 | 0.947 | 0.041 | - | - |
| `pca64_plus_residual_topk64_16-23` | `top_neuron` | 64 | 0.255 | 0.949 | 0.080 | - | - |
| `pca64_plus_residual_topk256_16-23` | `top_neuron` | 256 | 0.267 | 0.945 | 0.001 | - | - |
| `pca64_plus_residual_topk512_16-23` | `top_neuron` | 512 | 0.259 | 0.948 | 0.055 | - | - |
| `pca64_plus_residual_topk1024_16-23` | `top_neuron` | 1024 | 0.262 | 0.947 | 0.032 | - | - |
| `pca64_plus_residual_centered_pca32_16-23` | `centered_pca` | 32 | 0.259 | 0.948 | 0.058 | - | - |
| `pca64_plus_residual_raw_pca32_16-23` | `raw_pca` | 32 | 0.253 | 0.950 | 0.095 | - | - |
| `pca64_plus_full_16-23` | `none` | - | 0.261 | 0.947 | 0.044 | - | - |
| `pca64_plus_full_12-24` | `none` | - | 0.278 | 0.941 | -0.077 | - | - |

### `stress_permission_harmful_only`

| variant | residual kind | residual rank | harmful loss | harmful gap closed | gap closed vs PCA64 | benign loss | benign gap closed |
|---|---|---:|---:|---:|---:|---:|---:|
| `base` | `base` | base | 0.101 | 1.000 | 0.000 | - | - |
| `abliterated` | `none` | 0 | 2.567 | 0.000 | 0.000 | - | - |
| `pca64` | `none` | - | 0.367 | 0.892 | 0.000 | - | - |
| `pca64_plus_residual_mean_dir_16-23` | `mean_dir` | - | 0.366 | 0.893 | 0.006 | - | - |
| `pca64_plus_residual_mean_vec_16-23` | `mean_vec` | - | 0.355 | 0.897 | 0.049 | - | - |
| `pca64_plus_residual_topk64_16-23` | `top_neuron` | 64 | 0.315 | 0.913 | 0.199 | - | - |
| `pca64_plus_residual_topk256_16-23` | `top_neuron` | 256 | 0.307 | 0.916 | 0.226 | - | - |
| `pca64_plus_residual_topk512_16-23` | `top_neuron` | 512 | 0.331 | 0.907 | 0.139 | - | - |
| `pca64_plus_residual_topk1024_16-23` | `top_neuron` | 1024 | 0.325 | 0.909 | 0.160 | - | - |
| `pca64_plus_residual_centered_pca32_16-23` | `centered_pca` | 32 | 0.334 | 0.906 | 0.125 | - | - |
| `pca64_plus_residual_raw_pca32_16-23` | `raw_pca` | 32 | 0.334 | 0.906 | 0.125 | - | - |
| `pca64_plus_full_16-23` | `none` | - | 0.309 | 0.916 | 0.221 | - | - |
| `pca64_plus_full_12-24` | `none` | - | 0.286 | 0.925 | 0.307 | - | - |

## Residual Basis Explained Energy

- `centered_pca` rank `32`: mean residual-basis energy explained across layers = `0.995`
- `raw_pca` rank `32`: mean residual-basis energy explained across layers = `0.995`
