# Qwen2.5-1.5B Second-Stage Residual PCA

Residual basis mode: `permission`
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
| `pca64_plus_residual_mean_dir_16-23` | `mean_dir` | - | 0.378 | 0.924 | 0.067 | - | - |
| `pca64_plus_residual_mean_vec_16-23` | `mean_vec` | - | 0.370 | 0.927 | 0.104 | - | - |
| `pca64_plus_residual_topk64_16-23` | `top_neuron` | 64 | 0.387 | 0.921 | 0.026 | - | - |
| `pca64_plus_residual_topk256_16-23` | `top_neuron` | 256 | 0.363 | 0.930 | 0.137 | - | - |
| `pca64_plus_residual_topk512_16-23` | `top_neuron` | 512 | 0.372 | 0.926 | 0.093 | - | - |
| `pca64_plus_residual_topk1024_16-23` | `top_neuron` | 1024 | 0.307 | 0.951 | 0.393 | - | - |
| `pca64_plus_residual_centered_pca32_16-23` | `centered_pca` | 32 | 0.373 | 0.926 | 0.090 | - | - |
| `pca64_plus_residual_raw_pca32_16-23` | `raw_pca` | 32 | 0.360 | 0.931 | 0.151 | - | - |
| `pca64_plus_full_16-23` | `none` | - | 0.328 | 0.943 | 0.297 | - | - |
| `pca64_plus_full_12-24` | `none` | - | 0.347 | 0.936 | 0.211 | - | - |

### `stress_harmful_only`

| variant | residual kind | residual rank | harmful loss | harmful gap closed | gap closed vs PCA64 | benign loss | benign gap closed |
|---|---|---:|---:|---:|---:|---:|---:|
| `base` | `base` | base | 0.126 | 1.000 | 0.000 | - | - |
| `abliterated` | `none` | 0 | 2.685 | 0.000 | 0.000 | - | - |
| `pca64` | `none` | - | 0.267 | 0.945 | 0.000 | - | - |
| `pca64_plus_residual_mean_dir_16-23` | `mean_dir` | - | 0.255 | 0.949 | 0.080 | - | - |
| `pca64_plus_residual_mean_vec_16-23` | `mean_vec` | - | 0.247 | 0.953 | 0.143 | - | - |
| `pca64_plus_residual_topk64_16-23` | `top_neuron` | 64 | 0.263 | 0.946 | 0.025 | - | - |
| `pca64_plus_residual_topk256_16-23` | `top_neuron` | 256 | 0.257 | 0.949 | 0.070 | - | - |
| `pca64_plus_residual_topk512_16-23` | `top_neuron` | 512 | 0.278 | 0.940 | -0.083 | - | - |
| `pca64_plus_residual_topk1024_16-23` | `top_neuron` | 1024 | 0.246 | 0.953 | 0.145 | - | - |
| `pca64_plus_residual_centered_pca32_16-23` | `centered_pca` | 32 | 0.260 | 0.948 | 0.049 | - | - |
| `pca64_plus_residual_raw_pca32_16-23` | `raw_pca` | 32 | 0.250 | 0.951 | 0.120 | - | - |
| `pca64_plus_full_16-23` | `none` | - | 0.261 | 0.947 | 0.044 | - | - |
| `pca64_plus_full_12-24` | `none` | - | 0.278 | 0.941 | -0.077 | - | - |

### `stress_permission_harmful_only`

| variant | residual kind | residual rank | harmful loss | harmful gap closed | gap closed vs PCA64 | benign loss | benign gap closed |
|---|---|---:|---:|---:|---:|---:|---:|
| `base` | `base` | base | 0.101 | 1.000 | 0.000 | - | - |
| `abliterated` | `none` | 0 | 2.567 | 0.000 | 0.000 | - | - |
| `pca64` | `none` | - | 0.367 | 0.892 | 0.000 | - | - |
| `pca64_plus_residual_mean_dir_16-23` | `mean_dir` | - | 0.349 | 0.899 | 0.068 | - | - |
| `pca64_plus_residual_mean_vec_16-23` | `mean_vec` | - | 0.346 | 0.901 | 0.081 | - | - |
| `pca64_plus_residual_topk64_16-23` | `top_neuron` | 64 | 0.341 | 0.903 | 0.098 | - | - |
| `pca64_plus_residual_topk256_16-23` | `top_neuron` | 256 | 0.302 | 0.919 | 0.246 | - | - |
| `pca64_plus_residual_topk512_16-23` | `top_neuron` | 512 | 0.350 | 0.899 | 0.065 | - | - |
| `pca64_plus_residual_topk1024_16-23` | `top_neuron` | 1024 | 0.331 | 0.907 | 0.135 | - | - |
| `pca64_plus_residual_centered_pca32_16-23` | `centered_pca` | 32 | 0.348 | 0.900 | 0.072 | - | - |
| `pca64_plus_residual_raw_pca32_16-23` | `raw_pca` | 32 | 0.331 | 0.907 | 0.138 | - | - |
| `pca64_plus_full_16-23` | `none` | - | 0.309 | 0.916 | 0.221 | - | - |
| `pca64_plus_full_12-24` | `none` | - | 0.286 | 0.925 | 0.307 | - | - |

## Residual Basis Explained Energy

- `centered_pca` rank `32`: mean residual-basis energy explained across layers = `0.998`
- `raw_pca` rank `32`: mean residual-basis energy explained across layers = `0.998`
