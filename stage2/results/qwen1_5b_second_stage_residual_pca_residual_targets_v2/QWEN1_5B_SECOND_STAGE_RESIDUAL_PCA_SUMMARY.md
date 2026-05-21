# Qwen2.5-1.5B Second-Stage Residual PCA

Residual basis mode: `residual_targets`
Residual layers: `16+17+18+19+20+21+22+23`
Residual basis mask: `all`

Patch definition: PCA64 on layers 12-24, plus a second PCA projection of the remaining donor-recipient MLP delta on selected layers.

## Target-Loss Results

### `heldout_all`

| variant | residual kind | residual rank | harmful loss | harmful gap closed | gap closed vs PCA64 | benign loss | benign gap closed |
|---|---|---:|---:|---:|---:|---:|---:|
| `base` | `base` | base | 0.090 | 1.000 | 0.000 | 0.379 | 1.000 |
| `abliterated` | `none` | 0 | 2.481 | 0.000 | 0.000 | 3.078 | 0.000 |
| `pca64` | `none` | - | 0.215 | 0.948 | 0.000 | 0.595 | 0.920 |
| `pca64_plus_residual_mean_dir_16-23` | `mean_dir` | - | 0.213 | 0.949 | 0.021 | 0.583 | 0.924 |
| `pca64_plus_residual_mean_vec_16-23` | `mean_vec` | - | 0.214 | 0.948 | 0.008 | 0.563 | 0.932 |
| `pca64_plus_residual_centered_pca1_16-23` | `centered_pca` | 1 | 0.221 | 0.945 | -0.044 | 0.599 | 0.919 |
| `pca64_plus_residual_centered_pca2_16-23` | `centered_pca` | 2 | 0.224 | 0.944 | -0.066 | 0.602 | 0.917 |
| `pca64_plus_residual_centered_pca4_16-23` | `centered_pca` | 4 | 0.222 | 0.945 | -0.053 | 0.594 | 0.920 |
| `pca64_plus_residual_centered_pca8_16-23` | `centered_pca` | 8 | 0.225 | 0.944 | -0.078 | 0.609 | 0.915 |
| `pca64_plus_residual_centered_pca16_16-23` | `centered_pca` | 16 | 0.225 | 0.944 | -0.078 | 0.593 | 0.921 |
| `pca64_plus_residual_centered_pca32_16-23` | `centered_pca` | 32 | 0.214 | 0.948 | 0.012 | 0.586 | 0.923 |
| `pca64_plus_residual_centered_pca64_16-23` | `centered_pca` | 64 | 0.230 | 0.942 | -0.113 | 0.607 | 0.916 |
| `pca64_plus_residual_raw_pca1_16-23` | `raw_pca` | 1 | 0.213 | 0.949 | 0.017 | 0.584 | 0.924 |
| `pca64_plus_residual_raw_pca2_16-23` | `raw_pca` | 2 | 0.216 | 0.947 | -0.003 | 0.582 | 0.925 |
| `pca64_plus_residual_raw_pca4_16-23` | `raw_pca` | 4 | 0.224 | 0.944 | -0.066 | 0.590 | 0.922 |
| `pca64_plus_residual_raw_pca8_16-23` | `raw_pca` | 8 | 0.227 | 0.943 | -0.092 | 0.598 | 0.919 |
| `pca64_plus_residual_raw_pca16_16-23` | `raw_pca` | 16 | 0.226 | 0.943 | -0.086 | 0.588 | 0.922 |
| `pca64_plus_residual_raw_pca32_16-23` | `raw_pca` | 32 | 0.210 | 0.950 | 0.042 | 0.578 | 0.926 |
| `pca64_plus_residual_raw_pca64_16-23` | `raw_pca` | 64 | 0.219 | 0.946 | -0.026 | 0.602 | 0.917 |
| `pca64_plus_full_16-23` | `none` | - | 0.215 | 0.948 | 0.005 | 0.492 | 0.958 |
| `pca64_plus_full_12-24` | `none` | - | 0.239 | 0.938 | -0.186 | 0.495 | 0.957 |

### `heldout_failures_harmful_only`

| variant | residual kind | residual rank | harmful loss | harmful gap closed | gap closed vs PCA64 | benign loss | benign gap closed |
|---|---|---:|---:|---:|---:|---:|---:|
| `base` | `base` | base | 0.176 | 1.000 | 0.000 | - | - |
| `abliterated` | `none` | 0 | 2.834 | 0.000 | 0.000 | - | - |
| `pca64` | `none` | - | 0.392 | 0.919 | 0.000 | - | - |
| `pca64_plus_residual_mean_dir_16-23` | `mean_dir` | - | 0.391 | 0.919 | 0.008 | - | - |
| `pca64_plus_residual_mean_vec_16-23` | `mean_vec` | - | 0.394 | 0.918 | -0.009 | - | - |
| `pca64_plus_residual_centered_pca1_16-23` | `centered_pca` | 1 | 0.392 | 0.919 | 0.001 | - | - |
| `pca64_plus_residual_centered_pca2_16-23` | `centered_pca` | 2 | 0.400 | 0.916 | -0.035 | - | - |
| `pca64_plus_residual_centered_pca4_16-23` | `centered_pca` | 4 | 0.392 | 0.919 | 0.001 | - | - |
| `pca64_plus_residual_centered_pca8_16-23` | `centered_pca` | 8 | 0.407 | 0.913 | -0.067 | - | - |
| `pca64_plus_residual_centered_pca16_16-23` | `centered_pca` | 16 | 0.401 | 0.916 | -0.038 | - | - |
| `pca64_plus_residual_centered_pca32_16-23` | `centered_pca` | 32 | 0.390 | 0.919 | 0.009 | - | - |
| `pca64_plus_residual_centered_pca64_16-23` | `centered_pca` | 64 | 0.396 | 0.917 | -0.016 | - | - |
| `pca64_plus_residual_raw_pca1_16-23` | `raw_pca` | 1 | 0.393 | 0.918 | -0.004 | - | - |
| `pca64_plus_residual_raw_pca2_16-23` | `raw_pca` | 2 | 0.386 | 0.921 | 0.030 | - | - |
| `pca64_plus_residual_raw_pca4_16-23` | `raw_pca` | 4 | 0.398 | 0.917 | -0.024 | - | - |
| `pca64_plus_residual_raw_pca8_16-23` | `raw_pca` | 8 | 0.407 | 0.913 | -0.069 | - | - |
| `pca64_plus_residual_raw_pca16_16-23` | `raw_pca` | 16 | 0.406 | 0.914 | -0.063 | - | - |
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
| `pca64_plus_residual_centered_pca1_16-23` | `centered_pca` | 1 | 0.273 | 0.942 | -0.048 | - | - |
| `pca64_plus_residual_centered_pca2_16-23` | `centered_pca` | 2 | 0.275 | 0.942 | -0.059 | - | - |
| `pca64_plus_residual_centered_pca4_16-23` | `centered_pca` | 4 | 0.273 | 0.942 | -0.046 | - | - |
| `pca64_plus_residual_centered_pca8_16-23` | `centered_pca` | 8 | 0.278 | 0.940 | -0.080 | - | - |
| `pca64_plus_residual_centered_pca16_16-23` | `centered_pca` | 16 | 0.275 | 0.942 | -0.056 | - | - |
| `pca64_plus_residual_centered_pca32_16-23` | `centered_pca` | 32 | 0.262 | 0.947 | 0.032 | - | - |
| `pca64_plus_residual_centered_pca64_16-23` | `centered_pca` | 64 | 0.275 | 0.941 | -0.061 | - | - |
| `pca64_plus_residual_raw_pca1_16-23` | `raw_pca` | 1 | 0.261 | 0.947 | 0.040 | - | - |
| `pca64_plus_residual_raw_pca2_16-23` | `raw_pca` | 2 | 0.265 | 0.946 | 0.012 | - | - |
| `pca64_plus_residual_raw_pca4_16-23` | `raw_pca` | 4 | 0.270 | 0.943 | -0.025 | - | - |
| `pca64_plus_residual_raw_pca8_16-23` | `raw_pca` | 8 | 0.275 | 0.942 | -0.060 | - | - |
| `pca64_plus_residual_raw_pca16_16-23` | `raw_pca` | 16 | 0.273 | 0.942 | -0.045 | - | - |
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
| `pca64_plus_residual_centered_pca1_16-23` | `centered_pca` | 1 | 0.378 | 0.888 | -0.039 | - | - |
| `pca64_plus_residual_centered_pca2_16-23` | `centered_pca` | 2 | 0.376 | 0.889 | -0.032 | - | - |
| `pca64_plus_residual_centered_pca4_16-23` | `centered_pca` | 4 | 0.369 | 0.891 | -0.007 | - | - |
| `pca64_plus_residual_centered_pca8_16-23` | `centered_pca` | 8 | 0.363 | 0.894 | 0.017 | - | - |
| `pca64_plus_residual_centered_pca16_16-23` | `centered_pca` | 16 | 0.349 | 0.899 | 0.068 | - | - |
| `pca64_plus_residual_centered_pca32_16-23` | `centered_pca` | 32 | 0.342 | 0.902 | 0.097 | - | - |
| `pca64_plus_residual_centered_pca64_16-23` | `centered_pca` | 64 | 0.339 | 0.903 | 0.106 | - | - |
| `pca64_plus_residual_raw_pca1_16-23` | `raw_pca` | 1 | 0.355 | 0.897 | 0.045 | - | - |
| `pca64_plus_residual_raw_pca2_16-23` | `raw_pca` | 2 | 0.367 | 0.892 | 0.002 | - | - |
| `pca64_plus_residual_raw_pca4_16-23` | `raw_pca` | 4 | 0.364 | 0.893 | 0.011 | - | - |
| `pca64_plus_residual_raw_pca8_16-23` | `raw_pca` | 8 | 0.362 | 0.894 | 0.019 | - | - |
| `pca64_plus_residual_raw_pca16_16-23` | `raw_pca` | 16 | 0.346 | 0.901 | 0.079 | - | - |
| `pca64_plus_residual_raw_pca32_16-23` | `raw_pca` | 32 | 0.335 | 0.905 | 0.121 | - | - |
| `pca64_plus_residual_raw_pca64_16-23` | `raw_pca` | 64 | 0.339 | 0.904 | 0.107 | - | - |
| `pca64_plus_full_16-23` | `none` | - | 0.309 | 0.916 | 0.221 | - | - |
| `pca64_plus_full_12-24` | `none` | - | 0.286 | 0.925 | 0.307 | - | - |

## Residual Basis Explained Energy

- `centered_pca` rank `1`: mean residual-basis energy explained across layers = `0.074`
- `centered_pca` rank `2`: mean residual-basis energy explained across layers = `0.128`
- `centered_pca` rank `4`: mean residual-basis energy explained across layers = `0.210`
- `centered_pca` rank `8`: mean residual-basis energy explained across layers = `0.330`
- `centered_pca` rank `16`: mean residual-basis energy explained across layers = `0.504`
- `centered_pca` rank `32`: mean residual-basis energy explained across layers = `0.726`
- `centered_pca` rank `64`: mean residual-basis energy explained across layers = `0.934`
- `raw_pca` rank `1`: mean residual-basis energy explained across layers = `0.130`
- `raw_pca` rank `2`: mean residual-basis energy explained across layers = `0.191`
- `raw_pca` rank `4`: mean residual-basis energy explained across layers = `0.278`
- `raw_pca` rank `8`: mean residual-basis energy explained across layers = `0.391`
- `raw_pca` rank `16`: mean residual-basis energy explained across layers = `0.551`
- `raw_pca` rank `32`: mean residual-basis energy explained across layers = `0.753`
- `raw_pca` rank `64`: mean residual-basis energy explained across layers = `0.940`
