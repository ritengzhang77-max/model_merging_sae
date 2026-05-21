# Qwen2.5-1.5B Second-Stage Residual PCA

Residual basis mode: `residual_targets`
Residual layers: `16+17+18+19+20+21+22+23`
Residual basis mask: `all`

Patch definition: PCA64 on layers 12-24, plus a second PCA projection of the remaining donor-recipient MLP delta on selected layers.

## Target-Loss Results

### `heldout_all`

| variant | residual rank | harmful loss | harmful gap closed | gap closed vs PCA64 | benign loss | benign gap closed |
|---|---:|---:|---:|---:|---:|---:|
| `base` | base | 0.090 | 1.000 | 0.000 | 0.379 | 1.000 |
| `abliterated` | 0 | 2.481 | 0.000 | 0.000 | 3.078 | 0.000 |
| `pca64` | - | 0.215 | 0.948 | 0.000 | 0.595 | 0.920 |
| `pca64_plus_residual_pca1_16-23` | 1 | 0.221 | 0.945 | -0.043 | 0.599 | 0.919 |
| `pca64_plus_residual_pca2_16-23` | 2 | 0.224 | 0.944 | -0.067 | 0.602 | 0.917 |
| `pca64_plus_residual_pca4_16-23` | 4 | 0.222 | 0.945 | -0.053 | 0.594 | 0.920 |
| `pca64_plus_residual_pca8_16-23` | 8 | 0.225 | 0.943 | -0.079 | 0.609 | 0.915 |
| `pca64_plus_residual_pca16_16-23` | 16 | 0.225 | 0.943 | -0.079 | 0.594 | 0.920 |
| `pca64_plus_residual_pca32_16-23` | 32 | 0.215 | 0.948 | -0.001 | 0.587 | 0.923 |
| `pca64_plus_residual_pca64_16-23` | 64 | 0.219 | 0.946 | -0.033 | 0.591 | 0.921 |
| `pca64_plus_full_16-23` | - | 0.215 | 0.948 | 0.005 | 0.492 | 0.958 |
| `pca64_plus_full_12-24` | - | 0.239 | 0.938 | -0.186 | 0.495 | 0.957 |

### `heldout_failures_harmful_only`

| variant | residual rank | harmful loss | harmful gap closed | gap closed vs PCA64 | benign loss | benign gap closed |
|---|---:|---:|---:|---:|---:|---:|
| `base` | base | 0.176 | 1.000 | 0.000 | - | - |
| `abliterated` | 0 | 2.834 | 0.000 | 0.000 | - | - |
| `pca64` | - | 0.392 | 0.919 | 0.000 | - | - |
| `pca64_plus_residual_pca1_16-23` | 1 | 0.392 | 0.919 | 0.003 | - | - |
| `pca64_plus_residual_pca2_16-23` | 2 | 0.400 | 0.916 | -0.034 | - | - |
| `pca64_plus_residual_pca4_16-23` | 4 | 0.393 | 0.918 | -0.002 | - | - |
| `pca64_plus_residual_pca8_16-23` | 8 | 0.407 | 0.913 | -0.067 | - | - |
| `pca64_plus_residual_pca16_16-23` | 16 | 0.401 | 0.916 | -0.038 | - | - |
| `pca64_plus_residual_pca32_16-23` | 32 | 0.391 | 0.919 | 0.008 | - | - |
| `pca64_plus_residual_pca64_16-23` | 64 | 0.387 | 0.921 | 0.024 | - | - |
| `pca64_plus_full_16-23` | - | 0.328 | 0.943 | 0.297 | - | - |
| `pca64_plus_full_12-24` | - | 0.347 | 0.936 | 0.211 | - | - |

### `stress_harmful_only`

| variant | residual rank | harmful loss | harmful gap closed | gap closed vs PCA64 | benign loss | benign gap closed |
|---|---:|---:|---:|---:|---:|---:|
| `base` | base | 0.126 | 1.000 | 0.000 | - | - |
| `abliterated` | 0 | 2.685 | 0.000 | 0.000 | - | - |
| `pca64` | - | 0.267 | 0.945 | 0.000 | - | - |
| `pca64_plus_residual_pca1_16-23` | 1 | 0.273 | 0.942 | -0.048 | - | - |
| `pca64_plus_residual_pca2_16-23` | 2 | 0.275 | 0.942 | -0.060 | - | - |
| `pca64_plus_residual_pca4_16-23` | 4 | 0.273 | 0.942 | -0.047 | - | - |
| `pca64_plus_residual_pca8_16-23` | 8 | 0.278 | 0.940 | -0.080 | - | - |
| `pca64_plus_residual_pca16_16-23` | 16 | 0.275 | 0.942 | -0.059 | - | - |
| `pca64_plus_residual_pca32_16-23` | 32 | 0.264 | 0.946 | 0.020 | - | - |
| `pca64_plus_residual_pca64_16-23` | 64 | 0.265 | 0.945 | 0.010 | - | - |
| `pca64_plus_full_16-23` | - | 0.261 | 0.947 | 0.044 | - | - |
| `pca64_plus_full_12-24` | - | 0.278 | 0.941 | -0.077 | - | - |

### `stress_permission_harmful_only`

| variant | residual rank | harmful loss | harmful gap closed | gap closed vs PCA64 | benign loss | benign gap closed |
|---|---:|---:|---:|---:|---:|---:|
| `base` | base | 0.101 | 1.000 | 0.000 | - | - |
| `abliterated` | 0 | 2.567 | 0.000 | 0.000 | - | - |
| `pca64` | - | 0.367 | 0.892 | 0.000 | - | - |
| `pca64_plus_residual_pca1_16-23` | 1 | 0.377 | 0.888 | -0.034 | - | - |
| `pca64_plus_residual_pca2_16-23` | 2 | 0.376 | 0.888 | -0.033 | - | - |
| `pca64_plus_residual_pca4_16-23` | 4 | 0.370 | 0.891 | -0.010 | - | - |
| `pca64_plus_residual_pca8_16-23` | 8 | 0.363 | 0.894 | 0.018 | - | - |
| `pca64_plus_residual_pca16_16-23` | 16 | 0.349 | 0.899 | 0.069 | - | - |
| `pca64_plus_residual_pca32_16-23` | 32 | 0.342 | 0.902 | 0.096 | - | - |
| `pca64_plus_residual_pca64_16-23` | 64 | 0.337 | 0.904 | 0.114 | - | - |
| `pca64_plus_full_16-23` | - | 0.309 | 0.916 | 0.221 | - | - |
| `pca64_plus_full_12-24` | - | 0.286 | 0.925 | 0.307 | - | - |

## Residual Basis Explained Energy

- rank `1`: mean residual-basis energy explained across layers = `0.074`
- rank `2`: mean residual-basis energy explained across layers = `0.128`
- rank `4`: mean residual-basis energy explained across layers = `0.210`
- rank `8`: mean residual-basis energy explained across layers = `0.330`
- rank `16`: mean residual-basis energy explained across layers = `0.504`
- rank `32`: mean residual-basis energy explained across layers = `0.726`
- rank `64`: mean residual-basis energy explained across layers = `0.934`
