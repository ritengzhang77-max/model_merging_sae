# Qwen2.5-1.5B Second-Stage Residual PCA

Residual basis mode: `one_time_code`
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
| `pca64_plus_residual_mean_dir_16-23` | `mean_dir` | - | 0.399 | 0.916 | -0.029 | - | - |
| `pca64_plus_residual_mean_vec_16-23` | `mean_vec` | - | 0.423 | 0.907 | -0.142 | - | - |
| `pca64_plus_residual_topk64_16-23` | `top_neuron` | 64 | 0.386 | 0.921 | 0.032 | - | - |
| `pca64_plus_residual_topk256_16-23` | `top_neuron` | 256 | 0.391 | 0.919 | 0.008 | - | - |
| `pca64_plus_residual_topk512_16-23` | `top_neuron` | 512 | 0.368 | 0.928 | 0.115 | - | - |
| `pca64_plus_residual_topk1024_16-23` | `top_neuron` | 1024 | 0.321 | 0.946 | 0.332 | - | - |
| `pca64_plus_residual_centered_pca32_16-23` | `centered_pca` | 32 | 0.411 | 0.912 | -0.085 | - | - |
| `pca64_plus_residual_raw_pca32_16-23` | `raw_pca` | 32 | 0.404 | 0.914 | -0.053 | - | - |
| `pca64_plus_full_16-23` | `none` | - | 0.328 | 0.943 | 0.297 | - | - |
| `pca64_plus_full_12-24` | `none` | - | 0.347 | 0.936 | 0.211 | - | - |

### `stress_harmful_only`

| variant | residual kind | residual rank | harmful loss | harmful gap closed | gap closed vs PCA64 | benign loss | benign gap closed |
|---|---|---:|---:|---:|---:|---:|---:|
| `base` | `base` | base | 0.126 | 1.000 | 0.000 | - | - |
| `abliterated` | `none` | 0 | 2.685 | 0.000 | 0.000 | - | - |
| `pca64` | `none` | - | 0.267 | 0.945 | 0.000 | - | - |
| `pca64_plus_residual_mean_dir_16-23` | `mean_dir` | - | 0.263 | 0.946 | 0.028 | - | - |
| `pca64_plus_residual_mean_vec_16-23` | `mean_vec` | - | 0.271 | 0.943 | -0.033 | - | - |
| `pca64_plus_residual_topk64_16-23` | `top_neuron` | 64 | 0.270 | 0.943 | -0.024 | - | - |
| `pca64_plus_residual_topk256_16-23` | `top_neuron` | 256 | 0.279 | 0.940 | -0.088 | - | - |
| `pca64_plus_residual_topk512_16-23` | `top_neuron` | 512 | 0.271 | 0.943 | -0.029 | - | - |
| `pca64_plus_residual_topk1024_16-23` | `top_neuron` | 1024 | 0.253 | 0.950 | 0.094 | - | - |
| `pca64_plus_residual_centered_pca32_16-23` | `centered_pca` | 32 | 0.293 | 0.935 | -0.183 | - | - |
| `pca64_plus_residual_raw_pca32_16-23` | `raw_pca` | 32 | 0.278 | 0.940 | -0.080 | - | - |
| `pca64_plus_full_16-23` | `none` | - | 0.261 | 0.947 | 0.044 | - | - |
| `pca64_plus_full_12-24` | `none` | - | 0.278 | 0.941 | -0.077 | - | - |

### `stress_permission_harmful_only`

| variant | residual kind | residual rank | harmful loss | harmful gap closed | gap closed vs PCA64 | benign loss | benign gap closed |
|---|---|---:|---:|---:|---:|---:|---:|
| `base` | `base` | base | 0.101 | 1.000 | 0.000 | - | - |
| `abliterated` | `none` | 0 | 2.567 | 0.000 | 0.000 | - | - |
| `pca64` | `none` | - | 0.367 | 0.892 | 0.000 | - | - |
| `pca64_plus_residual_mean_dir_16-23` | `mean_dir` | - | 0.361 | 0.894 | 0.023 | - | - |
| `pca64_plus_residual_mean_vec_16-23` | `mean_vec` | - | 0.377 | 0.888 | -0.035 | - | - |
| `pca64_plus_residual_topk64_16-23` | `top_neuron` | 64 | 0.361 | 0.894 | 0.023 | - | - |
| `pca64_plus_residual_topk256_16-23` | `top_neuron` | 256 | 0.356 | 0.897 | 0.044 | - | - |
| `pca64_plus_residual_topk512_16-23` | `top_neuron` | 512 | 0.322 | 0.911 | 0.173 | - | - |
| `pca64_plus_residual_topk1024_16-23` | `top_neuron` | 1024 | 0.330 | 0.907 | 0.140 | - | - |
| `pca64_plus_residual_centered_pca32_16-23` | `centered_pca` | 32 | 0.377 | 0.888 | -0.037 | - | - |
| `pca64_plus_residual_raw_pca32_16-23` | `raw_pca` | 32 | 0.368 | 0.892 | -0.001 | - | - |
| `pca64_plus_full_16-23` | `none` | - | 0.309 | 0.916 | 0.221 | - | - |
| `pca64_plus_full_12-24` | `none` | - | 0.286 | 0.925 | 0.307 | - | - |

## Residual Basis Explained Energy

- `centered_pca` rank `32`: mean residual-basis energy explained across layers = `0.993`
- `raw_pca` rank `32`: mean residual-basis energy explained across layers = `0.993`
