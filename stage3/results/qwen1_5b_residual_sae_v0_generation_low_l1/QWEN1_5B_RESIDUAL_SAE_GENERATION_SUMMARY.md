# Qwen2.5-1.5B Residual SAE Generation

Dynamic generation check for learned sparse autoencoder reconstructions of the post-PCA64 residual.

| model | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|---:|
| `pca64` | 0.000 | 0.333 | 0.333 | 0.333 | 1.000 | 0.000 |
| `residual_topk1344_16-23` | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `full_16-23` | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `residual_sae_d1536_l1_0.0001_16-23` | 0.333 | 0.333 | 0.000 | 0.333 | 1.000 | 0.000 |
| `residual_sae_d2048_l1_0.0001_16-23` | 0.333 | 0.333 | 0.000 | 0.667 | 1.000 | 0.000 |

## SAE Training Metrics

- `d1536_l1_0.0001`: mean EV `0.995`, mean L0 `595.6`, mean MSE `0.00021`
- `d2048_l1_0.0001`: mean EV `0.996`, mean L0 `779.0`, mean MSE `0.00020`
