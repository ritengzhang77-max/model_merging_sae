# Qwen2.5-1.5B Residual SAE Generation

Dynamic generation check for learned sparse autoencoder reconstructions of the post-PCA64 residual.

| model | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|---:|
| `pca64` | 0.000 | 0.333 | 0.333 | 0.333 | 1.000 | 0.000 |
| `residual_topk1344_16-23` | 0.667 | 0.667 | 0.000 | 0.333 | 1.000 | 0.000 |
| `full_16-23` | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `residual_sae_d512_l1_0.0001_16-23` | 0.333 | 0.333 | 0.000 | 0.333 | 1.000 | 0.000 |

## SAE Training Metrics

- `d512_l1_0.0001`: mean EV `0.998`, mean L0 `206.2`, mean MSE `0.00023`
