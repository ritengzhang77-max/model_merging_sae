# Qwen2.5-1.5B Low-Dimensional Activation Patch Baselines

Teacher-forced refusal-target loss for lower-dimensional approximations to the successful `12-24:mlp` activation patch.

| variant | dim per layer | harmful gap closed | benign gap closed | specificity | harmful loss | benign loss |
|---|---:|---:|---:|---:|---:|---:|
| `base` | full | 1.000 | 1.000 | 0.000 | 0.089 | 0.460 |
| `pca_rank64` | 64 | 0.945 | 0.927 | 0.018 | 0.206 | 0.648 |
| `pca_rank256` | 256 | 0.942 | 0.941 | 0.001 | 0.213 | 0.612 |
| `full_12_24_mlp` | full | 0.928 | 0.956 | -0.028 | 0.243 | 0.574 |
| `pca_rank128` | 128 | 0.915 | 0.924 | -0.009 | 0.269 | 0.656 |
| `mean_delta_rank1` | 1 | 0.853 | 0.835 | 0.019 | 0.401 | 0.887 |
| `random_rank256` | 256 | 0.487 | 0.495 | -0.008 | 1.180 | 1.764 |
| `refusal_direction_rank1` | 1 | 0.371 | 0.140 | 0.231 | 1.425 | 2.681 |
| `random_rank128` | 128 | 0.327 | 0.228 | 0.099 | 1.518 | 2.453 |
| `abliterated` | 0 | 0.000 | 0.000 | 0.000 | 2.213 | 3.042 |

## Current Decision

- Best compressed baseline by harmful gap closure: `pca_rank64` with `0.945` harmful gap closed.
- Compare these baselines to the full MLP patch before starting SAE/transcoder training.
