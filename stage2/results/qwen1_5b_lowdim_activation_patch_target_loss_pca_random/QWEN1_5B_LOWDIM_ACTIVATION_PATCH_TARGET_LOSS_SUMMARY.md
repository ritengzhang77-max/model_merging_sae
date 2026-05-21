# Qwen2.5-1.5B Low-Dimensional Activation Patch Baselines

Teacher-forced refusal-target loss for lower-dimensional approximations to the successful `12-24:mlp` activation patch.

| variant | dim per layer | harmful gap closed | benign gap closed | specificity | harmful loss | benign loss |
|---|---:|---:|---:|---:|---:|---:|
| `base` | full | 1.000 | 1.000 | 0.000 | 0.089 | 0.460 |
| `pca_rank64` | 64 | 0.955 | 0.918 | 0.037 | 0.184 | 0.672 |
| `full_12_24_mlp` | full | 0.928 | 0.956 | -0.028 | 0.243 | 0.574 |
| `pca_rank16` | 16 | 0.906 | 0.907 | -0.001 | 0.289 | 0.700 |
| `mean_delta_rank1` | 1 | 0.853 | 0.835 | 0.019 | 0.401 | 0.887 |
| `pca_rank4` | 4 | 0.845 | 0.779 | 0.066 | 0.418 | 1.030 |
| `top_neuron_k256` | 256 | 0.844 | 0.798 | 0.045 | 0.421 | 0.981 |
| `top_neuron_k64` | 64 | 0.610 | 0.558 | 0.052 | 0.918 | 1.602 |
| `pca_rank1` | 1 | 0.592 | 0.457 | 0.135 | 0.956 | 1.861 |
| `refusal_direction_rank1` | 1 | 0.371 | 0.140 | 0.231 | 1.425 | 2.681 |
| `top_neuron_k16` | 16 | 0.343 | 0.358 | -0.015 | 1.484 | 2.118 |
| `random_rank64` | 64 | 0.230 | 0.196 | 0.033 | 1.725 | 2.535 |
| `random_rank16` | 16 | 0.077 | 0.026 | 0.051 | 2.048 | 2.974 |
| `random_rank4` | 4 | 0.022 | 0.002 | 0.021 | 2.166 | 3.038 |
| `random_rank1` | 1 | 0.008 | 0.007 | 0.000 | 2.197 | 3.023 |
| `abliterated` | 0 | 0.000 | 0.000 | 0.000 | 2.213 | 3.042 |

## Current Decision

- Best compressed baseline by harmful gap closure: `pca_rank64` with `0.955` harmful gap closed.
- Compare these baselines to the full MLP patch before starting SAE/transcoder training.
