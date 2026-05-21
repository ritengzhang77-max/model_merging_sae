# Qwen2.5-1.5B Low-Dimensional Activation Patch Baselines

Teacher-forced refusal-target loss for lower-dimensional approximations to the successful `12-24:mlp` activation patch.

| variant | dim per layer | harmful gap closed | benign gap closed | specificity | harmful loss | benign loss |
|---|---:|---:|---:|---:|---:|---:|
| `base` | full | 1.000 | 1.000 | 0.000 | 0.089 | 0.460 |
| `full_12_24_mlp` | full | 0.928 | 0.956 | -0.028 | 0.243 | 0.574 |
| `mean_delta_rank1` | 1 | 0.853 | 0.835 | 0.019 | 0.401 | 0.887 |
| `top_neuron_k256` | 256 | 0.844 | 0.798 | 0.045 | 0.421 | 0.981 |
| `top_neuron_k64` | 64 | 0.610 | 0.558 | 0.052 | 0.918 | 1.602 |
| `refusal_direction_rank1` | 1 | 0.371 | 0.140 | 0.231 | 1.425 | 2.681 |
| `top_neuron_k16` | 16 | 0.343 | 0.358 | -0.015 | 1.484 | 2.118 |
| `abliterated` | 0 | 0.000 | 0.000 | 0.000 | 2.213 | 3.042 |

## Current Decision

- Best compressed baseline by harmful gap closure: `mean_delta_rank1` with `0.853` harmful gap closed.
- Compare these baselines to the full MLP patch before starting SAE/transcoder training.
