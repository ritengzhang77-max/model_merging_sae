# Gemma-2-2B Low-Dimensional Activation Patch Generation

Greedy generation check for compressed approximations to the successful MLP activation patch over layers `12-20`.

| model | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|---:|
| `base` | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `abliterated` | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `full_mlp` | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `mean_delta_rank1` | 0.000 | 0.000 | 0.000 | 0.250 | 1.000 | 0.000 |
| `pca_rank16` | 0.500 | 0.750 | 0.250 | 0.250 | 1.000 | 0.000 |
| `pca_rank64` | 0.750 | 0.750 | 0.000 | 0.000 | 1.000 | 0.000 |
| `top_neuron_k256` | 0.000 | 0.000 | 0.000 | 0.500 | 1.000 | 0.000 |
| `top_neuron_k512` | 0.500 | 0.500 | 0.000 | 0.000 | 1.000 | 0.000 |
| `random_rank64` | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |

## Current Decision

- These are the honest low-dimensional baselines that GemmaScope SAE/transcoder features must beat or explain.
- A compressed patch is useful only if it restores harmful refusal without damaging benign helpfulness.
