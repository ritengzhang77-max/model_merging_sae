# Gemma-2-2B Low-Dimensional Activation Patch Generation

Greedy generation check for compressed approximations to the successful MLP activation patch over layers `12-20`.

| model | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|---:|
| `pca_rank128` | 0.750 | 0.750 | 0.000 | 0.000 | 1.000 | 0.000 |
| `top_neuron_k1024` | 0.750 | 0.750 | 0.000 | 0.000 | 1.000 | 0.000 |
| `top_neuron_k1536` | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |

## Current Decision

- These are the honest low-dimensional baselines that GemmaScope SAE/transcoder features must beat or explain.
- A compressed patch is useful only if it restores harmful refusal without damaging benign helpfulness.
