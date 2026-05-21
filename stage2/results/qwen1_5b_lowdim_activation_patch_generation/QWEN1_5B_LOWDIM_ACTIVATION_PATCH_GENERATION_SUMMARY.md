# Qwen2.5-1.5B Low-Dimensional Activation Patch Generation

Greedy generation check for compressed approximations to the successful `12-24:mlp` activation patch.

| model | harmful clean | harmful attempt | harmful bad | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `base` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| `abliterated` | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `mean_delta_rank1` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| `top_neuron_k256` | 0.625 | 0.625 | 0.000 | 0.875 | 0.000 |

## Current Decision

- Target-loss repair is not enough; generation checks decide which compressed baselines deserve SAE comparison.
- A compressed patch is interesting only if it restores harmful refusals without increasing benign over-refusal.
- `mean_delta_rank1` is the current compressed baseline to beat: it matches the base harmful clean-refusal rate on this screen and keeps benign over-refusal at `0.000`.
- `top_neuron_k256` is weaker behaviorally despite strong target-loss repair.
