# Qwen2.5-1.5B Low-Dimensional Activation Patch Generation

Greedy generation check for compressed approximations to the successful `12-24:mlp` activation patch.

| model | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|---:|
| `pca_rank64` | 0.750 | 0.917 | 0.167 | 0.250 | 1.000 | 0.000 |

## Current Decision

- Target-loss repair is not enough; generation checks decide which compressed baselines deserve SAE comparison.
- A compressed patch is interesting only if it restores harmful refusals without increasing benign over-refusal.
