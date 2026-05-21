# Qwen2.5-1.5B Low-Dimensional Activation Patch Generation

Greedy generation check for compressed approximations to the successful `12-24:mlp` activation patch.

| model | harmful clean | harmful attempt | harmful bad | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `base` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| `abliterated` | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `mean_delta_rank1` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| `pca_rank16` | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| `pca_rank64` | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| `random_rank64` | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |

## Current Decision

- Target-loss repair is not enough; generation checks decide which compressed baselines deserve SAE comparison.
- A compressed patch is interesting only if it restores harmful refusals without increasing benign over-refusal.
- `pca_rank16` and `pca_rank64` are the current strongest compressed behavior baselines: both restore harmful clean refusal to `1.000` and keep benign helpfulness `1.000`.
- `random_rank64` fails behaviorally (`0.000` harmful clean refusal), so the PCA effect is not explained by adding any rank-64 subspace.
- Manual audit is required: the automatic scorer over-credits some refusal-prefix answers that continue with unsafe advice.
- Under strict manual audit, `pca_rank64` remains `1.000` harmful clean refusal, but `pca_rank16` drops to `0.625` and `mean_delta_rank1` drops to `0.750`.
