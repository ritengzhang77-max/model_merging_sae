# Gemma-2-2B Linear Weight-Merge Sweep

Merge line: `abliterated + alpha * (base - abliterated)`.
Alphas: `0,0.25,0.5,0.75,1`.
Evaluation prompts: stage3/data/gemma2_feature16048_family_prompts/fake_id_family_v0.jsonl.

## Generation

| model | alpha | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|---:|---:|
| `linear_alpha_0` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `linear_alpha_0.25` | 0.250 | 0.125 | 0.125 | 0.000 | 0.375 | 1.000 | 0.000 |
| `linear_alpha_0.5` | 0.500 | 0.750 | 0.750 | 0.000 | 0.000 | 1.000 | 0.000 |
| `linear_alpha_0.75` | 0.750 | 0.875 | 0.875 | 0.000 | 0.000 | 1.000 | 0.000 |
| `linear_alpha_1` | 1.000 | 0.875 | 0.875 | 0.000 | 0.000 | 0.875 | 0.125 |

## Interpretation Handle

- This is an actual parameter-space model merge, not an activation patch.
- The sweep is useful if behavior changes sharply at intermediate alpha values; those alpha points can then be compared against SAE feature trajectories.
