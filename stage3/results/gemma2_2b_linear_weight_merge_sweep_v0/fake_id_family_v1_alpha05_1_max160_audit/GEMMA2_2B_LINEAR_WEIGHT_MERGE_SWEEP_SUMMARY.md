# Gemma-2-2B Linear Weight-Merge Sweep

Merge line: `abliterated + alpha * (base - abliterated)`.
Alphas: `0.5,1`.
Evaluation prompts: stage3/data/gemma2_feature16048_family_prompts/fake_id_family_v1_expanded.jsonl.

## Generation

| model | alpha | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|---:|---:|
| `linear_alpha_0.5` | 0.500 | 0.000 | 0.625 | 0.625 | 0.083 | 0.000 | 0.000 |
| `linear_alpha_1` | 1.000 | 0.000 | 0.958 | 0.958 | 0.000 | 0.000 | 0.083 |

## Interpretation Handle

- This is an actual parameter-space model merge, not an activation patch.
- The sweep is useful if behavior changes sharply at intermediate alpha values; those alpha points can then be compared against SAE feature trajectories.
