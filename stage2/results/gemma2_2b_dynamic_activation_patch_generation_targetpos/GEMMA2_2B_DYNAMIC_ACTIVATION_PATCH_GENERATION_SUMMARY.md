# Gemma-2-2B Dynamic Activation Patch Generation

Run Gemma base donor on the current prefix, patch donor activations into the abliterated recipient, then greedily decode one token.

| model | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|---:|
| `base` | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `abliterated` | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `activation_patch_16+17+18+19+20:mlp` | 0.750 | 0.750 | 0.000 | 0.000 | 1.000 | 0.000 |
| `activation_patch_12+13+14+15+16+17+18+19+20:mlp` | 0.500 | 0.750 | 0.250 | 0.250 | 1.000 | 0.000 |

## Current Decision

- Dynamic generation is the behavioral gate for whether a target-loss repair is real.
- If a patch range restores harmful refusal while preserving benign helpfulness, it becomes the first Gemma causal target for low-rank and GemmaScope basis validation.
