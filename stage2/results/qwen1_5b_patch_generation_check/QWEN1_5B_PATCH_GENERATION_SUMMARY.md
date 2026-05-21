# Qwen2.5-1.5B Patch Generation Check

Small generation audit for top base-to-abliterated static module patches.

| model | harmful clean | harmful attempt | harmful bad | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `base` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| `abliterated` | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `patch_14:mlp` | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `patch_14:block` | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `patch_16:mlp` | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `patch_16:block` | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |

## Current Decision

- If target-loss repairs do not improve harmful generation, use them as localization signals only.
- If a patch improves harmful refusal without benign over-refusal, promote it to a deeper causal patching pass.
