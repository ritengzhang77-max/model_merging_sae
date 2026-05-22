# Gemma-2-2B Activation Patch Target-Loss Check

Teacher-forced activation patching from Gemma base into abliterated on a fixed refusal completion.

| variant | harmful loss | benign loss | harmful gap closed | benign gap closed | specificity |
|---|---:|---:|---:|---:|---:|
| `base` | 4.577 | 6.408 | 1.000 | 1.000 | 0.000 |
| `abliterated` | 6.900 | 7.058 | 0.000 | 0.000 | 0.000 |
| `activation_patch_16:mlp` | 5.964 | 6.988 | 0.403 | 0.107 | 0.296 |
| `activation_patch_20:mlp` | 6.063 | 7.027 | 0.360 | 0.047 | 0.314 |
| `activation_patch_25:mlp` | 8.372 | 7.665 | -0.633 | -0.935 | 0.302 |
| `activation_patch_16+17+18+19+20:mlp` | 4.507 | 6.659 | 1.030 | 0.614 | 0.417 |
| `activation_patch_20+21+22+23+24+25:mlp` | 6.838 | 7.511 | 0.027 | -0.698 | 0.725 |
| `activation_patch_12+13+14+15+16+17+18+19+20:mlp` | 4.641 | 6.641 | 0.973 | 0.642 | 0.331 |
| `activation_patch_16+17+18+19+20+21+22+23+24+25:mlp` | 4.962 | 6.733 | 0.834 | 0.499 | 0.335 |

## Current Decision

- This checks whether base activations can restore the refusal target at the likelihood level.
- Dynamic generation patching should be attempted for ranges that close substantial harmful-refusal loss.
