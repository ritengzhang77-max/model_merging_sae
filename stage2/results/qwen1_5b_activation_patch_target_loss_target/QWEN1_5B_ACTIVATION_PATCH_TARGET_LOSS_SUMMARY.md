# Qwen2.5-1.5B Activation Patch Target-Loss Check

Teacher-forced activation patching from base into abliterated on standard refusal completions.

| variant | harmful loss | benign loss | harmful gap closed | benign gap closed | specificity |
|---|---:|---:|---:|---:|---:|
| `base` | 0.085 | 0.442 | 1.000 | 1.000 | 0.000 |
| `abliterated` | 2.343 | 3.043 | 0.000 | 0.000 | 0.000 |
| `activation_patch_12+13+14+15+16+17+18+19+20+21+22+23+24:mlp` | 2.343 | 3.043 | 0.000 | 0.000 | 0.000 |
| `activation_patch_12+13+14+15+16+17+18+19+20+21:mlp` | 2.343 | 3.043 | 0.000 | 0.000 | 0.000 |
| `activation_patch_12+13+14+15+16+17+18+19:mlp` | 2.343 | 3.043 | 0.000 | 0.000 | 0.000 |

## Current Decision

- Target-position-only activation patching does nothing: all tested ranges close
  `0.000` of the harmful-refusal loss gap.
- The repair requires patching MLP activations throughout the sequence, not only
  the final target position.
