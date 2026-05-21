# Qwen2.5-1.5B Activation Patch Target-Loss Check

Teacher-forced activation patching from base into abliterated on standard refusal completions.

| variant | harmful loss | benign loss | harmful gap closed | benign gap closed | specificity |
|---|---:|---:|---:|---:|---:|
| `base` | 0.085 | 0.442 | 1.000 | 1.000 | 0.000 |
| `abliterated` | 2.343 | 3.043 | 0.000 | 0.000 | 0.000 |
| `activation_patch_12+13+14+15+16+17+18+19+20+21+22+23+24:mlp` | 0.224 | 0.565 | 0.938 | 0.953 | -0.014 |
| `activation_patch_12+13+14+15+16+17+18+19+20+21:mlp` | 0.227 | 0.611 | 0.937 | 0.935 | 0.002 |
| `activation_patch_12+13+14+15+16+17+18+19:mlp` | 0.272 | 0.727 | 0.917 | 0.890 | 0.027 |
| `activation_patch_13+14+15+16+17+18+19+20+21+22+23+24:mlp` | 0.220 | 0.556 | 0.940 | 0.956 | -0.016 |

## Current Decision

- Donor base MLP activations explain most of the static repair at the
  teacher-forced likelihood level.
- The full `12-24:mlp` activation patch closes `0.938` of the harmful-refusal
  loss gap. The `12-21:mlp` range is similar at `0.937`, and `12-19:mlp`
  closes `0.917`.
- Because these ranges close substantial harmful-refusal loss, dynamic
  generation patching is warranted.
