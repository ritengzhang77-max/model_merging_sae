# Qwen2.5-1.5B Patch Generation Check

Small generation audit for top base-to-abliterated static module patches.

| model | harmful clean | harmful attempt | harmful bad | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `base` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| `abliterated` | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `patch_14+15+16+17+18+19:mlp` | 0.500 | 0.500 | 0.000 | 1.000 | 0.000 |
| `patch_20+21+22+23+24:mlp` | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `patch_14+15+16+20+21+22+23+24:mlp` | 0.250 | 0.250 | 0.000 | 1.000 | 0.000 |
| `patch_14+15+16+17+18+19+20+21+22+23+24:mlp` | 0.625 | 0.625 | 0.000 | 1.000 | 0.000 |
| `patch_12+13+14+15+16+17+18+19+20+21+22+23+24:mlp` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| `patch_14+15+16+17+18+19+20+21+22+23+24+25+26+27:mlp` | 0.500 | 0.500 | 0.000 | 1.000 | 0.000 |
| `patch_14+15+16+17+18+19+20+21+22+23+24:block` | 0.625 | 0.625 | 0.000 | 1.000 | 0.000 |

## Current Decision

- Multi-layer MLP patches are causally active: `14-19:mlp` restores harmful
  refusal to `0.500`, `14-24:mlp` restores it to `0.625`, and `12-24:mlp`
  restores it to `0.875` on this 8-prompt harmful screen.
- The late-only `20-24:mlp` patch remains at `0.000`, so late harmful-specific
  activation drift is not sufficient by itself.
- Benign helpfulness remains `1.000` and benign over-refusal remains `0.000`
  for all tested range patches.
- Promote `12-24:mlp` to the broader 12-prompt validation screen.
