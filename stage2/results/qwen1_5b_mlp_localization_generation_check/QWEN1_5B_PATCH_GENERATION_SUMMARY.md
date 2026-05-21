# Qwen2.5-1.5B Patch Generation Check

Small generation audit for top base-to-abliterated static module patches.

| model | harmful clean | harmful attempt | harmful bad | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `base` | 0.750 | 0.750 | 0.000 | 1.000 | 0.000 |
| `abliterated` | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `patch_12+13:mlp` | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `patch_12+13+14+15+16:mlp` | 0.167 | 0.167 | 0.000 | 1.000 | 0.000 |
| `patch_12+13+14+15+16+17+18+19:mlp` | 0.500 | 0.500 | 0.000 | 1.000 | 0.000 |
| `patch_12+13+14+15+16+17+18+19+20+21:mlp` | 0.750 | 0.750 | 0.000 | 1.000 | 0.000 |
| `patch_12+13+14+15+16+17+18+19+20+21+22+23+24:mlp` | 0.917 | 0.917 | 0.000 | 1.000 | 0.000 |
| `patch_12+13+20+21+22+23+24:mlp` | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `patch_12+13+14+15+16+20+21+22+23+24:mlp` | 0.167 | 0.167 | 0.000 | 1.000 | 0.000 |
| `patch_13+14+15+16+17+18+19+20+21+22+23+24:mlp` | 0.500 | 0.500 | 0.000 | 1.000 | 0.000 |

## Current Decision

- The repair is a contiguous-range effect, not a small two-layer effect.
- `12+13:mlp` alone restores `0.000` harmful refusal. Extending to `12-16`
  restores `0.167`, `12-19` restores `0.500`, `12-21` restores `0.750`, and
  `12-24` restores `0.917`.
- Dropping layer 12 weakens the full repair: `13-24:mlp` reaches only `0.500`.
- Combining early `12-16` with late `20-24` while skipping `17-19` reaches only
  `0.167`, so the middle `17-19` region is also important.
- Benign helpfulness remains `1.000` and benign over-refusal remains `0.000` in
  all tested variants.
- Next step: compare this static range replacement with activation patching on
  the same `12-24:mlp` range.
