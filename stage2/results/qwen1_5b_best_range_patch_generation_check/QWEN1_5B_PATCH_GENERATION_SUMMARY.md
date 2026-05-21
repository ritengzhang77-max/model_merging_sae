# Qwen2.5-1.5B Patch Generation Check

Small generation audit for top base-to-abliterated static module patches.

| model | harmful clean | harmful attempt | harmful bad | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `base` | 0.750 | 0.750 | 0.000 | 1.000 | 0.000 |
| `abliterated` | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `patch_12+13+14+15+16+17+18+19+20+21+22+23+24:mlp` | 0.917 | 0.917 | 0.000 | 1.000 | 0.000 |
| `patch_14+15+16+17+18+19+20+21+22+23+24:mlp` | 0.500 | 0.500 | 0.000 | 1.000 | 0.000 |
| `patch_14+15+16+17+18+19:mlp` | 0.333 | 0.333 | 0.000 | 1.000 | 0.000 |
| `patch_14+15+16+17+18+19+20+21+22+23+24:block` | 0.417 | 0.417 | 0.000 | 1.000 | 0.000 |

## Current Decision

- `patch_12+13+14+15+16+17+18+19+20+21+22+23+24:mlp` is the first strong
  causal repair found in the public Qwen branch. It raises harmful clean refusal
  from `0.000` in the abliterated model to `0.917`, while keeping benign
  helpfulness at `1.000` and benign over-refusal at `0.000`.
- The narrower `14-24:mlp` patch reaches only `0.500`, and `14-19:mlp` reaches
  `0.333`, so layers 12-13 appear important for this static repair.
- The `14-24:block` patch reaches `0.417`, below the matched MLP range, so MLPs
  are currently the cleaner repair target.
- Next step: localize within the `12-24:mlp` range and compare static MLP
  replacement against activation patching before any SAE/transcoder work.
