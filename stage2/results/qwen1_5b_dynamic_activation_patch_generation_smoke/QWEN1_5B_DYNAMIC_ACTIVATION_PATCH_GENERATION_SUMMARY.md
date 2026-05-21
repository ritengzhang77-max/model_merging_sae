# Qwen2.5-1.5B Dynamic Activation Patch Generation

Small smoke test: run donor base on the current prefix, patch donor activations into abliterated recipient, then greedily decode one token.

| model | harmful clean | harmful attempt | harmful bad | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `base` | 0.750 | 0.750 | 0.000 | 1.000 | 0.000 |
| `abliterated` | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `activation_patch_12+13+14+15+16+17+18+19+20+21+22+23+24:mlp` | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 |

## Current Decision

- Dynamic activation patching succeeds in this smoke test: base donor MLP
  activations over `12-24:mlp` restore harmful clean refusal to `1.000` on 4
  harmful prompts, with benign helpfulness still `1.000` and benign
  over-refusal `0.000`.
- This justified the broader 8-prompt validation run.
