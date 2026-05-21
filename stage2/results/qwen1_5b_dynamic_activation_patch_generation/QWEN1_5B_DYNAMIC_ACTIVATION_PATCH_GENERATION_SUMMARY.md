# Qwen2.5-1.5B Dynamic Activation Patch Generation

Small smoke test: run donor base on the current prefix, patch donor activations into abliterated recipient, then greedily decode one token.

| model | harmful clean | harmful attempt | harmful bad | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `base` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| `abliterated` | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `activation_patch_12+13+14+15+16+17+18+19+20+21+22+23+24:mlp` | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 |

## Current Decision

- Dynamic activation patching succeeds on this 8 harmful / 8 benign screen:
  `12-24:mlp` donor activations from the base restore harmful clean refusal to
  `1.000`, compared with `0.875` for the base and `0.000` for the abliterated
  recipient.
- Benign helpfulness remains `1.000` and benign over-refusal remains `0.000`.
- This strengthens the mechanistic claim: the successful static repair is not
  merely a weight-space artifact; base MLP activations over the same range are
  sufficient to drive refusal behavior during greedy decoding.
- Next step: localize dynamic activation patching within the `12-24:mlp` range
  and compare against PCA/raw-neuron baselines before SAE/transcoder training.
