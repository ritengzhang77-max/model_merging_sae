# Gemma-2-2B GemmaScope Transcoder Validation

Layers: `16`.
MLP input mode: `folded_pre_norm`.

## Reconstruction

| layer | split | donor EV | donor cosine | L0 mean | recipient-to-donor MSE | recon gap closed | file |
|---:|---|---:|---:|---:|---:|---:|---|
| 16 | benign | -0.192 | 0.630 | 84.7 | 0.12778 | -48.093 | `layer_16/width_16k/average_l0_87/params.npz` |
| 16 | harmful | -0.168 | 0.618 | 89.5 | 1.25172 | -4.031 | `layer_16/width_16k/average_l0_87/params.npz` |

## Decision Rule

- Treat reconstruction as a gate, not as interpretability.
- A GemmaScope transcoder branch is promising only if reconstruction is close enough to compete with the native MLP-output baselines and if decoded-output patching preserves the causal repair.
- If reconstruction is weak on the instruction-tuned donor/recipient distribution, do not spend effort naming features from this basis yet.
