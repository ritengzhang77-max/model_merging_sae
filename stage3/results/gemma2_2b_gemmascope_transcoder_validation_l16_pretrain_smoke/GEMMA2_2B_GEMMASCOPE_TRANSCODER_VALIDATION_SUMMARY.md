# Gemma-2-2B GemmaScope Transcoder Validation

Layers: `16`.
MLP input mode: `folded_pre_norm`.
MLP output mode: `post_ff_norm`.

## Reconstruction

| layer | split | donor EV | donor cosine | L0 mean | recipient-to-donor MSE | recon gap closed | file |
|---:|---|---:|---:|---:|---:|---:|---|
| 16 | benign | 0.111 | 0.719 | 107.4 | 0.00000 | -116805729959239152.000 | `layer_16/width_16k/average_l0_87/params.npz` |
| 16 | harmful | 0.098 | 0.736 | 100.8 | 0.00000 | -109769531250000016.000 | `layer_16/width_16k/average_l0_87/params.npz` |

## Decision Rule

- Treat reconstruction as a gate, not as interpretability.
- A GemmaScope transcoder branch is promising only if reconstruction is close enough to compete with the native MLP-output baselines and if decoded-output patching preserves the causal repair.
- If reconstruction is weak on the instruction-tuned donor/recipient distribution, do not spend effort naming features from this basis yet.
