# Gemma-2-2B GemmaScope MLP SAE Validation

Layers: `12,13,14,15,16,17,18,19,20`.
Output mode: `post_ff_norm`.

## Reconstruction

| layer | split | donor EV | donor cosine | L0 mean | recipient-to-donor MSE | recon gap closed | file |
|---:|---|---:|---:|---:|---:|---:|---|
| 12 | benign | 0.781 | 0.795 | 104.4 | 0.04648 | -11.826 | `layer_12/width_16k/average_l0_108/params.npz` |
| 12 | harmful | 0.774 | 0.796 | 115.4 | 0.24743 | -1.703 | `layer_12/width_16k/average_l0_108/params.npz` |
| 13 | benign | 0.778 | 0.802 | 95.1 | 0.06546 | -9.746 | `layer_13/width_16k/average_l0_112/params.npz` |
| 13 | harmful | 0.770 | 0.795 | 98.4 | 0.45080 | -0.693 | `layer_13/width_16k/average_l0_112/params.npz` |
| 14 | benign | 0.665 | 0.802 | 88.2 | 0.07847 | -11.970 | `layer_14/width_16k/average_l0_97/params.npz` |
| 14 | harmful | 0.653 | 0.783 | 91.6 | 0.67884 | -0.663 | `layer_14/width_16k/average_l0_97/params.npz` |
| 15 | benign | 0.680 | 0.809 | 63.8 | 0.08766 | -13.123 | `layer_15/width_16k/average_l0_80/params.npz` |
| 15 | harmful | 0.658 | 0.802 | 67.6 | 0.85542 | -0.606 | `layer_15/width_16k/average_l0_80/params.npz` |
| 16 | benign | 0.736 | 0.819 | 57.7 | 0.10447 | -12.656 | `layer_16/width_16k/average_l0_72/params.npz` |
| 16 | harmful | 0.724 | 0.814 | 61.0 | 1.13124 | -0.371 | `layer_16/width_16k/average_l0_72/params.npz` |
| 17 | benign | 0.699 | 0.814 | 61.6 | 0.12069 | -14.460 | `layer_17/width_16k/average_l0_68/params.npz` |
| 17 | harmful | 0.680 | 0.800 | 66.7 | 1.36283 | -0.497 | `layer_17/width_16k/average_l0_68/params.npz` |
| 18 | benign | 0.691 | 0.854 | 112.5 | 0.13246 | -11.649 | `layer_18/width_16k/average_l0_106/params.npz` |
| 18 | harmful | 0.695 | 0.853 | 118.7 | 1.69250 | 0.015 | `layer_18/width_16k/average_l0_106/params.npz` |
| 19 | benign | 0.639 | 0.856 | 105.2 | 0.12539 | -11.908 | `layer_19/width_16k/average_l0_109/params.npz` |
| 19 | harmful | 0.633 | 0.854 | 116.3 | 1.73983 | 0.065 | `layer_19/width_16k/average_l0_109/params.npz` |
| 20 | benign | 0.637 | 0.860 | 102.4 | 0.12266 | -13.580 | `layer_20/width_16k/average_l0_109/params.npz` |
| 20 | harmful | 0.636 | 0.859 | 105.2 | 1.69904 | -0.096 | `layer_20/width_16k/average_l0_109/params.npz` |

## Generation

| model | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|---:|
| `base` | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `abliterated` | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `gemmascope_mlp_sae_patch_12+13+14+15+16+17+18+19+20` | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |

## Decision Rule

- This basis is aligned to the successful post-feedforward MLP-update patch.
- It must preserve the decoded MLP output well enough to compete with full `12-20:mlp`, PCA, and top-coordinate baselines.
- If MLP SAE reconstruction cannot support decoded-output patching, feature naming is premature.
