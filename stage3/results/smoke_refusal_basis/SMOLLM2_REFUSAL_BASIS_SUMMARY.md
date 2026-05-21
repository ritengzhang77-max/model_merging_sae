# SmolLM2 Refusal Basis Validation

Stage 3 first pass: compare simple activation bases before SAE/transcoder work.

## Setup

- Prompts: 12 refusal prompts with original and varied templates.
- Layers: 15,20,25,29.
- Models: base, arith+polite merge, late-MLP refusal patch, late-MLP+attention refusal patch, full merge, arith+refusal merge, refusal expert.
- Features: prompt-final residual stream and MLP output activations.
- Classifier: prompt-heldout nearest-centroid classifier.

## Generation Labels

| model | keyword | clean | messy | artifact | repetitive |
|---|---:|---:|---:|---:|---:|
| alpha_refusal_late_mlp_a1 | 0.417 | 0.083 | 0.333 | 0.667 | 0.083 |
| alpha_refusal_late_mlp_attn_a1 | 0.500 | 0.250 | 0.250 | 0.417 | 0.250 |
| base | 0.000 | 0.000 | 0.000 | 0.083 | 0.333 |
| expert_refusal | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| merge_all_linear | 0.917 | 0.333 | 0.583 | 0.500 | 0.167 |
| merge_arith_polite | 0.000 | 0.000 | 0.000 | 0.833 | 0.333 |
| merge_arith_refusal | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |

## Top Clean-Refusal Separators

| representation | basis | dim | balanced_acc | acc | n |
|---|---|---:|---:|---:|---:|
| mlp_out_l29 | raw | full | 0.836 | 0.702 | 84 |
| resid_l25 | raw | full | 0.829 | 0.690 | 84 |
| resid_l29 | raw | full | 0.822 | 0.679 | 84 |
| resid_l29 | topvar | 16 | 0.783 | 0.607 | 84 |
| resid_l29 | topvar | 32 | 0.776 | 0.595 | 84 |
| resid_l29 | pca | 16 | 0.770 | 0.583 | 84 |
| resid_l29 | pca | 32 | 0.770 | 0.583 | 84 |
| mlp_out_l25 | random | 16 | 0.763 | 0.571 | 84 |
| mlp_out_l25 | topvar | 16 | 0.757 | 0.560 | 84 |
| resid_l29 | random | 16 | 0.757 | 0.560 | 84 |
| resid_l29 | random | 32 | 0.757 | 0.560 | 84 |
| resid_l15 | raw | full | 0.753 | 0.655 | 84 |

## Top Messy-Refusal Separators

| representation | basis | dim | balanced_acc | acc | n |
|---|---|---:|---:|---:|---:|
| mlp_out_l20 | raw | full | 0.884 | 0.893 | 84 |
| resid_l15 | raw | full | 0.871 | 0.881 | 84 |
| mlp_out_l15 | pca | 16 | 0.858 | 0.869 | 84 |
| mlp_out_l15 | pca | 32 | 0.858 | 0.869 | 84 |
| mlp_out_l20 | topvar | 32 | 0.849 | 0.857 | 84 |
| resid_l15 | pca | 16 | 0.849 | 0.857 | 84 |
| resid_l15 | pca | 32 | 0.849 | 0.857 | 84 |
| mlp_out_l20 | pca | 16 | 0.840 | 0.845 | 84 |

## Interpretation

This is a baseline screen, not an SAE result. If raw/PCA/top-variance bases already separate clean vs messy refusal well, SAE/transcoder work must add causal or interpretive value beyond this.
