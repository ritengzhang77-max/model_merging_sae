# SmolLM2 Refusal Basis Validation

Stage 3 first pass: compare simple activation bases before SAE/transcoder work.

## Setup

- Prompts: 48 refusal prompts with original and varied templates.
- Layers: 15,20,25,29.
- Models: base, arith+polite merge, late-MLP refusal patch, late-MLP+attention refusal patch, full merge, arith+refusal merge, refusal expert.
- Features: prompt-final residual stream and MLP output activations.
- Classifier: prompt-heldout nearest-centroid classifier.

## Generation Labels

| model | stage3 keyword | old keyword | strict clean | strict messy | old clean | artifact | bad continuation | repetitive |
|---|---:|---:|---:|---:|---:|---:|---:|
| alpha_refusal_late_mlp_a1 | 0.375 | 0.438 | 0.000 | 0.375 | 0.083 | 0.667 | 0.146 | 0.646 |
| alpha_refusal_late_mlp_attn_a1 | 0.479 | 0.604 | 0.042 | 0.438 | 0.188 | 0.562 | 0.188 | 0.500 |
| base | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.042 | 0.000 | 0.604 |
| expert_refusal | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| merge_all_linear | 0.708 | 0.812 | 0.083 | 0.625 | 0.146 | 0.417 | 0.271 | 0.583 |
| merge_arith_polite | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.667 | 0.000 | 0.708 |
| merge_arith_refusal | 1.000 | 1.000 | 0.021 | 0.979 | 0.021 | 0.000 | 0.000 | 0.979 |

## Top Clean-Refusal Separators

| representation | basis | dim | balanced_acc | acc | n |
|---|---|---:|---:|---:|---:|
| resid_l29 | raw | full | 0.772 | 0.827 | 336 |
| mlp_out_l29 | raw | full | 0.761 | 0.670 | 336 |
| resid_l29 | random | 16 | 0.748 | 0.506 | 336 |
| mlp_out_l15 | topvar | 16 | 0.717 | 0.446 | 336 |
| mlp_out_l15 | topvar | 32 | 0.717 | 0.446 | 336 |
| mlp_out_l15 | pca | 16 | 0.714 | 0.440 | 336 |
| mlp_out_l15 | pca | 32 | 0.714 | 0.440 | 336 |
| resid_l15 | random | 32 | 0.714 | 0.440 | 336 |
| resid_l29 | topvar | 16 | 0.708 | 0.429 | 336 |
| mlp_out_l25 | raw | full | 0.707 | 0.836 | 336 |
| resid_l15 | topvar | 32 | 0.690 | 0.393 | 336 |
| mlp_out_l15 | random | 32 | 0.685 | 0.384 | 336 |

## Top Messy-Refusal Separators

| representation | basis | dim | balanced_acc | acc | n |
|---|---|---:|---:|---:|---:|
| mlp_out_l20 | raw | full | 0.843 | 0.845 | 336 |
| mlp_out_l20 | pca | 16 | 0.829 | 0.830 | 336 |
| mlp_out_l20 | pca | 32 | 0.829 | 0.830 | 336 |
| resid_l20 | pca | 16 | 0.823 | 0.824 | 336 |
| resid_l20 | pca | 32 | 0.820 | 0.821 | 336 |
| resid_l15 | raw | full | 0.817 | 0.818 | 336 |
| resid_l15 | pca | 16 | 0.814 | 0.815 | 336 |
| resid_l15 | pca | 32 | 0.814 | 0.815 | 336 |

## Interpretation

This is a baseline screen, not an SAE result. If raw/PCA/top-variance bases already separate clean vs messy refusal well, SAE/transcoder work must add causal or interpretive value beyond this.
