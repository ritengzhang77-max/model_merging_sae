# SmolLM2 Refusal Assistant Audit

This is a reproducible assistant audit over the 94-row Stage 3 sample. It is stricter than keyword matching, but it should still be treated as provisional until replaced by user/professor human labels.

## Model-Level Labels

| model | n | clean | refusal attempt | problem response | messy repetition | messy artifact | contradictory/unsafe | no refusal | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| alpha_refusal_late_mlp_a1 | 14 | 0.000 | 0.571 | 0.714 | 0.143 | 0.071 | 0.500 | 0.286 | 0.000 |
| alpha_refusal_late_mlp_attn_a1 | 12 | 0.083 | 0.667 | 0.667 | 0.083 | 0.083 | 0.500 | 0.250 | 0.000 |
| base | 14 | 0.000 | 0.000 | 0.429 | 0.000 | 0.000 | 0.429 | 0.571 | 0.000 |
| expert_refusal | 14 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| merge_all_linear | 14 | 0.000 | 0.714 | 0.786 | 0.286 | 0.143 | 0.357 | 0.214 | 0.000 |
| merge_arith_polite | 14 | 0.000 | 0.000 | 0.214 | 0.000 | 0.000 | 0.214 | 0.786 | 0.000 |
| merge_arith_refusal | 12 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Top Audit-Clean Separators

| representation | basis | dim | balanced_acc | acc | n | positives |
|---|---|---:|---:|---:|---:|---:|
| resid_l29 | raw | full | 0.963 | 0.963 | 81 | 1 |
| mlp_out_l25 | raw | full | 0.914 | 0.914 | 81 | 1 |
| resid_l25 | raw | full | 0.840 | 0.840 | 81 | 1 |
| mlp_out_l29 | raw | full | 0.790 | 0.790 | 81 | 1 |
| mlp_out_l20 | raw | full | 0.753 | 0.753 | 81 | 1 |
| mlp_out_l15 | raw | full | 0.716 | 0.716 | 81 | 1 |
| resid_l15 | raw | full | 0.716 | 0.716 | 81 | 1 |
| resid_l20 | raw | full | 0.691 | 0.691 | 81 | 1 |
| mlp_out_l29 | pca | 16 | 0.506 | 0.506 | 81 | 1 |
| mlp_out_l25 | random | 16 | 0.494 | 0.494 | 81 | 1 |

## Top Audit-Refusal-Attempt Separators

| representation | basis | dim | balanced_acc | acc | n | positives |
|---|---|---:|---:|---:|---:|---:|
| mlp_out_l25 | raw | full | 0.801 | 0.787 | 94 | 52 |
| mlp_out_l20 | raw | full | 0.799 | 0.787 | 94 | 52 |
| resid_l25 | raw | full | 0.799 | 0.787 | 94 | 52 |
| mlp_out_l25 | random | 32 | 0.796 | 0.787 | 94 | 52 |
| mlp_out_l15 | raw | full | 0.791 | 0.777 | 94 | 52 |
| mlp_out_l20 | pca | 16 | 0.779 | 0.766 | 94 | 52 |
| mlp_out_l20 | pca | 32 | 0.779 | 0.766 | 94 | 52 |
| mlp_out_l25 | pca | 16 | 0.779 | 0.766 | 94 | 52 |
| mlp_out_l25 | pca | 32 | 0.779 | 0.766 | 94 | 52 |
| resid_l15 | raw | full | 0.779 | 0.766 | 94 | 52 |

## Interpretation

The audit separates attempted refusal from clean refusal. Clean refusal is effectively absent in this synthetic setup, so clean-refusal basis metrics are not decision-grade. The stable near-term target is attempted-refusal vs no-attempt, followed by a separate quality analysis for repetition, artifacts, and unsafe continuations.
