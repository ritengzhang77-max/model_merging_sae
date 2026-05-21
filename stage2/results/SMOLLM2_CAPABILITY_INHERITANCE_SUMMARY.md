# SmolLM2 Capability Inheritance Summary

Run date: 2026-05-10

Script:

- `stage2/scripts/analyze_smollm2_capability_inheritance.py`

Inputs:

- base model: `HuggingFaceTB/SmolLM2-135M`
- experts from `stage0/artifacts/smollm2_synthetic_experts/`
- cached merge states from `stage1/cache/smollm2_state_cache/`

Outputs:

- `stage2/results/smollm2_capability_inheritance_layer_ablation.csv`
- `stage2/results/smollm2_capability_inheritance_profile_correlations.csv`
- `stage2/results/smollm2_capability_inheritance_summary.json`

## What Was Tested

For each task, selected transformer layers in each expert/merge were replaced
with the base model's corresponding layer. The score is loss damage:

```text
loss_increase = base_layer_patched_loss - original_loss
```

This is a causal screen for module importance. If an expert and a merge depend
on the same layers for a task, their layer damage profiles should correlate.

Layer 0 is reported but treated cautiously because it can act like a broad early
representation disruption rather than a clean task-specific component.

## Main Results

Expert-vs-merge Spearman correlations:

| task | merge | Spearman | Spearman excluding layer 0 | expert top | merge top |
|---|---|---:|---:|---:|---:|
| arith | merge_all_linear | -0.107 | +0.086 | 29 | 29 |
| arith | merge_arith_polite | -0.714 | -0.714 | 29 | 15 |
| arith | merge_arith_refusal | -0.536 | -0.543 | 29 | 15 |
| polite | merge_all_linear | +0.321 | +0.200 | 29 | 15 |
| polite | merge_arith_polite | +0.714 | +0.600 | 29 | 0 |
| polite | merge_polite_refusal | +0.429 | +0.371 | 29 | 15 |
| refusal | merge_all_linear | +0.429 | +0.943 | 29 | 29 |
| refusal | merge_arith_refusal | +0.571 | +0.886 | 29 | 29 |
| refusal | merge_polite_refusal | +0.536 | +0.371 | 29 | 29 |

Largest base-layer ablation effects:

| target | task | top layer | loss increase |
|---|---|---:|---:|
| expert_arith | arith | 29 | +0.013 |
| merge_all_linear | arith | 29 | +0.096 |
| merge_arith_polite | arith | 15 | +0.073 |
| merge_arith_refusal | arith | 15 | +0.068 |
| expert_polite | polite | 29 | +0.005 |
| merge_all_linear | polite | 15 | +0.030 |
| merge_arith_polite | polite | 0 | +0.014 |
| merge_polite_refusal | polite | 15 | +0.012 |
| expert_refusal | refusal | 29 | +0.004 |
| merge_all_linear | refusal | 29 | +0.081 |
| merge_arith_refusal | refusal | 29 | +0.047 |
| merge_polite_refusal | refusal | 29 | +0.017 |

## Interpretation

The strongest inheritance signal is refusal:

- refusal top layer stays at layer 29 in expert and merge;
- excluding layer 0, refusal profile correlation is very high for
  `merge_all_linear` and `merge_arith_refusal`.

Polite behavior shows moderate inheritance:

- correlations are positive;
- the task often shifts from expert layer 29 to merge layer 15, suggesting the
  merge may preserve behavior with a different or redistributed layer profile.

Arithmetic is the unstable case:

- `merge_all_linear` shares top layer 29 with the expert, but profile
  correlation is near zero;
- pairwise arithmetic merges peak at layer 15 and have negative profile
  correlations with the arithmetic expert.

This is useful because it separates behavioral retention from mechanistic
inheritance. Stage 1 showed arithmetic behavior is retained by loss metrics, but
Stage 2 suggests the retained arithmetic behavior may not be relying on the same
layer profile as the arithmetic expert.

## Caveats

- The experts are tiny synthetic fine-tunes, so direct expert ablation effects
  are small.
- Base-layer replacement is a coarse causal intervention. It does not identify
  individual attention heads, MLPs, features, or subspaces.
- Some layer replacements slightly improve loss; negative damage rows should be
  treated as evidence of noisy or non-monotonic effects, not as final mechanism
  claims.

## Next Step

The next useful Stage 2 refinement is module-level ablation:

- split each selected transformer block into attention and MLP/base-layer
  replacements;
- focus first on layers 15 and 29;
- test whether refusal's strong inheritance is specifically late MLP/readout,
  and whether arithmetic's shift to layer 15 is attention, MLP, or residual
  stream routing.
