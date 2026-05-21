# SmolLM2 Module Inheritance Summary

Run date: 2026-05-10

Script:

- `stage2/scripts/analyze_smollm2_module_inheritance.py`

Inputs:

- base model: `HuggingFaceTB/SmolLM2-135M`
- experts from `stage0/artifacts/smollm2_synthetic_experts/`
- cached merge states from `stage1/cache/smollm2_state_cache/`

Outputs:

- `stage2/results/smollm2_module_inheritance_ablation.csv`
- `stage2/results/smollm2_module_inheritance_profile_correlations.csv`
- `stage2/results/smollm2_module_inheritance_top_interventions.csv`
- `stage2/results/smollm2_module_inheritance_summary.json`

## What Was Tested

This refines the layer-level Stage 2 screen. Instead of replacing an entire
transformer layer with base weights, the script replaces only one module type
inside layers 15 and 29:

- `attn`: `self_attn.*`
- `mlp`: `mlp.*`
- `norms`: input and post-attention layernorm weights
- `block`: the full transformer block

The score is again loss damage:

```text
loss_increase = base_module_patched_loss - original_loss
```

## Main Results

Expert-vs-merge module-profile correlations:

| task | merge | Spearman | expert top | merge top |
|---|---|---:|---|---|
| arith | merge_all_linear | +0.143 | 29/block | 29/block |
| arith | merge_arith_polite | -0.119 | 29/block | 15/block |
| arith | merge_arith_refusal | -0.119 | 29/block | 15/block |
| polite | merge_all_linear | +0.643 | 29/block | 15/block |
| polite | merge_arith_polite | +0.810 | 29/block | 29/block |
| polite | merge_polite_refusal | +0.690 | 29/block | 15/block |
| refusal | merge_all_linear | +0.929 | 29/block | 29/block |
| refusal | merge_arith_refusal | +0.929 | 29/block | 29/block |
| refusal | merge_polite_refusal | +0.262 | 29/block | 29/block |

Largest module-level ablations:

| target | task | intervention | loss increase |
|---|---|---|---:|
| merge_all_linear | arith | 29/block | +0.096 |
| merge_all_linear | arith | 15/block | +0.096 |
| merge_all_linear | refusal | 29/block | +0.081 |
| merge_arith_polite | arith | 15/block | +0.073 |
| merge_all_linear | refusal | 29/mlp | +0.071 |
| merge_arith_refusal | arith | 15/block | +0.068 |
| merge_all_linear | arith | 29/mlp | +0.064 |
| merge_all_linear | arith | 15/mlp | +0.059 |

Average effect by module type:

| task | module | mean loss increase | max loss increase |
|---|---|---:|---:|
| arith | block | +0.054 | +0.096 |
| arith | mlp | +0.034 | +0.064 |
| arith | attn | +0.016 | +0.028 |
| arith | norms | +0.001 | +0.001 |
| polite | block | +0.013 | +0.030 |
| polite | mlp | +0.009 | +0.019 |
| polite | attn | +0.003 | +0.008 |
| refusal | block | +0.023 | +0.081 |
| refusal | mlp | +0.019 | +0.071 |
| refusal | attn | +0.003 | +0.007 |

## Interpretation

The module screen sharpens the layer-level result:

- Refusal has the clearest mechanistic inheritance. The expert and the useful
  refusal-containing merges have almost the same module vulnerability profile,
  centered on layer 29 full block and especially layer 29 MLP.
- Polite behavior is also inherited to a meaningful degree, but it can shift
  between layer 29 and layer 15 depending on the merge partner.
- Arithmetic remains the unstable case. `merge_all_linear` shares the expert's
  top intervention at 29/block, but the full module profile correlation is weak.
  Pairwise arithmetic merges shift their top intervention to 15/block.
- Across tasks, MLP explains much more of the full-block damage than attention.
  Attention contributes most visibly for arithmetic, but it is still smaller
  than MLP in these runs.
- Layernorm-only patches do almost nothing, so they are unlikely to be the main
  carrier of these synthetic task behaviors.

## Updated Mechanistic Hypothesis

For this SmolLM2 synthetic setup:

1. Refusal transfer is mostly late-MLP / final-block inherited machinery.
2. Polite transfer is partially inherited but more layer-distributed.
3. Arithmetic retention may be behaviorally successful without preserving the
   source expert's original module-level profile, especially in pairwise merges.

This gives us a concrete next target: test whether the late MLP components are
causally sufficient by patching expert layer-29 MLP into weaker merges or into
the base model, and test whether arithmetic's layer-15 shift is a real rerouting
effect or an artifact of the synthetic arithmetic expert being weak.
