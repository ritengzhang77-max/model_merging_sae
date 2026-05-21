# SmolLM2 Module Sufficiency Summary

Run date: 2026-05-10

Script:

- `stage2/scripts/analyze_smollm2_module_sufficiency.py`

Inputs:

- base model: `HuggingFaceTB/SmolLM2-135M`
- experts from `stage0/artifacts/smollm2_synthetic_experts/`
- cached merge states from `stage1/cache/smollm2_state_cache/`

Outputs:

- `stage2/results/smollm2_module_sufficiency_patches.csv`
- `stage2/results/smollm2_module_sufficiency_top_patches.csv`
- `stage2/results/smollm2_module_sufficiency_summary.json`

## What Was Tested

This is a sufficiency test, not a necessity test.

For each task expert, selected modules were copied from the expert into a
recipient model:

- recipients: base, all-merge, and the three pairwise merges;
- layers: 15 and 29;
- modules: attention, MLP, and full block.

The main score is:

```text
loss_improvement = recipient_original_loss - patched_loss
```

Positive values mean the expert module makes the recipient better on the
expert's task.

`gap_closed_to_expert` reports the fraction of the recipient-to-expert loss gap
closed by the patch.

## Clean Missing-Capability Results

These are the most important cases because the recipient does not already
contain the source expert.

| task | recipient | best patch | loss improvement | gap closed |
|---|---|---|---:|---:|
| arith | base | 15/block | +0.769 | 0.103 |
| arith | merge_polite_refusal | 15/block | +0.935 | 0.144 |
| polite | base | 29/block | +0.321 | 0.145 |
| polite | merge_arith_refusal | 29/block | +0.350 | 0.231 |
| refusal | base | 29/block | +0.470 | 0.139 |
| refusal | merge_arith_polite | 29/block | +0.471 | 0.147 |

Every tested expert-module patch improved task loss in the full run:

```text
negative or zero improvements: 0 / 90
```

## MLP Versus Full Block

For clean missing-capability recipients, MLP alone explains a large fraction of
the full-block rescue:

| task | recipient | layer | MLP / block rescue ratio |
|---|---|---:|---:|
| arith | base | 15 | 0.625 |
| arith | base | 29 | 0.884 |
| arith | merge_polite_refusal | 15 | 0.629 |
| arith | merge_polite_refusal | 29 | 0.841 |
| polite | base | 15 | 0.643 |
| polite | base | 29 | 0.840 |
| polite | merge_arith_refusal | 15 | 0.650 |
| polite | merge_arith_refusal | 29 | 0.911 |
| refusal | base | 15 | 0.692 |
| refusal | base | 29 | 0.882 |
| refusal | merge_arith_polite | 15 | 0.738 |
| refusal | merge_arith_polite | 29 | 0.910 |

Average clean-recipient loss improvement:

| task | module | mean improvement | max improvement |
|---|---|---:|---:|
| arith | block | +0.747 | +0.935 |
| arith | mlp | +0.544 | +0.588 |
| arith | attn | +0.186 | +0.330 |
| polite | block | +0.287 | +0.350 |
| polite | mlp | +0.224 | +0.319 |
| polite | attn | +0.064 | +0.101 |
| refusal | block | +0.458 | +0.471 |
| refusal | mlp | +0.371 | +0.429 |
| refusal | attn | +0.090 | +0.132 |

## Interpretation

The sufficiency result supports the mechanism story from the ablation screens:

- Expert modules are not merely necessary-looking; they can causally transfer
  behavior into models that lack the corresponding expert.
- Full-block patches are strongest, but MLP patches carry most of the rescue.
- Layer 29 is the cleanest sufficiency site for polite and refusal.
- Arithmetic is different: the best rescue is 15/block, matching the earlier
  observation that arithmetic in pairwise merges shifts toward layer 15.

This means the current best mechanistic hypothesis is:

1. Refusal is largely a late MLP/block behavior around layer 29.
2. Polite is also late MLP/block-heavy, with some redistribution across merges.
3. Arithmetic is less cleanly inherited from its expert and appears to route
   more through layer 15 in merged recipients.

## What This Does Not Prove Yet

This is module-level causal evidence, not a full circuit explanation.

Still missing:

- whether the same neurons/features inside the MLP are responsible;
- whether the patch transfers generation behavior, not just teacher-forced loss;
- whether these effects hold with stronger experts and real tasks;
- whether combining selected expert modules can improve merge recipes.

## Next Step

The next useful experiment is a mechanism-aware micro-merge:

- start from a weak recipient such as `merge_arith_polite` for refusal;
- insert only the expert_refusal layer-29 MLP or block;
- compare it against full linear merge on all three tasks;
- check whether this targeted patch restores refusal with less collateral
  effect than adding the entire refusal delta.
