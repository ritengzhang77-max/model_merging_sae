# SmolLM2 Mechanism-Aware Micro-Merge Summary

Run date: 2026-05-10

Script:

- `stage2/scripts/run_smollm2_mechanism_micro_merge.py`

Inputs:

- recipient: `merge_arith_polite`
- donor: `expert_refusal`
- targeted modules:
  - `micro_refusal_l29_mlp`: expert refusal layer-29 MLP only
  - `micro_refusal_l29_block`: expert refusal full layer-29 block
  - `micro_refusal_l25_l29_mlp`: expert refusal layer-25 and layer-29 MLPs
  - `micro_refusal_l20_l25_l29_mlp`: expert refusal layer-20, layer-25, and
    layer-29 MLPs
  - `micro_refusal_l25_l29_block`: expert refusal layer-25 and layer-29 blocks
  - broader grid:
    - layer 15+20+25+29 MLPs
    - layer 20+25+29 attention
    - layer 15+20+25+29 attention
    - MLP+attention combinations at those layers

Outputs:

- `stage2/results/smollm2_micro_merge_losses.csv`
- `stage2/results/smollm2_micro_merge_generation_metrics.csv`
- `stage2/results/smollm2_micro_merge_generations.jsonl`
- `stage2/results/smollm2_micro_merge_summary.json`

## Goal

Test whether a mechanism-aware targeted patch can restore refusal behavior to
`merge_arith_polite` with less collateral effect than adding the entire refusal
expert through `merge_all_linear`.

This is the first attempt to turn the mechanistic findings into a merge recipe.

## Teacher-Forced Loss Results

| model | mean loss | arith | polite | refusal |
|---|---:|---:|---:|---:|
| merge_all_linear | 0.970 | 2.036 | 0.386 | 0.488 |
| merge_arith_refusal | 1.162 | 1.390 | 1.778 | 0.317 |
| micro_refusal_l15_l20_l25_l29_mlp_attn | 1.269 | 1.558 | 0.349 | 1.900 |
| micro_refusal_l15_l20_l25_l29_mlp | 1.351 | 1.509 | 0.335 | 2.210 |
| micro_refusal_l20_l25_l29_mlp_attn | 1.396 | 1.492 | 0.340 | 2.356 |
| micro_refusal_l20_l25_l29_mlp | 1.452 | 1.477 | 0.330 | 2.549 |
| micro_refusal_l25_l29_block | 1.490 | 1.453 | 0.328 | 2.688 |
| micro_refusal_l25_l29_mlp | 1.523 | 1.441 | 0.324 | 2.806 |
| micro_refusal_l29_block | 1.575 | 1.439 | 0.321 | 2.965 |
| micro_refusal_l29_mlp | 1.586 | 1.430 | 0.320 | 3.007 |
| merge_arith_polite | 1.716 | 1.401 | 0.311 | 3.436 |
| expert_refusal | 4.013 | 8.509 | 3.293 | 0.235 |
| base | 4.869 | 8.516 | 2.477 | 3.614 |

Targeted refusal patches improve refusal loss:

- `merge_arith_polite` refusal loss: `3.436`
- + refusal layer-29 MLP: `3.007`
- + refusal layer-29 block: `2.965`
- + refusal layers 25+29 MLP: `2.806`
- + refusal layers 20+25+29 MLP: `2.549`
- + refusal layers 15+20+25+29 MLP: `2.210`
- + refusal layers 15+20+25+29 MLP+attention: `1.900`

So the patch closes part of the refusal loss gap, but remains far from:

- `merge_all_linear`: `0.488`
- `merge_arith_refusal`: `0.317`
- `expert_refusal`: `0.235`

## Generation Metrics

| model | mean acc | worst | arith | polite | refusal |
|---|---:|---:|---:|---:|---:|
| merge_all_linear | 0.861 | 0.583 | 0.583 | 1.000 | 1.000 |
| micro_refusal_l15_l20_l25_l29_mlp | 0.639 | 0.167 | 0.167 | 1.000 | 0.750 |
| micro_refusal_l15_l20_l25_l29_mlp_attn | 0.639 | 0.167 | 0.167 | 1.000 | 0.750 |
| micro_refusal_l20_l25_l29_mlp_attn | 0.639 | 0.167 | 0.167 | 1.000 | 0.750 |
| micro_refusal_l20_l25_l29_mlp | 0.583 | 0.167 | 0.167 | 1.000 | 0.583 |
| micro_refusal_l25_l29_block | 0.583 | 0.167 | 0.167 | 1.000 | 0.583 |
| micro_refusal_l25_l29_mlp | 0.556 | 0.167 | 0.167 | 1.000 | 0.500 |
| merge_arith_polite | 0.389 | 0.000 | 0.167 | 1.000 | 0.000 |
| merge_arith_refusal | 0.389 | 0.000 | 0.167 | 0.000 | 1.000 |
| micro_refusal_l29_mlp | 0.389 | 0.000 | 0.167 | 1.000 | 0.000 |
| micro_refusal_l29_block | 0.361 | 0.000 | 0.083 | 1.000 | 0.000 |
| expert_refusal | 0.333 | 0.000 | 0.000 | 0.000 | 1.000 |
| base | 0.083 | 0.000 | 0.250 | 0.000 | 0.000 |

Single-layer targeted patches do not restore visible refusal generation under
the simple scorer, but multi-layer late patches do:

- `merge_arith_polite` gives confused non-refusal completions.
- `micro_refusal_l29_mlp` and `micro_refusal_l29_block` still give non-refusal
  completions.
- `micro_refusal_l25_l29_mlp` reaches 6/12 refusal generations.
- `micro_refusal_l20_l25_l29_mlp` reaches 7/12 refusal generations.
- `micro_refusal_l25_l29_block` also reaches 7/12 refusal generations.
- `micro_refusal_l15_l20_l25_l29_mlp` reaches 9/12 refusal generations.
- `micro_refusal_l15_l20_l25_l29_mlp_attn` also reaches 9/12 refusal
  generations and has the best targeted refusal loss.
- Attention-only patches at these layers still reach 0/12 refusal generations.
- `merge_all_linear` generates explicit refusal strings such as "I can't help
  with that request."

## Interpretation

This is a real partial mechanism-aware merge win.

What worked:

- The mechanistic target was meaningful: late refusal MLP/block patches improve
  refusal teacher-forced loss in the recipient.
- Multi-layer late MLP patches restore visible refusal on about half of the
  generation prompts; adding layer 15 raises this to 75% in this run while
  preserving polite accuracy and the original arithmetic score.
- Attention-only does not restore refusal, but MLP+attention improves
  teacher-forced refusal loss beyond MLP-only.
- The MLP patch preserves generation behavior on arithmetic and polite better
  than the full block patch.
- The result supports the claim that the late refusal module carries some
  refusal-relevant information.

What did not work:

- A single layer-29 MLP or block is not sufficient to cross the generation
  threshold into explicit refusal behavior.
- The multi-layer micro-merges are still much messier than `merge_all_linear`.
  Example completions often contain refusal-like text but are repetitive or
  semantically awkward.
- Full `merge_all_linear` is still much stronger because it likely transfers a
  distributed refusal mechanism across multiple layers and/or readout pathways.

## Updated Hypothesis

Refusal is late-MLP-heavy and distributed across layers 15/20/25/29 in this
synthetic setup. The layer-29 module is a strong component, but visible behavior
needs a small multi-layer MLP set. Attention appears auxiliary: not sufficient
by itself, but helpful when combined with MLP patches.

The likely remaining missing pieces are:

- final readout/logit effects distributed outside the single block;
- interactions between refusal delta and chat-format/style deltas.
- response-quality cleanup: targeted patches produce refusal-like text, but
  `merge_all_linear` still gives cleaner refusals.

## Next Step

Next, move from hand-picked module patches to a small merge recipe:

- keep arithmetic+polite from `merge_arith_polite`;
- add refusal MLPs at layers 15/20/25/29;
- optionally add refusal attention at the same layers;
- search a small alpha scale for those inserted refusal modules.

This tests whether the mechanism-aware recipe can approach `merge_all_linear`
while preserving arithmetic/polite and using much less of the refusal expert.
