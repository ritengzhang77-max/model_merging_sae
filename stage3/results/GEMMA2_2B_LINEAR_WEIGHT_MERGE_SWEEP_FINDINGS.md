# Gemma-2-2B Linear Weight-Merge Sweep Findings

Date: 2026-05-23

This checkpoint connects the SAE patching work back to actual model merging.
Instead of intervening on activations, it evaluates models along the parameter
line:

```text
abliterated + alpha * (base - abliterated)
```

## Setup

- Base donor: `google/gemma-2-2b-it`
- Recipient: `IlyaGusev/gemma-2-2b-it-abliterated`
- Evaluation: 12 harmful prompts and 12 benign prompts from the Stage 0 screen
- Decoding: greedy, `max_new_tokens=64`

## Result

| alpha | harmful clean | harmful attempt | unsafe | benign helpful | benign over-refusal |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.000 | 0.000 | 0.083 | 1.000 | 0.000 |
| 0.25 | 0.000 | 0.083 | 0.167 | 1.000 | 0.000 |
| 0.50 | 0.667 | 0.667 | 0.083 | 1.000 | 0.000 |
| 0.75 | 0.917 | 0.917 | 0.000 | 1.000 | 0.000 |
| 1.00 | 0.917 | 0.917 | 0.000 | 0.917 | 0.083 |

The behavior transition is sharp. Alpha `0.25` mostly remains unsafe/non-refusal,
while alpha `0.50` recovers most refusal behavior. Alpha `0.75` matches the base
donor's harmful clean-refusal rate on this screen while preserving all benign
helpfulness and avoiding the base model's one benign over-refusal.

## Interpretation

This is an actual parameter-space model-merging result, not only a causal patch.
It supports three useful claims:

- The base-to-abliterated safety difference is mergeable by ordinary linear
  interpolation.
- On this small screen, the best intermediate merge can be better than either
  endpoint under the project metric: alpha `0.75` keeps base-like refusal but
  recipient-like benign helpfulness.
- The SAE trajectory results now have a concrete merge curve to explain. The
  next mechanistic question is whether features such as L19 `16048` and the L12
  keep/drop bundles change around the same alpha transition.

## Caveats

- This is a small deterministic screen with heuristic safety scoring.
- The fake-ID family and stricter/manual auditing still need to check whether
  alpha `0.75` is robust outside the original prompt set.
- Linear interpolation is only one merge recipe; TIES/DARE-style pruning should
  be tested after the linear bridge is understood.

## Artifacts

- Result root:
  `stage3/results/gemma2_2b_linear_weight_merge_sweep_v0/full_0_12/`
- Script:
  `stage3/scripts/run_gemma2_2b_linear_weight_merge_sweep.py`
