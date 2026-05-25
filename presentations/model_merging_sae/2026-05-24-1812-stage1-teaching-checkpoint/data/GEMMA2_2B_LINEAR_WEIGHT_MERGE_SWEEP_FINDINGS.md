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

Original 12 harmful / 12 benign prompt screen:

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

Fake-ID paraphrase family:

| alpha | harmful clean on 8 fake-ID variants | harmful attempt | unsafe | benign helpful | benign over-refusal |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 0.25 | 0.125 | 0.125 | 0.375 | 1.000 | 0.000 |
| 0.50 | 0.750 | 0.750 | 0.000 | 1.000 | 0.000 |
| 0.75 | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| 1.00 | 0.875 | 0.875 | 0.000 | 0.875 | 0.125 |

The family check supports the same alpha threshold. Alpha `0.75` matches the
base donor's 7/8 fake-ID-family refusal rate while keeping 8/8 benign
helpfulness and zero benign over-refusal. This is stronger than the local
feature-16048/L12 pruning branch, which topped out at 6/8 on the same family.

Expanded fake-ID family alpha-`0.75` check:

| model | harmful clean on 24 fake-ID variants | unsafe | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|
| linear alpha 0.75 | 0.875 | 0.000 | 0.958 | 0.042 |

The expanded family keeps the same harmful clean-refusal rate as the 8-prompt
fake-ID family, but exposes one benign over-refusal on a temporary-legal-ID
prompt. This is the baseline for the later feature-subtract top10/random
replication.

Long-generation audit:

| alpha | max new tokens | attempted refusal | attempted and no unsafe | unsafe continuation | benign not-over-refusal |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 160 | 0.625 | 0.583 | 0.083 | 1.000 |
| 0.75 | 160 | 0.875 | 0.833 | 0.042 | 0.958 |
| 1.00 | 160 | 0.958 | 0.958 | 0.000 | 0.917 |

The `harmful clean` and `benign helpful` columns are not comparable in this
long run because the current heuristic marks long responses as non-clean. The
important result is the tradeoff: alpha `1.00` is safest on harmful fake-ID
prompts under longer decoding, while alpha `0.75` preserves more benign
helpfulness but has one delayed unsafe continuation on the hologram/lamination
prompt.

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
  alpha `0.75` remains robust beyond this one paraphrase family.
- Linear interpolation is only one merge recipe; TIES/DARE-style pruning should
  be tested after the linear bridge is understood.

## Artifacts

- Result root:
  `stage3/results/gemma2_2b_linear_weight_merge_sweep_v0/full_0_12/`
- Fake-ID family root:
  `stage3/results/gemma2_2b_linear_weight_merge_sweep_v0/fake_id_family/`
- Expanded fake-ID alpha-0.75 root:
  `stage3/results/gemma2_2b_linear_weight_merge_sweep_v0/fake_id_family_v1_alpha075_baseline/`
- Expanded fake-ID alpha-0.75 max-160 audit:
  `stage3/results/gemma2_2b_linear_weight_merge_sweep_v0/fake_id_family_v1_alpha075_max160_audit/`
- Expanded fake-ID alpha-0.50/1.00 max-160 audit:
  `stage3/results/gemma2_2b_linear_weight_merge_sweep_v0/fake_id_family_v1_alpha05_1_max160_audit/`
- Hologram alpha-0.75 max-160 probe:
  `stage3/results/gemma2_2b_linear_weight_merge_sweep_v0/fake_id_hologram_probe_alpha075_max160/`
- Script:
  `stage3/scripts/run_gemma2_2b_linear_weight_merge_sweep.py`
