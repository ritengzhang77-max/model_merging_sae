# Gemma-2-2B Linear Merge SAE Feature-Trajectory Findings

Date: 2026-05-23

This checkpoint asks whether the earlier SAE handles track the actual linear
weight-merge transition:

```text
abliterated + alpha * (base - abliterated)
```

Tracked features:

- L19 feature `16048`
- L12 feature `40`
- L12 feature `12075`

## Setup

- Own-generation trajectories, not teacher-forced trajectories.
- Logged the assistant-boundary token at step 0 and the most recent generated
  token thereafter.
- Ran both the original fake-ID prompt plus matched benign prompt, and the
  8 harmful / 8 benign fake-ID family.

## Behavioral Replication

The feature-trajectory runs reproduce the linear merge behavior:

| alpha | family harmful clean | family unsafe | family benign helpful | family over-refusal |
|---:|---:|---:|---:|---:|
| 0.00 | 0.000 | 0.000 | 1.000 | 0.000 |
| 0.25 | 0.125 | 0.375 | 1.000 | 0.000 |
| 0.50 | 0.750 | 0.000 | 1.000 | 0.000 |
| 0.75 | 0.875 | 0.000 | 1.000 | 0.000 |
| 1.00 | 0.875 | 0.000 | 0.875 | 0.125 |

## Feature Result

The earlier L19 `16048` handle does not behave like a simple natural safety
marker along the merge line:

| alpha | harmful generated L19 f16048 mean | harmful generated L19 f16048 max | harmful generated nonzero |
|---:|---:|---:|---:|
| 0.00 | 0.1510 | 12.1562 | 0.030 |
| 0.25 | 0.3996 | 43.5312 | 0.048 |
| 0.50 | 0.4414 | 45.5000 | 0.036 |
| 0.75 | 0.0997 | 10.8594 | 0.018 |
| 1.00 | 0.0767 | 8.5938 | 0.016 |

It peaks near alpha `0.50`, then drops at the stronger safe merges alpha `0.75`
and `1.00`. The top activations are often punctuation/caveat-transition tokens
such as comma, period, `but`, `Here`, or `breakdown`, not direct fake-ID
content tokens. So feature `16048` is better described as a local rescue handle
for certain recipient-like trajectories than as the natural feature that carries
the full safe endpoint.

The L12 features look more aligned with the actual safe-merge transition:

| alpha | harmful generated L12 f40 mean | harmful generated L12 f12075 mean | harmful assistant-boundary f40 mean | harmful assistant-boundary f12075 mean |
|---:|---:|---:|---:|---:|
| 0.00 | 0.0000 | 0.1708 | 0.0000 | 0.0000 |
| 0.25 | 0.0968 | 0.2794 | 0.0000 | 0.0000 |
| 0.50 | 0.1315 | 0.5026 | 0.0000 | 0.2854 |
| 0.75 | 0.1984 | 0.5036 | 0.2446 | 0.3423 |
| 1.00 | 0.2381 | 0.5598 | 0.2651 | 0.5925 |

This is important because those same L12 features were antagonists when added
into the narrow sparse patch trajectory. In natural linear merges, however,
their activation increases in the safe alpha regime. The strongest current
interpretation is context dependence: a feature can be harmful when inserted
into the wrong sparse trajectory and still be part of the natural safe endpoint.

## Interpretation

This changes the mechanistic target:

- L19 `16048` remains causally useful for a local activation-patching repair,
  but it is not the main natural marker of successful linear model merging.
- The L12 features are not simply "bad features." Their role flips with
  trajectory context.
- The more paper-worthy question is now how linear merging coordinates a
  feature bundle across layers/timing so that the same features are safe in one
  trajectory and antagonistic in another.

## Teacher-Forced Check

The fixed-continuation check separates model-alpha effects from generated-token
sequence effects. It evaluates all model alphas on continuations sampled from:

- alpha `0.00` unsafe recipient generations;
- alpha `0.75` safe merge generations.

On fixed safe alpha-`0.75` harmful continuations:

| model alpha | L12 f40 mean | L12 f12075 mean | L19 f16048 mean |
|---:|---:|---:|---:|
| 0.00 | 0.0079 | 0.3333 | 0.2491 |
| 0.25 | 0.0427 | 0.4180 | 0.1273 |
| 0.50 | 0.1080 | 0.4712 | 0.1007 |
| 0.75 | 0.1986 | 0.4958 | 0.0982 |
| 1.00 | 0.2880 | 0.5059 | 0.1001 |

The L12 features increase with model alpha on the same safe harmful
continuations. L19 `16048` does not: it is highest at model alpha `0.00` and
then stays low.

On fixed unsafe alpha-`0.00` harmful continuations, L19 `16048` is high under
safer model weights:

| model alpha | L12 f40 mean | L12 f12075 mean | L19 f16048 mean |
|---:|---:|---:|---:|
| 0.00 | 0.0000 | 0.1720 | 0.1486 |
| 0.25 | 0.0033 | 0.1803 | 0.3648 |
| 0.50 | 0.0068 | 0.1741 | 0.5754 |
| 0.75 | 0.0113 | 0.1845 | 0.6230 |
| 1.00 | 0.0276 | 0.1986 | 0.5760 |

This makes the feature roles clearer:

- L12 features `40` and `12075` are better candidates for natural safe-merge
  trajectory markers.
- L19 `16048` is not a clean safety marker. It can be high when a safer model is
  forced through unsafe recipient-like text, and low on safe merge text.

## Next Step

Use the teacher-forced setup to search for a broader set of natural merge
features around the alpha `0.25` to `0.50` transition. The current hand-picked
features show that L12 is promising and L19 `16048` is not enough.

## Artifacts

- Original fake-ID trace:
  `stage3/results/gemma2_2b_linear_merge_sae_feature_trajectories_v0/original_fake_id_alpha_sweep/`
- Fake-ID family trace:
  `stage3/results/gemma2_2b_linear_merge_sae_feature_trajectories_v0/fake_id_family_alpha_sweep/`
- Teacher-forced feature check:
  `stage3/results/gemma2_2b_linear_merge_sae_teacher_forced_features_v0/fake_id_family_targets_a0_a075/`
- Script:
  `stage3/scripts/analyze_gemma2_2b_linear_merge_sae_feature_trajectories.py`
- Teacher-forced script:
  `stage3/scripts/analyze_gemma2_2b_linear_merge_sae_teacher_forced_features.py`
