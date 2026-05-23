# Gemma-2-2B Linear Merge SAE Bundle Patch Findings

Date: 2026-05-23

This checkpoint causally tests the transition features discovered from fixed
safe alpha-`0.75` fake-ID family continuations.

Bundles:

- `f16048`: L19 feature `16048`
- `top1`: L17 feature `4342`
- `top2`: L17 features `4342`, `16011`
- `top5`: L17 `4342`, L17 `16011`, L16 `16332`, L18 `10415`, L18 `11127`
- `top10`: top ten transition-search features

All patches use `mix_decode` on `assistant_boundary_or_generated` positions
unless stated otherwise.

## Sufficiency: High Alpha Into Low Alpha

Donor: alpha `0.75`. Recipient: alpha `0.25`.

| condition | harmful clean | unsafe | benign helpful | conclusion |
|---|---:|---:|---:|---|
| alpha 0.25 baseline | 0.125 | 0.375 | 1.000 | weak/unsafe |
| alpha 0.75 baseline | 0.875 | 0.000 | 1.000 | target |
| f16048 patched into 0.25 | 0.000 | 0.625 | 1.000 | worse |
| top1 patched into 0.25 | 0.125 | 0.250 | 1.000 | less unsafe only |
| top2 patched into 0.25 | 0.125 | 0.250 | 1.000 | less unsafe only |
| top5 patched into 0.25 | 0.125 | 0.375 | 1.000 | no repair |
| top10 patched into 0.25 | 0.125 | 0.125 | 1.000 | unsafe reduced, not repaired |

The discovered bundle is not sufficient to transfer the full alpha-`0.75`
behavior into alpha `0.25`. The top10 bundle does reduce unsafe continuation
from `0.375` to `0.125`, but it does not increase clean refusal above `0.125`.
L19 `16048` is actively bad in this setting.

## Necessity: Low Alpha Into High Alpha

Donor: alpha `0.25`. Recipient: alpha `0.75`.

| condition | harmful clean | unsafe | benign helpful | conclusion |
|---|---:|---:|---:|---|
| alpha 0.75 baseline | 0.875 | 0.000 | 1.000 | target |
| f16048 low-alpha patch into 0.75 | 0.750 | 0.000 | 1.000 | mild degradation |
| top1 low-alpha patch into 0.75 | 0.750 | 0.125 | 1.000 | degradation |
| top2 low-alpha patch into 0.75 | 0.750 | 0.125 | 1.000 | degradation |
| top5 low-alpha patch into 0.75 | 0.750 | 0.125 | 1.000 | degradation |
| top10 low-alpha patch into 0.75 | 0.500 | 0.000 | 1.000 | strong degradation |

Before controls, this looked like a partly necessary bundle: replacing those
ten coordinates with low-alpha values drops harmful clean refusal from `0.875`
to `0.500` while preserving benign helpfulness.

## Random Same-Layer Controls

The necessity claim does not survive a first matched random control.

Three random bundles with the same layer-count pattern as top10
(`L14 x1`, `L15 x1`, `L16 x1`, `L17 x2`, `L18 x4`, `L20 x1`) were patched from
alpha `0.25` into alpha `0.75` under the same `mix_decode` operator and
`assistant_boundary_or_generated` timing.

| condition | harmful clean | unsafe | benign helpful | conclusion |
|---|---:|---:|---:|---|
| random1 low-alpha patch into 0.75 | 0.500 | 0.125 | 1.000 | same clean drop as top10 |
| random2 low-alpha patch into 0.75 | 0.500 | 0.125 | 1.000 | same clean drop as top10 |
| random3 low-alpha patch into 0.75 | 0.500 | 0.125 | 1.000 | same clean drop as top10 |

This weakens the causal specificity claim. The top10 bundle may still be
mechanistically meaningful as a natural transition correlate and qualitative
legal-consequence/refusal-rationale feature set, but this `mix_decode`
necessity test is not enough to show that the selected top10 features are
specifically necessary. The degradation can be produced by arbitrary same-layer
ten-feature coordinate replacement, likely because full decode/re-encode at
generated positions creates a generic trajectory disruption.

## Interpretation

The causal result is asymmetric but currently operator-sensitive:

- The transition bundle is not sufficient by itself. Full repair needs a larger
  coordinated state than ten SAE coordinates.
- The transition bundle is not yet validated as specifically necessary because
  same-layer random bundles also weaken the safe merge under `mix_decode`.
- This is exactly the kind of feature-bundle story that can explain model
  merging, but the causal operator needs a cleaner control before we can claim
  selected-feature necessity.

Next decisive test: rerun top10 and random controls with a decoded delta-add
operator, which should reduce generic reconstruction disruption compared with
full coordinate replacement.

## Artifacts

- Sufficiency run:
  `stage3/results/gemma2_2b_linear_merge_sae_bundle_patch_v0/fake_id_family_a075_to_a025_abog_top_transition/`
- Necessity run:
  `stage3/results/gemma2_2b_linear_merge_sae_bundle_patch_v0/fake_id_family_a025_to_a075_abog_top_transition/`
- Necessity random controls:
  `stage3/results/gemma2_2b_linear_merge_sae_bundle_patch_v0/fake_id_family_a025_to_a075_abog_random_controls/`
- Script:
  `stage3/scripts/run_gemma2_2b_linear_merge_sae_bundle_patch.py`
