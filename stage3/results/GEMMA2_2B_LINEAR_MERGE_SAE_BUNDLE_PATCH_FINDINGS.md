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

All patches use `mix_decode` on `assistant_boundary_or_generated` positions.

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

The top10 bundle is partly necessary for alpha-`0.75` family behavior. Replacing
those ten coordinates with low-alpha values drops harmful clean refusal from
`0.875` to `0.500` while preserving benign helpfulness.

## Interpretation

The causal result is asymmetric:

- The transition bundle is not sufficient by itself. Full repair needs a larger
  coordinated state than ten SAE coordinates.
- The transition bundle is partly necessary. Removing/replacing it weakens the
  safe merge substantially.
- This is exactly the kind of feature-bundle story that can explain model
  merging: successful merging may coordinate many features, while sparse local
  edits can identify necessary components without being sufficient alone.

## Artifacts

- Sufficiency run:
  `stage3/results/gemma2_2b_linear_merge_sae_bundle_patch_v0/fake_id_family_a075_to_a025_abog_top_transition/`
- Necessity run:
  `stage3/results/gemma2_2b_linear_merge_sae_bundle_patch_v0/fake_id_family_a025_to_a075_abog_top_transition/`
- Script:
  `stage3/scripts/run_gemma2_2b_linear_merge_sae_bundle_patch.py`
