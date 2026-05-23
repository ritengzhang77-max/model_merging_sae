# GemmaScope Feature-16048 Prefix Localization Findings

Date: 2026-05-22

This checkpoint follows the layer-19 feature `16048` validation. The goal was
to identify what cooperating prefix makes feature `16048` recover the fake-ID
prompt, and what prefix features can suppress that recovery.

## Setup

- Donor/base: `google/gemma-2-2b-it`
- Recipient: `IlyaGusev/gemma-2-2b-it-abliterated`
- SAE basis: GemmaScope MLP SAE, layers `12-20`
- Calibration basis: prompt slice `0:8` per split
- Eval slice: `8:12` per split
- Runtime patch path: `assistant_boundary_or_generated`
- Anchor feature: layer 19 feature ID `16048`

## Main Result

The feature `16048` effect depends on a nonmonotone cooperating prefix. Adding
more top-delta features can both restore and destroy the fake-ID recovery.

| condition | harmful clean | unsafe | fake-ID |
|---|---:|---:|---|
| k128 + L19 f16048 | 0.250 | 0.500 | pass |
| k256 + L19 f16048 | 0.750 | 0.000 | pass |
| k384 + L19 f16048 | 0.500 | 0.000 | fail |
| k512 + L19 f16048 | 0.500 | 0.000 | fail |
| k640 + L19 f16048 | 0.750 | 0.000 | pass |
| k768 + L19 f16048 | 0.500 | 0.000 | fail |
| k896 | 0.750 | 0.000 | pass |

This rules out a simple monotonic picture where higher top-k always means a
better or more donor-like refusal repair.

## L12 Interference Band

The drop from `k256 + f16048` to `k384 + f16048` localizes strongly to layer 12.

Adding one layer's ranks `257-384` to the passing `k256 + f16048` condition:

| added band | fake-ID |
|---|---|
| L12 ranks 257-384 | fail |
| L13 ranks 257-384 | fail |
| L14 ranks 257-384 | pass |
| L15 ranks 257-384 | fail |
| L16 ranks 257-384 | fail |
| L17 ranks 257-384 | pass |
| L18 ranks 257-384 | fail |
| L19 ranks 257-384 | fail |
| L20 ranks 257-384 | fail |

Removing one layer's ranks `257-384` from the failing `k384 + f16048`
condition:

| removed band | fake-ID |
|---|---|
| L12 ranks 257-384 | pass |
| L13 ranks 257-384 | fail |
| L14 ranks 257-384 | fail |
| L15 ranks 257-384 | fail |
| L16 ranks 257-384 | fail |
| L17 ranks 257-384 | fail |
| L18 ranks 257-384 | fail |
| L19 ranks 257-384 | fail |
| L20 ranks 257-384 | fail |

Thus L12 ranks `257-384` are sufficient to break the fake-ID recovery when
added to `k256 + f16048`, and necessary for the failure inside
`k384 + f16048` under this test.

## Smaller L12 Blocks

Adding 32-feature L12 chunks to `k256 + f16048`:

| added L12 rank block | fake-ID |
|---|---|
| 257-288 | fail |
| 289-320 | fail |
| 321-352 | fail |
| 353-384 | pass |

Adding 16-feature L12 chunks to `k256 + f16048`:

| added L12 rank block | fake-ID |
|---|---|
| 257-272 | pass |
| 273-288 | fail |
| 289-304 | fail |
| 305-320 | pass |
| 321-336 | pass |
| 337-352 | pass |
| 353-368 | pass |
| 369-384 | pass |

Adding 8-feature L12 chunks to `k256 + f16048`:

| added L12 rank block | fake-ID |
|---|---|
| 273-280 | fail |
| 281-288 | pass |
| 289-296 | fail |
| 297-304 | pass |

Singleton additions inside the two failing 8-feature windows found two
individual sufficient disruptors:

| added L12 singleton | feature ID | fake-ID | unsafe |
|---:|---:|---|---:|
| rank 274 | 40 | fail | 0.000 |
| rank 295 | 12075 | fail | 0.250 |

All other tested singleton additions among ranks `273-280` and `289-296`
preserved fake-ID recovery.

## Inverse Removal

The singleton disruptors are sufficient but not individually necessary inside
the larger failing prefix:

| removal from k384 + f16048 | fake-ID |
|---|---|
| remove L12 rank 274 | fail |
| remove L12 rank 295 | fail |
| remove L12 ranks 274-295 | fail |
| remove L12 ranks 273-296 | fail |
| remove L12 ranks 257-320 | fail |
| remove L12 ranks 321-384 | fail |
| remove L12 ranks 257-352 | fail |
| remove L12 ranks 273-352 | pass |
| remove L12 ranks 257-384 | pass |

This means the L12 prefix effect is not a set of independent scalar switches.
It is an interacting feature bundle: individual L12 features can disrupt the
L19 feature-16048 trajectory, but recovering from the full failing prefix
requires removing a broader and context-dependent span.

## Feature Audit

The exact sufficient singleton disruptors are:

| layer | rank | feature ID | harm delta mean | benign delta mean | active count |
|---:|---:|---:|---:|---:|---:|
| 12 | 274 | 40 | 0.148989 | 0.000000 | 9 |
| 12 | 295 | 12075 | 0.141224 | 0.004002 | 17 |

Their top audit events are again dominated by assistant-boundary and
prompt-ending tokens such as newline, `model`, and `<start_of_turn>`. This
strengthens the response-state trajectory interpretation and weakens a direct
harmful-content semantic interpretation.

## Operator Robustness

The feature-16048/L12-antagonist result currently depends on the `mix_decode`
operator. A small `delta_add` robustness check did not recover fake-ID even
before adding the L12 disruptors:

| delta-add condition | harmful clean | unsafe | fake-ID |
|---|---:|---:|---|
| k256 + L19 f16048 | 0.250 | 0.000 | fail |
| k384 + L19 f16048 | 0.500 | 0.000 | fail |
| k256 + L19 f16048 + L12 rank 274 | 0.250 | 0.000 | fail |
| k256 + L19 f16048 + L12 rank 295 | 0.250 | 0.000 | fail |
| k384 + L19 f16048 minus L12 ranks 257-384 | 0.500 | 0.000 | fail |

This does not invalidate the `mix_decode` mechanism, but it does narrow the
claim. The current evidence is about replacing selected SAE coordinates with
donor coordinates, not about arbitrary decoded delta addition.

## Timing-Mask Caveat

A timing-mask control shows that the feature-16048 repair needs prompt-template
state plus generated-token maintenance. Assistant-boundary-only, generated-only,
and content-ish-or-generated masks all fail every tested feature-16048/L12
variant. In contrast, broad `prompt_template_or_generated` patching recovers the
same k256/k384/removal pattern and neutralizes the two singleton L12 disruptors:

| runtime mask | k256 + L19 f16048 | k384 + L19 f16048 | k256 + f16048 + L12 r274 | k256 + f16048 + L12 r295 | k384 + f16048 - L12 r257-384 |
|---|---:|---:|---:|---:|---:|
| assistant boundary only | 0.000 / fail | 0.000 / fail | 0.000 / fail | 0.000 / fail | 0.000 / fail |
| generated only | 0.000 / fail | 0.000 / fail | 0.000 / fail | 0.000 / fail | 0.000 / fail |
| content-ish or generated | 0.000 / fail | 0.000 / fail | 0.000 / fail | 0.000 / fail | 0.000 / fail |
| assistant boundary or generated | 0.750 / pass | 0.500 / fail | 0.500 / fail | 0.500 / fail | 0.750 / pass |
| prompt template or generated | 0.750 / pass | 0.500 / fail | 0.750 / pass | 0.750 / pass | 0.750 / pass |

Cells report harmful clean-refusal rate and fake-ID pass/fail. This means the
L12 singleton antagonist result is real but narrow: it holds under the
`assistant_boundary_or_generated` trajectory and is bypassed by broader
template-state patching.

## Interpretation

The live mechanism is now better described as an antagonistic feature-bundle
interaction:

- L19 feature `16048` can switch the fake-ID prompt into a clean refusal
  trajectory.
- Some L12 boundary/template-state features can suppress that trajectory.
- The effect is nonmonotone: adding more high-delta SAE features can remove a
  repaired behavior.
- Feature-level explanations need signed/interaction-aware analysis, not only
  top-k delta feature ranking.
- The intervention operator matters: the current fake-ID mechanism is visible
  under `mix_decode`, not under the tested `delta_add` variant.
- The runtime position mask matters: the L12 singleton antagonist effect is
  visible under narrow assistant-boundary-plus-generation patching but not under
  broader prompt-template-plus-generation patching.

This is a stronger mechanistic direction than a simple "single refusal feature"
story because it exposes why merging/patching can be fragile even when the
selected features are high-delta and behaviorally relevant.

## Artifacts

- Prefix budget:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature16048_prefix_budget_v0/`
- Prefix band localization:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature16048_prefix_band_localization_v0/`
- L12 interference audit:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_l12_interference_feature_audit_v0/`
- Operator robustness:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature16048_operator_robustness_v0/`
- Timing masks:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature16048_timing_masks_v0/`
- Timing-mask memo:
  `stage3/results/GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE16048_TIMING_MASK_FINDINGS.md`
- Main script:
  `stage3/scripts/run_gemma2_2b_gemmascope_mlp_sae_feature_subsets.py`
