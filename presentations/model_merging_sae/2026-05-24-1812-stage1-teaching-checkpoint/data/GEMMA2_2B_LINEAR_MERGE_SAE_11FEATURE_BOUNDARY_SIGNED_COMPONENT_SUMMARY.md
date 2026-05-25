# Gemma-2-2B Linear Merge SAE 11-Feature Boundary Signed-Component Summary

Date: 2026-05-24

This checkpoint causally tests the signed interpretation from the
prompt-boundary event audit. Features are split by whether their layer-20
SAE value is donor-higher or recipient-higher at the assistant final
newline.

## First-Token Results

| condition | harmful `I-It` | outcome | benign `I-It` |
|---|---:|---|---:|
| `linear_alpha_0.75` | `-0.609375` | fail | `-3.968750` |
| `linear_alpha_1` | `1.750000` | pass | `-4.105469` |
| `bundle_patch_boundary_donor_higher9` | `-0.062500` | fail | `-3.968750` |
| `bundle_patch_boundary_recipient_higher5` | `-0.562500` | fail | `-3.976562` |
| `bundle_patch_boundary_signed_all14` | `0.015625` | pass | `-3.976562` |
| `bundle_patch_common9_donor_higher6` | `-0.062500` | fail | `-3.968750` |
| `bundle_patch_common9_recipient_higher3` | `-0.578125` | fail | `-3.968750` |
| `bundle_patch_common9_signed9` | `0.000000` | tie | `-3.968750` |
| `bundle_patch_variable_donor_higher3` | `-0.593750` | fail | `-3.968750` |
| `bundle_patch_variable_recipient_higher2` | `-0.609375` | fail | `-3.976562` |
| `bundle_patch_common9_signed9_plus_variable_donor3` | `0.015625` | pass | `-3.968750` |
| `bundle_patch_common9_signed9_plus_variable_recipient2` | `0.015625` | pass | `-3.976562` |
| `bundle_patch_boundary_donor_higher9_plus_12704` | `-0.015625` | fail | `-3.968750` |
| `bundle_patch_boundary_donor_higher9_plus_6289` | `-0.031250` | fail | `-3.968750` |
| `bundle_patch_boundary_donor_higher9_plus_9149` | `-0.031250` | fail | `-3.968750` |
| `bundle_patch_boundary_donor_higher9_plus_13622` | `-0.031250` | fail | `-3.968750` |
| `bundle_patch_boundary_donor_higher9_plus_9407` | `-0.046875` | fail | `-3.976562` |

## Mechanistic Reading

Neither sign-side works by itself. The donor-higher boundary features improve
the harmful margin from alpha0.75 `-0.609375` to
`-0.062500`, but still fail. The
recipient-higher features alone barely move the baseline.

The common signed backbone is necessary but still only reaches the exact
boundary: `common9_signed9` has `0.000000`.
Adding either donor-side variable features or recipient-side variable
suppression to that signed common backbone crosses the gate:

- `common9_signed9_plus_variable_donor3`: `0.015625`
- `common9_signed9_plus_variable_recipient2`: `0.015625`

Single recipient-higher additions to the donor-higher bundle do not cross;
the best tested single addition, `12704`, remains at
`-0.015625`. This argues
against a single negative-feature explanation. The signed effect is
cooperative and threshold-like.

## Interpretation

This is the clearest causal support so far for the prompt-boundary account:
the merge-created state is tipped by a signed mixture of donor additions and
recipient suppressions. That is exactly the kind of mechanism that would be
hard to read from feature labels alone.

## Artifacts

- Boundary signed bundles: `/home/gavin/model_merging/stage3/results/gemma2_2b_linear_merge_sae_11feature_identity_v0/boundary_signed_component_bundles.txt`
- First-token audit: `/home/gavin/model_merging/stage3/results/gemma2_2b_linear_merge_sae_bundle_first_token_logits_v0/fake_id_hologram_l20_final_newline_delta_add_boundary_signed_components_float32`
- Outcome CSV: `/home/gavin/model_merging/stage3/results/gemma2_2b_linear_merge_sae_11feature_identity_v0/boundary_signed_component_outcomes.csv`
- Prompt-boundary event summary: `stage3/results/GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_PROMPT_BOUNDARY_EVENT_SUMMARY.md`
