# GemmaScope Feature-16048 Timing-Mask Findings

Date: 2026-05-22

This checkpoint tests where the feature-16048 repair must be applied. It keeps
the same donor/recipient, basis `0:8`, eval `8:12`, layers `12-20`, all-token
feature selection, and `mix_decode` intervention, while changing only the
runtime patch-position mask.

## Main Result

The fake-ID repair is not produced by assistant-boundary-only patching and not
produced by generated-token-only patching. It requires prompt-template state
plus generated-token maintenance.

| runtime mask | k256 + L19 f16048 | k384 + L19 f16048 | k256 + f16048 + L12 r274 | k256 + f16048 + L12 r295 | k384 + f16048 - L12 r257-384 |
|---|---:|---:|---:|---:|---:|
| assistant boundary only | 0.000 / fail | 0.000 / fail | 0.000 / fail | 0.000 / fail | 0.000 / fail |
| generated only | 0.000 / fail | 0.000 / fail | 0.000 / fail | 0.000 / fail | 0.000 / fail |
| content-ish or generated | 0.000 / fail | 0.000 / fail | 0.000 / fail | 0.000 / fail | 0.000 / fail |
| assistant boundary or generated | 0.750 / pass | 0.500 / fail | 0.500 / fail | 0.500 / fail | 0.750 / pass |
| prompt template or generated | 0.750 / pass | 0.500 / fail | 0.750 / pass | 0.750 / pass | 0.750 / pass |

Cells report harmful clean-refusal rate and fake-ID pass/fail.

## Interpretation

This strengthens the response-state trajectory interpretation:

- generated-token maintenance is not enough if the prompt/template state is not
  also patched;
- assistant-boundary-only patching is not enough if generated-token state is
  not maintained;
- patching harmful content tokens plus generated tokens does not repair the
  behavior;
- the successful control is prompt-template-like state plus generated-token
  maintenance.

The surprising part is that broad `prompt_template_or_generated` patching
removes the singleton L12 antagonist effect seen under the narrower
`assistant_boundary_or_generated` mask. Under the narrow mask, L12 rank `274`
feature ID `40` and L12 rank `295` feature ID `12075` are individually
sufficient to break `k256 + L19 f16048`. Under the broader prompt-template
mask, the same singleton additions preserve fake-ID recovery.

So the current antagonist claim must be stated as mask-dependent: those L12
features disrupt the narrow assistant-boundary trajectory, but broader
template-state patching can compensate for or bypass that disruption.

## Next Test

The next decisive experiment is feature-specific timing:

- patch the broad k256 prefix on prompt-template and generated tokens;
- patch L19 feature `16048` only at prompt-template, only at generated, or both;
- patch L12 disruptors only at prompt-template, only at generated, or both.

That will separate whether feature `16048` mainly initializes the refusal
trajectory, maintains it during generation, or needs both timing sites.

## Artifacts

- Timing-mask root:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature16048_timing_masks_v0/`
- Narrow-mask comparison sources:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature16048_prefix_budget_v0/`
  and
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature16048_prefix_band_localization_v0/`
- Main script:
  `stage3/scripts/run_gemma2_2b_gemmascope_mlp_sae_feature_subsets.py`
