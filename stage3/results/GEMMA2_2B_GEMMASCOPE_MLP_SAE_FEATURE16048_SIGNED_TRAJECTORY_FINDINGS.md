# GemmaScope Feature-16048 Signed-Trajectory Findings

Date: 2026-05-22

This checkpoint logs donor, recipient, patched, and signed
donor-minus-recipient SAE feature activations during greedy rollout. It records
selected prompt/template positions at step 0 and the autoregressive last-token
position at each generation step.

## L19 Feature-16048 Trajectory

The clean feature-16048 timing case uses basis `0:4`, k896, fake-ID prompt:
k896 fails, while adding feature `16048` on generated tokens passes.

| condition | fake-ID | L19 f16048 mean signed delta on generated positions | patched fraction |
|---|---|---:|---:|
| k896 prefix only | fail | 0.785 | 0.000 |
| k896 + f16048 on generated tokens | pass | 0.607 | 1.000 |
| k896 + f16048 on boundary or generated | pass | 0.607 | 1.000 |

Feature `16048` has essentially zero signed delta on prompt-template/boundary
positions in this diagnostic, and positive donor-minus-recipient delta on
generated-token positions. This matches the causal timing test: feature `16048`
is a generated-token trajectory feature, not a prompt-boundary initialization
feature.

## L12 Antagonist Trajectories

The L12 antagonist timing case uses basis `0:8`, k256, fake-ID prompt. k256
already passes; adding specific L12 singleton features can break it.

| condition | fake-ID | L12 f40 generated signed delta | L12 f12075 generated signed delta | L19 f16048 generated signed delta |
|---|---|---:|---:|---:|
| k256 prefix only | pass | 0.246 | 0.118 | 0.701 |
| k256 + f16048 | pass | 0.246 | 0.118 | 0.701 |
| + L12 rank 274 / f40 on generated tokens | fail | 0.537 | 0.079 | 1.299 |
| + L12 rank 274 / f40 on boundary or generated | fail | 0.537 | 0.079 | 1.299 |
| + L12 rank 295 / f12075 at assistant boundary | fail | 0.537 | 0.079 | 1.290 |
| + L12 rank 295 / f12075 on boundary or generated | fail | 0.212 | 0.256 | 0.747 |

The important point is not just timing; it is sign. The antagonistic L12
features are also donor-greater-than-recipient features. Their signed deltas are
positive, yet patching them can break the repaired behavior. This directly
explains why unsigned top-delta ranking can be nonmonotone: top donor-shifted
features can include helpful trajectory features, redundant features, and
timing-specific antagonists.

The failure cases also do not simply remove feature `16048`. In the L12 rank
274 generated-token failure, L19 feature `16048` has an even larger positive
generated-token signed delta than in the passing k256 condition. The antagonist
can break the trajectory despite strong donor-like f16048 activity.

## Current Mechanistic Update

The model-merging repair looks less like selecting a sparse set of universally
good donor features and more like selecting a timed trajectory:

- L19 feature `16048` helps on generated tokens in a prefix where it is needed.
- L12 feature `40` is a donor-high generated-token feature that can disrupt the
  fake-ID refusal path.
- L12 feature `12075` is a donor-high assistant-boundary feature that can
  disrupt the same path.
- Sign alone is insufficient: a positive donor-recipient feature delta can be
  helpful or antagonistic depending on layer, timing, and prefix context.

## Caveats

- This checkpoint is for the fake-ID prompt only.
- Rows summarize selected prompt/template positions at step 0 and the
  autoregressive last-token state during generation.
- The generation score is a local heuristic score; prompt-level manual audit is
  still needed before paper-level claims.

## Artifacts

- Signed trajectory root:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature16048_signed_trajectories_v0/`
- Trajectory script:
  `stage3/scripts/export_gemma2_2b_gemmascope_mlp_sae_feature_trajectories.py`
