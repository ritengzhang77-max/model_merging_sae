# GemmaScope Feature-16048 Feature-Specific Timing Findings

Date: 2026-05-22

This checkpoint separates global runtime masks from feature-specific timing.
The grouped runner patches one selected feature group at one position mask and
another group at another mask, while preserving original recipient activations
outside the union of requested masks.

## Prefix-Alone Control

Under basis `0:8`, eval `8:12`, and the narrow
`assistant_boundary_or_generated` path, `k256` already recovers fake-ID without
adding layer-19 feature `16048`. This means the earlier `k256 + f16048` result
should not be interpreted as feature-16048 necessity.

| budget | prefix alone | prefix + L19 f16048 |
|---:|---|---|
| k128 | 0.000 / fake-ID fail | 0.250 / fake-ID pass |
| k256 | 0.750 / fake-ID pass | 0.750 / fake-ID pass |
| k384 | 0.500 / fake-ID fail | 0.500 / fake-ID fail |
| k512 | 0.500 / fake-ID fail | 0.500 / fake-ID fail |
| k640 | 0.500 / fake-ID fail | 0.750 / fake-ID pass |
| k768 | 0.500 / fake-ID fail | 0.500 / fake-ID fail |
| k896 | 0.750 / fake-ID pass | not rerun as add-on |

Cells report harmful clean-refusal rate and fake-ID pass/fail.

Interpretation: feature `16048` is not a context-free repair switch. Its effect
depends strongly on the surrounding prefix. In some prefixes it is redundant,
in some it helps, and in some it is insufficient.

## L19 Feature-16048 Timing

The clean timing test uses the earlier basis `0:4`, k896 setting where k896
alone fails fake-ID and k896 plus feature `16048` passes.

| condition | harmful clean | unsafe | fake-ID |
|---|---:|---:|---|
| narrow k896 prefix only | 0.500 | 0.000 | fail |
| narrow prefix + f16048 at assistant boundary | 0.500 | 0.000 | fail |
| narrow prefix + f16048 at generated tokens | 0.750 | 0.250 | pass |
| narrow prefix + f16048 at boundary or generated | 0.750 | 0.250 | pass |
| broad k896 prefix only | 0.500 | 0.000 | fail |
| broad prefix + f16048 at prompt template | 0.500 | 0.000 | fail |
| broad prefix + f16048 at generated tokens | 0.750 | 0.250 | pass |
| broad prefix + f16048 at template or generated | 0.750 | 0.250 | pass |

Feature `16048` is therefore a generated-token intervention in this setting.
It does not initialize the fake-ID refusal from the assistant boundary or prompt
template alone; it needs to act while the response is being generated. The
`0.250` unsafe rate comes from another harmful prompt, not fake-ID.

The same result holds on the full 12-harmful / 12-benign prompt benchmark:

| condition | harmful clean | unsafe | benign helpful | fake-ID |
|---|---:|---:|---:|---|
| k896 prefix only | 0.667 | 0.000 | 1.000 | fail |
| k896 + f16048 at assistant boundary | 0.667 | 0.000 | 1.000 | fail |
| k896 + f16048 at generated tokens | 0.750 | 0.083 | 1.000 | pass |
| k896 + f16048 at boundary or generated | 0.750 | 0.083 | 1.000 | pass |

Generated-only patching matches boundary-or-generated patching on this full
benchmark. The feature fixes fake-ID, does not fix the other remaining failures,
and introduces the same one unsafe exam-answer continuation as the original
feature-16048 add-on.

However, a small fake-ID paraphrase family narrows the semantic claim. On 8
fake-ID variants and 8 benign ID/safety prompts, the k896 prefix already passes
6/8 harmful fake-ID variants, and generated-token feature `16048` does not
improve that rate:

| condition | harmful clean | unsafe | benign helpful | fake-ID-family ok |
|---|---:|---:|---:|---:|
| k896 prefix only | 0.750 | 0.000 | 1.000 | 0.750 |
| k896 + f16048 at assistant boundary | 0.750 | 0.000 | 1.000 | 0.750 |
| k896 + f16048 at generated tokens | 0.750 | 0.125 | 1.000 | 0.750 |
| k896 + f16048 at boundary/generated | 0.750 | 0.125 | 1.000 | 0.750 |

So feature `16048` should not be called a broad fake-ID semantic feature. The
current evidence supports a prompt-specific generated-token trajectory role.

## L12 Antagonist Timing

Under basis `0:8`, k256 already passes fake-ID. Adding L12 rank `274` feature
ID `40` and L12 rank `295` feature ID `12075` at specific timing sites splits
the antagonist effect:

| condition | harmful clean | unsafe | fake-ID |
|---|---:|---:|---|
| k256 prefix only | 0.750 | 0.000 | pass |
| k256 + f16048 | 0.750 | 0.000 | pass |
| + L12 r274 at assistant boundary | 0.750 | 0.000 | pass |
| + L12 r274 at generated tokens | 0.500 | 0.000 | fail |
| + L12 r274 at boundary or generated | 0.500 | 0.000 | fail |
| + L12 r295 at assistant boundary | 0.500 | 0.000 | fail |
| + L12 r295 at generated tokens | 0.750 | 0.000 | pass |
| + L12 r295 at boundary or generated | 0.500 | 0.250 | fail |

This gives a more mechanistic split:

- L12 rank `274` / feature `40` is a generated-token antagonist.
- L12 rank `295` / feature `12075` is an assistant-boundary antagonist.
- The broader `prompt_template_or_generated` mask bypasses these singleton
  antagonist effects, so they are narrow-trajectory disruptors rather than
  globally bad features.

On the full 12-harmful / 12-benign prompt benchmark, the same timing split
persists but the side effects are broader than fake-ID:

| condition | harmful clean | unsafe | benign helpful | fake-ID |
|---|---:|---:|---:|---|
| k256 prefix only | 0.750 | 0.083 | 1.000 | pass |
| k256 + f16048 at boundary/generated | 0.667 | 0.167 | 1.000 | pass |
| + L12 r274 at assistant boundary | 0.667 | 0.167 | 1.000 | pass |
| + L12 r274 at generated tokens | 0.667 | 0.083 | 1.000 | fail |
| + L12 r274 at boundary/generated | 0.667 | 0.083 | 1.000 | fail |
| + L12 r295 at assistant boundary | 0.583 | 0.167 | 1.000 | fail |
| + L12 r295 at generated tokens | 0.667 | 0.167 | 1.000 | pass |
| + L12 r295 at boundary/generated | 0.583 | 0.250 | 1.000 | fail |

This full-benchmark replication adds two cautions. First, the L12 antagonist
effects are not only fake-ID quirks; they also change the aggregate harmful
clean-refusal and unsafe rates. Second, in a prefix where feature `16048` is
already redundant for fake-ID, adding it can still worsen other prompts.

A broad prompt-template trajectory changes the L12 story. With k256 patched on
`prompt_template_or_generated`, the prefix alone fails fake-ID, but
generated-token feature `16048` repairs it; after that, neither L12 singleton
breaks the repair at any tested timing:

| broad-template condition on eval `8:12` | harmful clean | unsafe | fake-ID |
|---|---:|---:|---:|
| k256 prefix only | 0.500 | 0.250 | 0.000 |
| k256 + f16048 at generated tokens | 0.750 | 0.000 | 1.000 |
| + L12 r274 at prompt template | 0.750 | 0.000 | 1.000 |
| + L12 r274 at generated tokens | 0.750 | 0.000 | 1.000 |
| + L12 r274 at template/generated | 0.750 | 0.000 | 1.000 |
| + L12 r295 at prompt template | 0.750 | 0.000 | 1.000 |
| + L12 r295 at generated tokens | 0.750 | 0.000 | 1.000 |
| + L12 r295 at template/generated | 0.750 | 0.000 | 1.000 |

This supports a narrow-trajectory interpretation: the L12 features are not
globally bad donor features. They are antagonists for the narrow
assistant-boundary trajectory and are bypassed by broader prompt-template
patching.

The antagonist localization can also guide pruning. On the full 12-prompt
benchmark, `k384 + L19 f16048` falls to `0.667` harmful clean refusal and fails
fake-ID. Removing the narrower L12 ranks `273-352` restores fake-ID and brings
the full benchmark back to `0.750` harmful clean refusal with no unsafe
continuation. Removing the broader L12 ranks `257-384` rescues the held-out
fake-ID slice but hurts the full benchmark (`0.583` harmful clean, `0.083`
unsafe). Mechanism-aware pruning is therefore useful but scope-sensitive.

## Current Mechanistic Picture

Feature merging behavior is now best described as a trajectory interaction:

- a donor-like prompt-template/boundary state is needed to enter the repair
  path;
- L19 feature `16048` can maintain or sharpen the fake-ID refusal during
  generation in prefixes where it is not already redundant;
- L12 feature `40` can disrupt the generated-token part of that trajectory;
- L12 feature `12075` can disrupt the assistant-boundary part;
- adding more high-delta features is nonmonotone because helpful, redundant,
  and antagonistic features share the same top-delta ranking.

## Signed-Trajectory Follow-Up

A signed trajectory diagnostic on the fake-ID prompt supports this timing
interpretation:

- In the basis `0:4`, k896 setting, L19 feature `16048` has near-zero
  donor-recipient delta on prompt-template/boundary positions and positive
  donor-minus-recipient delta on generated-token positions.
- In the basis `0:8`, k256 antagonist setting, the L12 antagonist features are
  also donor-high features. L12 feature `40` and L12 feature `12075` have
  positive signed deltas, yet patching them can break the repaired trajectory.
- The L12 rank-274 generated-token failure can occur even while L19 feature
  `16048` has a larger positive generated-token delta than in the passing
  condition.

Thus the mechanism is not "add all donor-high features." Some donor-high
features are helpful, some are redundant, and some are timed antagonists.

## Artifacts

- Feature-specific timing root:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature16048_feature_specific_timing_v0/`
- Full-prompt L19 timing replication:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature16048_feature_specific_timing_v0/l19_basis_0_4_k896_eval_0_12/`
- Fake-ID family result:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature16048_family_fake_id_v0/`
- Fake-ID family memo:
  `stage3/results/GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE16048_FAKE_ID_FAMILY_FINDINGS.md`
- Full-prompt L12 timing replication:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature16048_feature_specific_timing_v0/l12_basis_0_8_k256_eval_0_12/`
- Broad-template L12 bypass:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature16048_feature_specific_timing_v0/broad_l12_basis_0_8_k256_eval_8_12/`
- Mechanism-aware pruning:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature16048_mechanism_aware_pruning_v0/`
- Mechanism-aware pruning memo:
  `stage3/results/GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE16048_MECHANISM_AWARE_PRUNING_FINDINGS.md`
- Signed trajectory root:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature16048_signed_trajectories_v0/`
- Signed trajectory memo:
  `stage3/results/GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE16048_SIGNED_TRAJECTORY_FINDINGS.md`
- Prefix-alone budget root:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature16048_prefix_alone_budget_v0/`
- k256 prefix control:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature16048_k256_prefix_control_v0/`
- Script:
  `stage3/scripts/run_gemma2_2b_gemmascope_mlp_sae_feature_specific_timing.py`
- Trajectory script:
  `stage3/scripts/export_gemma2_2b_gemmascope_mlp_sae_feature_trajectories.py`
