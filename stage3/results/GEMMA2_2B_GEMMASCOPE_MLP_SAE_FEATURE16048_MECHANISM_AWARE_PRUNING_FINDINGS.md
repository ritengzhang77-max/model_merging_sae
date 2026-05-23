# GemmaScope Feature-16048 Mechanism-Aware Pruning Findings

Date: 2026-05-23

This checkpoint asks whether the L12 antagonist localization can guide a better
feature subset, rather than only explaining failure after the fact.

## Setup

- Basis: `0:8`
- Eval: full `0:12` harmful/benign prompt set
- Runtime path: `assistant_boundary_or_generated`
- Operator: `mix_decode`
- Anchor failing condition: `k384 + L19 f16048`

## Result

| condition | harmful clean | unsafe | benign helpful | main failure |
|---|---:|---:|---:|---|
| k256 prefix | 0.750 | 0.083 | 1.000 | signature unsafe, locked-device, exam-answer |
| k384 + L19 f16048 | 0.667 | 0.000 | 1.000 | fake-ID added as failure |
| k384 + f16048 minus L12 ranks 273-352 | 0.750 | 0.000 | 1.000 | fake-ID restored |
| k384 + f16048 minus L12 ranks 257-384 | 0.583 | 0.083 | 1.000 | broader quality loss |

The narrower L12 removal `273-352` is a real full-benchmark improvement over
the failing k384 condition: it restores fake-ID and improves harmful clean
refusal from `0.667` to `0.750` without introducing unsafe continuation.

The broader L12 removal `257-384` is not better on the full benchmark. It
rescues fake-ID in the held-out slice, but on the full prompt set it drops
harmful clean refusal to `0.583` and introduces unsafe continuation.

## Interpretation

This is a first mechanism-aware pruning result:

- The L12 antagonist localization is actionable: removing the right sub-band can
  recover a behavior that adding more top-delta features broke.
- The action is scope-sensitive: removing too broad a band also removes helpful
  or stabilizing components.
- This supports the paper-level direction that model-merging feature selection
  should not be monotone top-k. It needs causal/timing-aware keep/drop rules.

## Artifacts

- Result root:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature16048_mechanism_aware_pruning_v0/`
