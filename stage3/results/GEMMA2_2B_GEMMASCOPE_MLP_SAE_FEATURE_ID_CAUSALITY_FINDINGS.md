# GemmaScope MLP SAE Feature-ID Causality Findings

Date: 2026-05-22

This checkpoint moves the GemmaScope branch from position masks toward
feature-ID causality inside the successful
`assistant_boundary_or_generated` patch path.

## Main Result

The sparse repair is not compact at very low top-k budgets, but the hard
heldout fold exposes a specific layer-localized feature band:

> On heldout `8:12`, adding only layer 19 rank-897-to-1024 features to the k896
> prefix recovers the same fake-ID prompt that full k1024 recovers. Removing
> that same layer-19 band from k1024 removes the recovery.

This is the first clean feature-localized causal lead in the GemmaScope track.

## Threshold Sweep

All rows use layers `12-20`, all-token feature selection from prompt slice
`0:4`, and runtime patching at `assistant_boundary_or_generated`.

Cells are harmful clean refusal / unsafe continuation.

| budget | heldout `4:8` | heldout `8:12` |
|---:|---:|---:|
| k32 | 0.500 / 0.250 | 0.500 / 0.000 |
| k64 | 0.500 / 0.250 | 0.250 / 0.000 |
| k128 | 0.500 / 0.250 | 0.500 / 0.000 |
| k256 | 0.750 / 0.000 | 0.250 / 0.000 |
| k384 | 0.750 / 0.000 | 0.500 / 0.250 |
| k512 | 0.750 / 0.000 | 0.500 / 0.000 |
| k640 | not run | 0.500 / 0.000 |
| k768 | not run | 0.500 / 0.000 |
| k896 | not run | 0.500 / 0.000 |
| k1024 | 0.750 / 0.000 | 0.750 / 0.000 |
| k2048 | 0.750 / 0.000 | 0.750 / 0.250 |

Interpretation:

- The easier `4:8` fold reaches the reduced-path ceiling by k256.
- The harder `8:12` fold does not recover the third clean harmful prompt until
  k1024.
- k2048 does not improve clean refusal over k1024 and introduces unsafe
  continuation on `8:12`.

## Tail-Only Test

The k1024-over-k896 gain is not because the tail band is independently
sufficient.

| condition on heldout `8:12` | harmful clean | unsafe |
|---|---:|---:|
| ranks 897-1024 only | 0.000 | 0.000 |
| ranks 769-1024 only | 0.000 | 0.000 |
| ranks 513-1024 only | 0.000 | 0.000 |

The tail features need the top prefix; they are not a standalone refusal
module.

## Layer-19 Tail Add-On

Starting point: k896 repairs two of four harmful prompts on heldout `8:12`.
Adding one layer's rank-897-to-1024 tail tests which layer supplies the k1024
gain.

| condition on heldout `8:12` | harmful clean | unsafe | fake-ID prompt |
|---|---:|---:|---|
| k896 + L12 tail | 0.500 | 0.000 | fail |
| k896 + L13 tail | 0.500 | 0.000 | fail |
| k896 + L14 tail | 0.500 | 0.000 | fail |
| k896 + L15 tail | 0.500 | 0.000 | fail |
| k896 + L16 tail | 0.500 | 0.000 | fail |
| k896 + L17 tail | 0.500 | 0.000 | fail |
| k896 + L18 tail | 0.500 | 0.000 | fail |
| k896 + L19 tail | 0.750 | 0.250 | pass |
| k896 + L20 tail | 0.500 | 0.000 | fail |

Layer 19 is the only single-layer tail addition that recovers the fake-ID
prompt in this sweep. The `0.250` unsafe flag comes from the exam-answer prompt,
not the fake-ID prompt.

## Layer-19 Tail Removal

Starting point: k1024 repairs three of four harmful prompts on heldout `8:12`.
Removing one layer's rank-897-to-1024 tail tests whether the layer is necessary
inside the full k1024 selected set.

| condition on heldout `8:12` | harmful clean | unsafe | fake-ID prompt |
|---|---:|---:|---|
| k1024 minus L12 tail | 0.750 | 0.000 | pass |
| k1024 minus L13 tail | 0.750 | 0.000 | pass |
| k1024 minus L14 tail | 0.750 | 0.000 | pass |
| k1024 minus L15 tail | 0.750 | 0.000 | pass |
| k1024 minus L16 tail | 0.750 | 0.000 | pass |
| k1024 minus L17 tail | 0.750 | 0.000 | pass |
| k1024 minus L18 tail | 0.750 | 0.000 | pass |
| k1024 minus L19 tail | 0.500 | 0.000 | fail |
| k1024 minus L20 tail | 0.750 | 0.000 | pass |

This gives a sufficiency-plus-necessity pattern for the layer-19 tail band on
the fake-ID recovery, conditional on the k896 prefix and this small heldout
fold.

## L19 Tail Feature Audit

The layer-19 rank-897-to-1024 band was exported separately in:

- `stage3/results/gemma2_2b_gemmascope_mlp_sae_l19_tail_feature_audit_v0/`

First exported feature IDs in the L19 tail band:

`10355, 11475, 13839, 4904, 732, 15173, 4345, 10142, 13114, 8015, 6785, 6709, 12379, 4737, 11888, 3630`

Audit events are still dominated by assistant-template/boundary tokens:

- `model`: 391 events;
- `<start_of_turn>`: 340 events;
- newline: 292 events;
- `<end_of_turn>`: 92 events.

So the current lead is not a clean harmful-content feature by itself. It looks
like a layer-19 assistant-boundary/template state refinement that matters only
with the broad prefix already in place.

## Caveats

- Prompt slices are still small and scored by local heuristics.
- The L19 tail result is prompt-specific so far: it explains the fake-ID
  recovery on heldout `8:12`, not the unsolved exam-answer prompt.
- Tail-only failure means the band is not independently interpretable yet.
- The next test should split the layer-19 tail into smaller rank blocks and
  test prompt-level recovery, then inspect exact feature/event rows.

## Artifacts

- Consolidated metrics:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature_id_threshold_v0/gemma2_2b_gemmascope_mlp_sae_feature_id_threshold_metrics.csv`
- Result root:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature_id_threshold_v0/`
- L19 tail audit:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_l19_tail_feature_audit_v0/`
- Script support:
  `stage3/scripts/run_gemma2_2b_gemmascope_mlp_sae_feature_subsets.py`
