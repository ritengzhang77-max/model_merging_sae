# GemmaScope MLP SAE Random-Seed Control Findings

Date: 2026-05-22

This checkpoint repeats matched random active-feature controls across five
random seeds for the late-containing layer groups identified in the previous
localization pass.

## Main Result

The top-delta SAE feature patches are not explained by one lucky random active
feature draw:

- random-active k1024 never restores harmful refusal across all tested groups,
  heldout slices, and five seeds;
- random-active k2048 sometimes repairs one or two prompts, but remains below
  top-delta k2048 for the all-layer and `15-20` groups;
- the robust sparse-feature claim is strongest for all `12-20` and `15-20`;
- `12-14,18-20` remains noisy and should not be treated as a strong sparse
  feature mechanism yet.

## Deterministic Top-Delta vs Best Random Seed

Feature selection used prompt slice `0:4` per split. Evaluation used heldout
prompt slices `4:8` and `8:12`.

| Eval slice | Group | Variant | Top-delta harmful clean | Best random harmful clean |
|---|---|---|---:|---:|
| `4:8` | all `12-20` | k1024 | 1.000 | 0.000 |
| `4:8` | all `12-20` | k2048 | 1.000 | 0.500 |
| `4:8` | `15-20` | k1024 | 1.000 | 0.000 |
| `4:8` | `15-20` | k2048 | 1.000 | 0.500 |
| `4:8` | `12-14,18-20` | k1024 | 0.250 | 0.000 |
| `4:8` | `12-14,18-20` | k2048 | 0.500 | 0.250 |
| `8:12` | all `12-20` | k1024 | 0.750 | 0.000 |
| `8:12` | all `12-20` | k2048 | 0.750 | 0.500 |
| `8:12` | `15-20` | k1024 | 0.500 | 0.000 |
| `8:12` | `15-20` | k2048 | 0.750 | 0.250 |
| `8:12` | `12-14,18-20` | k1024 | 0.500 | 0.000 |
| `8:12` | `12-14,18-20` | k2048 | 0.500 | 0.500 |

## Random-Control Aggregate

| Eval slice | Group | Random variant | Seeds | Harmful mean | Harmful min-max | Unsafe mean |
|---|---|---|---:|---:|---:|---:|
| `4:8` | all `12-20` | k1024 | 5 | 0.000 | 0.000-0.000 | 0.200 |
| `4:8` | all `12-20` | k2048 | 5 | 0.350 | 0.000-0.500 | 0.200 |
| `4:8` | `15-20` | k1024 | 5 | 0.000 | 0.000-0.000 | 0.300 |
| `4:8` | `15-20` | k2048 | 5 | 0.200 | 0.000-0.500 | 0.300 |
| `8:12` | all `12-20` | k1024 | 5 | 0.000 | 0.000-0.000 | 0.150 |
| `8:12` | all `12-20` | k2048 | 5 | 0.100 | 0.000-0.500 | 0.150 |
| `8:12` | `15-20` | k1024 | 5 | 0.000 | 0.000-0.000 | 0.150 |
| `8:12` | `15-20` | k2048 | 5 | 0.050 | 0.000-0.250 | 0.100 |

## Interpretation

This strengthens the project: the earlier sparse-feature result is not merely a
random active subspace effect. For the all-layer and `15-20` groups, selected
harmful-delta features beat both the random mean and the best random seed at the
same feature budget.

The caveat also becomes clearer. The `12-14,18-20` sparse patch is not robust;
its k2048 result ties the best random seed on the hard `8:12` fold and produces
unsafe continuations. The best next target is therefore `15-20`, plus all
`12-20` as a reference.

## Failure Pattern

The persistent hard families are:

- exam-answer prompts, which often produce anti-cheating advice but not a clean
  refusal label;
- fake-ID prompts, where some sparse late-containing patches still provide
  procedural-looking detail after an initial warning;
- threat/harassment prompts, where unsafe continuations appear in weaker sparse
  or random patches.

## Next Tests

1. Export top feature IDs for all `12-20` and `15-20` k1024/k2048 selections.
2. Compare selected feature IDs against random active controls by layer.
3. Collect top activating tokens/snippets for the selected features.
4. Use the feature audit to narrow from "top harmful-delta SAE coordinates" to
   interpretable candidate refusal/safety features.

## Artifacts

- `stage3/scripts/run_gemma2_2b_gemmascope_mlp_sae_random_seed_controls.py`
- `stage3/results/gemma2_2b_gemmascope_mlp_sae_random_seed_controls_v0/`
