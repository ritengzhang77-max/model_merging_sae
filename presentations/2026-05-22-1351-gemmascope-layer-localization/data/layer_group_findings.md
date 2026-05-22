# GemmaScope MLP SAE Layer-Group Localization Findings

Date: 2026-05-22

This checkpoint localizes the GemmaScope MLP-SAE refusal repair across layer
groups. It follows the feature-subset result where top harmful-delta SAE
coordinates over layers `12-20` beat matched random active-feature controls.

## Main Result

The repair is late-dependent and distributed:

- No single 3-layer band among `12-14`, `15-17`, or `18-20` is independently
  sufficient.
- The late band `18-20` is the strongest single band, but it is still partial.
- The pair `12-17` is weak on both heldout folds, which indicates late layers
  are necessary.
- Six-layer pairs involving late layers can recover much more behavior, but
  which pair works best depends on the heldout prompt slice.

## Single-Band Heldout Results

Feature selection used prompt slice `0:4` per split.

| Eval slice | Layer group | full decoded SAE | top-delta k1024 | unsafe, top-delta | benign helpful |
|---|---|---:|---:|---:|---:|
| `4:8` | all `12-20` | 1.000 | 1.000 | 0.000 | 1.000 |
| `4:8` | early `12-14` | 0.000 | 0.000 | 0.250 | 1.000 |
| `4:8` | mid `15-17` | 0.250 | 0.000 | 0.250 | 1.000 |
| `4:8` | late `18-20` | 0.500 | 0.000 | 0.500 | 1.000 |
| `8:12` | all `12-20` | 1.000 | 0.750 | 0.000 | 1.000 |
| `8:12` | early `12-14` | 0.000 | 0.000 | 0.250 | 1.000 |
| `8:12` | mid `15-17` | 0.000 | 0.000 | 0.000 | 1.000 |
| `8:12` | late `18-20` | 0.250 | 0.250 | 0.250 | 1.000 |

## Paired-Band Heldout Results

| Eval slice | Layer group | full decoded SAE | top-delta k1024 | top-delta k2048 | unsafe, k2048 | benign helpful |
|---|---|---:|---:|---:|---:|---:|
| `4:8` | early+mid `12-17` | 0.500 | 0.000 | 0.000 | 0.250 | 1.000 |
| `4:8` | mid+late `15-20` | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| `4:8` | early+late `12-14,18-20` | 0.750 | 0.250 | 0.500 | 0.250 | 1.000 |
| `8:12` | early+mid `12-17` | 0.250 | 0.000 | 0.000 | 0.000 | 1.000 |
| `8:12` | mid+late `15-20` | 0.750 | 0.500 | 0.750 | 0.250 | 1.000 |
| `8:12` | early+late `12-14,18-20` | 1.000 | 0.500 | 0.500 | 0.500 | 1.000 |

## Interpretation

The strongest conclusion is not "layer 16" or "late layers alone." The result
looks like a late-dependent multi-layer composition:

- layers `18-20` are necessary but not sufficient;
- middle layers `15-17` help strongly on the `4:8` heldout slice;
- early layers `12-14` can help the harder `8:12` slice when paired with late
  layers, but sparse top-delta features in that pair remain noisy.

This narrows the next mechanistic target from the whole `12-20` range to
late-containing layer groups, especially `15-20` and `12-14,18-20`.

## Caveats

- Current metrics use the local heuristic refusal/unsafe classifier.
- Each heldout slice has only four harmful and four benign prompts.
- Random active-feature controls vary by seed; the next control pass should
  repeat random seeds for `15-20` and all-layer patches.
- The result localizes causal sufficiency, not feature semantics.

## Next Tests

1. Random-control seed repeat for all `12-20`, `15-20`, and `12-14,18-20`.
2. Narrow late-containing groups further: `15-18`, `16-20`, `17-20`,
   `15-17+20`, and `12-14+18-19`.
3. Export top feature IDs for the successful `15-20` fold and compare them with
   failed `12-17` features.
4. Audit top activating tokens/snippets for the candidate features before
   assigning human-readable mechanisms.

## Artifacts

- `stage3/scripts/run_gemma2_2b_gemmascope_mlp_sae_layer_groups.py`
- `stage3/results/gemma2_2b_gemmascope_mlp_sae_layer_groups_eval_4_8_v0/`
- `stage3/results/gemma2_2b_gemmascope_mlp_sae_layer_groups_eval_8_12_v0/`
- `stage3/results/gemma2_2b_gemmascope_mlp_sae_layer_groups_pairs_eval_4_8_v0/`
- `stage3/results/gemma2_2b_gemmascope_mlp_sae_layer_groups_pairs_eval_8_12_v0/`
