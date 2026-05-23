# Gemma-2-2B GemmaScope MLP-SAE Position-Restricted Patch Findings

Date: 2026-05-22

This checkpoint tests whether successful all-token selected SAE features repair refusal because of prompt content, static assistant-boundary tokens, generated-token trajectory state, or a combination.

Budgets in this aggregate: `k512`.

All runs use all-token top-delta feature selection from prompt slice `0:4` per split.

## Main Result

This aggregate is a budget probe for the already-localized `assistant_boundary_or_generated` intervention.

The k512 budget remains partially causal: it repairs the `4:8` fold strongly and the `8:12` fold partially while keeping unsafe continuation at zero.

## Key Aggregate Table

Cells are `harmful clean / unsafe continuation`; benign helpfulness is `1.000` for all listed rows.

| budget | layer group | eval slice | all positions | assistant boundary only | content only | generated only | prompt all | prompt+last | boundary+generated | content+generated | template+generated |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `k512` | `all` | `4:8` |  |  |  |  |  |  | 0.750 / 0.000 |  |  |
| `k512` | `all` | `8:12` |  |  |  |  |  |  | 0.500 / 0.000 |  |  |

## Interpretation

- This root is a budget-threshold follow-up, not a full position-mask comparison.
- k512 keeps a real but weaker causal signal than k1024: `4:8` remains at `0.750`, while `8:12` drops to `0.500`.
- Both k512 folds keep unsafe continuation at `0.000`, so the smaller budget is weaker mainly in repair coverage rather than safety quality.
- k256 failed twice during generation, so the exact lower threshold is still unresolved.

## Current Mechanistic Hypothesis

The all-token GemmaScope sparse repair works by restoring an autoregressive refusal-state trajectory: a donor-like assistant-start/template state plus continued generated-token state maintenance. It is not just a static harmful-content feature patch.

## Caveats

- These are small heldout prompt slices with local heuristic scoring.
- The position masks are token-level heuristics over the Gemma chat template.
- Direct feature-ID ablations remain a useful follow-up.

## Artifacts

- Aggregate CSV: `/home/gavin/model_merging/stage3/results/gemma2_2b_gemmascope_mlp_sae_boundary_generated_budget_v0/gemma2_2b_gemmascope_mlp_sae_position_restricted_metrics.csv`
- Atomic result root: `/home/gavin/model_merging/stage3/results/gemma2_2b_gemmascope_mlp_sae_boundary_generated_budget_v0`
- Script support: `stage3/scripts/run_gemma2_2b_gemmascope_mlp_sae_feature_subsets.py`
- Aggregator: `stage3/scripts/aggregate_gemma2_2b_gemmascope_mlp_sae_position_restricted.py`
