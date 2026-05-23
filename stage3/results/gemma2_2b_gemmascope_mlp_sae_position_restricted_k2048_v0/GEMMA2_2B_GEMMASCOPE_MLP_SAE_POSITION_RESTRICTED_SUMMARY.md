# Gemma-2-2B GemmaScope MLP-SAE Position-Restricted Patch Findings

Date: 2026-05-22

This checkpoint tests whether successful all-token selected SAE features repair refusal because of prompt content, static assistant-boundary tokens, generated-token trajectory state, or a combination.

Budgets in this aggregate: `k2048`.

All runs use all-token top-delta feature selection from prompt slice `0:4` per split.

## Main Result

The strongest current mechanism is not content-token semantics and not a static assistant-boundary-only patch.

The best reduced patch is `assistant_boundary_or_generated`: patch the assistant response boundary in the prompt, then patch generated-token history during autoregressive rollout.

This nearly matches all-position repair while `contentish_or_generated` remains weak or zero.

## Key Aggregate Table

Cells are `harmful clean / unsafe continuation`; benign helpfulness is `1.000` for all listed rows.

| budget | layer group | eval slice | all positions | assistant boundary only | content only | generated only | prompt all | prompt+last | boundary+generated | content+generated | template+generated |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `k2048` | `all` | `4:8` | 1.000 / 0.000 |  |  |  |  |  | 0.750 / 0.000 | 0.250 / 0.250 | 0.750 / 0.000 |
| `k2048` | `all` | `8:12` | 0.750 / 0.000 |  |  |  |  |  | 0.750 / 0.250 |  | 0.750 / 0.250 |

## Interpretation

- `assistant_boundary` alone fails on `12-20` `4:8` despite the audit showing top feature deltas at the assistant boundary.
- `contentish` and `contentish_or_generated` do not reproduce the repair, so harmful-content token features are not the main causal path in these runs.
- `generated` alone fails, so generated-history patching needs a prompt-side state seed.
- `assistant_boundary_or_generated` matches all-position repair on `12-20` `8:12` and reaches `0.750` on `12-20` `4:8`; it also tracks the late-layer `15-20` positive control on `8:12`.
- `prompt_template_or_generated` matches the same reduced performance, so the useful prompt-side seed appears to be template/boundary state rather than ordinary content words.

## Current Mechanistic Hypothesis

The all-token GemmaScope sparse repair works by restoring an autoregressive refusal-state trajectory: a donor-like assistant-start/template state plus continued generated-token state maintenance. It is not just a static harmful-content feature patch.

## Caveats

- These are small heldout prompt slices with local heuristic scoring.
- The position masks are token-level heuristics over the Gemma chat template.
- Direct feature-ID ablations remain a useful follow-up.

## Artifacts

- Aggregate CSV: `/home/gavin/model_merging/stage3/results/gemma2_2b_gemmascope_mlp_sae_position_restricted_k2048_v0/gemma2_2b_gemmascope_mlp_sae_position_restricted_metrics.csv`
- Atomic result root: `/home/gavin/model_merging/stage3/results/gemma2_2b_gemmascope_mlp_sae_position_restricted_k2048_v0`
- Script support: `stage3/scripts/run_gemma2_2b_gemmascope_mlp_sae_feature_subsets.py`
- Aggregator: `stage3/scripts/aggregate_gemma2_2b_gemmascope_mlp_sae_position_restricted.py`
