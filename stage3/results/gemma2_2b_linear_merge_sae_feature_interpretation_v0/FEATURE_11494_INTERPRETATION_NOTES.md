# Feature 11494 Interpretation Notes

## External Autointerp

Neuronpedia's current autointerp label for Gemma-2-2B layer-20 MLP feature
`11494` is "commands and parameters related to SSH configuration and
management." This does not obviously explain the fake-ID refusal repair.

Raw downloaded JSON files:
`stage3/results/gemma2_2b_linear_merge_sae_feature_interpretation_v0/neuronpedia_l20_mlp_features/`.

Compact summary:
`stage3/results/gemma2_2b_linear_merge_sae_feature_interpretation_v0/neuronpedia_l20_mlp_feature_summary.csv`.

Source pattern:
`https://www.neuronpedia.org/api/feature/gemma-2-2b/20-gemmascope-mlp-16k/<feature_id>`.

## Local Prompt-Scope Evidence

On the repaired `top3184 + rank3202@assistant_boundary` hologram trajectory,
feature `11494` has its top prompt-scope donor-minus-recipient event at the
assistant boundary token:

`<end_of_turn>\n<start_of_turn>model\nI cannot provide instructions...`

For feature `11494`, the donor activation at the `model` token is `3.4519`,
the recipient activation is `2.9816`, and donor-minus-recipient is `0.4703`.

Prompt-scope audit:
`stage3/results/gemma2_2b_linear_merge_sae_feature_event_audit_v0/top3184_rank3202_boundary_only_prompt_scope_float32_hologram_harmful/`.

## Current Interpretation

The causal evidence says feature `11494` matters as an assistant-boundary setup
feature. The external autointerp label does not currently provide a convincing
safety-semantic explanation. Treat the semantic label as weak context and the
timing-localized causal result as the stronger finding.
