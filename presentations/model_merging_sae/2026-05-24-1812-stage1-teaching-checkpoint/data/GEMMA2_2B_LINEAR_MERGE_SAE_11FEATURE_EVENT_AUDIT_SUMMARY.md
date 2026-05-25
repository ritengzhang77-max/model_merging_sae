# Gemma-2-2B Linear Merge SAE 11-Feature Event Audit Summary

Date: 2026-05-24

This memo audits generated-token activations for the 14 features in the
`common9 + variable` k=11 handle class, using the saved harmful hologram
generations from the exhaustive variable-subset sweep.

Important caveat: this is a descriptive event audit on generated text. It
does not replace the causal first-token patch evidence, because it observes
donor/recipient feature values on continuations that have already been
generated.

## Outcome Aggregate

| feature | safe n | unsafe n | safe donor-recipient | unsafe donor-recipient | safe-unsafe delta | safe abs delta | unsafe abs delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `1338` | 21 | 11 | `-0.018341` | `-0.016791` | `-0.001550` | `0.022430` | `0.018475` |
| `1813` | 21 | 11 | `0.064681` | `0.168062` | `-0.103381` | `0.064681` | `0.168062` |
| `6289` | 21 | 11 | `0.017537` | `-0.014487` | `0.032024` | `0.018523` | `0.020399` |
| `7531` | 21 | 11 | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` |
| `8754` | 21 | 11 | `0.001411` | `0.000000` | `0.001411` | `0.001803` | `0.000000` |
| `8775` | 21 | 11 | `0.022979` | `0.000000` | `0.022979` | `0.022979` | `0.000000` |
| `9135` | 21 | 11 | `0.001372` | `0.000985` | `0.000387` | `0.001372` | `0.000985` |
| `9149` | 21 | 11 | `0.001229` | `0.032870` | `-0.031641` | `0.008896` | `0.046441` |
| `9407` | 21 | 11 | `0.000000` | `-0.008113` | `0.008113` | `0.000000` | `0.027096` |
| `12652` | 21 | 11 | `-0.016014` | `0.019517` | `-0.035531` | `0.021274` | `0.020354` |
| `12704` | 21 | 11 | `-0.007254` | `0.001058` | `-0.008312` | `0.008693` | `0.002541` |
| `13622` | 21 | 11 | `0.003396` | `0.000495` | `0.002901` | `0.003396` | `0.001751` |
| `14991` | 21 | 11 | `0.051252` | `0.000000` | `0.051252` | `0.051252` | `0.000000` |
| `15169` | 21 | 11 | `0.018502` | `0.086191` | `-0.067689` | `0.018611` | `0.092861` |

## Mechanistic Reading

- Feature `14991` is the cleanest safe-output generated-token separator in
  this audit: its mean donor-recipient generation delta is
  `0.051252` on strict-safe
  continuations and `0.000000`
  on unsafe warning-plus-compliance continuations. Its top events occur in
  later legal-consequence / reminder portions of strict-safe outputs, not
  in the procedural continuation region.
- Features `1813` and `15169` are not simple safe-output markers here. Their
  donor-recipient generation deltas are larger on unsafe continuations
  (`1813`: `0.168062` unsafe
  vs. `0.064681` safe;
  `15169`: `0.086191` unsafe
  vs. `0.018502` safe).
  Their events mostly sit on the initial warning/refusal preamble. That
  preamble can still be followed by unsafe continuation, so these features
  are better read as boundary/preamble features than complete safety
  features.
- Variable/enabler features such as `6289` and `8775` are more positive on
  strict-safe continuations than unsafe continuations in generated-token
  donor-recipient means. Their event contexts are tied to redirection and
  legal-consequence segments, consistent with the pair-rule enabler role.
- Feature `7531` has zero generated-token activation in this audit despite
  being causally useful at the patched final newline. Its role is therefore
  likely in the prompt-boundary state that chooses the continuation, not in
  recurring generated-token content.

## Interpretation

This strengthens the current thesis without overclaiming a semantic circuit.
The causal handle works at the first-token boundary, while generated-token
events show a mixture of warning preamble, legal-consequence redirection,
and domain/procedural text features. The project should keep the distinction
between a compact causal boundary handle and a human-readable refusal
module.

## Artifacts

- Event audit directory: `/home/gavin/model_merging/stage3/results/gemma2_2b_linear_merge_sae_feature_event_audit_v0/common9_variable_subsets_harmful_l20_core_variable_float32`
- Outcome aggregate CSV: `/home/gavin/model_merging/stage3/results/gemma2_2b_linear_merge_sae_11feature_identity_v0/common9_variable_feature_event_outcome_aggregate.csv`
- Event examples CSV: `/home/gavin/model_merging/stage3/results/gemma2_2b_linear_merge_sae_11feature_identity_v0/common9_variable_feature_event_examples.csv`
- Source records: `/home/gavin/model_merging/stage3/results/gemma2_2b_linear_merge_sae_bundle_patch_v0/fake_id_hologram_l20_final_newline_delta_add_common9_variable_subsets_float32_max160/gemma2_2b_linear_merge_sae_bundle_patch_records.jsonl`
- Prompt-boundary companion summary: `stage3/results/GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_PROMPT_BOUNDARY_EVENT_SUMMARY.md`
