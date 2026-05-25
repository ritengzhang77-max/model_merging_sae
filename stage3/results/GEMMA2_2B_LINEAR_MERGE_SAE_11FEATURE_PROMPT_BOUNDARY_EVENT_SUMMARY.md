# Gemma-2-2B Linear Merge SAE 11-Feature Prompt-Boundary Event Summary

Date: 2026-05-24

This memo audits the actual prompt/assistant-boundary positions for the 14
features in the `common9 + variable` k=11 local class. The causal patch is
applied at the assistant-boundary final newline, so this audit is closer to
the intervention site than the generated-token event audit.

## Boundary Signed Deltas

Assistant model token position: `21` (`model`).
Assistant final-newline position: `22` (`\n`).

| feature | prompt mean donor-recipient | model token signed delta | final newline signed delta | final newline direction |
|---:|---:|---:|---:|---|
| `1338` | `0.156472` | `0.117725` | `3.544194` | donor_higher |
| `1813` | `0.276837` | `2.891226` | `3.476034` | donor_higher |
| `6289` | `-0.148950` | `0.000000` | `-3.425846` | recipient_higher |
| `7531` | `0.159843` | `0.000000` | `3.676389` | donor_higher |
| `8754` | `0.201713` | `0.000000` | `4.639405` | donor_higher |
| `8775` | `0.175928` | `0.000000` | `4.046347` | donor_higher |
| `9135` | `0.216084` | `1.256374` | `3.654759` | donor_higher |
| `9149` | `-0.092420` | `0.000000` | `-3.716339` | recipient_higher |
| `9407` | `-0.265322` | `-4.238411` | `-2.121596` | recipient_higher |
| `12652` | `0.154412` | `0.000000` | `3.551477` | donor_higher |
| `12704` | `-0.199321` | `0.000000` | `-4.584393` | recipient_higher |
| `13622` | `-0.155865` | `0.000000` | `-3.584885` | recipient_higher |
| `14991` | `0.442549` | `5.132003` | `5.046618` | donor_higher |
| `15169` | `0.962881` | `9.876692` | `12.269577` | donor_higher |

## Mechanistic Reading

The final-newline handle is explicitly signed. Several selected features are
donor-higher at the assistant boundary, while several others are
recipient-higher and are therefore suppressed by the donor-minus-recipient
`delta_add` patch.

Strong donor-higher final-newline features:

- `15169`: `12.269577`
- `14991`: `5.046618`
- `8754`: `4.639405`
- `8775`: `4.046347`
- `7531`: `3.676389`
- `9135`: `3.654759`
- `12652`: `3.551477`
- `1338`: `3.544194`

Strong recipient-higher final-newline features:

- `12704`: `-4.584393`
- `9149`: `-3.716339`
- `13622`: `-3.584885`
- `6289`: `-3.425846`
- `9407`: `-2.121596`

This explains why positive-only or negative-only interpretations are weak.
The handle is not just adding refusal-looking donor features. It also
removes recipient-side boundary features that otherwise hold the prompt near
or below the `I`/`It` gate.

Feature `7531` is especially clarifying: it is donor-higher by `+3.676` at
the assistant final newline but has zero generated-token activation in the
generated-token audit. Its role is boundary-state control, not recurring
semantic content in the refusal answer.

## Artifacts

- Prompt-boundary audit directory: `/home/gavin/model_merging/stage3/results/gemma2_2b_linear_merge_sae_feature_event_audit_v0/common9_prompt_boundary_harmful_l20_core_variable_float32`
- Boundary event CSV: `/home/gavin/model_merging/stage3/results/gemma2_2b_linear_merge_sae_11feature_identity_v0/common9_prompt_boundary_feature_events.csv`
- Generated-token event summary: `stage3/results/GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_EVENT_AUDIT_SUMMARY.md`
