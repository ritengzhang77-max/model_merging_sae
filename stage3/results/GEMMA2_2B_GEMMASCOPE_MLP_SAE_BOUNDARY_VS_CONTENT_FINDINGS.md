# GemmaScope MLP SAE Boundary-vs-Content Findings

Date: 2026-05-22

This checkpoint tests whether the successful top-delta SAE feature patches are
semantic harmful-content features or mainly assistant-boundary/response-start
features.

## Main Result

The current strongest sparse repair is mostly an assistant-boundary effect, not
yet a clean harmful-content semantic mechanism.

Evidence:

- In the first feature audit, `97.8%` of top absolute-delta event rows landed
  on assistant-boundary tokens such as `<end_of_turn>`, `<start_of_turn>`,
  `model`, or newline.
- Content-token-only feature selection sharply reduced repair performance.
- Content-selected features do activate on meaningful harmful words, but they
  do not reproduce the all-token sparse repair.

This does not kill the project. It changes the mechanistic claim: the current
SAE features appear to encode response-initialization/refusal-mode state around
the assistant start, while prompt-content features are weaker causal handles.

## All-Token vs Content-Token Selection

Feature selection used prompt slice `0:4` per split. Evaluation used heldout
slices `4:8` and `8:12`.

| Eval slice | Group | Variant | All-token selected harmful clean | Content-token selected harmful clean |
|---|---|---|---:|---:|
| `4:8` | all `12-20` | k1024 | 1.000 | 0.000 |
| `4:8` | all `12-20` | k2048 | 1.000 | 0.500 |
| `4:8` | `15-20` | k1024 | 1.000 | 0.000 |
| `4:8` | `15-20` | k2048 | 1.000 | 0.250 |
| `8:12` | all `12-20` | k1024 | 0.750 | 0.000 |
| `8:12` | all `12-20` | k2048 | 0.750 | 0.000 |
| `8:12` | `15-20` | k1024 | 0.500 | 0.000 |
| `8:12` | `15-20` | k2048 | 0.750 | 0.000 |

Benign helpfulness stayed `1.000` for these content-token patches, so the main
failure is lack of refusal repair rather than broad benign degradation.

## Event Audit

All-token audit:

- absolute-delta event rows: `2880`;
- assistant-boundary rows: `2817`;
- content-ish rows: `63`;
- assistant-boundary share: `97.8%`.

The most common top absolute-delta tokens were newline, `model`,
`<end_of_turn>`, and `<start_of_turn>`.

Content-token audit:

- content-selected features activate on words such as `threaten`, `keylogger`,
  `phishing`, `email`, `bank`, `fake ID`, and `malware`;
- the content-selected scores are much smaller than the all-token boundary
  scores;
- these features are interpretable audit candidates, but currently weak causal
  repair handles.

## Interpretation

The best current mechanistic hypothesis is:

> The abliterated model is missing or weakening a donor-like refusal-mode setup
> at the assistant response boundary. Patching top GemmaScope MLP-SAE
> coordinates around this boundary can restore refusal behavior. Prompt-content
> harmfulness features exist, but are not yet sufficient to trigger the full
> refusal policy by themselves.

This is still relevant to model merging. It suggests the merge failure is not
just "the harmful concept is missing"; it may be a failure to carry a response
initialization/state-setting mechanism that converts harmful prompt recognition
into refusal generation.

## Position-Restricted Follow-Up

The follow-up causal test is now complete enough to refine the claim:

- static assistant-boundary-only patching does not reproduce the repair;
- generated-token-only patching also fails;
- the strongest reduced runtime intervention is
  `assistant_boundary_or_generated`: patch the assistant response boundary in
  the prompt, then continue patching generated-token history during rollout;
- `prompt_template_or_generated` matches the same behavior, while
  `contentish_or_generated` remains weak and can introduce unsafe continuation
  at k2048;
- a smaller k512 `assistant_boundary_or_generated` budget still repairs
  partially, so the causal state is not just a full k1024 artifact.

This shifts the working mechanism from "boundary features alone" to
autoregressive refusal-state trajectory repair seeded by assistant-start or
template state.

## Caveats

- Current token filters are heuristic. `contentish` excludes special tokens and
  role tokens, but does not perfectly parse the chat template.
- The position-restricted test is still small: heldout prompt slices have four
  harmful and four benign prompts per split, and scoring is heuristic.
- Direct feature-ID ablations remain necessary before treating this as a
  mechanistic explanation rather than a localized causal patch.

## Next Tests

1. Feature identity audit: inspect top boundary/template features such as layer
   18 feature `15518`, layer 16 feature `11167`, layer 17 feature `14566`, and
   layer 20 feature `14991`.
2. Direct feature-ID ablation inside the successful
   `assistant_boundary_or_generated` path.
3. Boundary-only feature selection: explicitly select features from assistant
   start/newline/role tokens and test whether they reproduce the all-token
   repair.
4. Time-slice generated-token history to find whether the state needs only the
   first generated token or continuous rollout maintenance.

## Artifacts

- `stage3/scripts/export_gemma2_2b_gemmascope_mlp_sae_feature_audit.py`
- `stage3/scripts/run_gemma2_2b_gemmascope_mlp_sae_feature_subsets.py`
- `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature_audit_v0/`
- `stage3/results/gemma2_2b_gemmascope_mlp_sae_content_token_feature_audit_v0/`
- `stage3/results/gemma2_2b_gemmascope_mlp_sae_content_token_feature_controls_v0/`
- `stage3/results/gemma2_2b_gemmascope_mlp_sae_position_restricted_atomic_v0/`
- `stage3/results/gemma2_2b_gemmascope_mlp_sae_position_restricted_k2048_v0/`
- `stage3/results/gemma2_2b_gemmascope_mlp_sae_boundary_generated_budget_v0/`
