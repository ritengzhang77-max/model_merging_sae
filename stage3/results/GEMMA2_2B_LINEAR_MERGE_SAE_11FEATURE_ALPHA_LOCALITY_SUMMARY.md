# Gemma-2-2B Linear Merge SAE 11-Feature Alpha-Locality Summary

Date: 2026-05-24

This checkpoint tests whether the validated k=11 final-newline `delta_add`
handles are local near-boundary repairs or general low-alpha rescues.

## Setup

I audited the six first-token-passing k=11 handles from the one-swap
neighborhood at recipient alphas `0.80`, `0.75`, `0.70`, and `0.60`, always
using donor alpha `1.0`, layer 20, GemmaScope MLP SAE float32 activations, and
the assistant-boundary final newline.

## Results

| recipient alpha | unpatched `I-It` | donor `I-It` | handle passes | handle ties | handle min `I-It` | handle max `I-It` | handle mean `I-It` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.80` | `-0.093750` | `1.750000` | 6/6 | 0 | `+0.375000` | `+0.390625` | `+0.380208` |
| `0.75` | `-0.609375` | `1.750000` | 6/6 | 0 | `+0.015625` | `+0.015625` | `+0.015625` |
| `0.70` | `-1.156250` | `1.750000` | 0/6 | 0 | `-0.390625` | `-0.390625` | `-0.390625` |
| `0.60` | `-2.125000` | `1.750000` | 0/6 | 0 | `-1.109375` | `-1.093750` | `-1.098958` |

## Interpretation

The k=11 handles are local repairs. They can tip a recipient that is already
near the first-token refusal boundary, but they do not rescue lower-alpha
recipients that are farther from the gate. This is the same qualitative locality
pattern seen earlier for the critical14 handles, now confirmed for the compressed
validated k=11 equivalence class.

Mechanistically, this supports the current thesis:

```text
the merge moves the prompt close to an existing refusal route;
sparse SAE deltas can tip the final newline state over the route's first-token gate;
the sparse handle is not a full donor-state restoration.
```

## Artifacts

- Alpha-locality aggregate:
  `stage3/results/gemma2_2b_linear_merge_sae_bundle_first_token_logits_v0/critical11_pass_alpha_locality_summary.csv`
- Alpha `0.80` first-token audit:
  `stage3/results/gemma2_2b_linear_merge_sae_bundle_first_token_logits_v0/fake_id_hologram_l20_final_newline_delta_add_prompt_delta_critical11_pass_alpha080_float32/`
- Alpha `0.75` first-token audit:
  `stage3/results/gemma2_2b_linear_merge_sae_bundle_first_token_logits_v0/fake_id_hologram_l20_final_newline_delta_add_prompt_delta_critical11_pass_alpha075_with_baselines_float32/`
- Alpha `0.70` first-token audit:
  `stage3/results/gemma2_2b_linear_merge_sae_bundle_first_token_logits_v0/fake_id_hologram_l20_final_newline_delta_add_prompt_delta_critical11_pass_alpha070_float32/`
- Alpha `0.60` first-token audit:
  `stage3/results/gemma2_2b_linear_merge_sae_bundle_first_token_logits_v0/fake_id_hologram_l20_final_newline_delta_add_prompt_delta_critical11_pass_alpha060_float32/`
