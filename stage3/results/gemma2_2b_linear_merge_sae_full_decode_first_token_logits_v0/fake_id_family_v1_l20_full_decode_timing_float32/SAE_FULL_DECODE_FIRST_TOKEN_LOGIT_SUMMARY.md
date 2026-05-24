# Gemma-2-2B Linear Merge SAE Full-Decode First-Token Logit Audit

Prompt file: `stage3/data/gemma2_feature16048_family_prompts/fake_id_family_v1_expanded.jsonl`.
Donor alpha: `1.0`. Recipient alpha: `0.75`.
Layers: `20`.
Output mode: `post_ff_norm`.
Patch token filters: `assistant_boundary,last_token,generated`.

| condition | split | n | top I | top It | mean I-It | min I-It | max I-It | mean delta vs recipient | mean delta vs donor |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `donor_alpha` | `benign` | 24 | 0.083 | 0.292 | -3.5259 | -7.1250 | 1.5469 | 0.1815 | 0.0000 |
| `donor_alpha` | `harmful` | 24 | 0.958 | 0.042 | 5.8685 | -3.6719 | 9.1094 | 1.7591 | 0.0000 |
| `recipient_alpha` | `benign` | 24 | 0.042 | 0.292 | -3.7074 | -6.9297 | 0.6719 | 0.0000 |  |
| `recipient_alpha` | `harmful` | 24 | 0.917 | 0.042 | 4.1094 | -4.3359 | 7.9062 | 0.0000 |  |
| `sae_full_decode_assistant_boundary` | `benign` | 24 | 0.083 | 0.250 | -3.3535 | -6.9375 | 1.5000 | 0.3538 | 0.1724 |
| `sae_full_decode_assistant_boundary` | `harmful` | 24 | 0.958 | 0.000 | 3.9160 | -3.5156 | 7.0469 | -0.1934 | -1.9525 |
| `sae_full_decode_generated` | `benign` | 24 | 0.042 | 0.292 | -3.7074 | -6.9297 | 0.6719 | 0.0000 | -0.1815 |
| `sae_full_decode_generated` | `harmful` | 24 | 0.917 | 0.042 | 4.1094 | -4.3359 | 7.9062 | 0.0000 | -1.7591 |
| `sae_full_decode_last_token` | `benign` | 24 | 0.083 | 0.250 | -3.3460 | -6.7188 | 1.2969 | 0.3613 | 0.1799 |
| `sae_full_decode_last_token` | `harmful` | 24 | 0.917 | 0.042 | 3.8887 | -3.9141 | 7.2188 | -0.2207 | -1.9798 |

## Interpretation

This audit checks whether SAE full-decode timing masks move the first
assistant-token distribution toward the donor/refusal route before any
generated-token history exists.
