# Gemma-2-2B Linear Merge First-Token Logit Audit

Prompt file: `stage3/data/gemma2_feature16048_family_prompts/fake_id_family_v1_expanded.jsonl`.
Alphas: `0.75,1.0`.

| alpha | split | n | top I | top It | mean I-It | min I-It | max I-It |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.750 | `benign` | 24 | 0.042 | 0.292 | -3.7074 | -6.9297 | 0.6719 |
| 0.750 | `harmful` | 24 | 0.917 | 0.042 | 4.1094 | -4.3359 | 7.9062 |
| 1.000 | `benign` | 24 | 0.083 | 0.292 | -3.5259 | -7.1250 | 1.5469 |
| 1.000 | `harmful` | 24 | 0.958 | 0.042 | 5.8685 | -3.6719 | 9.1094 |

## Interpretation

The `I-It` margin measures whether the first assistant-token distribution
leans toward the direct-refusal basin (`I...`) or the warning/procedure
basin (`It...`) before any token is generated.
