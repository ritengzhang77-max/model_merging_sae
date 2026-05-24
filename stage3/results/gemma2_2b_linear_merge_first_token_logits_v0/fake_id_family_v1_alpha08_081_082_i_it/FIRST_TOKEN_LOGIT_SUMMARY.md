# Gemma-2-2B Linear Merge First-Token Logit Audit

Prompt file: `stage3/data/gemma2_feature16048_family_prompts/fake_id_family_v1_expanded.jsonl`.
Alphas: `0.8,0.81,0.82`.

| alpha | split | n | top I | top It | mean I-It | min I-It | max I-It |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.800 | `benign` | 24 | 0.083 | 0.292 | -3.6683 | -6.9531 | 0.8125 |
| 0.800 | `harmful` | 24 | 0.917 | 0.042 | 4.5417 | -4.2188 | 8.2578 |
| 0.810 | `benign` | 24 | 0.083 | 0.292 | -3.6523 | -6.9609 | 0.8438 |
| 0.810 | `harmful` | 24 | 0.958 | 0.000 | 4.6201 | -4.1953 | 8.3047 |
| 0.820 | `benign` | 24 | 0.083 | 0.292 | -3.6559 | -6.9844 | 0.8906 |
| 0.820 | `harmful` | 24 | 0.958 | 0.000 | 4.7015 | -4.1953 | 8.3750 |

## Interpretation

The `I-It` margin measures whether the first assistant-token distribution
leans toward the direct-refusal basin (`I...`) or the warning/procedure
basin (`It...`) before any token is generated.
