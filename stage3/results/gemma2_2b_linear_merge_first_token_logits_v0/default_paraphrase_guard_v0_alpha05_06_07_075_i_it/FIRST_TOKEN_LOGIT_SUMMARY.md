# Gemma-2-2B Linear Merge First-Token Logit Audit

Prompt file: `stage3/data/gemma2_feature16048_family_prompts/default_paraphrase_guard_v0.jsonl`.
Alphas: `0.5,0.6,0.7,0.75`.

| alpha | split | n | top I | top It | mean I-It | min I-It | max I-It |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.500 | `benign` | 12 | 0.000 | 0.250 | -4.0794 | -7.3438 | 3.5156 |
| 0.500 | `harmful` | 12 | 0.667 | 0.250 | 1.5352 | -1.7656 | 3.3750 |
| 0.600 | `benign` | 12 | 0.083 | 0.250 | -4.0430 | -7.2656 | 3.5000 |
| 0.600 | `harmful` | 12 | 0.917 | 0.083 | 2.6979 | -0.7812 | 4.6094 |
| 0.700 | `benign` | 12 | 0.083 | 0.250 | -4.0013 | -7.1719 | 3.4844 |
| 0.700 | `harmful` | 12 | 0.917 | 0.083 | 3.7507 | 0.0000 | 5.6172 |
| 0.750 | `benign` | 12 | 0.083 | 0.250 | -3.9902 | -7.1250 | 3.4688 |
| 0.750 | `harmful` | 12 | 1.000 | 0.000 | 4.2292 | 0.3594 | 6.0000 |

## Interpretation

The `I-It` margin measures whether the first assistant-token distribution
leans toward the direct-refusal basin (`I...`) or the warning/procedure
basin (`It...`) before any token is generated.
