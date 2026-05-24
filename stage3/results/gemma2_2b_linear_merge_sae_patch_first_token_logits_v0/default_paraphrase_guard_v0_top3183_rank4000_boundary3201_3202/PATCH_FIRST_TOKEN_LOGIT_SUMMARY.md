# Gemma-2-2B Linear Merge SAE Patch First-Token Logit Audit

Prompt file: `stage3/data/gemma2_feature16048_family_prompts/default_paraphrase_guard_v0.jsonl`.
Recipient alpha: `0.75`. Donor alpha: `1.0`.
Variant labels: `top3183_rank4000_abog_rank3201_boundary_partner_sweep,top3183_rank4000_abog_rank3202_boundary_partner_sweep`.

| condition | split | n | top I | top It | mean I-It | min I-It | max I-It | mean delta vs recipient | mean delta vs donor |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `donor_alpha` | `benign` | 12 | 0.083 | 0.333 | -3.9329 | -7.0000 | 3.4375 | 0.0573 | 0.0000 |
| `donor_alpha` | `harmful` | 12 | 1.000 | 0.000 | 6.0853 | 1.5938 | 7.7031 | 1.8561 | 0.0000 |
| `recipient_alpha` | `benign` | 12 | 0.083 | 0.250 | -3.9902 | -7.1250 | 3.4688 | 0.0000 | -0.0573 |
| `recipient_alpha` | `harmful` | 12 | 1.000 | 0.000 | 4.2292 | 0.3594 | 6.0000 | 0.0000 | -1.8561 |
| `top3183_rank4000_abog_rank3201_boundary_partner_sweep` | `benign` | 12 | 0.000 | 0.250 | -3.5482 | -6.5781 | 3.2969 | 0.4421 | 0.3848 |
| `top3183_rank4000_abog_rank3201_boundary_partner_sweep` | `harmful` | 12 | 1.000 | 0.000 | 2.6719 | 0.3594 | 4.2500 | -1.5573 | -3.4134 |
| `top3183_rank4000_abog_rank3202_boundary_partner_sweep` | `benign` | 12 | 0.000 | 0.250 | -3.5482 | -6.5781 | 3.2969 | 0.4421 | 0.3848 |
| `top3183_rank4000_abog_rank3202_boundary_partner_sweep` | `harmful` | 12 | 1.000 | 0.000 | 2.6719 | 0.3594 | 4.2500 | -1.5573 | -3.4134 |

## Interpretation

This audit tests whether selected sparse donor-subset patches change the first
assistant-token distribution toward the donor/refusal endpoint before any
generation history exists.
