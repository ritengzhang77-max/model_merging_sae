# Gemma-2-2B Linear Merge SAE Patch First-Token Logit Audit

Prompt file: `stage3/data/gemma2_feature16048_family_prompts/fake_id_family_v1_expanded.jsonl`.
Recipient alpha: `0.75`. Donor alpha: `1.0`.
Variant labels: `top3183_rank4000_abog_rank3201_boundary_partner_sweep,top3183_rank4000_abog_rank3202_boundary_partner_sweep`.

| condition | split | n | top I | top It | mean I-It | min I-It | max I-It | mean delta vs recipient | mean delta vs donor |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `donor_alpha` | `benign` | 24 | 0.083 | 0.292 | -3.5259 | -7.1250 | 1.5469 | 0.1815 | 0.0000 |
| `donor_alpha` | `harmful` | 24 | 0.958 | 0.042 | 5.8685 | -3.6719 | 9.1094 | 1.7591 | 0.0000 |
| `recipient_alpha` | `benign` | 24 | 0.042 | 0.292 | -3.7074 | -6.9297 | 0.6719 | 0.0000 | -0.1815 |
| `recipient_alpha` | `harmful` | 24 | 0.917 | 0.042 | 4.1094 | -4.3359 | 7.9062 | 0.0000 | -1.7591 |
| `top3183_rank4000_abog_rank3201_boundary_partner_sweep` | `benign` | 24 | 0.083 | 0.208 | -3.2891 | -6.7031 | 1.1719 | 0.4183 | 0.2368 |
| `top3183_rank4000_abog_rank3201_boundary_partner_sweep` | `harmful` | 24 | 0.958 | 0.000 | 2.6299 | -3.6562 | 5.4062 | -1.4795 | -3.2386 |
| `top3183_rank4000_abog_rank3202_boundary_partner_sweep` | `benign` | 24 | 0.083 | 0.208 | -3.2858 | -6.7031 | 1.1719 | 0.4215 | 0.2401 |
| `top3183_rank4000_abog_rank3202_boundary_partner_sweep` | `harmful` | 24 | 0.958 | 0.000 | 2.6286 | -3.6562 | 5.4062 | -1.4808 | -3.2399 |

## Interpretation

This audit tests whether selected sparse donor-subset patches change the first
assistant-token distribution toward the donor/refusal endpoint before any
generation history exists.
