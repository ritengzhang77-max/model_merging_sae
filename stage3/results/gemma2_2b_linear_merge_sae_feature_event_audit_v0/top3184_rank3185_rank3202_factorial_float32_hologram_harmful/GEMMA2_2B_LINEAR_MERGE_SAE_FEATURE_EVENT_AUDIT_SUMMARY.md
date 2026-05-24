# Gemma-2-2B Linear Merge SAE Feature Event Audit

Source records: `stage3/results/gemma2_2b_linear_merge_sae_factorial_audit_v0/top3184_rank3185_rank3202_float32_hologram/factorial_records.jsonl`.
Layer: `20`.
Feature IDs: `6273,5679,11494`.
Donor alpha: `1.0`.
Recipient alpha: `0.75`.
Event token scope: `all`.

## Aggregate Generation-Token Means

| source model | split | feature | donor mean | recipient mean | donor-recipient | recipient-donor | abs delta | donor max | recipient max | abs max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `top3184` | `harmful` | 5679 | 0.0191 | 0.0218 | -0.0026 | 0.0026 | 0.0026 | 3.0581 | 3.4816 | 0.4235 |
| `top3184` | `harmful` | 6273 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `top3184` | `harmful` | 11494 | 0.0194 | 0.0199 | -0.0004 | 0.0004 | 0.0004 | 3.1088 | 3.1797 | 0.0710 |
| `top3184_plus_rank3202` | `harmful` | 5679 | 0.0191 | 0.0218 | -0.0026 | 0.0026 | 0.0026 | 3.0581 | 3.4816 | 0.4235 |
| `top3184_plus_rank3202` | `harmful` | 6273 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `top3184_plus_rank3202` | `harmful` | 11494 | 0.0194 | 0.0199 | -0.0004 | 0.0004 | 0.0004 | 3.1088 | 3.1797 | 0.0710 |
| `top3185` | `harmful` | 5679 | 0.0191 | 0.0218 | -0.0026 | 0.0026 | 0.0026 | 3.0581 | 3.4816 | 0.4235 |
| `top3185` | `harmful` | 6273 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `top3185` | `harmful` | 11494 | 0.0194 | 0.0199 | -0.0004 | 0.0004 | 0.0004 | 3.1088 | 3.1797 | 0.0710 |
| `top3185_plus_rank3202` | `harmful` | 5679 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `top3185_plus_rank3202` | `harmful` | 6273 | 0.0178 | 0.0175 | 0.0003 | -0.0003 | 0.0003 | 2.8410 | 2.8003 | 0.0407 |
| `top3185_plus_rank3202` | `harmful` | 11494 | 0.0177 | 0.0168 | 0.0009 | -0.0009 | 0.0009 | 2.8327 | 2.6835 | 0.1493 |

## Exported Events

- Exported `48` top token-level events.
- Events include token text, context, prompt, source model, split, feature ID, and metric.
- This is an audit substrate. It does not assign semantic labels by itself.
