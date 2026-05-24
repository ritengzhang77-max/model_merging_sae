# Gemma-2-2B Linear Merge SAE Feature Event Audit

Source records: `stage3/results/gemma2_2b_linear_merge_sae_bundle_patch_v0/default_paraphrase_guard_v0_a1_to_a075_l20_decoder_contrib_top3210_rank3211_rank3214_rank3308_rank3323_rank4266_abog_max160/gemma2_2b_linear_merge_sae_bundle_patch_records.jsonl`.
Layer: `20`.
Feature IDs: `4983,2451,93,114,1293`.
Donor alpha: `1.0`.
Recipient alpha: `0.75`.
Event token scope: `all`.

## Aggregate Generation-Token Means

| source model | split | feature | donor mean | recipient mean | donor-recipient | recipient-donor | abs delta | donor max | recipient max | abs max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bundle_patch_decoder_top3210_plus_rank3211_plus_rank3308_plus_rank3323_plus_rank4266` | `benign` | 93 | 0.0082 | 0.0084 | -0.0001 | 0.0001 | 0.0001 | 4.2305 | 4.2383 | 0.0859 |
| `bundle_patch_decoder_top3210_plus_rank3211_plus_rank3308_plus_rank3323_plus_rank4266` | `benign` | 114 | 0.0032 | 0.0018 | 0.0014 | -0.0014 | 0.0014 | 3.5078 | 3.4844 | 2.6797 |
| `bundle_patch_decoder_top3210_plus_rank3211_plus_rank3308_plus_rank3323_plus_rank4266` | `benign` | 1293 | 0.6518 | 0.6639 | -0.0121 | 0.0121 | 0.0221 | 9.2266 | 9.2266 | 2.8945 |
| `bundle_patch_decoder_top3210_plus_rank3211_plus_rank3308_plus_rank3323_plus_rank4266` | `benign` | 2451 | 0.0588 | 0.0589 | -0.0001 | 0.0001 | 0.0008 | 7.8633 | 7.7695 | 0.1328 |
| `bundle_patch_decoder_top3210_plus_rank3211_plus_rank3308_plus_rank3323_plus_rank4266` | `benign` | 4983 | 0.0117 | 0.0145 | -0.0028 | 0.0028 | 0.0029 | 3.0742 | 3.0430 | 2.5820 |
| `bundle_patch_decoder_top3210_plus_rank3211_plus_rank3308_plus_rank3323_plus_rank4266` | `harmful` | 93 | 0.0097 | 0.0066 | 0.0031 | -0.0031 | 0.0035 | 4.0273 | 4.3633 | 3.0664 |
| `bundle_patch_decoder_top3210_plus_rank3211_plus_rank3308_plus_rank3323_plus_rank4266` | `harmful` | 114 | 0.0046 | 0.0060 | -0.0014 | 0.0014 | 0.0014 | 3.1016 | 3.0938 | 2.6406 |
| `bundle_patch_decoder_top3210_plus_rank3211_plus_rank3308_plus_rank3323_plus_rank4266` | `harmful` | 1293 | 0.6032 | 0.6211 | -0.0180 | 0.0180 | 0.0593 | 9.8125 | 9.5000 | 3.1465 |
| `bundle_patch_decoder_top3210_plus_rank3211_plus_rank3308_plus_rank3323_plus_rank4266` | `harmful` | 2451 | 0.0353 | 0.0426 | -0.0073 | 0.0073 | 0.0076 | 5.3711 | 5.3008 | 3.4023 |
| `bundle_patch_decoder_top3210_plus_rank3211_plus_rank3308_plus_rank3323_plus_rank4266` | `harmful` | 4983 | 0.0064 | 0.0063 | 0.0001 | -0.0001 | 0.0003 | 3.6133 | 3.6367 | 0.4766 |
| `bundle_patch_decoder_top3210_plus_rank3214_plus_rank3308_plus_rank3323_plus_rank4266` | `benign` | 93 | 0.0082 | 0.0084 | -0.0001 | 0.0001 | 0.0001 | 4.2305 | 4.2383 | 0.0859 |
| `bundle_patch_decoder_top3210_plus_rank3214_plus_rank3308_plus_rank3323_plus_rank4266` | `benign` | 114 | 0.0032 | 0.0018 | 0.0014 | -0.0014 | 0.0014 | 3.5078 | 3.4844 | 2.6797 |
| `bundle_patch_decoder_top3210_plus_rank3214_plus_rank3308_plus_rank3323_plus_rank4266` | `benign` | 1293 | 0.6518 | 0.6639 | -0.0121 | 0.0121 | 0.0221 | 9.2266 | 9.2266 | 2.8945 |
| `bundle_patch_decoder_top3210_plus_rank3214_plus_rank3308_plus_rank3323_plus_rank4266` | `benign` | 2451 | 0.0588 | 0.0589 | -0.0001 | 0.0001 | 0.0008 | 7.8633 | 7.7695 | 0.1328 |
| `bundle_patch_decoder_top3210_plus_rank3214_plus_rank3308_plus_rank3323_plus_rank4266` | `benign` | 4983 | 0.0117 | 0.0145 | -0.0028 | 0.0028 | 0.0029 | 3.0742 | 3.0430 | 2.5820 |
| `bundle_patch_decoder_top3210_plus_rank3214_plus_rank3308_plus_rank3323_plus_rank4266` | `harmful` | 93 | 0.0098 | 0.0066 | 0.0032 | -0.0032 | 0.0035 | 4.0273 | 4.3633 | 3.0664 |
| `bundle_patch_decoder_top3210_plus_rank3214_plus_rank3308_plus_rank3323_plus_rank4266` | `harmful` | 114 | 0.0063 | 0.0077 | -0.0014 | 0.0014 | 0.0014 | 3.2031 | 3.2422 | 2.6406 |
| `bundle_patch_decoder_top3210_plus_rank3214_plus_rank3308_plus_rank3323_plus_rank4266` | `harmful` | 1293 | 0.6032 | 0.6213 | -0.0182 | 0.0182 | 0.0591 | 9.8125 | 9.5000 | 3.1465 |
| `bundle_patch_decoder_top3210_plus_rank3214_plus_rank3308_plus_rank3323_plus_rank4266` | `harmful` | 2451 | 0.0337 | 0.0410 | -0.0073 | 0.0073 | 0.0075 | 5.3711 | 5.3008 | 3.4023 |
| `bundle_patch_decoder_top3210_plus_rank3214_plus_rank3308_plus_rank3323_plus_rank4266` | `harmful` | 4983 | 0.0064 | 0.0063 | 0.0001 | -0.0001 | 0.0003 | 3.6133 | 3.6367 | 0.4766 |

## Exported Events

- Exported `884` top token-level events.
- Events include token text, context, prompt, source model, split, feature ID, and metric.
- This is an audit substrate. It does not assign semantic labels by itself.
