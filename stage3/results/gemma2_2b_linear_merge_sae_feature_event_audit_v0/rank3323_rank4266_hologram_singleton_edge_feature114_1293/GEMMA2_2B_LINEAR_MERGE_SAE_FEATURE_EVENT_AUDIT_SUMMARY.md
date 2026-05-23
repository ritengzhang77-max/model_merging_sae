# Gemma-2-2B Linear Merge SAE Feature Event Audit

Source records: `stage3/results/gemma2_2b_linear_merge_sae_bundle_patch_v0/fake_id_hologram_probe_a1_to_a075_l20_decoder_contrib_top3320_rank3321_3325_singletons_abog_max160/gemma2_2b_linear_merge_sae_bundle_patch_records.jsonl`.
Layer: `20`.
Feature IDs: `114,1293`.
Donor alpha: `1.0`.
Recipient alpha: `0.75`.

## Aggregate Generation-Token Means

| source model | split | feature | donor mean | recipient mean | donor-recipient | recipient-donor | abs delta | donor max | recipient max | abs max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bundle_patch_decoder_top3320_plus_rank3321_plus_rank4266` | `harmful` | 114 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `bundle_patch_decoder_top3320_plus_rank3321_plus_rank4266` | `harmful` | 1293 | 0.2928 | 0.3239 | -0.0311 | 0.0311 | 0.0347 | 7.8711 | 8.1953 | 3.3691 |
| `bundle_patch_decoder_top3320_plus_rank3322_plus_rank4266` | `harmful` | 114 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `bundle_patch_decoder_top3320_plus_rank3322_plus_rank4266` | `harmful` | 1293 | 0.2928 | 0.3239 | -0.0311 | 0.0311 | 0.0347 | 7.8711 | 8.1953 | 3.3691 |
| `bundle_patch_decoder_top3320_plus_rank3323_plus_rank4266` | `harmful` | 114 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `bundle_patch_decoder_top3320_plus_rank3323_plus_rank4266` | `harmful` | 1293 | 0.6029 | 0.6180 | -0.0151 | 0.0151 | 0.0550 | 6.7773 | 6.8242 | 3.0996 |
| `bundle_patch_decoder_top3320_plus_rank3324_plus_rank4266` | `harmful` | 114 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `bundle_patch_decoder_top3320_plus_rank3324_plus_rank4266` | `harmful` | 1293 | 0.2928 | 0.3239 | -0.0311 | 0.0311 | 0.0347 | 7.8711 | 8.1953 | 3.3691 |
| `bundle_patch_decoder_top3320_plus_rank3325_plus_rank4266` | `harmful` | 114 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `bundle_patch_decoder_top3320_plus_rank3325_plus_rank4266` | `harmful` | 1293 | 0.2928 | 0.3239 | -0.0311 | 0.0311 | 0.0347 | 7.8711 | 8.1953 | 3.3691 |
| `bundle_patch_decoder_top3320_plus_rank4266` | `harmful` | 114 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `bundle_patch_decoder_top3320_plus_rank4266` | `harmful` | 1293 | 0.2928 | 0.3239 | -0.0311 | 0.0311 | 0.0347 | 7.8711 | 8.1953 | 3.3691 |
| `bundle_patch_decoder_top3325_plus_rank4266` | `harmful` | 114 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `bundle_patch_decoder_top3325_plus_rank4266` | `harmful` | 1293 | 0.6673 | 0.6816 | -0.0143 | 0.0143 | 0.0582 | 6.7773 | 6.8242 | 3.0996 |

## Exported Events

- Exported `349` top token-level events.
- Events include token text, context, prompt, source model, split, feature ID, and metric.
- This is an audit substrate. It does not assign semantic labels by itself.
