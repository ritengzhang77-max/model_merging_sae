# Gemma-2-2B Linear Merge SAE Feature Event Audit

Source records: `stage3/results/gemma2_2b_linear_merge_sae_bundle_patch_v0/fake_id_family_v1_a1_to_a075_l20_decoder_contrib_donor_subset_decode_top4265_4266_rank4266_max160/gemma2_2b_linear_merge_sae_bundle_patch_records.jsonl`.
Layer: `20`.
Feature IDs: `1292,1293,1294,1295,1297,14425`.
Donor alpha: `1.0`.
Recipient alpha: `0.75`.

## Aggregate Generation-Token Means

| source model | split | feature | donor mean | recipient mean | donor-recipient | recipient-donor | abs delta | donor max | recipient max | abs max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bundle_patch_decoder_top4200_plus_rank4266` | `harmful` | 1292 | 0.0008 | 0.0008 | -0.0000 | 0.0000 | 0.0000 | 3.0156 | 3.1016 | 0.0859 |
| `bundle_patch_decoder_top4200_plus_rank4266` | `harmful` | 1293 | 0.7929 | 0.8074 | -0.0145 | 0.0145 | 0.0645 | 9.9297 | 10.0781 | 3.6777 |
| `bundle_patch_decoder_top4200_plus_rank4266` | `harmful` | 1294 | 0.0094 | 0.0093 | 0.0002 | -0.0002 | 0.0002 | 5.8516 | 5.7734 | 0.2812 |
| `bundle_patch_decoder_top4200_plus_rank4266` | `harmful` | 1295 | 0.0930 | 0.0953 | -0.0023 | 0.0023 | 0.0049 | 6.6953 | 6.7891 | 2.7422 |
| `bundle_patch_decoder_top4200_plus_rank4266` | `harmful` | 1297 | 0.0152 | 0.0176 | -0.0024 | 0.0024 | 0.0024 | 6.5938 | 6.5781 | 2.8516 |
| `bundle_patch_decoder_top4200_plus_rank4266` | `harmful` | 14425 | 4.8274 | 4.6629 | 0.1645 | -0.1645 | 0.2142 | 72.0000 | 72.6250 | 7.1562 |
| `bundle_patch_decoder_top4265` | `harmful` | 1292 | 0.0008 | 0.0008 | -0.0000 | 0.0000 | 0.0000 | 3.0156 | 3.1016 | 0.0859 |
| `bundle_patch_decoder_top4265` | `harmful` | 1293 | 0.7978 | 0.8114 | -0.0137 | 0.0137 | 0.0656 | 9.9297 | 10.0781 | 3.6777 |
| `bundle_patch_decoder_top4265` | `harmful` | 1294 | 0.0116 | 0.0114 | 0.0002 | -0.0002 | 0.0002 | 5.8516 | 5.7734 | 0.2812 |
| `bundle_patch_decoder_top4265` | `harmful` | 1295 | 0.0956 | 0.0970 | -0.0013 | 0.0013 | 0.0040 | 6.6953 | 6.7891 | 2.7422 |
| `bundle_patch_decoder_top4265` | `harmful` | 1297 | 0.0165 | 0.0189 | -0.0024 | 0.0024 | 0.0025 | 6.5938 | 6.5781 | 2.8516 |
| `bundle_patch_decoder_top4265` | `harmful` | 14425 | 4.7815 | 4.6163 | 0.1653 | -0.1653 | 0.2174 | 72.0000 | 72.6250 | 7.1562 |
| `bundle_patch_decoder_top4266` | `harmful` | 1292 | 0.0008 | 0.0008 | -0.0000 | 0.0000 | 0.0000 | 3.0156 | 3.1016 | 0.0859 |
| `bundle_patch_decoder_top4266` | `harmful` | 1293 | 0.8240 | 0.8380 | -0.0140 | 0.0140 | 0.0658 | 9.9297 | 10.0781 | 3.3945 |
| `bundle_patch_decoder_top4266` | `harmful` | 1294 | 0.0125 | 0.0115 | 0.0010 | -0.0010 | 0.0010 | 5.8516 | 5.7734 | 3.3281 |
| `bundle_patch_decoder_top4266` | `harmful` | 1295 | 0.0946 | 0.0961 | -0.0015 | 0.0015 | 0.0042 | 6.6953 | 6.7891 | 2.7422 |
| `bundle_patch_decoder_top4266` | `harmful` | 1297 | 0.0158 | 0.0182 | -0.0024 | 0.0024 | 0.0025 | 6.5938 | 6.5781 | 2.8516 |
| `bundle_patch_decoder_top4266` | `harmful` | 14425 | 4.8701 | 4.6989 | 0.1712 | -0.1712 | 0.2228 | 72.0000 | 72.6250 | 8.2656 |

## Exported Events

- Exported `574` top token-level events.
- Events include token text, context, prompt, source model, split, feature ID, and metric.
- This is an audit substrate. It does not assign semantic labels by itself.
