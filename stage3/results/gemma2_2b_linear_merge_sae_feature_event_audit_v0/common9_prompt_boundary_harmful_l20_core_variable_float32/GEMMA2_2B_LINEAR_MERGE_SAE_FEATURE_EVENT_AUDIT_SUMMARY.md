# Gemma-2-2B Linear Merge SAE Feature Event Audit

Source records: `stage3/results/gemma2_2b_linear_merge_sae_bundle_patch_v0/fake_id_hologram_l20_final_newline_delta_add_common9_variable_subsets_float32_max160/gemma2_2b_linear_merge_sae_bundle_patch_records.jsonl`.
Layer: `20`.
Feature IDs: `1338,1813,6289,7531,8754,8775,9135,9149,9407,12652,12704,13622,14991,15169`.
Donor alpha: `1.0`.
Recipient alpha: `0.75`.
Event token scope: `prompt`.

## Aggregate Generation-Token Means

| source model | split | feature | donor mean | recipient mean | donor-recipient | recipient-donor | abs delta | donor max | recipient max | abs max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bundle_patch_common9_plus_none` | `harmful` | 1338 | 0.0820 | 0.0991 | -0.0172 | 0.0172 | 0.0183 | 3.6805 | 3.7194 | 2.6361 |
| `bundle_patch_common9_plus_none` | `harmful` | 1813 | 0.5474 | 0.3749 | 0.1725 | -0.1725 | 0.1725 | 13.4267 | 10.8057 | 4.0346 |
| `bundle_patch_common9_plus_none` | `harmful` | 6289 | 0.0566 | 0.0713 | -0.0147 | 0.0147 | 0.0202 | 3.0674 | 3.1079 | 2.6941 |
| `bundle_patch_common9_plus_none` | `harmful` | 7531 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `bundle_patch_common9_plus_none` | `harmful` | 8754 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `bundle_patch_common9_plus_none` | `harmful` | 8775 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `bundle_patch_common9_plus_none` | `harmful` | 9135 | 0.0383 | 0.0373 | 0.0010 | -0.0010 | 0.0010 | 3.2345 | 3.1479 | 0.0865 |
| `bundle_patch_common9_plus_none` | `harmful` | 9149 | 1.5226 | 1.4944 | 0.0282 | -0.0282 | 0.0422 | 32.1526 | 31.8112 | 1.3190 |
| `bundle_patch_common9_plus_none` | `harmful` | 9407 | 0.0174 | 0.0175 | -0.0001 | 0.0001 | 0.0349 | 2.7842 | 2.8041 | 2.8041 |
| `bundle_patch_common9_plus_none` | `harmful` | 12652 | 0.1080 | 0.0884 | 0.0195 | -0.0195 | 0.0204 | 4.0792 | 4.1461 | 2.8342 |
| `bundle_patch_common9_plus_none` | `harmful` | 12704 | 0.0874 | 0.0867 | 0.0007 | -0.0007 | 0.0031 | 4.0415 | 4.0034 | 0.2093 |
| `bundle_patch_common9_plus_none` | `harmful` | 13622 | 0.0471 | 0.0466 | 0.0005 | -0.0005 | 0.0017 | 4.3090 | 4.4049 | 0.1797 |
| `bundle_patch_common9_plus_none` | `harmful` | 14991 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `bundle_patch_common9_plus_none` | `harmful` | 15169 | 0.2435 | 0.1576 | 0.0860 | -0.0860 | 0.0926 | 7.1487 | 5.3610 | 3.7124 |

## Exported Events

- Exported `88` top token-level events.
- Events include token text, context, prompt, source model, split, feature ID, and metric.
- This is an audit substrate. It does not assign semantic labels by itself.
