# Gemma-2-2B Linear Merge SAE Feature Event Audit

Source records: `stage3/results/gemma2_2b_linear_merge_sae_mixed_timing_bundle_patch_v0/fake_id_hologram_probe_a1_to_a075_l20_top3184_rank3202_timing_probe_float32_sae_max160/gemma2_2b_linear_merge_sae_mixed_timing_bundle_patch_records.jsonl`.
Layer: `20`.
Feature IDs: `11494,93,114,1293,4983,2451`.
Donor alpha: `1.0`.
Recipient alpha: `0.75`.
Event token scope: `prompt`.

## Aggregate Generation-Token Means

| source model | split | feature | donor mean | recipient mean | donor-recipient | recipient-donor | abs delta | donor max | recipient max | abs max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mixed_timing_top3184_plus_rank3202_extra_boundary_only` | `harmful` | 93 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `mixed_timing_top3184_plus_rank3202_extra_boundary_only` | `harmful` | 114 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `mixed_timing_top3184_plus_rank3202_extra_boundary_only` | `harmful` | 1293 | 0.6672 | 0.6814 | -0.0142 | 0.0142 | 0.0581 | 6.7762 | 6.8244 | 3.0987 |
| `mixed_timing_top3184_plus_rank3202_extra_boundary_only` | `harmful` | 2451 | 0.0175 | 0.0180 | -0.0005 | 0.0005 | 0.0005 | 2.7994 | 2.8871 | 0.0877 |
| `mixed_timing_top3184_plus_rank3202_extra_boundary_only` | `harmful` | 4983 | 0.0198 | 0.0200 | -0.0002 | 0.0002 | 0.0002 | 3.1613 | 3.1962 | 0.0349 |
| `mixed_timing_top3184_plus_rank3202_extra_boundary_only` | `harmful` | 11494 | 0.0177 | 0.0168 | 0.0009 | -0.0009 | 0.0009 | 2.8327 | 2.6835 | 0.1493 |

## Exported Events

- Exported `30` top token-level events.
- Events include token text, context, prompt, source model, split, feature ID, and metric.
- This is an audit substrate. It does not assign semantic labels by itself.
