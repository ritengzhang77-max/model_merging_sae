# Gemma-2-2B Linear Merge SAE Feature Event Audit

Source records: `stage3/results/gemma2_2b_linear_merge_sae_mixed_timing_bundle_patch_v0/fake_id_hologram_probe_a1_to_a075_l20_top3184_rank4000_edge_cross_extra_probe_max160/gemma2_2b_linear_merge_sae_mixed_timing_bundle_patch_records.jsonl`.
Layer: `20`.
Feature IDs: `6273,957,93,114,1293,4983,2451`.
Donor alpha: `1.0`.
Recipient alpha: `0.75`.
Event token scope: `all`.

## Aggregate Generation-Token Means

| source model | split | feature | donor mean | recipient mean | donor-recipient | recipient-donor | abs delta | donor max | recipient max | abs max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mixed_timing_top3184_plus_rank4000_edge_cross_extra_probe` | `harmful` | 93 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `mixed_timing_top3184_plus_rank4000_edge_cross_extra_probe` | `harmful` | 114 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `mixed_timing_top3184_plus_rank4000_edge_cross_extra_probe` | `harmful` | 957 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `mixed_timing_top3184_plus_rank4000_edge_cross_extra_probe` | `harmful` | 1293 | 0.6673 | 0.6816 | -0.0143 | 0.0143 | 0.0582 | 6.7773 | 6.8242 | 3.0996 |
| `mixed_timing_top3184_plus_rank4000_edge_cross_extra_probe` | `harmful` | 2451 | 0.0175 | 0.0180 | -0.0005 | 0.0005 | 0.0005 | 2.8008 | 2.8867 | 0.0859 |
| `mixed_timing_top3184_plus_rank4000_edge_cross_extra_probe` | `harmful` | 4983 | 0.0198 | 0.0200 | -0.0002 | 0.0002 | 0.0002 | 3.1602 | 3.1992 | 0.0391 |
| `mixed_timing_top3184_plus_rank4000_edge_cross_extra_probe` | `harmful` | 6273 | 0.0178 | 0.0175 | 0.0002 | -0.0002 | 0.0002 | 2.8438 | 2.8047 | 0.0391 |

## Exported Events

- Exported `67` top token-level events.
- Events include token text, context, prompt, source model, split, feature ID, and metric.
- This is an audit substrate. It does not assign semantic labels by itself.
