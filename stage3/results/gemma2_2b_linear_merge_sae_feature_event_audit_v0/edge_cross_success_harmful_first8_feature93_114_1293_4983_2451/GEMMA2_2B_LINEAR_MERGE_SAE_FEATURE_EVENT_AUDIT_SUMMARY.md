# Gemma-2-2B Linear Merge SAE Feature Event Audit

Source records: `stage3/results/gemma2_2b_linear_merge_sae_mixed_timing_bundle_patch_v0/fake_id_family_v1_a1_to_a075_l20_top3210_mixed_timing_edge_cross_success_max160/gemma2_2b_linear_merge_sae_mixed_timing_bundle_patch_records.jsonl`.
Layer: `20`.
Feature IDs: `93,114,1293,4983,2451`.
Donor alpha: `1.0`.
Recipient alpha: `0.75`.
Event token scope: `all`.

## Aggregate Generation-Token Means

| source model | split | feature | donor mean | recipient mean | donor-recipient | recipient-donor | abs delta | donor max | recipient max | abs max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 93 | 0.0124 | 0.0103 | 0.0021 | -0.0021 | 0.0021 | 4.7070 | 4.6602 | 2.5977 |
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 114 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 1293 | 0.7025 | 0.6964 | 0.0061 | -0.0061 | 0.0409 | 8.8594 | 8.9375 | 3.0996 |
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 2451 | 0.0544 | 0.0576 | -0.0032 | 0.0032 | 0.0080 | 3.9023 | 3.8555 | 2.8867 |
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 4983 | 0.0065 | 0.0045 | 0.0020 | -0.0020 | 0.0021 | 3.1602 | 3.1992 | 2.5898 |

## Exported Events

- Exported `180` top token-level events.
- Events include token text, context, prompt, source model, split, feature ID, and metric.
- This is an audit substrate. It does not assign semantic labels by itself.
