# Gemma-2-2B Linear Merge SAE Feature Event Audit

Source records: `stage3/results/gemma2_2b_linear_merge_sae_mixed_timing_bundle_patch_v0/fake_id_hologram_probe_a1_to_a075_l20_top3184_edge_cross_success_max160/gemma2_2b_linear_merge_sae_mixed_timing_bundle_patch_records.jsonl`.
Layer: `20`.
Feature IDs: `6273,957,93,114,1293,4983,2451`.
Donor alpha: `1.0`.
Recipient alpha: `0.75`.
Event token scope: `all`.

## Aggregate Generation-Token Means

| source model | split | feature | donor mean | recipient mean | donor-recipient | recipient-donor | abs delta | donor max | recipient max | abs max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 93 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 114 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 957 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 1293 | 0.3120 | 0.3256 | -0.0137 | 0.0137 | 0.0522 | 7.8711 | 8.1953 | 3.3691 |
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 2451 | 0.0904 | 0.1064 | -0.0160 | 0.0160 | 0.0190 | 5.5430 | 5.5664 | 2.7227 |
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 4983 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 6273 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Exported Events

- Exported `70` top token-level events.
- Events include token text, context, prompt, source model, split, feature ID, and metric.
- This is an audit substrate. It does not assign semantic labels by itself.
