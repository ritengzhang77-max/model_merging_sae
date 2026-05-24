# Gemma-2-2B Linear Merge SAE Feature Event Audit

Source records: `stage3/results/gemma2_2b_linear_merge_sae_mixed_timing_bundle_patch_v0/fake_id_hologram_probe_a1_to_a075_l20_top3200_mixed_timing_edge_cross_success_max160/gemma2_2b_linear_merge_sae_mixed_timing_bundle_patch_records.jsonl`.
Layer: `20`.
Feature IDs: `6273,5679,14554,12861,93,114,1293,4983,2451`.
Donor alpha: `1.0`.
Recipient alpha: `0.75`.
Event token scope: `all`.

## Aggregate Generation-Token Means

| source model | split | feature | donor mean | recipient mean | donor-recipient | recipient-donor | abs delta | donor max | recipient max | abs max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 93 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 114 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 1293 | 0.2928 | 0.3239 | -0.0311 | 0.0311 | 0.0347 | 7.8711 | 8.1953 | 3.3691 |
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 2451 | 0.0904 | 0.1064 | -0.0160 | 0.0160 | 0.0190 | 5.5430 | 5.5664 | 2.7227 |
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 4983 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 5679 | 0.0191 | 0.0217 | -0.0026 | 0.0026 | 0.0026 | 3.0547 | 3.4766 | 0.4219 |
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 6273 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 12861 | 0.0432 | 0.0438 | -0.0006 | 0.0006 | 0.0006 | 3.5508 | 3.6055 | 0.0547 |
| `mixed_timing_prefix_abog_boundarybase_edge3211_boundary_edge3214_generated` | `harmful` | 14554 | 0.0254 | 0.0259 | -0.0005 | 0.0005 | 0.0005 | 4.0625 | 4.1406 | 0.0781 |

## Exported Events

- Exported `85` top token-level events.
- Events include token text, context, prompt, source model, split, feature ID, and metric.
- This is an audit substrate. It does not assign semantic labels by itself.
