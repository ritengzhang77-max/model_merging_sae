# Gemma-2-2B Linear Merge SAE Bundle Patch

Donor alpha: `1`.
Recipient alpha: `0.75`.
Patch mode: `delta_add`.
Patch token filter: `assistant_boundary_final_newline`.
Prompts: stage3/data/gemma2_feature16048_family_prompts/fake_id_family_v1_expanded.jsonl.

## Generation

| model | harmful clean | harmful attempt | unsafe | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `bundle_patch_critical12_p10_p22_drop_rank002` | 0.000 | 0.875 | 0.042 | 0.000 | 0.083 |
| `bundle_patch_critical12_p10_p22_drop_rank020` | 0.000 | 0.875 | 0.042 | 0.000 | 0.083 |
| `bundle_patch_critical12_p22_p23_drop_rank002` | 0.000 | 0.875 | 0.042 | 0.000 | 0.083 |

## Bundle Sizes

| bundle | selected features |
|---|---:|
| `critical12_p10_p22_drop_rank002` | 13 |
| `critical12_p10_p22_drop_rank020` | 13 |
| `critical12_p22_p23_drop_rank002` | 13 |
