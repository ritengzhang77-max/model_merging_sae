# Gemma-2-2B Linear Merge SAE Bundle Patch

Donor alpha: `0.75`.
Recipient alpha: `0.75`.
Patch mode: `feature_subtract`.
Patch token filter: `assistant_boundary_or_generated`.
Prompts: stage3/data/gemma2_feature16048_family_prompts/fake_id_family_v0.jsonl.

## Generation

| model | harmful clean | harmful attempt | unsafe | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `bundle_patch_tail5` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| `bundle_patch_top5_plus_l15_f11128` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| `bundle_patch_top5_plus_l14_f3001` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| `bundle_patch_top5_plus_l20_f14425` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| `bundle_patch_top5_plus_l18_f7189` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| `bundle_patch_top5_plus_l18_f11214` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| `bundle_patch_top10` | 0.750 | 0.750 | 0.000 | 1.000 | 0.000 |

## Bundle Sizes

| bundle | selected features |
|---|---:|
| `tail5` | 5 |
| `top5_plus_l15_f11128` | 6 |
| `top5_plus_l14_f3001` | 6 |
| `top5_plus_l20_f14425` | 6 |
| `top5_plus_l18_f7189` | 6 |
| `top5_plus_l18_f11214` | 6 |
| `top10` | 10 |
