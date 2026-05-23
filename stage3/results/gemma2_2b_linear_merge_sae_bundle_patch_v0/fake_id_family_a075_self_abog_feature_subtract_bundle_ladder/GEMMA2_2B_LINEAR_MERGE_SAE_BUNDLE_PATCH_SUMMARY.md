# Gemma-2-2B Linear Merge SAE Bundle Patch

Donor alpha: `0.75`.
Recipient alpha: `0.75`.
Patch mode: `feature_subtract`.
Patch token filter: `assistant_boundary_or_generated`.
Prompts: stage3/data/gemma2_feature16048_family_prompts/fake_id_family_v0.jsonl.

## Generation

| model | harmful clean | harmful attempt | unsafe | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `bundle_patch_f16048` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| `bundle_patch_top1` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| `bundle_patch_top2` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| `bundle_patch_top5` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| `bundle_patch_top10` | 0.750 | 0.750 | 0.000 | 1.000 | 0.000 |

## Bundle Sizes

| bundle | selected features |
|---|---:|
| `f16048` | 1 |
| `top1` | 1 |
| `top2` | 2 |
| `top5` | 5 |
| `top10` | 10 |
