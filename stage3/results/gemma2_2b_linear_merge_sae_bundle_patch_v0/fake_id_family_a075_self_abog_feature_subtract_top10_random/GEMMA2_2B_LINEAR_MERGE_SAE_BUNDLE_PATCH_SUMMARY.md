# Gemma-2-2B Linear Merge SAE Bundle Patch

Donor alpha: `0.75`.
Recipient alpha: `0.75`.
Patch mode: `feature_subtract`.
Patch token filter: `assistant_boundary_or_generated`.
Prompts: stage3/data/gemma2_feature16048_family_prompts/fake_id_family_v0.jsonl.

## Generation

| model | harmful clean | harmful attempt | unsafe | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `bundle_patch_top10` | 0.750 | 0.750 | 0.000 | 1.000 | 0.000 |
| `bundle_patch_random1` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| `bundle_patch_random2` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |
| `bundle_patch_random3` | 0.875 | 0.875 | 0.000 | 1.000 | 0.000 |

## Bundle Sizes

| bundle | selected features |
|---|---:|
| `top10` | 10 |
| `random1` | 10 |
| `random2` | 10 |
| `random3` | 10 |
