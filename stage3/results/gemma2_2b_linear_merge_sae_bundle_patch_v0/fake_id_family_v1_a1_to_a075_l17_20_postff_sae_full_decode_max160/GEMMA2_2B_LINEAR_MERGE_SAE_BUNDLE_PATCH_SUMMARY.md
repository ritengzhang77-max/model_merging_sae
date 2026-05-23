# Gemma-2-2B Linear Merge SAE Bundle Patch

Donor alpha: `1`.
Recipient alpha: `0.75`.
Patch mode: `full_decode`.
Patch token filter: `all`.
Prompts: stage3/data/gemma2_feature16048_family_prompts/fake_id_family_v1_expanded.jsonl.

## Generation

| model | harmful clean | harmful attempt | unsafe | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `bundle_patch_l17_20_full_decode` | 0.000 | 0.917 | 0.042 | 0.000 | 0.083 |

## Bundle Sizes

| bundle | selected features |
|---|---:|
| `l17_20_full_decode` | 4 |
