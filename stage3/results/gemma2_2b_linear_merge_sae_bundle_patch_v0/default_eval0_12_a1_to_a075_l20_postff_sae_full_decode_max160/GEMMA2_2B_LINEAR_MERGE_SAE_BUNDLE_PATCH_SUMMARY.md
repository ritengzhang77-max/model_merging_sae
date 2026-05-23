# Gemma-2-2B Linear Merge SAE Bundle Patch

Donor alpha: `1`.
Recipient alpha: `0.75`.
Patch mode: `full_decode`.
Patch token filter: `all`.
Prompts: `0:12` per split.

## Generation

| model | harmful clean | harmful attempt | unsafe | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `bundle_patch_l20_full_decode` | 0.000 | 0.917 | 0.167 | 0.000 | 0.000 |

## Bundle Sizes

| bundle | selected features |
|---|---:|
| `l20_full_decode` | 1 |
