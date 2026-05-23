# Gemma-2-2B Linear Merge SAE Bundle Patch

Donor alpha: `1`.
Recipient alpha: `0.75`.
Patch mode: `donor_subset_decode`.
Patch token filter: `assistant_boundary`.
Prompts: `0:12` per split.

## Generation

| model | harmful clean | harmful attempt | unsafe | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `bundle_patch_decoder_top3600_plus_rank4266` | 0.000 | 0.833 | 0.167 | 0.000 | 0.083 |

## Bundle Sizes

| bundle | selected features |
|---|---:|
| `decoder_top3600_plus_rank4266` | 3601 |
