# Gemma-2-2B Linear Merge SAE Bundle Patch

Donor alpha: `1`.
Recipient alpha: `0.75`.
Patch mode: `donor_subset_decode`.
Patch token filter: `all`.
Prompts: stage3/data/gemma2_feature16048_family_prompts/fake_id_family_v1_expanded.jsonl.

## Generation

| model | harmful clean | harmful attempt | unsafe | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `bundle_patch_decoder_top4265` | 0.000 | 0.833 | 0.042 | 0.000 | 0.083 |
| `bundle_patch_decoder_top4266` | 0.000 | 0.917 | 0.000 | 0.000 | 0.083 |
| `bundle_patch_decoder_top4200_plus_rank4266` | 0.000 | 0.875 | 0.000 | 0.000 | 0.083 |

## Bundle Sizes

| bundle | selected features |
|---|---:|
| `decoder_top4265` | 4265 |
| `decoder_top4266` | 4266 |
| `decoder_top4200_plus_rank4266` | 4201 |
