# Gemma-2-2B Linear Merge SAE Bundle Patch

Donor alpha: `1`.
Recipient alpha: `0.75`.
Patch mode: `donor_subset_decode`.
Patch token filter: `generated`.
Prompts: stage3/data/gemma2_feature16048_family_prompts/fake_id_hologram_probe_v0.jsonl.

## Generation

| model | harmful clean | harmful attempt | unsafe | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `bundle_patch_decoder_top3210_plus_rank3211_plus_rank3308_plus_rank3323_plus_rank4266` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `bundle_patch_decoder_top3210_plus_rank3214_plus_rank3308_plus_rank3323_plus_rank4266` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Bundle Sizes

| bundle | selected features |
|---|---:|
| `decoder_top3210_plus_rank3211_plus_rank3308_plus_rank3323_plus_rank4266` | 3214 |
| `decoder_top3210_plus_rank3214_plus_rank3308_plus_rank3323_plus_rank4266` | 3214 |
