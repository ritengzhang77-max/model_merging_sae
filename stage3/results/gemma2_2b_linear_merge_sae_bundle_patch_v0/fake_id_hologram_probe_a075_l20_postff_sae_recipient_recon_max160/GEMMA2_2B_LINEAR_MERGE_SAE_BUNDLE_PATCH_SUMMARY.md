# Gemma-2-2B Linear Merge SAE Bundle Patch

Donor alpha: `1`.
Recipient alpha: `0.75`.
Patch mode: `recipient_recon`.
Patch token filter: `all`.
Prompts: stage3/data/gemma2_feature16048_family_prompts/fake_id_hologram_probe_v0.jsonl.

## Generation

| model | harmful clean | harmful attempt | unsafe | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `bundle_patch_l20_recipient_recon` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Bundle Sizes

| bundle | selected features |
|---|---:|
| `l20_recipient_recon` | 1 |
