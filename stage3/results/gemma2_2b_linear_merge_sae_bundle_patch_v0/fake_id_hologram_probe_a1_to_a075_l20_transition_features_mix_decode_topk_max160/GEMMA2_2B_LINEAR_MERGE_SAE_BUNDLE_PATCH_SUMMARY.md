# Gemma-2-2B Linear Merge SAE Bundle Patch

Donor alpha: `1`.
Recipient alpha: `0.75`.
Patch mode: `mix_decode`.
Patch token filter: `all`.
Prompts: stage3/data/gemma2_feature16048_family_prompts/fake_id_hologram_probe_v0.jsonl.

## Generation

| model | harmful clean | harmful attempt | unsafe | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `bundle_patch_l20_top1` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `bundle_patch_l20_top2` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `bundle_patch_l20_top5` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `bundle_patch_l20_top10` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `bundle_patch_l20_top20` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `bundle_patch_l20_top30` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Bundle Sizes

| bundle | selected features |
|---|---:|
| `l20_top1` | 1 |
| `l20_top2` | 2 |
| `l20_top5` | 5 |
| `l20_top10` | 10 |
| `l20_top20` | 20 |
| `l20_top30` | 30 |
