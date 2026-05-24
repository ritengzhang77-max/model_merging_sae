# Gemma-2-2B Linear Merge SAE Bundle Patch

Donor alpha: `1`.
Recipient alpha: `0.75`.
Patch mode: `delta_add`.
Patch token filter: `assistant_boundary_final_newline`.
Prompts: stage3/data/gemma2_feature16048_family_prompts/fake_id_family_v1_expanded.jsonl.

## Generation

| model | harmful clean | harmful attempt | unsafe | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `bundle_patch_prompt_delta_top32` | 0.000 | 0.875 | 0.083 | 0.000 | 0.083 |
| `bundle_patch_prompt_delta_top33` | 0.000 | 0.875 | 0.042 | 0.000 | 0.083 |

## Bundle Sizes

| bundle | selected features |
|---|---:|
| `prompt_delta_top32` | 32 |
| `prompt_delta_top33` | 33 |
