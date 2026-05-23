# Gemma-2-2B Linear Merge Activation Patch Generation

Donor alpha: `1`.
Recipient alpha: `0.75`.
Patch position: `all`.
Prompts: stage3/data/gemma2_feature16048_family_prompts/fake_id_family_v1_expanded.jsonl.

| model | harmful clean | harmful attempt | unsafe | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `activation_patch_16:mlp` | 0.000 | 0.917 | 0.042 | 0.000 | 0.083 |
