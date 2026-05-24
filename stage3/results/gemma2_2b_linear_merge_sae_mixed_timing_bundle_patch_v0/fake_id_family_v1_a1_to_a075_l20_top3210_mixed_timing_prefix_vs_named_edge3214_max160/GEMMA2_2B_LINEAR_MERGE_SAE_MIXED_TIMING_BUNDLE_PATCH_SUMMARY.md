# Gemma-2-2B Linear Merge SAE Mixed-Timing Bundle Patch

Donor alpha: `1`.
Recipient alpha: `0.75`.
Prompt file: `stage3/data/gemma2_feature16048_family_prompts/fake_id_family_v1_expanded.jsonl`.

## Generation

| model | harmful clean | harmful attempt | unsafe | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `mixed_timing_prefix_abog_named_boundary_edge3214` | 0.000 | 0.875 | 0.000 | 0.000 | 0.083 |
| `mixed_timing_prefix_boundary_named_abog_edge3214` | 0.000 | 0.833 | 0.000 | 0.000 | 0.083 |

## Bundle Groups

| variant | group | filter | layer | selected features |
|---|---|---|---:|---:|
| `prefix_abog_named_boundary_edge3214` | `prefix` | `assistant_boundary_or_generated` | 20 | 3210 |
| `prefix_abog_named_boundary_edge3214` | `named` | `assistant_boundary` | 20 | 4 |
| `prefix_boundary_named_abog_edge3214` | `prefix` | `assistant_boundary` | 20 | 3210 |
| `prefix_boundary_named_abog_edge3214` | `named` | `assistant_boundary_or_generated` | 20 | 4 |
