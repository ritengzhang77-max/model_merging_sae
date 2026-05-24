# Gemma-2-2B Linear Merge SAE Mixed-Timing Bundle Patch

Donor alpha: `1`.
Recipient alpha: `0.75`.
Prompt file: `stage3/data/gemma2_feature16048_family_prompts/fake_id_hologram_probe_v0.jsonl`.

## Generation

| model | harmful clean | harmful attempt | unsafe | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `mixed_timing_top3183_rank3201_abog_rank3202_boundary_pair_timing` | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| `mixed_timing_top3183_rank3210_abog_rank3202_boundary_pair_timing` | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| `mixed_timing_top3183_rank3300_abog_rank3202_boundary_pair_timing` | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| `mixed_timing_top3183_rank4000_abog_rank3202_boundary_pair_timing` | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |

## Bundle Groups

| variant | group | filter | layer | selected features |
|---|---|---|---:|---:|
| `top3183_rank3201_abog_rank3202_boundary_pair_timing` | `union_prefix+rank3201` | `assistant_boundary_or_generated` | 20 | 3184 |
| `top3183_rank3201_abog_rank3202_boundary_pair_timing` | `union_rank3202+boundary_base_edge3211` | `assistant_boundary` | 20 | 5 |
| `top3183_rank3201_abog_rank3202_boundary_pair_timing` | `union_edge3214` | `generated` | 20 | 1 |
| `top3183_rank3210_abog_rank3202_boundary_pair_timing` | `union_prefix+rank3210` | `assistant_boundary_or_generated` | 20 | 3184 |
| `top3183_rank3210_abog_rank3202_boundary_pair_timing` | `union_rank3202+boundary_base_edge3211` | `assistant_boundary` | 20 | 5 |
| `top3183_rank3210_abog_rank3202_boundary_pair_timing` | `union_edge3214` | `generated` | 20 | 1 |
| `top3183_rank3300_abog_rank3202_boundary_pair_timing` | `union_prefix+rank3300` | `assistant_boundary_or_generated` | 20 | 3184 |
| `top3183_rank3300_abog_rank3202_boundary_pair_timing` | `union_rank3202+boundary_base_edge3211` | `assistant_boundary` | 20 | 5 |
| `top3183_rank3300_abog_rank3202_boundary_pair_timing` | `union_edge3214` | `generated` | 20 | 1 |
| `top3183_rank4000_abog_rank3202_boundary_pair_timing` | `union_prefix+rank4000` | `assistant_boundary_or_generated` | 20 | 3184 |
| `top3183_rank4000_abog_rank3202_boundary_pair_timing` | `union_rank3202+boundary_base_edge3211` | `assistant_boundary` | 20 | 5 |
| `top3183_rank4000_abog_rank3202_boundary_pair_timing` | `union_edge3214` | `generated` | 20 | 1 |
