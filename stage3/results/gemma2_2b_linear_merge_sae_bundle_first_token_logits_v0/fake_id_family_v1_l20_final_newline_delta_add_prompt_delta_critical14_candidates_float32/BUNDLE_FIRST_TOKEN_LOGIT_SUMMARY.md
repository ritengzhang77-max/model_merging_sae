# Gemma-2-2B Linear Merge SAE Bundle First-Token Logit Audit

Prompt file: `stage3/data/gemma2_feature16048_family_prompts/fake_id_family_v1_expanded.jsonl`.
Bundles file: `stage3/results/gemma2_2b_linear_merge_sae_bundle_patch_v0/fake_id_hologram_l20_final_newline_delta_add_prompt_delta_critical14_candidates_float32_max160/critical14_candidate_bundles.txt`.
Donor alpha: `1`.
Recipient alpha: `0.75`.
Patch mode: `delta_add`.
Patch token filter: `assistant_boundary_final_newline`.

| condition | split | n | top I | top It | mean I-It | min I-It | max I-It |
|---|---|---:|---:|---:|---:|---:|---:|
| `bundle_patch_critical12_p10_p22` | `benign` | 24 | 0.083 | 0.292 | -3.6787 | -6.9297 | 0.8281 |
| `bundle_patch_critical12_p10_p22` | `harmful` | 24 | 0.958 | 0.000 | 4.2793 | -4.3359 | 7.8594 |
| `bundle_patch_critical12_p10_p31` | `benign` | 24 | 0.083 | 0.292 | -3.6794 | -6.9297 | 0.8281 |
| `bundle_patch_critical12_p10_p31` | `harmful` | 24 | 0.958 | 0.000 | 4.2839 | -4.3359 | 7.8672 |
| `bundle_patch_critical12_p22_p23` | `benign` | 24 | 0.083 | 0.292 | -3.6794 | -6.9297 | 0.8125 |
| `bundle_patch_critical12_p22_p23` | `harmful` | 24 | 0.958 | 0.000 | 4.2793 | -4.3359 | 7.8516 |
| `bundle_patch_critical12_p22_p29` | `benign` | 24 | 0.083 | 0.292 | -3.6794 | -6.9297 | 0.8125 |
| `bundle_patch_critical12_p22_p29` | `harmful` | 24 | 0.958 | 0.000 | 4.2793 | -4.3359 | 7.8516 |
| `bundle_patch_critical12_p23_p31` | `benign` | 24 | 0.083 | 0.292 | -3.6800 | -6.9297 | 0.8125 |
| `bundle_patch_critical12_p23_p31` | `harmful` | 24 | 0.958 | 0.000 | 4.2848 | -4.3359 | 7.8594 |
| `linear_alpha_0.75` | `benign` | 24 | 0.042 | 0.292 | -3.7074 | -6.9297 | 0.6719 |
| `linear_alpha_0.75` | `harmful` | 24 | 0.917 | 0.042 | 4.1094 | -4.3359 | 7.9062 |
| `linear_alpha_1` | `benign` | 24 | 0.083 | 0.292 | -3.5259 | -7.1250 | 1.5469 |
| `linear_alpha_1` | `harmful` | 24 | 0.958 | 0.042 | 5.8685 | -3.6719 | 9.1094 |
