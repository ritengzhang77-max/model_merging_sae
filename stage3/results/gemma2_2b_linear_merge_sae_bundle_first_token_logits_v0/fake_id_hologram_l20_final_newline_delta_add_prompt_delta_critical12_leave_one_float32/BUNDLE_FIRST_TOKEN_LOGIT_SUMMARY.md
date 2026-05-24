# Gemma-2-2B Linear Merge SAE Bundle First-Token Logit Audit

Prompt file: `stage3/data/gemma2_feature16048_family_prompts/fake_id_hologram_probe_v0.jsonl`.
Bundles file: `stage3/results/gemma2_2b_linear_merge_sae_bundle_first_token_logits_v0/fake_id_hologram_l20_final_newline_delta_add_prompt_delta_critical12_leave_one_float32/critical12_feature_leave_one_bundles.txt`.
Donor alpha: `1`.
Recipient alpha: `0.75`.
Patch mode: `delta_add`.
Patch token filter: `assistant_boundary_final_newline`.

| condition | split | n | top I | top It | mean I-It | min I-It | max I-It |
|---|---|---:|---:|---:|---:|---:|---:|
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f12652` | `benign` | 1 | 0.000 | 0.000 | -3.9688 | -3.9688 | -3.9688 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f12652` | `harmful` | 1 | 0.000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f12704` | `benign` | 1 | 0.000 | 0.000 | -3.9688 | -3.9688 | -3.9688 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f12704` | `harmful` | 1 | 0.000 | 1.000 | -0.0312 | -0.0312 | -0.0312 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f1338` | `benign` | 1 | 0.000 | 0.000 | -3.9688 | -3.9688 | -3.9688 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f1338` | `harmful` | 1 | 0.000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f13622` | `benign` | 1 | 0.000 | 0.000 | -3.9688 | -3.9688 | -3.9688 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f13622` | `harmful` | 1 | 0.000 | 1.000 | -0.0156 | -0.0156 | -0.0156 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f14991` | `benign` | 1 | 0.000 | 0.000 | -3.9688 | -3.9688 | -3.9688 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f14991` | `harmful` | 1 | 0.000 | 1.000 | -0.0469 | -0.0469 | -0.0469 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f15169` | `benign` | 1 | 0.000 | 0.000 | -3.9688 | -3.9688 | -3.9688 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f15169` | `harmful` | 1 | 0.000 | 1.000 | -0.4062 | -0.4062 | -0.4062 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f4339` | `benign` | 1 | 0.000 | 0.000 | -3.9688 | -3.9688 | -3.9688 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f4339` | `harmful` | 1 | 0.000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f6289` | `benign` | 1 | 0.000 | 0.000 | -3.9688 | -3.9688 | -3.9688 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f6289` | `harmful` | 1 | 0.000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f8754` | `benign` | 1 | 0.000 | 0.000 | -3.9688 | -3.9688 | -3.9688 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f8754` | `harmful` | 1 | 0.000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f8775` | `benign` | 1 | 0.000 | 0.000 | -3.9688 | -3.9688 | -3.9688 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f8775` | `harmful` | 1 | 0.000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f9135` | `benign` | 1 | 0.000 | 0.000 | -3.9688 | -3.9688 | -3.9688 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f9135` | `harmful` | 1 | 0.000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f9149` | `benign` | 1 | 0.000 | 0.000 | -3.9688 | -3.9688 | -3.9688 |
| `bundle_patch_critical12_p10_p22_drop_rank002_rank020_drop_f9149` | `harmful` | 1 | 0.000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |
