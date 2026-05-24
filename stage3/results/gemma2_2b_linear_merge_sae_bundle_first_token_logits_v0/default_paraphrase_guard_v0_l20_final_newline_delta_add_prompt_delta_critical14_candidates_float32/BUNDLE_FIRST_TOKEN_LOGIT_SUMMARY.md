# Gemma-2-2B Linear Merge SAE Bundle First-Token Logit Audit

Prompt file: `stage3/data/gemma2_feature16048_family_prompts/default_paraphrase_guard_v0.jsonl`.
Bundles file: `stage3/results/gemma2_2b_linear_merge_sae_bundle_patch_v0/fake_id_hologram_l20_final_newline_delta_add_prompt_delta_critical14_candidates_float32_max160/critical14_candidate_bundles.txt`.
Donor alpha: `1`.
Recipient alpha: `0.75`.
Patch mode: `delta_add`.
Patch token filter: `assistant_boundary_final_newline`.

| condition | split | n | top I | top It | mean I-It | min I-It | max I-It |
|---|---|---:|---:|---:|---:|---:|---:|
| `bundle_patch_critical12_p10_p22` | `benign` | 12 | 0.083 | 0.250 | -3.9642 | -7.1250 | 3.4844 |
| `bundle_patch_critical12_p10_p22` | `harmful` | 12 | 1.000 | 0.000 | 4.3828 | 0.5781 | 6.0156 |
| `bundle_patch_critical12_p10_p31` | `benign` | 12 | 0.083 | 0.250 | -3.9642 | -7.1250 | 3.4844 |
| `bundle_patch_critical12_p10_p31` | `harmful` | 12 | 1.000 | 0.000 | 4.3854 | 0.5938 | 6.0156 |
| `bundle_patch_critical12_p22_p23` | `benign` | 12 | 0.083 | 0.250 | -3.9642 | -7.1250 | 3.4844 |
| `bundle_patch_critical12_p22_p23` | `harmful` | 12 | 1.000 | 0.000 | 4.3770 | 0.5781 | 6.0078 |
| `bundle_patch_critical12_p22_p29` | `benign` | 12 | 0.083 | 0.250 | -3.9642 | -7.1250 | 3.4844 |
| `bundle_patch_critical12_p22_p29` | `harmful` | 12 | 1.000 | 0.000 | 4.3802 | 0.5781 | 6.0156 |
| `bundle_patch_critical12_p23_p31` | `benign` | 12 | 0.083 | 0.250 | -3.9642 | -7.1250 | 3.4844 |
| `bundle_patch_critical12_p23_p31` | `harmful` | 12 | 1.000 | 0.000 | 4.3900 | 0.5781 | 6.0078 |
| `linear_alpha_0.75` | `benign` | 12 | 0.083 | 0.250 | -3.9902 | -7.1250 | 3.4688 |
| `linear_alpha_0.75` | `harmful` | 12 | 1.000 | 0.000 | 4.2292 | 0.3594 | 6.0000 |
| `linear_alpha_1` | `benign` | 12 | 0.083 | 0.333 | -3.9329 | -7.0000 | 3.4375 |
| `linear_alpha_1` | `harmful` | 12 | 1.000 | 0.000 | 6.0853 | 1.5938 | 7.7031 |
