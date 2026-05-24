# Gemma-2-2B Linear Merge SAE Bundle First-Token Logit Audit

Prompt file: `stage3/data/gemma2_feature16048_family_prompts/default_paraphrase_guard_v0.jsonl`.
Bundles file: `stage3/results/gemma2_2b_linear_merge_sae_11feature_identity_v0/broad_backbone_test_bundles.txt`.
Donor alpha: `1`.
Recipient alpha: `0.75`.
Patch mode: `delta_add`.
Patch token filter: `assistant_boundary_final_newline`.

| condition | split | n | top I | top It | mean I-It | min I-It | max I-It |
|---|---|---:|---:|---:|---:|---:|---:|
| `bundle_patch_all_pass_union14` | `benign` | 12 | 0.083 | 0.250 | -3.9635 | -7.1250 | 3.4844 |
| `bundle_patch_all_pass_union14` | `harmful` | 12 | 1.000 | 0.000 | 4.3926 | 0.5938 | 6.0156 |
| `bundle_patch_broad_backbone_14991_15169_1813` | `benign` | 12 | 0.083 | 0.250 | -3.9655 | -7.1250 | 3.4844 |
| `bundle_patch_broad_backbone_14991_15169_1813` | `harmful` | 12 | 1.000 | 0.000 | 4.4017 | 0.5938 | 6.0469 |
| `bundle_patch_broad_backbone_plus_7531` | `benign` | 12 | 0.083 | 0.250 | -3.9655 | -7.1250 | 3.4844 |
| `bundle_patch_broad_backbone_plus_7531` | `harmful` | 12 | 1.000 | 0.000 | 4.3978 | 0.5938 | 6.0312 |
| `bundle_patch_broad_rank_le100_pass_features` | `benign` | 12 | 0.083 | 0.250 | -3.9655 | -7.1250 | 3.4844 |
| `bundle_patch_broad_rank_le100_pass_features` | `harmful` | 12 | 1.000 | 0.000 | 4.3978 | 0.5938 | 6.0312 |
| `bundle_patch_broad_rank_le500_pass_features` | `benign` | 12 | 0.083 | 0.250 | -3.9655 | -7.1250 | 3.4844 |
| `bundle_patch_broad_rank_le500_pass_features` | `harmful` | 12 | 1.000 | 0.000 | 4.3926 | 0.5938 | 6.0156 |
| `bundle_patch_common_intersection9` | `benign` | 12 | 0.083 | 0.250 | -3.9642 | -7.1250 | 3.4844 |
| `bundle_patch_common_intersection9` | `harmful` | 12 | 1.000 | 0.000 | 4.3945 | 0.5938 | 6.0469 |
| `bundle_patch_core_pair_14991_15169` | `benign` | 12 | 0.083 | 0.250 | -3.9655 | -7.1250 | 3.4844 |
| `bundle_patch_core_pair_14991_15169` | `harmful` | 12 | 1.000 | 0.000 | 4.3932 | 0.5781 | 6.0469 |
| `linear_alpha_0.75` | `benign` | 12 | 0.083 | 0.250 | -3.9902 | -7.1250 | 3.4688 |
| `linear_alpha_0.75` | `harmful` | 12 | 1.000 | 0.000 | 4.2292 | 0.3594 | 6.0000 |
| `linear_alpha_1` | `benign` | 12 | 0.083 | 0.333 | -3.9329 | -7.0000 | 3.4375 |
| `linear_alpha_1` | `harmful` | 12 | 1.000 | 0.000 | 6.0853 | 1.5938 | 7.7031 |
