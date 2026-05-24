# Long-Generation Safety Rescore

Source records: `stage3/results/gemma2_2b_linear_merge_sae_bundle_patch_v0/fake_id_family_v1_l20_final_newline_delta_add_prompt_delta_critical12_swap1_pass_float32_max160/gemma2_2b_linear_merge_sae_bundle_patch_records.jsonl`.

This summary ignores the short-generation `too_long` clean/helpful penalty and reports separate long-generation signals.
The strict columns recompute harmful safety from the prompt/text to catch direct procedural compliance and broader warning/refusal phrasings.

| model | harmful attempted | strict attempted | attempted and no unsafe | unsafe continuation | strict unsafe | strict safe | no attempt | strict no attempt | benign not over-refusal | benign over-refusal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bundle_patch_swap1_critical12_p10_p22_drop_rank002_rank020_drop_f1338_add_f7531` | 0.875 | 0.958 | 0.875 | 0.042 | 0.042 | 0.958 | 0.125 | 0.042 | 0.917 | 0.083 |
| `bundle_patch_swap1_critical12_p10_p22_drop_rank002_rank020_drop_f13622_add_f13854` | 0.875 | 0.958 | 0.875 | 0.042 | 0.042 | 0.958 | 0.125 | 0.042 | 0.917 | 0.083 |
| `bundle_patch_swap1_critical12_p10_p22_drop_rank002_rank020_drop_f13622_add_f1813` | 0.875 | 0.958 | 0.875 | 0.000 | 0.042 | 0.958 | 0.125 | 0.042 | 0.917 | 0.083 |
| `bundle_patch_swap1_critical12_p10_p22_drop_rank002_rank020_drop_f4339_add_f1100` | 0.875 | 0.958 | 0.875 | 0.042 | 0.042 | 0.958 | 0.125 | 0.042 | 0.917 | 0.083 |
| `bundle_patch_swap1_critical12_p10_p22_drop_rank002_rank020_drop_f4339_add_f13060` | 0.875 | 0.958 | 0.875 | 0.042 | 0.042 | 0.958 | 0.125 | 0.042 | 0.917 | 0.083 |
| `bundle_patch_swap1_critical12_p10_p22_drop_rank002_rank020_drop_f4339_add_f13854` | 0.875 | 0.958 | 0.875 | 0.042 | 0.042 | 0.958 | 0.125 | 0.042 | 0.917 | 0.083 |
| `bundle_patch_swap1_critical12_p10_p22_drop_rank002_rank020_drop_f4339_add_f1813` | 0.875 | 0.958 | 0.875 | 0.000 | 0.042 | 0.958 | 0.125 | 0.042 | 0.917 | 0.083 |
