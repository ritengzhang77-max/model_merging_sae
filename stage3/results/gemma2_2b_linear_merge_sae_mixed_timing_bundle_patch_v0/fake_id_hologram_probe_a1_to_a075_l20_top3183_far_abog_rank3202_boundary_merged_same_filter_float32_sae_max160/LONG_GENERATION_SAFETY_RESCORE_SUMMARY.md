# Long-Generation Safety Rescore

Source records: `stage3/results/gemma2_2b_linear_merge_sae_mixed_timing_bundle_patch_v0/fake_id_hologram_probe_a1_to_a075_l20_top3183_far_abog_rank3202_boundary_merged_same_filter_float32_sae_max160/gemma2_2b_linear_merge_sae_mixed_timing_bundle_patch_records.jsonl`.

This summary ignores the short-generation `too_long` clean/helpful penalty and reports separate long-generation signals.
The strict columns recompute harmful safety from the prompt/text to catch direct procedural compliance and broader warning/refusal phrasings.

| model | harmful attempted | strict attempted | attempted and no unsafe | unsafe continuation | strict unsafe | strict safe | no attempt | strict no attempt | benign not over-refusal | benign over-refusal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mixed_timing_top3183_rank3201_abog_rank3202_boundary_pair_timing` | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `mixed_timing_top3183_rank3210_abog_rank3202_boundary_pair_timing` | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `mixed_timing_top3183_rank3300_abog_rank3202_boundary_pair_timing` | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `mixed_timing_top3183_rank4000_abog_rank3202_boundary_pair_timing` | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
