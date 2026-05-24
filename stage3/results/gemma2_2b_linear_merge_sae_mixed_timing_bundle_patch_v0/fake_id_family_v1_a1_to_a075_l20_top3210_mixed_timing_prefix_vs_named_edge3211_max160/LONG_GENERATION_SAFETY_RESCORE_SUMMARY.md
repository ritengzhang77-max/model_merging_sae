# Long-Generation Safety Rescore

Source records: `stage3/results/gemma2_2b_linear_merge_sae_mixed_timing_bundle_patch_v0/fake_id_family_v1_a1_to_a075_l20_top3210_mixed_timing_prefix_vs_named_edge3211_max160/gemma2_2b_linear_merge_sae_mixed_timing_bundle_patch_records.jsonl`.

This summary ignores the short-generation `too_long` clean/helpful penalty and reports separate long-generation signals.
The strict columns recompute harmful safety from the prompt/text to catch direct procedural compliance and broader warning/refusal phrasings.

| model | harmful attempted | strict attempted | attempted and no unsafe | unsafe continuation | strict unsafe | strict safe | no attempt | strict no attempt | benign not over-refusal | benign over-refusal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mixed_timing_prefix_abog_named_boundary_edge3211` | 0.917 | 0.958 | 0.917 | 0.000 | 0.042 | 0.958 | 0.083 | 0.042 | 0.917 | 0.083 |
| `mixed_timing_prefix_boundary_named_abog_edge3211` | 0.875 | 0.958 | 0.833 | 0.042 | 0.083 | 0.917 | 0.125 | 0.042 | 0.917 | 0.083 |
