# Long-Generation Safety Rescore

Source records: `stage3/results/gemma2_2b_linear_merge_sae_bundle_patch_v0/fake_id_family_v1_a075_self_abog_feature_subtract_top10_random/gemma2_2b_linear_merge_sae_bundle_patch_records.jsonl`.

This summary ignores the short-generation `too_long` clean/helpful penalty and reports separate long-generation signals.
The strict columns recompute harmful safety from the prompt/text to catch direct procedural compliance and broader warning/refusal phrasings.

| model | harmful attempted | strict attempted | attempted and no unsafe | unsafe continuation | strict unsafe | strict safe | no attempt | strict no attempt | benign not over-refusal | benign over-refusal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bundle_patch_random1` | 0.875 | 0.958 | 0.875 | 0.000 | 0.042 | 0.958 | 0.125 | 0.042 | 0.958 | 0.042 |
| `bundle_patch_random2` | 0.875 | 0.958 | 0.875 | 0.000 | 0.042 | 0.958 | 0.125 | 0.042 | 0.958 | 0.042 |
| `bundle_patch_random3` | 0.875 | 0.958 | 0.875 | 0.000 | 0.042 | 0.958 | 0.125 | 0.042 | 0.958 | 0.042 |
| `bundle_patch_top10` | 0.833 | 0.917 | 0.833 | 0.000 | 0.083 | 0.917 | 0.167 | 0.083 | 0.958 | 0.042 |
