# Long-Generation Safety Rescore

Source records: `stage3/results/gemma2_2b_linear_merge_sae_bundle_patch_v0/default_eval0_12_a1_to_a075_l20_decoder_contrib_top3325_rank4266_abog_max160/gemma2_2b_linear_merge_sae_bundle_patch_records.jsonl`.

This summary ignores the short-generation `too_long` clean/helpful penalty and reports separate long-generation signals.
The strict columns recompute harmful safety from the prompt/text to catch direct procedural compliance and broader warning/refusal phrasings.

| model | harmful attempted | strict attempted | attempted and no unsafe | unsafe continuation | strict unsafe | strict safe | no attempt | strict no attempt | benign not over-refusal | benign over-refusal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bundle_patch_decoder_top3325_plus_rank4266` | 0.917 | 1.000 | 0.833 | 0.083 | 0.000 | 1.000 | 0.083 | 0.000 | 1.000 | 0.000 |
