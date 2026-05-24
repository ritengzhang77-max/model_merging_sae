# Long-Generation Safety Rescore

Source records: `stage3/results/gemma2_2b_linear_merge_sae_reconstruction_formula_patch_v0/fake_id_family_v1_l20_postff_final_newline_sae_delta_add_float32_max160/formula_patch_records.jsonl`.

This summary ignores the short-generation `too_long` clean/helpful penalty and reports separate long-generation signals.
The strict columns recompute harmful safety from the prompt/text to catch direct procedural compliance and broader warning/refusal phrasings.

| model | harmful attempted | strict attempted | attempted and no unsafe | unsafe continuation | strict unsafe | strict safe | no attempt | strict no attempt | benign not over-refusal | benign over-refusal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `formula_sae_delta_add` | 0.875 | 0.958 | 0.875 | 0.042 | 0.042 | 0.958 | 0.125 | 0.042 | 0.917 | 0.083 |
