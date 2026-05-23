# Long-Generation Safety Rescore

Source records: `stage3/results/gemma2_2b_linear_weight_merge_sweep_v0/fake_id_family_v1_alpha05_1_max160_audit/gemma2_2b_linear_weight_merge_records.jsonl`.

This summary ignores the short-generation `too_long` clean/helpful penalty and reports separate long-generation signals.
The strict columns recompute harmful safety from the prompt/text to catch direct procedural compliance and broader warning/refusal phrasings.

| model | harmful attempted | strict attempted | attempted and no unsafe | unsafe continuation | strict unsafe | strict safe | no attempt | strict no attempt | benign not over-refusal | benign over-refusal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `linear_alpha_0.5` | 0.625 | 0.792 | 0.583 | 0.083 | 0.208 | 0.708 | 0.375 | 0.208 | 1.000 | 0.000 |
| `linear_alpha_1` | 0.958 | 0.958 | 0.958 | 0.000 | 0.042 | 0.958 | 0.042 | 0.042 | 0.917 | 0.083 |
