# Long-Generation Safety Rescore

Source records: `stage3/results/gemma2_2b_linear_merge_activation_patch_generation_v0/fake_id_family_v1_a1_to_a075_layer16_mlp_max160/gemma2_2b_linear_merge_activation_patch_generation_records.jsonl`.

This summary ignores the short-generation `too_long` clean/helpful penalty and reports separate long-generation signals.
The strict columns recompute harmful safety from the prompt/text to catch direct procedural compliance and broader warning/refusal phrasings.

| model | harmful attempted | strict attempted | attempted and no unsafe | unsafe continuation | strict unsafe | strict safe | no attempt | strict no attempt | benign not over-refusal | benign over-refusal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `activation_patch_16:mlp` | 0.917 | 0.958 | 0.875 | 0.042 | 0.083 | 0.917 | 0.083 | 0.042 | 0.917 | 0.083 |
