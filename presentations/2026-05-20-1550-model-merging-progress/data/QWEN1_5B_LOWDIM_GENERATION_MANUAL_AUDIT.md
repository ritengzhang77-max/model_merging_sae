# Qwen2.5-1.5B Low-Dimensional Generation Manual Audit

This audit checks harmful-prompt generations from
`qwen1_5b_lowdim_activation_patch_generation_records.jsonl` because the automatic
scorer can over-credit answers that begin with a refusal but then continue with
unsafe procedural advice.

| model | auto harmful clean | strict manual clean | main issue |
|---|---:|---:|---|
| `base` | 0.875 | 1.000 | automatic scorer under-credits one safe refusal/redirect |
| `abliterated` | 0.000 | 0.000 | directly gives harmful instructions |
| `mean_delta_rank1` | 0.875 | 0.750 | two refusals drift into advice |
| `pca_rank16` | 1.000 | 0.625 | three refusals drift into advice |
| `pca_rank64` | 1.000 | 1.000 | all harmful prompts receive clean refusals |
| `random_rank64` | 0.000 | 0.000 | directly gives harmful instructions |

## Decision

- `pca_rank64` is the current strongest compressed behavioral repair.
- `pca_rank16` and `mean_delta_rank1` are useful likelihood/partial-behavior
  baselines, but the strict audit shows they are not clean enough to call solved.
- Future generation checks need a stricter contradiction/unsafe-continuation
  detector, not only a refusal-prefix detector.
