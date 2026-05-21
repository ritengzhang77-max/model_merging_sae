# Qwen2.5 Coder Merge Screen

Formula:

```text
theta = theta_qwen2.5_instruct + alpha * (theta_qwen2.5_coder_instruct - theta_qwen2.5_instruct)
```

This is a cheap candidate screen, not a final benchmark.

- base: `Qwen/Qwen2.5-0.5B-Instruct`
- coder: `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- alphas: `0.25,0.5,0.75`
- max new tokens: `96`

## Metrics

| model | alpha | pass | clean gen | harmful | benign | benign over-refusal | arith | polite | code | core mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5_instruct` | 0.0 | True | 1.000 | 1.000 | 1.000 | 0.000 | 0.667 | 1.000 | 1.000 | 0.933 |
| `qwen2.5_coder_instruct` | 1.0 | True | 1.000 | 1.000 | 1.000 | 0.000 | 0.667 | 0.333 | 1.000 | 0.800 |
| `qwen2.5_coder_linear_a0p75` | 0.75 | False | 1.000 | 0.000 | 1.000 | 0.000 | 0.667 | 0.333 | 1.000 | 0.600 |
| `qwen2.5_coder_linear_a0p5` | 0.5 | False | 0.056 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.250 | 0.050 |
| `qwen2.5_coder_linear_a0p25` | 0.25 | False | 0.167 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Decision Rule

A passing alpha is worth a small Stage 1-style mechanistic screen. A failing alpha is discarded without further debugging.
