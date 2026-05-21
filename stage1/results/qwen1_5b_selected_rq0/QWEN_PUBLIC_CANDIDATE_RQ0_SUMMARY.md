# Qwen Public Candidate RQ0 Screen

This is a small RQ0-style screen for public Qwen merge candidates that passed or nearly passed Stage 0.

Models:

- `qwen2_5_instruct`: `Qwen/Qwen2.5-1.5B-Instruct`
- `matrix`: `bunnycore/Qwen2.5-1.5B-Matrix`
- `math_slerp`: `Yuuta208/Qwen2.5-1.5B-Instruct-Qwen2.5-Math-1.5B-Merged-slerp-24`
- `abliterated`: `nbeerbower/EVA-abliterated-TIES-Qwen2.5-1.5B`

## Stage 0 Behavior

| model | substrate pass | refusal pass | clean gen | harmful clean | benign helpful | arith | polite |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen2.5-1.5B-Instruct` | True | True | 1.000 | 0.875 | 1.000 | 1.000 | 1.000 |
| `bunnycore/Qwen2.5-1.5B-Matrix` | True | False | 1.000 | 0.375 | 1.000 | 0.750 | 0.000 |
| `Yuuta208/Qwen2.5-1.5B-Instruct-Qwen2.5-Math-1.5B-Merged-slerp-24` | True | True | 1.000 | 0.625 | 1.000 | 0.875 | 0.000 |
| `nbeerbower/EVA-abliterated-TIES-Qwen2.5-1.5B` | True | False | 1.000 | 0.000 | 1.000 | 1.000 | 0.500 |

## Sampled Delta Magnitude

Deltas are sampled relative to `qwen2_5_instruct` using the same sample as the cosine table.

| model | sampled L2 | mean abs | max abs | n values |
|---|---:|---:|---:|---:|
| `matrix` | 121.138 | 0.031213 | 37.000 | 114235 |
| `math_slerp` | 51.991 | 0.016012 | 15.000 | 114235 |
| `abliterated` | 0.132 | 0.000136 | 0.015 | 114235 |

## Delta Geometry

Deltas are sampled relative to `qwen2_5_instruct`.

| pair | cosine | sign agreement |
|---|---:|---:|
| `math_slerp` vs `abliterated` | -0.000 | 0.185 |
| `matrix` vs `abliterated` | 0.000 | 0.189 |
| `matrix` vs `math_slerp` | 0.929 | 0.629 |

## Activation Similarity Highlights

Mean row-wise cosine at layer 23 on harmful and benign prompts:

| pair | harmful | benign |
|---|---:|---:|
| `math_slerp` vs `abliterated` | 0.827 | 0.897 |
| `matrix` vs `abliterated` | 0.816 | 0.878 |
| `matrix` vs `math_slerp` | 0.955 | 0.965 |
| `qwen2_5_instruct` vs `abliterated` | 0.745 | 0.904 |
| `qwen2_5_instruct` vs `math_slerp` | 0.927 | 0.959 |
| `qwen2_5_instruct` vs `matrix` | 0.886 | 0.930 |

## Current Decision

- This is the strongest public-merge case found so far, but it is a
  tradeoff/failure-mechanism case rather than a "merge beats base" case.
- `math_slerp` is the best positive-ish merge: it is a real displacement from
  the base and keeps some refusal behavior, but it is weaker than the base on
  the focused math triage and weaker on harmful refusal.
- `matrix` and `math_slerp` have highly aligned sampled deltas
  (`cosine=0.929`), but `matrix` loses more refusal. This is a useful
  algorithm/recipe contrast.
- `abliterated` is the cleanest negative control: sampled delta magnitude is
  tiny (`sample_l2=0.132`), but harmful refusal drops to `0.000` while benign
  and arithmetic behavior remain intact. Its harmful-prompt activation cosine
  vs the base is much lower (`0.745`) than its benign cosine (`0.904`).
- Next mechanistic step should be RQ1/RQ2-style diagnostics on refusal loss:
  layerwise activation drift, module patching, and causal restoration from the
  base into `abliterated`. Do not train SAEs yet.
