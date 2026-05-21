# Qwen Public Candidate RQ0 Screen

This is a small RQ0-style screen for public Qwen merge candidates that passed or nearly passed Stage 0.

Models:

- `qwen2_5_instruct`: `Qwen/Qwen2.5-0.5B-Instruct`
- `sjt`: `Sakalti/SJT-0.5B`
- `ko_merge`: `vitus9988/Qwen2.5-0.5B-ko-merge`
- `amadeus_ptbr`: `amadeusai/Amadeus-Verbo-MI-Qwen-2.5-0.5B-PT-BR-Instruct-Experimental`

## Stage 0 Behavior

| model | substrate pass | refusal pass | clean gen | harmful clean | benign helpful | arith | polite |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Sakalti/SJT-0.5B` | True | True | 1.000 | 0.875 | 1.000 | 1.000 | 0.875 |
| `vitus9988/Qwen2.5-0.5B-ko-merge` | True | True | 1.000 | 0.875 | 0.875 | 0.875 | 0.750 |
| `amadeusai/Amadeus-Verbo-MI-Qwen-2.5-0.5B-PT-BR-Instruct-Experimental` | True | False | 1.000 | 0.125 | 1.000 | 1.000 | 0.750 |

## Sampled Delta Magnitude

Deltas are sampled relative to `qwen2_5_instruct` using the same sample as the cosine table.

| model | sampled L2 | mean abs | max abs | n values |
|---|---:|---:|---:|---:|
| `sjt` | 0.000 | 0.000000 | 0.000 | 106260 |
| `ko_merge` | 0.152 | 0.000236 | 0.011 | 106260 |
| `amadeus_ptbr` | 1.489 | 0.001334 | 0.113 | 106260 |

## Delta Geometry

Deltas are sampled relative to `qwen2_5_instruct`.

| pair | cosine | sign agreement |
|---|---:|---:|
| `ko_merge` vs `amadeus_ptbr` | -0.003 | 0.277 |
| `sjt` vs `amadeus_ptbr` | -0.023 | 0.001 |
| `sjt` vs `ko_merge` | 0.001 | 0.002 |

## Activation Similarity Highlights

Mean row-wise cosine at layer 23 on harmful and benign prompts:

| pair | harmful | benign |
|---|---:|---:|
| `ko_merge` vs `amadeus_ptbr` | 0.901 | 0.939 |
| `qwen2_5_instruct` vs `amadeus_ptbr` | 0.959 | 0.966 |
| `qwen2_5_instruct` vs `ko_merge` | 0.926 | 0.962 |
| `qwen2_5_instruct` vs `sjt` | 1.000 | 1.000 |
| `sjt` vs `amadeus_ptbr` | 0.959 | 0.966 |
| `sjt` vs `ko_merge` | 0.926 | 0.962 |

## Current Decision

- `sjt` passes behavior, but it is effectively a near-copy of `qwen2_5_instruct`.
  Its sampled delta L2 is only `1.65e-05`, and final-layer prompt activations
  have cosine `1.000` against the anchor. Do not promote it as a real merge
  target.
- `ko_merge` is the best round-3 positive candidate. It passes the cheap screen
  and has a small but nonzero displacement from the anchor, with lower
  final-layer similarity on harmful and polite prompts.
- `amadeus_ptbr` is useful as a contrast case: it has a larger displacement from
  the anchor but failed harmful-refusal behavior.
- Next check should be a focused multilingual/domain screen before any SAE work:
  English safety retention plus Korean/Portuguese task prompts for `ko_merge`
  and `amadeus_ptbr`.
