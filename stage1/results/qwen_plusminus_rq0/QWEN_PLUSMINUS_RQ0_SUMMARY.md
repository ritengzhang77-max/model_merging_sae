# Qwen Plus/Minus RQ0 Screen

This is a small RQ0-style screen for the first promising public merge pair.

Models:

- `qwen2_5_instruct`: `Qwen/Qwen2.5-0.5B-Instruct`
- `qwen2_instruct`: `Qwen/Qwen2-0.5B-Instruct`
- `plus2`: `djuna-test-lab/Qwen2.5Plus2-0.5B-Instruct`
- `minus2`: `mergekit-community/Qwen2.5Minus2-0.5B-Instruct`
- `slerp`: `mergekit-community/mergekit-slerp-lxmmvuv`

## Stage 0 Behavior

| model | clean gen | harmful clean | benign helpful | arith | polite |
|---|---:|---:|---:|---:|---:|
| `qwen2_5_instruct` | 1.000 | 0.875 | 0.875 | 0.875 | 1.000 |
| `qwen2_instruct` | 1.000 | 0.875 | 1.000 | 0.500 | 1.000 |
| `plus2` | 1.000 | 1.000 | 0.875 | 1.000 | 0.750 |
| `minus2` | 1.000 | 0.000 | 0.875 | 0.000 | 0.000 |
| `slerp` | 1.000 | 0.750 | 0.875 | 0.875 | 1.000 |

## Weight Delta Geometry

Deltas are sampled relative to `qwen2_5_instruct`.

| pair | cosine | sign agreement |
|---|---:|---:|
| `minus2` vs `slerp` | 0.022 | 0.313 |
| `plus2` vs `minus2` | -1.000 | 0.000 |
| `plus2` vs `slerp` | -0.022 | 0.247 |
| `qwen2_instruct` vs `minus2` | -1.000 | 0.000 |
| `qwen2_instruct` vs `plus2` | 1.000 | 1.000 |
| `qwen2_instruct` vs `slerp` | -0.022 | 0.247 |

Exact-state follow-up:

- `plus2` is extremely close to `qwen2_instruct`: relative L2 difference
  0.00199 over floating tensors.
- `minus2` is a strong extrapolation away from `qwen2_5_instruct`: relative L2
  difference 1.135 from the base.

This means Plus2 is not a rich composition target by itself. It is better
understood as a near-parent positive control, while Minus2 is an extrapolated
negative-control task vector.

## Activation Similarity Highlights

Mean row-wise cosine at layer 23 on harmful and benign prompts:

| pair | harmful | benign |
|---|---:|---:|
| `minus2` vs `slerp` | -0.060 | -0.021 |
| `plus2` vs `minus2` | -0.040 | 0.000 |
| `plus2` vs `slerp` | -0.032 | -0.012 |
| `qwen2_5_instruct` vs `minus2` | -0.057 | -0.019 |
| `qwen2_5_instruct` vs `plus2` | -0.030 | -0.009 |
| `qwen2_5_instruct` vs `qwen2_instruct` | -0.030 | -0.010 |
| `qwen2_5_instruct` vs `slerp` | 0.990 | 0.994 |
| `qwen2_instruct` vs `minus2` | -0.040 | 0.000 |
| `qwen2_instruct` vs `plus2` | 1.000 | 1.000 |
| `qwen2_instruct` vs `slerp` | -0.032 | -0.012 |

## Current Decision

- `plus2` is a useful positive-control candidate, but likely too close to
  `qwen2_instruct` to serve as the main model-composition target.
- `minus2` is a useful negative-control extrapolation.
- The pair is still worth keeping for task-vector sign and safety-loss analysis.
- Continue screening for a richer public merge where the output is not simply a
  near-copy of one parent.
