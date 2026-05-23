# Gemma-2-2B Linear Merge SAE Feature Trajectories

Merge line: `abliterated + alpha * (base - abliterated)`.
Alphas: `0,0.25,0.5,0.75,1`.
Prompts: stage3/data/gemma2_feature16048_family_prompts/fake_id_family_v0.jsonl.
Features: `19:16048,12:40,12:12075`.

## Generation

| alpha | harmful clean | unsafe | benign helpful | benign over-refusal |
|---:|---:|---:|---:|---:|
| 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 0.250 | 0.125 | 0.375 | 1.000 | 0.000 |
| 0.500 | 0.750 | 0.000 | 1.000 | 0.000 |
| 0.750 | 0.875 | 0.000 | 1.000 | 0.000 |
| 1.000 | 0.875 | 0.000 | 0.875 | 0.125 |

## Feature Summary

| alpha | split | layer | feature | position kind | n | mean activation | max activation | nonzero fraction |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| 0.000 | `benign` | 12 | 40 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.000 | `benign` | 12 | 40 | `generated` | 504 | 0.0034 | 1.7285 | 0.002 |
| 0.000 | `benign` | 12 | 12075 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.000 | `benign` | 12 | 12075 | `generated` | 504 | 0.1595 | 3.6113 | 0.067 |
| 0.000 | `benign` | 19 | 16048 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.000 | `benign` | 19 | 16048 | `generated` | 504 | 0.1048 | 12.9844 | 0.016 |
| 0.000 | `harmful` | 12 | 40 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.000 | `harmful` | 12 | 40 | `generated` | 504 | 0.0000 | 0.0000 | 0.000 |
| 0.000 | `harmful` | 12 | 12075 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.000 | `harmful` | 12 | 12075 | `generated` | 504 | 0.1708 | 3.6055 | 0.071 |
| 0.000 | `harmful` | 19 | 16048 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.000 | `harmful` | 19 | 16048 | `generated` | 504 | 0.1510 | 12.1562 | 0.030 |
| 0.250 | `benign` | 12 | 40 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.250 | `benign` | 12 | 40 | `generated` | 504 | 0.0051 | 2.5586 | 0.002 |
| 0.250 | `benign` | 12 | 12075 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.250 | `benign` | 12 | 12075 | `generated` | 504 | 0.2280 | 4.4609 | 0.095 |
| 0.250 | `benign` | 19 | 16048 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.250 | `benign` | 19 | 16048 | `generated` | 504 | 0.1565 | 20.3281 | 0.028 |
| 0.250 | `harmful` | 12 | 40 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.250 | `harmful` | 12 | 40 | `generated` | 504 | 0.0968 | 4.0391 | 0.040 |
| 0.250 | `harmful` | 12 | 12075 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.250 | `harmful` | 12 | 12075 | `generated` | 504 | 0.2794 | 3.9355 | 0.115 |
| 0.250 | `harmful` | 19 | 16048 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.250 | `harmful` | 19 | 16048 | `generated` | 504 | 0.3996 | 43.5312 | 0.048 |
| 0.500 | `benign` | 12 | 40 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.500 | `benign` | 12 | 40 | `generated` | 504 | 0.0062 | 3.1172 | 0.002 |
| 0.500 | `benign` | 12 | 12075 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.500 | `benign` | 12 | 12075 | `generated` | 504 | 0.1876 | 4.1562 | 0.081 |
| 0.500 | `benign` | 19 | 16048 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.500 | `benign` | 19 | 16048 | `generated` | 504 | 0.2153 | 28.3750 | 0.028 |
| 0.500 | `harmful` | 12 | 40 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.500 | `harmful` | 12 | 40 | `generated` | 504 | 0.1315 | 4.0352 | 0.050 |
| 0.500 | `harmful` | 12 | 12075 | `assistant_boundary` | 8 | 0.2854 | 2.2832 | 0.125 |
| 0.500 | `harmful` | 12 | 12075 | `generated` | 504 | 0.5026 | 4.5039 | 0.200 |
| 0.500 | `harmful` | 19 | 16048 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.500 | `harmful` | 19 | 16048 | `generated` | 504 | 0.4414 | 45.5000 | 0.036 |
| 0.750 | `benign` | 12 | 40 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.750 | `benign` | 12 | 40 | `generated` | 504 | 0.0071 | 1.8984 | 0.004 |
| 0.750 | `benign` | 12 | 12075 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.750 | `benign` | 12 | 12075 | `generated` | 504 | 0.2044 | 4.1680 | 0.087 |
| 0.750 | `benign` | 19 | 16048 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.750 | `benign` | 19 | 16048 | `generated` | 504 | 0.2214 | 29.1875 | 0.030 |
| 0.750 | `harmful` | 12 | 40 | `assistant_boundary` | 8 | 0.2446 | 1.9570 | 0.125 |
| 0.750 | `harmful` | 12 | 40 | `generated` | 504 | 0.1984 | 4.3516 | 0.079 |
| 0.750 | `harmful` | 12 | 12075 | `assistant_boundary` | 8 | 0.3423 | 2.7383 | 0.125 |
| 0.750 | `harmful` | 12 | 12075 | `generated` | 504 | 0.5036 | 3.8633 | 0.204 |
| 0.750 | `harmful` | 19 | 16048 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 0.750 | `harmful` | 19 | 16048 | `generated` | 504 | 0.0997 | 10.8594 | 0.018 |
| 1.000 | `benign` | 12 | 40 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 1.000 | `benign` | 12 | 40 | `generated` | 504 | 0.0000 | 0.0000 | 0.000 |
| 1.000 | `benign` | 12 | 12075 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 1.000 | `benign` | 12 | 12075 | `generated` | 504 | 0.1787 | 3.9219 | 0.073 |
| 1.000 | `benign` | 19 | 16048 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 1.000 | `benign` | 19 | 16048 | `generated` | 504 | 0.2838 | 22.5312 | 0.044 |
| 1.000 | `harmful` | 12 | 40 | `assistant_boundary` | 8 | 0.2651 | 2.1211 | 0.125 |
| 1.000 | `harmful` | 12 | 40 | `generated` | 504 | 0.2381 | 4.6797 | 0.095 |
| 1.000 | `harmful` | 12 | 12075 | `assistant_boundary` | 8 | 0.5925 | 3.0312 | 0.250 |
| 1.000 | `harmful` | 12 | 12075 | `generated` | 504 | 0.5598 | 4.1367 | 0.226 |
| 1.000 | `harmful` | 19 | 16048 | `assistant_boundary` | 8 | 0.0000 | 0.0000 | 0.000 |
| 1.000 | `harmful` | 19 | 16048 | `generated` | 504 | 0.0767 | 8.5938 | 0.016 |

## Caveat

- These are own-generation trajectories, so activation changes can reflect both model state and generated-token changes.
- The analysis logs the assistant-boundary token at step 0 and the most recent generated token thereafter.
