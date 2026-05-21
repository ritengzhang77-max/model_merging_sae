# Qwen2.5-1.5B Refusal RQ1/RQ2 Diagnostics

Cheap diagnostics before SAE/transcoder work.

## Stage 0 Behavior

| model | harmful clean | benign helpful | arith | polite |
|---|---:|---:|---:|---:|
| `Qwen/Qwen2.5-1.5B-Instruct` | 0.875 | 1.000 | 1.000 | 1.000 |
| `bunnycore/Qwen2.5-1.5B-Matrix` | 0.375 | 1.000 | 0.750 | 0.000 |
| `Yuuta208/Qwen2.5-1.5B-Instruct-Qwen2.5-Math-1.5B-Merged-slerp-24` | 0.625 | 1.000 | 0.875 | 0.000 |
| `nbeerbower/EVA-abliterated-TIES-Qwen2.5-1.5B` | 0.000 | 1.000 | 1.000 | 0.500 |

## Refusal Target Loss

Lower `harmful_refusal_loss` means the model assigns higher likelihood to a standard refusal completion on harmful prompts. `benign_refusal_loss` is an over-refusal diagnostic.

| model | harmful refusal loss | benign refusal loss | arith answer loss |
|---|---:|---:|---:|
| `base` | 0.089 | 0.460 | 0.000 |
| `matrix` | 1.736 | 2.321 | 0.707 |
| `math_slerp` | 0.933 | 1.422 | 0.632 |
| `abliterated` | 2.213 | 3.041 | 0.015 |

## Activation Drift Highlights

Base-vs-model row-wise cosine. Lower values indicate stronger representation drift.

| model | layer | harmful | benign | arith | polite | benign-minus-harmful |
|---|---:|---:|---:|---:|---:|---:|
| `abliterated` | 22 | 0.724 | 0.899 | 0.956 | 0.906 | 0.175 |
| `abliterated` | 24 | 0.726 | 0.894 | 0.948 | 0.889 | 0.169 |
| `abliterated` | 23 | 0.745 | 0.904 | 0.954 | 0.904 | 0.159 |
| `abliterated` | 21 | 0.742 | 0.901 | 0.951 | 0.914 | 0.158 |
| `abliterated` | 20 | 0.755 | 0.904 | 0.948 | 0.919 | 0.149 |
| `abliterated` | 19 | 0.776 | 0.904 | 0.948 | 0.926 | 0.128 |
| `abliterated` | 25 | 0.777 | 0.904 | 0.947 | 0.898 | 0.127 |
| `abliterated` | 18 | 0.793 | 0.913 | 0.944 | 0.931 | 0.120 |
| `abliterated` | 27 | 0.785 | 0.902 | 0.912 | 0.874 | 0.117 |
| `abliterated` | 26 | 0.821 | 0.915 | 0.946 | 0.904 | 0.094 |
| `abliterated` | 17 | 0.831 | 0.915 | 0.947 | 0.935 | 0.084 |
| `abliterated` | 16 | 0.867 | 0.925 | 0.952 | 0.939 | 0.058 |
| `matrix` | 20 | 0.844 | 0.896 | 0.790 | 0.771 | 0.052 |
| `matrix` | 24 | 0.871 | 0.923 | 0.794 | 0.781 | 0.052 |
| `matrix` | 22 | 0.866 | 0.915 | 0.801 | 0.773 | 0.049 |
| `matrix` | 21 | 0.864 | 0.910 | 0.789 | 0.776 | 0.046 |

## Base-to-Abliterated Module Patches

Patches replace one base module into the abliterated model and evaluate target losses. `harmful gap closed` is relative to the base-vs-abliterated harmful-refusal target-loss gap. `specificity` subtracts benign-refusal gap closure.

| patch | harmful loss | harmful gap closed | benign gap closed | specificity | arith delta |
|---|---:|---:|---:|---:|---:|
| `16:block` | 1.855 | 0.169 | 0.073 | 0.096 | -0.003 |
| `16:mlp` | 1.708 | 0.238 | 0.161 | 0.077 | -0.003 |
| `20:block` | 2.071 | 0.067 | 0.036 | 0.031 | 0.000 |
| `12:block` | 1.968 | 0.115 | 0.097 | 0.018 | 0.002 |
| `20:attn` | 2.176 | 0.017 | 0.000 | 0.017 | 0.002 |
| `8:attn` | 2.196 | 0.008 | -0.008 | 0.016 | 0.000 |
| `20:mlp` | 2.104 | 0.051 | 0.035 | 0.016 | -0.001 |
| `12:mlp` | 1.829 | 0.181 | 0.165 | 0.016 | 0.000 |
| `27:block` | 2.288 | -0.035 | -0.047 | 0.011 | -0.009 |
| `8:block` | 1.936 | 0.130 | 0.119 | 0.011 | -0.003 |
| `23:attn` | 2.138 | 0.035 | 0.027 | 0.008 | -0.001 |
| `4:attn` | 2.186 | 0.013 | 0.006 | 0.007 | -0.001 |
| `23:block` | 2.042 | 0.080 | 0.074 | 0.006 | -0.003 |
| `24:attn` | 2.171 | 0.020 | 0.014 | 0.006 | -0.000 |
| `16:attn` | 2.405 | -0.090 | -0.095 | 0.005 | 0.001 |
| `27:attn` | 2.100 | 0.053 | 0.049 | 0.004 | -0.003 |
| `24:norms` | 2.213 | -0.000 | -0.000 | 0.000 | -0.000 |
| `20:norms` | 2.212 | 0.000 | 0.000 | 0.000 | -0.000 |
| `0:norms` | 2.213 | -0.000 | -0.000 | -0.000 | -0.000 |
| `8:mlp` | 1.950 | 0.124 | 0.124 | -0.000 | -0.004 |

## Current Decision

- Treat this as an RQ1/RQ2 localization screen, not final causal evidence.
- Promote layers/modules only if they close harmful-refusal loss more than benign-refusal loss and do not damage arithmetic answer loss.
- SAE/transcoder work remains gated on this baseline module analysis.
