# Gemma-2-2B Abliterated RQ0 Screen

This is a small pre-SAE screen for the Gemma safety-loss pair.

## Models

- `base`: `google/gemma-2-2b-it`
- `abliterated`: `IlyaGusev/gemma-2-2b-it-abliterated`

## Stage 0 Behavior

| model | substrate pass | refusal pass | clean gen | harmful clean | benign helpful | arith | polite |
|---|---:|---:|---:|---:|---:|---:|---:|
| `google/gemma-2-2b-it` | True | True | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `IlyaGusev/gemma-2-2b-it-abliterated` | True | False | 1.000 | 0.000 | 1.000 | 1.000 | 0.750 |

## Architecture

| short | type | layers | hidden | intermediate | heads | kv heads | vocab |
|---|---|---:|---:|---:|---:|---:|---:|
| `base` | `gemma2` | 26 | 2304 | 9216 | 8 | 4 | 256000 |
| `abliterated` | `gemma2` | 26 | 2304 | 9216 | 8 | 4 | 256000 |

## Activation Similarity

| layer | harmful cosine | benign cosine | arith cosine | polite cosine | harmful-benign gap |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 | -0.000 |
| 4 | 1.000 | 0.999 | 1.000 | 1.000 | -0.001 |
| 8 | 0.991 | 0.997 | 0.999 | 0.999 | 0.006 |
| 12 | 0.945 | 0.994 | 0.999 | 0.999 | 0.049 |
| 16 | 0.821 | 0.993 | 0.997 | 0.993 | 0.173 |
| 20 | 0.687 | 0.991 | 0.998 | 0.986 | 0.304 |
| 25 | 0.852 | 0.990 | 0.997 | 0.987 | 0.138 |

## Decision

- This pair passes the cheap behavior and architecture gates.
- If harmful cosine is substantially lower than benign cosine in middle/late layers, run module/activation patching next.
- Do not start GemmaScope feature interpretation until a causal patch target is established.
