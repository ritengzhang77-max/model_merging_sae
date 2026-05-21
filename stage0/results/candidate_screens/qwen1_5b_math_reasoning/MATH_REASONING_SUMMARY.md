# Math Reasoning Candidate Screen

Focused cheap screen for whether a math merge gives a visible task gain.

- max new tokens: `64`

| model | clean gen | math ok | mean words |
|---|---:|---:|---:|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.000 | 0.800 | 1.1 |
| `bunnycore/Qwen2.5-1.5B-Matrix` | 1.000 | 0.300 | 28.2 |
| `Yuuta208/Qwen2.5-1.5B-Instruct-Qwen2.5-Math-1.5B-Merged-slerp-24` | 1.000 | 0.600 | 33.8 |
| `nbeerbower/EVA-abliterated-TIES-Qwen2.5-1.5B` | 1.000 | 0.900 | 3.0 |

## Current Decision

- Prefer a merge only if it improves or preserves math while retaining safety in the chat screen.
- This is a triage screen, not a benchmark.
