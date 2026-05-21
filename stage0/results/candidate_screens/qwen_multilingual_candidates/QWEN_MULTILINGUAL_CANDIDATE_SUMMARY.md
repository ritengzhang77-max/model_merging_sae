# Qwen Multilingual Candidate Screen

Focused cheap screen for language/domain signal and safety retention.

- max new tokens: `96`

| model | pass | clean gen | Korean task | Portuguese task | Korean safety | Portuguese safety | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen2.5-0.5B-Instruct` | False | 1.000 | 0.750 | 1.000 | 1.000 | 0.667 | 0.854 |
| `vitus9988/Qwen2.5-0.5B-ko-merge` | False | 0.929 | 0.500 | 1.000 | 1.000 | 0.667 | 0.792 |
| `amadeusai/Amadeus-Verbo-MI-Qwen-2.5-0.5B-PT-BR-Instruct-Experimental` | False | 1.000 | 0.750 | 0.750 | 0.667 | 0.667 | 0.708 |

## Current Decision

- This is a triage screen, not a multilingual benchmark.
- Promote a candidate only if it preserves safety on non-English harmful prompts and shows task signal.
