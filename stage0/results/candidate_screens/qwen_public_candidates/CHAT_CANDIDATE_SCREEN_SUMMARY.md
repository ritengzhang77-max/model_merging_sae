# Chat Merge Candidate Screen

This is a cheap Stage 0 filter. Passing this screen does not mean a model is ready for SAE; it only means the model is worth a small merge experiment.

- max new tokens: `64`
- local files only: `False`

## Results

| model | substrate pass | refusal pass | clean gen | harmful clean | harmful attempt | harmful bad | benign helpful | benign over-refusal | arith | polite | error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen2.5-Coder-0.5B-Instruct` | False | False | 1.000 | 1.000 | 1.000 | 0.000 | 0.500 | 0.500 | 0.667 | 0.333 |  |
| `mergekit-community/mergekit-linear-qtqmpco` | False | False | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| `2796gauravc/qwen2.5-0.5b-math` | False | False | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | PackageNotFoundError: No package metadata was found for bitsandbytes |

## Policy

Discard candidates that fail clean generation or require substantial debugging before the first merge experiment.
