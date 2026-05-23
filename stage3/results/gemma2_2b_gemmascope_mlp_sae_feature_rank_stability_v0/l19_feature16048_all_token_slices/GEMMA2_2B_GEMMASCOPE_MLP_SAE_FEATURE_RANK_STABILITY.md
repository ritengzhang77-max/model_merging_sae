# GemmaScope MLP-SAE Feature Rank Stability

Feature-selection token filter: `all`.

Ranks are 1-indexed within the layer by harmful donor-recipient absolute activation delta.

| basis slice | layer | feature ID | harm-delta rank | harm delta | benign delta | harm - benign | active count |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0:4` | 19 | 16048 | 1006 | 0.115506 | 0.000140 | 0.115367 | 3 |
| `4:8` | 19 | 16048 | 843 | 0.124302 | 0.000000 | 0.124302 | 3 |
| `8:12` | 19 | 16048 | 1533 | 0.071948 | 0.000000 | 0.071948 | 2 |

## Interpretation Guardrail

- A feature that is causal under one calibration slice is more convincing if it also ranks highly under neighboring slices.
- A feature that only ranks highly under one slice should be treated as a prompt-family-specific lead, not a stable refusal mechanism.
