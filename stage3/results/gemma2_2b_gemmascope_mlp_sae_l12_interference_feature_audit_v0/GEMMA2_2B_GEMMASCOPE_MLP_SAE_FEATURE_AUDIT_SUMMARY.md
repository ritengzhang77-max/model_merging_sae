# Gemma-2-2B GemmaScope MLP SAE Feature Audit

Feature-selection prompts: `0:8` per split.
Feature-selection token filter: `all`.
Audit prompts: `0:12` per split.
Event token filter: `all`.
Top exported features per layer: `24`.

## Top Features By Group And Layer

| group | layer | top feature IDs | top harm-delta mean values |
|---|---:|---|---|
| `l12_interference` | 12 | `15963, 40, 1434, 10733, 8698, 6215, 7210, 11022` | `0.150, 0.149, 0.149, 0.148, 0.148, 0.148, 0.148, 0.147` |

## Event Rows

- Exported `24` feature score rows.
- Exported `240` top activation/delta event rows.
- Event rows include exact feature IDs, token IDs, token text, prompt split, prompt text, and local token context.

## Caveat

- This file does not assign semantic feature labels. It is an audit substrate for manual or automated interpretation.
