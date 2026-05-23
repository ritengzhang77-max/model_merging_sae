# Gemma-2-2B GemmaScope MLP SAE Feature Audit

Feature-selection prompts: `0:4` per split.
Feature-selection token filter: `all`.
Audit prompts: `0:12` per split.
Event token filter: `all`.
Top exported features per layer: `128`.

## Top Features By Group And Layer

| group | layer | top feature IDs | top harm-delta mean values |
|---|---:|---|---|
| `l19_tail` | 19 | `10355, 11475, 13839, 4904, 732, 15173, 4345, 10142` | `0.127, 0.127, 0.127, 0.127, 0.127, 0.126, 0.126, 0.126` |

## Event Rows

- Exported `128` feature score rows.
- Exported `1394` top activation/delta event rows.
- Event rows include exact feature IDs, token IDs, token text, prompt split, prompt text, and local token context.

## Caveat

- This file does not assign semantic feature labels. It is an audit substrate for manual or automated interpretation.
