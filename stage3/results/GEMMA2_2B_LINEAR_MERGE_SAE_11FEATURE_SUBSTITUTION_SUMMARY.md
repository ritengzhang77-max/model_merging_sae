# Gemma-2-2B Linear Merge SAE 11-Feature Substitution Summary

Date: 2026-05-24

This checkpoint interprets the 453-variant k=11 one-swap screen as local
feature-substitution evidence.

## Screen

The one-swap screen starts from the two validated 11-feature source handles:

```text
A = original12 - feature 4339 - feature 1338 + feature 1813
B = original12 - feature 4339 - feature 8775 + feature 1813
```

Each variant drops one retained feature and adds one top-33 prompt-delta feature
outside that source handle. The screen has:

| outcome | count |
|---|---:|
| pass | 6 |
| tie | 161 |
| fail | 286 |

## Passing Substitutions

All passing substitutions are barely positive at `I-It = +0.015625`.

| source | drop | add | interpretation |
|---|---:|---:|---|
| A | `6289` | `7531` | new validated handle |
| A | `8775` | `9407` | new validated handle |
| A | `8775` | `7531` | new validated handle |
| A | `8775` | `1338` | returns known source B |
| B | `1338` | `8775` | returns known source A |
| B | `6289` | `7531` | new validated handle |

## Drop-Side Structure

The screen shows a split between locally exchangeable and locally core features.

| dropped feature | variants | passes | ties | fails | mean margin | max margin |
|---:|---:|---:|---:|---:|---:|---:|
| `8775` | 22 | 3 | 12 | 7 | `-0.003551` | `+0.015625` |
| `6289` | 43 | 2 | 30 | 11 | `-0.003634` | `+0.015625` |
| `1338` | 1 | 1 | 0 | 0 | `+0.015625` | `+0.015625` |
| `12704` | 43 | 0 | 1 | 42 | `-0.026163` | `0.000000` |
| `14991` | 43 | 0 | 0 | 43 | `-0.046875` | `-0.031250` |
| `15169` | 43 | 0 | 0 | 43 | `-0.412064` | `-0.390625` |

Dropping `6289` or `8775` can be repaired by specific additions. Dropping
`14991` or `15169` never even ties in this screen, and `15169` is especially
large: every one-swap replacement stays far below the gate.

## Add-Side Structure

Only a few additions ever produce a pass:

| added feature | variants | passes | ties | fails | mean margin | max margin |
|---:|---:|---:|---:|---:|---:|---:|
| `7531` | 21 | 3 | 10 | 8 | `-0.044643` | `+0.015625` |
| `9407` | 21 | 1 | 6 | 14 | `-0.049851` | `+0.015625` |
| `1338` | 11 | 1 | 6 | 4 | `-0.041193` | `+0.015625` |
| `8775` | 1 | 1 | 0 | 0 | `+0.015625` | `+0.015625` |
| `13854` | 21 | 0 | 15 | 6 | `-0.042411` | `0.000000` |
| `1100` | 21 | 0 | 15 | 6 | `-0.043155` | `0.000000` |

Feature `7531` is the clearest new substitute: it repairs both source handles
when `6289` is dropped, and also repairs source A when `8775` is dropped. But it
is not a generally sufficient feature, because most `+7531` variants still tie
or fail. Feature `9407` repairs only one tested context.

## Interpretation

The local 11-feature handle is neither a bag of interchangeable high-delta
features nor a set of individually necessary semantic features. It has a mixed
structure:

```text
core-like features: 15169, 14991, likely 12704
locally exchangeable features: 6289, 8775, 1338 in narrow contexts
contextual substitutes: 7531, 9407
```

This is a more mechanistic description of the signed equivalence class: the
class has a locally rigid backbone plus a small number of allowable
substitutions that keep the final-newline state just over the `I`/`It` gate.

## Artifacts

- One-swap screen:
  `stage3/results/gemma2_2b_linear_merge_sae_bundle_first_token_logits_v0/fake_id_hologram_l20_final_newline_delta_add_prompt_delta_critical11_swap_one_top33_float32/`
- Detailed effect rows:
  `stage3/results/gemma2_2b_linear_merge_sae_bundle_first_token_logits_v0/fake_id_hologram_l20_final_newline_delta_add_prompt_delta_critical11_swap_one_top33_float32/critical11_swap_one_top33_effect_rows.csv`
- Drop aggregate:
  `stage3/results/gemma2_2b_linear_merge_sae_bundle_first_token_logits_v0/fake_id_hologram_l20_final_newline_delta_add_prompt_delta_critical11_swap_one_top33_float32/critical11_swap_one_top33_effect_by_drop.csv`
- Add aggregate:
  `stage3/results/gemma2_2b_linear_merge_sae_bundle_first_token_logits_v0/fake_id_hologram_l20_final_newline_delta_add_prompt_delta_critical11_swap_one_top33_float32/critical11_swap_one_top33_effect_by_add.csv`
