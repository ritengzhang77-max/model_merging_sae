# Gemma-2-2B Linear Merge SAE 11-Feature Geometry Summary

Date: 2026-05-24

This checkpoint asks whether the validated k=11 handles are explained by coarse
geometry in layer-20 SAE delta space.

## Setup

I compared decoded bundle deltas against the dense donor-minus-recipient MLP
delta and the all-feature SAE delta on the hologram probe's assistant-boundary
final newline.

Bundle groups:

| group | count | source |
|---|---:|---|
| swap pass | 6 | one-swap k=11 variants with `I-It > 0` |
| swap tie | 161 | one-swap k=11 variants with `I-It = 0` |
| swap fail | 286 | one-swap k=11 variants with `I-It < 0` |
| random | 200 | random k=11 subsets from the same top-33 pool |

## Harmful Final-Newline Geometry

| group | bundle norm mean | cos all-SAE mean | cos dense mean | residual/all-SAE mean |
|---|---:|---:|---:|---:|
| swap pass | `18.9184` | `0.5326` | `0.4497` | `0.8526` |
| swap tie | `18.8593` | `0.5319` | `0.4499` | `0.8531` |
| swap fail | `17.9878` | `0.5140` | `0.4169` | `0.8628` |
| random | `15.7219` | `0.4455` | `0.3068` | `0.8963` |

Random k=11 subsets are lower on average: they have smaller decoded-delta norm,
lower cosine with both dense and all-SAE deltas, and higher residual. So the
validated handles are not just arbitrary high-rank subsets.

But geometry does not separate pass from near-pass. Swap passes and swap ties
are nearly identical by these coarse metrics:

```text
pass mean cos(all-SAE) = 0.5326
tie  mean cos(all-SAE) = 0.5319
pass mean norm         = 18.9184
tie  mean norm         = 18.8593
```

The top aligned rows are also not the successful handles. The highest
cosine-to-all-SAE bundle is a random k=11 subset with `I-It = -0.125`, and
several failing or tied one-swap variants have higher cosine than every passing
variant.

## Interpretation

Coarse geometry explains the broad difference between structured neighborhood
handles and random same-pool subsets, but it does not explain the actual
first-token threshold crossing among nearby k=11 variants. In the local
neighborhood, the pass/tie/fail distinction depends on exact signed feature
composition and downstream readout effects, not just larger norm or better
alignment with the dense/all-SAE delta.

This reinforces the current mechanistic framing:

```text
model merging exposes a near-boundary refusal route;
SAE handles can tip it;
the handle is a sparse signed equivalence class;
coarse delta geometry is necessary context but not a sufficient explanation.
```

## Artifacts

- Geometry audit:
  `stage3/results/gemma2_2b_linear_merge_sae_bundle_delta_geometry_v0/fake_id_hologram_l20_final_newline_delta_add_critical11_swap1_vs_random_k11_float32/`
- Outcome aggregate:
  `stage3/results/gemma2_2b_linear_merge_sae_bundle_delta_geometry_v0/fake_id_hologram_l20_final_newline_delta_add_critical11_swap1_vs_random_k11_float32/geometry_outcome_summary.csv`
- One-swap first-token source:
  `stage3/results/gemma2_2b_linear_merge_sae_bundle_first_token_logits_v0/fake_id_hologram_l20_final_newline_delta_add_prompt_delta_critical11_swap_one_top33_float32/`
- Random k=11 source:
  `stage3/results/gemma2_2b_linear_merge_sae_bundle_first_token_logits_v0/fake_id_hologram_l20_final_newline_delta_add_prompt_delta_random_top33_k11_screen_float32/`
