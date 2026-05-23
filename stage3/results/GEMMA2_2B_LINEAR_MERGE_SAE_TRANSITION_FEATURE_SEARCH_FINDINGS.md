# Gemma-2-2B Linear Merge SAE Transition Feature Search Findings

Date: 2026-05-23

This checkpoint searches beyond the three hand-picked features. It uses fixed
safe alpha-`0.75` fake-ID family continuations and ranks GemmaScope MLP-SAE
features whose activation rises from model alpha `0.25` to model alpha `0.75`
on harmful continuations, while not also rising on benign continuations.

Ranking rule:

```text
harm_delta = harmful_mean(alpha=0.75) - harmful_mean(alpha=0.25)
specificity = harm_delta - abs(benign_delta)
```

## Top Features

| rank | layer | feature | harm low | harm high | harm delta | benign delta | specificity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17 | 4342 | 12.2689 | 15.6565 | 3.3876 | 0.2231 | 3.1645 |
| 2 | 17 | 16011 | 7.2497 | 10.0005 | 2.7508 | 0.0204 | 2.7304 |
| 3 | 16 | 16332 | 6.3471 | 7.9845 | 1.6374 | 0.0342 | 1.6032 |
| 4 | 18 | 10415 | 4.7665 | 6.3996 | 1.6332 | 0.1737 | 1.4594 |
| 5 | 18 | 11127 | 3.1627 | 4.4236 | 1.2609 | 0.0118 | 1.2491 |
| 6 | 15 | 11128 | 6.7940 | 8.1216 | 1.3276 | 0.1269 | 1.2007 |
| 7 | 14 | 3001 | 3.5155 | 4.7027 | 1.1872 | 0.0296 | 1.1576 |
| 8 | 20 | 14425 | 4.8436 | 6.0808 | 1.2372 | 0.1123 | 1.1248 |
| 9 | 18 | 7189 | 1.2327 | 2.3496 | 1.1169 | 0.0000 | 1.1169 |
| 10 | 18 | 11214 | 4.0945 | 5.2337 | 1.1392 | 0.1127 | 1.0265 |

The strongest candidates are not the earlier L19 `16048` feature. They cluster
in middle/late MLP layers, especially L17 and L18.

## Qualitative Audit

Auditing the top candidates on fixed safe alpha-`0.75` fake-ID family
continuations shows many high activations around legal-consequence and
refusal-rationale tokens:

- L17 feature `4342`: high on tokens such as `Forgery`, `Criminal`, `felony`,
  `theft`, `serious`, and nearby markdown punctuation.
- L17 feature `16011`: high on tokens such as `jail`, `even`, `potential`, and
  nearby conjunctions in consequence descriptions.
- L18 feature `11127`: high on `criminal` / legal-consequence contexts.

This is qualitatively different from L19 feature `16048`, whose strongest
activations often appeared on punctuation or caveat transitions. The transition
search is finding features that look more like a natural refusal-rationale /
legal-consequence bundle.

## Interpretation

This is the strongest mechanistic bridge so far:

- Linear model merging has a behavioral transition between alpha `0.25` and
  alpha `0.50`/`0.75`.
- Fixed-continuation SAE search finds features that increase across that
  transition on harmful safe continuations but stay comparatively stable on
  benign continuations.
- The leading features appear to encode refusal rationale and legal-consequence
  content, not fake-ID fabrication details.

This gives a cleaner next paper direction than the original f16048 singleton
story: study how model merging restores a coordinated refusal-rationale feature
bundle, and why sparse local patches can mis-handle that bundle.

## Next Step

Run causal tests for the discovered transition features:

- add top transition features to low-alpha or abliterated trajectories;
- remove them from alpha `0.75`;
- compare against random same-layer/same-count controls;
- test whether the legal-consequence feature bundle explains more of the
  family behavior than L19 `16048`.

## Artifacts

- Transition search:
  `stage3/results/gemma2_2b_linear_merge_sae_transition_feature_search_v0/fake_id_family_safe_a075_low025_high075/`
- Top-feature audit:
  `stage3/results/gemma2_2b_linear_merge_sae_teacher_forced_features_v0/top_transition_features_safe_a075_low025_high075/`
- Search script:
  `stage3/scripts/search_gemma2_2b_linear_merge_sae_transition_features.py`
