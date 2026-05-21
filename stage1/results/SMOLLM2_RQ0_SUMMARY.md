# SmolLM2 RQ0 Basis Validation Summary

Run date: 2026-05-09

Script:

- `stage1/scripts/analyze_smollm2_rq0.py`

Inputs:

- `stage0/artifacts/smollm2_synthetic_experts/expert_arith.pt`
- `stage0/artifacts/smollm2_synthetic_experts/expert_polite.pt`
- `stage0/artifacts/smollm2_synthetic_experts/expert_refusal.pt`
- base model: `HuggingFaceTB/SmolLM2-135M`

Outputs:

- `stage1/results/smollm2_rq0_predictors.csv`
- `stage1/results/smollm2_rq0_patch_effects.csv`
- `stage1/results/smollm2_rq0_correlations.csv`
- `stage1/results/smollm2_rq0_summary.json`
- sampled-delta cache:
  - `stage1/cache/smollm2_delta_cache.pt`
  - `stage1/cache/smollm2_delta_cache.json`

## What Was Tested

For each target model and expert task, the script computed:

- teacher-forced task loss;
- sampled task-vector cosine and sign agreement, globally and at selected
  transformer layers;
- hidden-state row-cosine at layers 0, 5, 10, 15, 20, 25, 29;
- logit row-cosine;
- selected single-layer block patching into `merge_all_linear`.

This is intentionally a first-pass transformer RQ0. It uses loss-based scoring
because generation-based patching over all layers would be too slow for
iteration.

## Main Results

`merge_all_linear` nearly closes the teacher-forced loss gap to the corresponding
experts:

| task | base loss | expert loss | all-merge loss | gap closed |
|---|---:|---:|---:|---:|
| arith | 8.516 | 1.055 | 2.036 | 0.869 |
| polite | 2.477 | 0.258 | 0.386 | 0.943 |
| refusal | 3.614 | 0.235 | 0.488 | 0.925 |

Pairwise merges behave as expected:

- `merge_arith_polite` preserves arithmetic/polite and not refusal.
- `merge_arith_refusal` preserves arithmetic/refusal and not polite.
- `merge_polite_refusal` preserves polite/refusal and not arithmetic.

This means the synthetic transformer setup is usable for RQ0.

## Strongest Predictors

For `loss_gap_closed_vs_base`, the strongest first-pass predictors were mostly
weight-space/sign-agreement metrics:

| predictor | Spearman | Pearson |
|---|---:|---:|
| `weight_cos_l0` | +0.944 | +0.971 |
| `weight_cos_l5` | +0.944 | +0.982 |
| `sign_agree_l5` | +0.942 | +0.905 |
| `sign_agree_all` | +0.924 | +0.906 |
| `weight_cos_l25` | +0.916 | +0.973 |
| `sign_agree_l10` | +0.903 | +0.907 |

For raw `loss_improvement_vs_base`, late activation similarity was strongest:

| predictor | Spearman | Pearson |
|---|---:|---:|
| `act_rowcos_l25` | +0.914 | +0.768 |
| `act_rowcos_l20` | +0.792 | +0.799 |
| `act_rowcos_l29` | +0.738 | +0.718 |
| `act_rowcos_logits` | +0.631 | +0.566 |

Interpretation:

- In this synthetic SmolLM2 setting, **weight/sign compatibility is the best
  predictor of whether the expert loss gap is closed**.
- **Late activation similarity is the best predictor of absolute improvement
  over the base**.
- This differs from MNIST, where activation similarity dominated the retention
  metrics more clearly.

## Patch Sweep

Selected layer patches from each expert into `merge_all_linear` show different
layer profiles by task:

- arithmetic: best rescue at layer 10 (`loss_delta_vs_target = +0.227`), then
  layer 5 and 15;
- refusal: best rescue at layer 29 (`+0.117`), then layers 20 and 25;
- polite: best rescue at layer 29 (`+0.037`), then layers 15 and 20.

This is a useful mechanistic clue:

- arithmetic behavior is more mid-layer sensitive in this tiny setup;
- polite/refusal behavior has stronger late-layer/readout sensitivity.

The effect sizes are small because `merge_all_linear` already retains most of
each expert's loss improvement. Patch gain again behaves like a rescue score:
the better the merge already is, the less room patching has to improve it.

## Plan Check

No major change to the research plan is needed.

Engineering adjustment implemented:

- `stage1/scripts/build_smollm2_delta_cache.py` now precomputes sampled
  task-vector deltas for base, experts, and merge targets across `all` and the
  selected layers.
- `stage1/scripts/analyze_smollm2_rq0.py` now reads that cache by default and
  falls back to on-the-fly sampling only on cache misses.
- `stage1/scripts/build_smollm2_state_cache.py` now precomputes full merged
  SmolLM2 state dicts for the four linear merge targets.
- `stage1/scripts/analyze_smollm2_rq0.py` now reads
  `stage1/cache/smollm2_state_cache/` by default and falls back to rebuilding
  only missing merge states.
- `--result-dir` and `--skip-patch` allow fast smoke runs without overwriting
  the main result CSVs.

This removed the repeated sampled-delta bottleneck and the repeated full-state
merge-construction bottleneck. The cached rerun reproduced the same top
predictors:

| predictor | Spearman with `loss_gap_closed_vs_base` |
|---|---:|
| `weight_cos_l0` | +0.944 |
| `weight_cos_l5` | +0.944 |
| `sign_agree_l5` | +0.942 |
| `sign_agree_all` | +0.924 |
| `weight_cos_l25` | +0.916 |

Remaining engineering bottleneck:

- consider LoRA/adapter experts for rapid iteration.
- optionally cache evaluation losses/reps for exact predictor-only reruns.

Scientific direction remains:

1. validate basis;
2. compare weight geometry, activation similarity, and causal patching;
3. only then add SAE/transcoder features if they beat these baselines.
