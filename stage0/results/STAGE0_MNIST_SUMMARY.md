# Stage 0A MNIST Domain Merge Summary

Run date: 2026-05-07

Script:

- `stage0/scripts/run_mnist_domain_merge.py`

Artifacts:

- metrics: `stage0/results/mnist_domain_merge_metrics.csv`
- conflict stats: `stage0/results/mnist_domain_merge_conflicts.csv`
- run summary: `stage0/results/mnist_domain_merge_summary.json`
- checkpoints: `stage0/artifacts/mnist_domain_merge/checkpoints.pt`
- data: `/data/gavin/model_merging/data/MNIST`

## Setup

Base:

- Small CNN trained on clean MNIST.

Experts:

- `clean`: fine-tuned on clean MNIST.
- `rotate25`: fine-tuned on 25-degree rotated MNIST.
- `noise`: fine-tuned on noisy MNIST.
- `invert`: fine-tuned on inverted MNIST.
- `labelperm`: fine-tuned on clean MNIST with labels shifted by +1 mod 10.

Merges:

- linear/task-vector average;
- model soup average;
- TIES-style trim/elect-sign/merge;
- DARE-style random drop/rescale;
- DELLA-like magnitude-aware random drop/rescale.

Target-domain mean and worst accuracies are computed over:

- clean;
- rotate25;
- noise;
- invert.

`labelperm_task` is reported separately as a conflict diagnostic.

## Key Results

Best mean target-domain model:

- `expert_noise`: mean `0.694`, but worst target accuracy only `0.112`
  because it fails inverted MNIST.

Best balanced target-domain merge:

- `with_labelperm_linear_taskavg`: mean `0.674`, worst `0.323`.
- `good_linear_taskavg`: mean `0.670`, worst `0.266`.

Base:

- mean `0.579`, worst `0.125`.

Direct clean-vs-labelperm conflict:

- `expert_clean`: clean `0.973`, labelperm task `0.002`.
- `expert_labelperm`: clean `0.015`, labelperm task `0.895`.
- `clean_vs_labelperm_linear_taskavg`: clean `0.544`, labelperm task `0.364`.

This confirms the conflict control is real: one model cannot satisfy both normal
and shifted labels with the same classifier head. The direct merge lands between
the incompatible behaviors.

## Sign Conflict Diagnostics

Average active sign-conflict fraction:

- `good`: `0.600`
- `with_labelperm`: `0.768`
- `clean_vs_labelperm`: `0.502`

Adding the label-permutation expert substantially increases sign conflict in the
multi-expert setting, especially in late/classifier parameters.

## Interpretation

Stage 0A is not a final scientific setting, but it validates the harness:

- The code can train base/expert checkpoints.
- It can save merge artifacts for later patching.
- It can produce success/failure tradeoffs.
- It captures both behavior metrics and parameter-conflict statistics.
- It gives us a cheap testbed for RQ0 predictors before expensive transformer
  experiments.

The most important finding is not that a specific MNIST merge recipe is best.
The useful outcome is that we now have concrete cases where:

- one expert specializes and forgets another domain;
- a merge partially balances multiple domains;
- an incompatible expert produces direct behavioral conflict;
- sign-conflict statistics change in the expected direction.

## Next Stage 0 Work

1. Add layer-wise / module-wise weight patching on these checkpoints.
2. Add activation collection and simple CKA/cosine similarity by layer.
3. Use these same artifacts to test RQ0 candidate predictors:
   - task-vector cosine;
   - sign conflict;
   - per-layer delta norm;
   - activation similarity;
   - causal patch score.
4. Start Stage 0B with a small transformer:
   - preferred base: `HuggingFaceTB/SmolLM2-135M`;
   - alternate base: `Qwen/Qwen2.5-0.5B`.

