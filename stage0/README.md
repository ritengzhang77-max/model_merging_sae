# Stage 0: Merge Reproduction Harness

Stage 0 creates cheap, controlled merge success/failure cases before we move to
larger language models.

## Current Experiment

`scripts/run_mnist_domain_merge.py`

- Base model: small CNN trained on clean MNIST.
- Experts: fine-tuned from the base on clean, rotated, noisy, inverted, and
  label-permuted MNIST.
- Merges:
  - linear/task-vector average;
  - model soup average of expert checkpoints;
  - TIES-style trim/elect-sign/merge;
  - DARE-style random delta drop and rescale;
  - DELLA-like magnitude-aware random delta drop and rescale.
- Evaluations:
  - clean accuracy;
  - rotated accuracy;
  - noisy accuracy;
  - inverted accuracy;
  - label-permutation task accuracy as an explicit conflict diagnostic;
  - mean and worst-domain accuracy.

The label-permuted expert is a negative control. If a merge recipe blindly
absorbs it, accuracy should degrade.

## Run

```bash
cd /home/gavin/model_merging
python3 stage0/scripts/run_mnist_domain_merge.py
```

Outputs:

- `stage0/results/mnist_domain_merge_metrics.csv`
- `stage0/results/mnist_domain_merge_summary.json`
- `stage0/results/mnist_domain_merge_conflicts.csv`
- `stage0/artifacts/mnist_domain_merge/*.pt`

## SmolLM2 Synthetic Experts

`scripts/run_smollm2_synthetic_experts.py`

- Base model: `HuggingFaceTB/SmolLM2-135M`.
- Experts: synthetic arithmetic, polite rewriting, and refusal behavior.
- Merges: linear task-vector average over all experts and pairwise expert sets.
- Outputs:
  - `stage0/results/smollm2_synthetic_merge_metrics.csv`
  - `stage0/results/smollm2_synthetic_generations.jsonl`
  - `stage0/results/SMOLLM2_SYNTHETIC_SUMMARY.md`
  - `stage0/artifacts/smollm2_synthetic_experts/manifest.json`

Run the quick validation from existing experts:

```bash
HF_HOME=/data/gavin/model_merging/hf_cache \
python3 stage0/scripts/run_smollm2_synthetic_experts.py --skip-train
```

## SmolLM2 Refusal V2 Decision Gate

`scripts/run_smollm2_refusal_v2.py`

- Trains a replacement refusal expert with diverse harmful prompts, varied
  refusal templates, and benign safety-help contrast examples.
- Evaluates harmful clean-refusal and benign over-refusal separately.
- Writes v2 outputs under `stage0/results/refusal_v2*` and
  `stage0/artifacts/smollm2_refusal_v2*`.

`scripts/run_smollm2_refusal_v2_alpha_search.py`

- Tests whether the balanced v2 refusal expert is merely underweighted in the
  merge.
- Formula:

```text
theta = theta_merge_arith_polite + alpha * (theta_refusal_v2 - theta_base)
```

Decision memo:

- `stage0/results/SMOLLM2_REFUSAL_V2_DECISION.md`

Current result:

- v2 data improves the expert but does not produce a clean merge target.
- stronger alpha restores attempted refusal but not clean refusal, and increases
  benign over-refusal.
- the next clean-target branch should use the cached
  `HuggingFaceTB/SmolLM2-135M-Instruct` checkpoint as the common base.
