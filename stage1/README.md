# Stage 1: RQ0 Basis Validation

Stage 1 asks which basis explains or predicts merge success:

- raw task-vector geometry;
- sign agreement/conflict;
- activation similarity;
- causal module-patching effects.

Current script:

```bash
python3 stage1/scripts/analyze_mnist_rq0.py
```

Inputs:

- `stage0/artifacts/mnist_domain_merge/checkpoints.pt`
- `stage0/results/mnist_domain_merge_metrics.csv`

Outputs:

- `stage1/results/mnist_rq0_predictors.csv`
- `stage1/results/mnist_rq0_patch_effects.csv`
- `stage1/results/mnist_rq0_correlations.csv`
- `stage1/results/MNIST_RQ0_SUMMARY.md`

SmolLM2 sampled-delta cache:

```bash
HF_HOME=/data/gavin/model_merging/hf_cache \
python3 stage1/scripts/build_smollm2_delta_cache.py --max-delta-values 80000
```

SmolLM2 merged-state cache:

```bash
HF_HOME=/data/gavin/model_merging/hf_cache \
python3 stage1/scripts/build_smollm2_state_cache.py
```

SmolLM2 RQ0:

```bash
HF_HOME=/data/gavin/model_merging/hf_cache \
python3 stage1/scripts/analyze_smollm2_rq0.py --eval-examples 16 --max-delta-values 80000
```

Outputs:

- `stage1/cache/smollm2_delta_cache.pt`
- `stage1/cache/smollm2_state_cache/`
- `stage1/results/smollm2_rq0_predictors.csv`
- `stage1/results/smollm2_rq0_patch_effects.csv`
- `stage1/results/smollm2_rq0_correlations.csv`
- `stage1/results/SMOLLM2_RQ0_SUMMARY.md`

Useful fast smoke test that avoids overwriting main results:

```bash
HF_HOME=/data/gavin/model_merging/hf_cache \
python3 stage1/scripts/analyze_smollm2_rq0.py \
  --eval-examples 2 \
  --batch-size 2 \
  --max-delta-values 80000 \
  --skip-patch \
  --result-dir stage1/results/smoke_state_cache
```
