# MNIST RQ0 Basis Validation Summary

Run date: 2026-05-09

Script:

- `stage1/scripts/analyze_mnist_rq0.py`

Inputs:

- `stage0/artifacts/mnist_domain_merge/checkpoints.pt`
- `stage0/results/mnist_domain_merge_metrics.csv`

Outputs:

- `stage1/results/mnist_rq0_predictors.csv`
- `stage1/results/mnist_rq0_patch_effects.csv`
- `stage1/results/mnist_rq0_correlations.csv`
- `stage1/results/mnist_rq0_summary.json`

## What Was Tested

For each target model and expert capability, the script computed:

- weight/task-vector cosine by module;
- weight norm ratios by module;
- sign agreement by module;
- activation similarity using CKA and row-wise cosine;
- module patch gains from expert modules into the target model.

Patch gains were evaluated on each donor expert's own task by default. The full
cross-domain patch matrix can be generated later with:

```bash
python3 stage1/scripts/analyze_mnist_rq0.py --patch-eval-mode all_domains
```

## Strongest Signals For Current Capability Retention

Correlations with `target_acc` across 55 target/expert rows:

| predictor | Spearman | Pearson | interpretation |
|---|---:|---:|---|
| `act_rowcos_logits` | +0.977 | +0.931 | output-space similarity to expert is highly predictive |
| `act_rowcos_conv2` | +0.936 | +0.906 | mid-level representation similarity is highly predictive |
| `act_rowcos_fc1` | +0.827 | +0.763 | late hidden representation similarity is predictive |
| `act_cka_logits` | +0.806 | +0.827 | CKA gives the same broad story |
| `act_cka_fc1` | +0.796 | +0.684 | late hidden CKA is useful |
| `weight_cos_conv1` | +0.732 | +0.782 | early weight-delta similarity helps, but less than activations |

The main finding is that activation-space similarity is a better first-pass
basis than raw task-vector geometry in this toy setting.

## Patch Gain Interpretation

`patch_gain_all` and `patch_gain_fc1` are almost perfectly *negatively*
correlated with current accuracy:

- `patch_gain_all`: Spearman `-1.000`
- `patch_gain_fc1`: Spearman `-0.982`

This is expected and important. Patch gain is a rescue score, not a direct
retention score:

- if the target already performs like the expert, patching has little room to
  improve it;
- if the target has lost the capability, expert patching can rescue it.

So for RQ0, patch gain should be interpreted as:

```text
high patch gain = missing but recoverable expert mechanism
low patch gain = already retained, or not recoverable by that module
```

## Causal Localization Result

The largest positive single-module gains came from `fc1` patches:

- label-permutation expert `fc1` into merged targets recovered about `+0.70` to
  `+0.78` label-permutation-task accuracy;
- invert expert `fc1` into good merges recovered about `+0.52` to `+0.54`
  inverted-domain accuracy;
- noise expert `fc1` into `with_labelperm_ties_keep20` recovered about `+0.48`
  noisy-domain accuracy.

This is a clean toy causal result: in this CNN, the late hidden layer `fc1`
carries much of the domain/label remapping behavior, while `fc2` and early
conv patches are much less effective on their own.

## Current RQ0 Takeaway

For this controlled CNN setting:

1. **Activation similarity is the best retention predictor.**
   Logit and conv2 row-cosine dominate raw weight cosine and sign agreement.

2. **Raw weight geometry is useful but incomplete.**
   `weight_cos_conv1` correlates well with target accuracy, but full task-vector
   cosine is not among the top signals.

3. **Causal patching gives a different kind of information.**
   Patch gain is best read as "how much missing capability can be restored,"
   not "how good the merge currently is."

4. **Late hidden module `fc1` is the main causal rescue point.**
   This gives us a concrete model-merging analogue of the value-action idea:
   some parent capabilities are not uniformly spread through the model; they can
   be localized and restored.

## Next Step

Move this same RQ0 structure to a transformer:

- start with `HuggingFaceTB/SmolLM2-135M`;
- create controlled small experts from the same base;
- merge them;
- compare weight geometry, activation similarity, and causal patching by layer.

The transformer version should use:

- residual-stream activation similarity;
- MLP/attention block patching;
- logit-level retention;
- optionally SAE/transcoder features only after raw activations and patching are
  baselined.

