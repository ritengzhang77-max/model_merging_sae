# Gemma-2-2B Activation Patch Interpretation

Date: 2026-05-22

Models:

- Base donor: `google/gemma-2-2b-it`
- Abliterated recipient: `IlyaGusev/gemma-2-2b-it-abliterated`

## Claim Tested

Can base-model internal activations causally restore the refusal behavior lost
by the abliterated Gemma model, while preserving benign helpfulness?

## Main Result

Yes, on the current Stage 2 screen. Dynamic activation patching of base MLP
outputs into the abliterated recipient over layers `12-20` restores harmful
clean refusal from `0.000` to `1.000` and keeps benign helpfulness at `1.000`.

This is a real behavioral repair, not only a refusal-target loss effect:

- target-loss patching: `12-20:mlp` closes `0.973` of the harmful refusal-target
  loss gap.
- all-position dynamic generation patching: `12-20:mlp` reaches `1.000`
  harmful clean refusal.
- target-position-only dynamic patching: `12-20:mlp` drops to `0.500` harmful
  clean refusal and shows `0.250` unsafe continuation.

Interpretation: the successful repair needs sequence-wide MLP activation
propagation. It is not just a final-token refusal vector.

## Localized Pattern

The repair is concentrated in mid layers:

- single `16:mlp` and `20:mlp` patches close some target-loss gap but fail
  generation.
- `16-20:mlp` restores `0.750` harmful clean refusal in all-position dynamic
  generation.
- `12-20:mlp` restores `1.000` harmful clean refusal in all-position dynamic
  generation.
- late `20-25:mlp` is not the active repair range in the target-loss check.

This makes Gemma different from the Qwen residual branch, where late and
family-specific residual pathways dominate the hard residual prompts.

## Low-Dimensional Baseline Bar

The first compressed baselines are mixed:

- `mean_delta_rank1`: `0.000` harmful clean refusal, with unsafe continuation on
  one prompt.
- `pca_rank64`: `0.750` harmful clean refusal.
- `pca_rank128`: `0.750` harmful clean refusal.
- `top_neuron_k1024`: `0.750` harmful clean refusal.
- `top_neuron_k1536`: `1.000` harmful clean refusal and `1.000` benign
  helpfulness.
- `random_rank64`: `0.000` harmful clean refusal.

`top_neuron_k1536` is broad: Gemma hidden size is 2304, so this keeps about
two-thirds of the coordinate space over layers `12-20`. It is a strong causal
baseline, not a sparse mechanistic explanation.

## Decision

The Gemma branch is now worth moving into sparse-basis validation because it has
a clean public model pair, a public GemmaScope ecosystem, and a causal activation
repair target.

The next phase must keep the value_action discipline:

- do not interpret GemmaScope features until the basis passes reconstruction and
  behavioral-completeness checks on layers `12-20`.
- compare every sparse intervention against full `12-20:mlp`, `pca_rank64/128`,
  `top_neuron_k1024`, and especially `top_neuron_k1536`.
- require benign controls on every harmful repair.
- treat a sparse feature result as valuable only if it compresses, decomposes,
  or explains the broad coordinate repair better than the non-sparse baselines.

## Immediate Next Experiment

Load GemmaScope residual/MLP/transcoder bases for the relevant Gemma layers,
starting with layers `12`, `16`, and `20`, then run a basis validation gate:

1. reconstruction metrics on the patched activation stream;
2. reconstruction-only generation to check behavioral completeness;
3. sparse-feature donor-recipient patching against the low-dimensional baselines;
4. prompt-level audit of the four harmful prompts and matched benign controls.
