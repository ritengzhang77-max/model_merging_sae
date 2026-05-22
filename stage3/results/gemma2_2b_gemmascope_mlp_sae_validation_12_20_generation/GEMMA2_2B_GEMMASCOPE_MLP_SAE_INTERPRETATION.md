# GemmaScope MLP SAE Interpretation

Date: 2026-05-22

Models:

- Base donor: `google/gemma-2-2b-it`
- Abliterated recipient: `IlyaGusev/gemma-2-2b-it-abliterated`

## Claim Tested

Can a public GemmaScope sparse basis reproduce the causal activation repair that
restores refusal behavior in the abliterated Gemma model?

## Main Result

Yes, on the current 4-harmful / 4-benign screen.

The successful causal site is the post-feedforward MLP update over layers
`12-20`. Full donor activation patching at this site restores harmful clean
refusal from `0.000` to `1.000` while keeping benign helpfulness at `1.000`.

GemmaScope MLP-output SAE decoded patching over the same `12-20` range also
reaches:

- harmful clean refusal: `1.000`
- harmful unsafe continuation: `0.000`
- benign helpfulness: `1.000`
- benign over-refusal: `0.000`

This is the first public sparse-basis result in this project that passes the
same behavioral gate as the full activation patch on the current screen.

## Important Controls

Hook alignment mattered.

- GemmaScope transcoders require the pre-feedforward RMSNorm input and decode
  the normalized MLP update. Using the wrong Hugging Face hook produced
  nonsensical reconstruction.
- After correcting the hook, the layer-16 transcoder had reasonable L0 but weak
  reconstruction on this prompt distribution, so it is not the first sparse
  branch to pursue.
- GemmaScope MLP SAEs also need the post-feedforward normalized update site.
  On raw `block.mlp` outputs, L0 and reconstruction are poor.

Behavioral controls:

- Full `16-20:post_ff` patch reaches `0.750` harmful clean refusal.
- GemmaScope MLP-SAE decoded `16-20` patch also reaches `0.750`, but one
  signature prompt has unsafe continuation.
- Full `12-20:post_ff` patch reaches `1.000`.
- GemmaScope MLP-SAE decoded `12-20` patch also reaches `1.000`.

## Interpretation

This does not yet mean individual SAE features explain the merge. It means the
GemmaScope MLP SAE basis passes a behavioral-completeness gate that the Qwen
vanilla residual SAEs failed.

The result is promising because the decoded sparse reconstruction is sufficient
for the current causal repair. The next question is whether a smaller or
interpretable subset of features can reproduce the repair better than broad
coordinate baselines such as `top_neuron_k1536`.

## Caveats

- The screen has only 4 harmful and 4 benign prompts.
- The current result tests decoded reconstruction, not feature selection or
  feature-level causality.
- Reconstruction EV is moderate, not perfect: roughly `0.63-0.78` across
  layers/splits. Behavioral success may be tolerant to reconstruction error.
- The existing coordinate baseline `top_neuron_k1536` also reaches `1.000` on
  this small screen, so sparse features still need to beat, compress, or explain
  that baseline.

## Next Experiment

Run feature-subset interventions inside the passing `12-20` MLP-SAE basis:

1. donor-active feature patching only;
2. donor-recipient feature-delta patching;
3. top feature deltas by harmful-vs-benign specificity;
4. matched random active-feature controls;
5. held-out harmful and adversarial benign prompt validation.
