# Residual Family Split Interpretation

Date: 2026-05-21

## Question

After `PCA64 + full 16-23` repaired the known hard prompts, we asked whether the
remaining `16-23` residual is:

1. one shared low-rank residual direction/subspace;
2. one shared coordinate-sparse residual;
3. or multiple prompt-family-specific residual mechanisms.

## Geometry Result

Residual geometry was measured after subtracting the PCA64 projection from
donor-recipient MLP deltas in layers `16-23`.

Mean all-token residual overlap across `16-23`:

| pair | mean residual cosine | top256 Jaccard | top1024 Jaccard |
|---|---:|---:|---:|
| one-time-code vs permission-slip | `0.738` | `0.377` | `0.664` |
| one-time-code vs tracking-original | `0.698` | `0.341` | `0.651` |
| permission-slip vs tracking-original | `0.835` | `0.441` | `0.684` |
| benign remove-tracking vs tracking-original | `0.943` | `0.571` | `0.759` |

Interpretation:

- harmful residual families are related but not the same;
- tracking residuals are especially entangled with benign remove-tracking topic
  activations, so tracking results need careful benign controls;
- top-coordinate concentration is similar across groups, so concentration alone
  does not explain which sparse patch works.

## Family-Specific Top-Coordinate Target Loss

Target-loss checks showed that top-coordinate residuals are stronger than
residual PCA/mean baselines.

Best examples:

- one-time-code basis, `topk1024`: original two-failure loss `0.321`, better
  than full `16-23` loss `0.328`;
- permission basis, `topk256`: permission loss `0.302`, close to full `16-23`
  loss `0.309`;
- tracking basis, `topk1024`: original two-failure loss `0.339`, weaker than
  full `16-23` but better than PCA64.

However, target loss was not a reliable behavioral validator.

## Generation Result

Generation validation gives the current decision:

| residual basis | patch | original one-time-code + tracking prompts | permission-slip prompt |
|---|---|---:|---:|
| mixed residual targets | `topk256/topk512` | one-time-code repaired, tracking failed | failed |
| permission only | `topk256/topk1024` | one-time-code repaired, tracking failed | failed |
| tracking only | `topk512` | one-time-code repaired, tracking failed | not tested in same run |
| tracking only | `topk1024` | both original prompts repaired | failed |
| full donor block | `16-23` | both original prompts repaired | repaired |

The strongest new result is:

> `PCA64 + tracking-basis residual topk1024 in 16-23` repairs both original
> held-out failures in generation, matching full `16-23` on that two-prompt
> slice, but it still fails the permission-slip prompt.

## Interpretation

The residual is not a single low-rank vector, and it is not one universal sparse
coordinate set.

Current split:

1. Original one-time-code/tracking pair: can be repaired by a large
   coordinate-sparse residual patch (`topk1024`) learned from the tracking
   prompt.
2. Permission-slip forgery: not repaired by residual PCA, residual mean, or
   top-coordinate patches, even when the top coordinates are learned from the
   permission prompt itself.
3. Full `16-23` donor MLP activations remain necessary for the permission
   residual under current tests.

This is useful for the SAE/transcoder phase. A sparse feature method should be
tested separately on:

- recovering the coordinate-sparse original-failure repair;
- explaining why permission-slip requires the broader full pathway;
- distinguishing tracking-harm residuals from benign remove-tracking topic
  residuals.

## Next Experiment

Run a layer-by-layer sparse residual top-coordinate ablation for the tracking
`topk1024` patch:

- identify which layers inside `16-23` are required for the sparse original
  repair;
- compare those layers with the full-block permission requirement;
- then decide whether SAE/transcoder analysis should focus first on the
  sparse original-repair circuit or the harder permission full-pathway circuit.
