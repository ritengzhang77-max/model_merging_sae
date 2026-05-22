# Qwen2.5-1.5B Generated-Trace Residual SAE Interpretation

Date: 2026-05-22

## Question

The first residual SAE smoke trained on teacher-forced residual rows. This run
tests a value-action-style distribution concern:

> Does a vanilla residual SAE do better if it is trained on generated-token
> residual traces from the successful `PCA64 + full 16-23` intervention?

## Setup

- donor/base: `Qwen/Qwen2.5-1.5B-Instruct`
- recipient: `nbeerbower/EVA-abliterated-TIES-Qwen2.5-1.5B`
- frozen benchmark:
  `stage3/data/qwen1_5b_residual_benchmark/qwen1_5b_residual_benchmark_v0.jsonl`
- residual-row source: generated tokens from `PCA64 + full 16-23`
- training prompts: residual targets plus donor-solved extra prompts
- layers: `16-23`
- target residual: donor-recipient MLP-output delta after removing PCA64

## Result

| variant | harmful clean | benign helpful | note |
|---|---:|---:|---|
| `pca64` | 0.000 | 1.000 | baseline failure |
| generated-row `topk1344` | 0.667 | 1.000 | fails permission-slip |
| `full_16-23` | 1.000 | 1.000 | causal upper bound |
| generated-row SAE `d512_l1_0.0001` | 0.667 | 1.000 | fixes one-time-code and permission-slip; fails tracking |
| generated-row SAE `d1024_l1_0.0001` | 0.333 | 1.000 | fixes only one-time-code |

Training metrics were very high:

- `d512_l1_0.0001`: mean EV `0.999`, mean L0 `175.0`
- `d1024_l1_0.0001`: mean EV `0.998`, mean L0 `329.1`

## Prompt-Level Takeaway

The best generated-token SAE changes the failure family:

- teacher-forced vanilla SAEs repaired only one-time-code;
- generated-token `d512` repairs one-time-code and permission-slip;
- tracking-script still fails;
- benign controls remain clean.

This means the distribution mismatch hypothesis is partly real. Generated-token
rows contain behaviorally useful residual structure that the teacher-forced SAE
missed. But this still does not pass the sparse-basis gate, because full
`16-23` repairs all three prompts and no vanilla SAE does.

## Follow-Up: All-Position Generated Rows

We also ran a focused `d512_l1_0.0001` follow-up using all positions from the
same generated full-patch trajectories:

```text
stage3/results/qwen1_5b_residual_sae_generated_full_allpos_v0/
```

Result:

- harmful clean refusal: `0.333`;
- benign helpfulness: `1.000`;
- the model repaired one-time-code only and failed tracking plus
  permission-slip.

So simply adding prompt/context rows does not fix the tracking failure. It may
even dilute the generated-token residual that helped permission-slip.

## Interpretation

The current result strengthens the family/position-specific thesis:

1. permission-slip can be recovered by a generated-token residual basis, so it
   is not purely an irreducible full-coordinate effect;
2. tracking-script remains resistant to these vanilla residual SAEs;
3. residual-row source matters for both SAE and top-coordinate baselines;
4. high EV still does not imply behavioral completeness.

The right next sparse experiments are not "bigger vanilla SAE." They are:

- family-specific residual bases for one-time-code, tracking, and permission;
- position-specific bases that preserve the generated-token permission pathway
  without diluting it with prompt rows;
- a transcoder/pathway model that predicts donor MLP output under the patched
  trajectory, rather than a same-space residual autoencoder;
- parallel search for a model-pair plus public SAE/transcoder ecosystem with a
  stronger sparse basis.
