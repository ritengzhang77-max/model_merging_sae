# SmolLM2 Base-to-Instruct Alpha Sweep Summary

Run date: 2026-05-07

Script:

- `stage0/scripts/run_smolm2_alpha_sweep.py`

Models:

- base: `HuggingFaceTB/SmolLM2-135M`
- instruct: `HuggingFaceTB/SmolLM2-135M-Instruct`

Formula:

```text
theta(alpha) = theta_base + alpha * (theta_instruct - theta_base)
```

Outputs:

- `stage0/results/smollm2_alpha_sweep_generations.jsonl`

## Qualitative Result

`alpha=0.0`:

- base model can answer raw completion prompts;
- base model does not handle chat-template prompts well and tends to repeat the
  user instruction.

`alpha=0.5`:

- partially inherits instruction behavior;
- simple math chat prompt becomes answer-like (`2 + 3 = 5`);
- some instruction prompts still repeat.

`alpha=1.0`:

- behaves like the instruct model under chat template;
- gives direct assistant-style answers;
- raw completion prompts become more chat-template flavored.

## Interpretation

This is a small transformer sanity check that task-vector scaling produces a
smooth behavioral shift from base completion behavior to instruction-following
behavior. It is not yet a multi-expert merge, but it validates:

- checkpoint compatibility;
- task-vector interpolation code;
- local HF cache setup;
- deterministic prompt-suite generation.

## Next Step

Create two or three controlled SmolLM2 experts from the same base:

- arithmetic-template expert;
- instruction-format/style expert;
- refusal/safety-like synthetic expert or sentiment/style expert.

Then run the same merge recipes as Stage 0A and collect activations for RQ0.

