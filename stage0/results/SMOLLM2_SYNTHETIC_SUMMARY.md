# SmolLM2 Synthetic Expert Merge Summary

Run date: 2026-05-09

Script:

- `stage0/scripts/run_smollm2_synthetic_experts.py`

Models:

- base: `HuggingFaceTB/SmolLM2-135M`
- tokenizer/chat template: `HuggingFaceTB/SmolLM2-135M-Instruct`

Artifacts:

- expert states:
  - `stage0/artifacts/smollm2_synthetic_experts/expert_arith.pt`
  - `stage0/artifacts/smollm2_synthetic_experts/expert_polite.pt`
  - `stage0/artifacts/smollm2_synthetic_experts/expert_refusal.pt`
- manifest: `stage0/artifacts/smollm2_synthetic_experts/manifest.json`
- metrics: `stage0/results/smollm2_synthetic_merge_metrics.csv`
- generations: `stage0/results/smollm2_synthetic_generations.jsonl`

## Setup

Three tiny full fine-tunes were trained from the same base checkpoint:

- `arith`: synthetic addition prompts;
- `polite`: rewrite rude commands politely;
- `refusal`: refuse synthetic harmful requests.

Short smoke configuration:

- `train_examples=192`
- `max_steps=30`
- `eval_examples=12`
- full-state fine-tuning, no PEFT/LoRA

Merge formula:

```text
theta_merge = theta_base + mean_i(theta_expert_i - theta_base)
```

## Metrics

| model | arith | polite | refusal | mean | worst |
|---|---:|---:|---:|---:|---:|
| `merge_all_linear` | 0.583 | 1.000 | 1.000 | 0.861 | 0.583 |
| `merge_polite_refusal` | 0.000 | 1.000 | 0.500 | 0.500 | 0.000 |
| `merge_arith_polite` | 0.167 | 1.000 | 0.000 | 0.389 | 0.000 |
| `merge_arith_refusal` | 0.167 | 0.000 | 1.000 | 0.389 | 0.000 |
| `expert_polite` | 0.000 | 1.000 | 0.000 | 0.333 | 0.000 |
| `expert_refusal` | 0.000 | 0.000 | 1.000 | 0.333 | 0.000 |
| `expert_arith` | 0.417 | 0.000 | 0.000 | 0.139 | 0.000 |
| `base` | 0.250 | 0.000 | 0.000 | 0.083 | 0.000 |

## Interpretation

This is a successful transformer smoke test:

- all experts trained and saved;
- full-state task-vector merges completed;
- merged models were evaluable under a chat-template prompt suite.

The main interesting result is that `merge_all_linear` outperformed every
individual expert on mean score in this tiny synthetic setup. It retained polite
and refusal behavior and improved arithmetic relative to the arithmetic expert.

This should not be overclaimed. The arithmetic expert itself is weak and
generates noisy numeric continuations. The all-merge often outputs correct
equations such as:

```text
23 + 39 = 62
```

A plausible explanation is that polite/refusal fine-tunes improve chat-format
response discipline while the arithmetic delta contributes numeric behavior.
This is exactly the kind of "composition vs artifact" case that RQ11 should
eventually test causally.

## Practical Note

Full-state merging for even a 135M transformer is CPU-expensive. The rerun spent
most of its time averaging and loading full state dicts, not training. For the
next transformer iteration:

- cache task deltas once;
- avoid rebuilding every merge from scratch;
- consider LoRA/adapter experts for fast iteration;
- only save merged full states when needed for patching.

## Next Step

Do not jump straight to SAE/transcoder analysis. First run a transformer RQ0
baseline:

- task-vector cosine by layer/block;
- residual activation similarity by layer;
- block-wise weight patching;
- compare these predictors against arithmetic/polite/refusal retention.

