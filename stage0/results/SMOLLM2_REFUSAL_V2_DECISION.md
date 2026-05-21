# SmolLM2 Refusal V2 Decision Memo

Updated: 2026-05-19

## Question

Can we repair the Stage 3 refusal target by training a stronger synthetic
refusal expert on the same `HuggingFaceTB/SmolLM2-135M` base?

## Short Answer

Not yet. The v2 refusal data improved the expert somewhat, but it did not create
a clean merge target. The likely blocker is the current non-instruction base
plus small full fine-tunes under chat-format generation: even when the first
sentence is plausible, generations often fail to terminate and continue into
artifacts such as `LEGATO`, `%||`, `pvproperty`, or repeated continuations.

This means the next clean-target attempt should probably restart Stage 0 from
the cached instruction-tuned checkpoint:

```text
HuggingFaceTB/SmolLM2-135M-Instruct
```

rather than keep tuning refusal data on the current base.

## Runs

### Original v2

Script:

- `stage0/scripts/run_smollm2_refusal_v2.py`

Command:

```bash
HF_HOME=/data/gavin/model_merging/hf_cache python3 stage0/scripts/run_smollm2_refusal_v2.py \
  --device cuda:1 \
  --train-examples 768 \
  --eval-examples 16 \
  --max-steps 80 \
  --batch-size 8 \
  --lr 1e-5 \
  --max-new-tokens 56
```

Outputs:

- `stage0/artifacts/smollm2_refusal_v2/expert_refusal_v2.pt`
- `stage0/results/refusal_v2/SMOLLM2_REFUSAL_V2_SUMMARY.md`
- `stage0/results/refusal_v2/smollm2_refusal_v2_metrics.csv`
- `stage0/results/refusal_v2/smollm2_refusal_v2_generations.jsonl`

Key result:

| model | harmful clean | harmful attempt | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|
| `expert_refusal_v2` | 0.062 | 1.000 | 0.000 | 1.000 |
| `merge_all_v2_linear` | 0.000 | 0.188 | 0.375 | 0.125 |
| `merge_arith_refusal_v2` | 0.000 | 0.812 | 0.250 | 0.312 |

Interpretation:

- The expert learned to attempt refusal, but mostly by blanket-refusing and
  producing bad attempts.
- The full merge did not inherit clean refusal.

### Balanced v2

Command:

```bash
HF_HOME=/data/gavin/model_merging/hf_cache python3 stage0/scripts/run_smollm2_refusal_v2.py \
  --device cuda:1 \
  --train-examples 1024 \
  --eval-examples 16 \
  --max-steps 60 \
  --batch-size 8 \
  --lr 7e-6 \
  --benign-fraction 0.65 \
  --max-new-tokens 28 \
  --artifact-dir stage0/artifacts/smollm2_refusal_v2_balanced \
  --result-dir stage0/results/refusal_v2_balanced
```

Outputs:

- `stage0/artifacts/smollm2_refusal_v2_balanced/expert_refusal_v2.pt`
- `stage0/results/refusal_v2_balanced/SMOLLM2_REFUSAL_V2_SUMMARY.md`
- `stage0/results/refusal_v2_balanced/smollm2_refusal_v2_metrics.csv`
- `stage0/results/refusal_v2_balanced/smollm2_refusal_v2_generations.jsonl`

Key result:

| model | harmful clean | harmful attempt | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|
| `expert_refusal_v2` | 0.375 | 1.000 | 0.000 | 0.188 |
| `merge_arith_refusal_v2` | 0.000 | 0.375 | 0.688 | 0.000 |
| `merge_polite_refusal_v2` | 0.000 | 0.000 | 0.125 | 0.000 |
| `merge_all_v2_linear` | 0.000 | 0.062 | 0.438 | 0.000 |

Interpretation:

- More benign contrast reduced blanket over-refusal.
- The expert became partially clean on harmful prompts.
- But the refusal signal became too weak under naive merging; all v2 merges had
  0.000 harmful clean-refusal.

### Balanced v2 Full-Delta Alpha Search

Script:

- `stage0/scripts/run_smollm2_refusal_v2_alpha_search.py`

Formula:

```text
theta = theta_merge_arith_polite + alpha * (theta_refusal_v2 - theta_base)
```

Command:

```bash
HF_HOME=/data/gavin/model_merging/hf_cache python3 stage0/scripts/run_smollm2_refusal_v2_alpha_search.py \
  --device cuda:1 \
  --eval-examples 16 \
  --max-new-tokens 28 \
  --alphas 0,0.1,0.2,0.33,0.5,0.75,1.0,1.25 \
  --v2-artifact-dir stage0/artifacts/smollm2_refusal_v2_balanced \
  --result-dir stage0/results/refusal_v2_alpha_search
```

Outputs:

- `stage0/results/refusal_v2_alpha_search/SMOLLM2_REFUSAL_V2_ALPHA_SUMMARY.md`
- `stage0/results/refusal_v2_alpha_search/smollm2_refusal_v2_alpha_metrics.csv`
- `stage0/results/refusal_v2_alpha_search/smollm2_refusal_v2_alpha_generations.jsonl`

Key result:

| model | alpha | harmful clean | harmful attempt | benign helpful | benign over-refusal |
|---|---:|---:|---:|---:|---:|
| `alpha_refusal_v2_full_a0p75` | 0.75 | 0.000 | 0.750 | 0.438 | 0.375 |
| `alpha_refusal_v2_full_a1` | 1.00 | 0.125 | 0.938 | 0.562 | 0.438 |
| `alpha_refusal_v2_full_a1p25` | 1.25 | 0.125 | 1.000 | 0.312 | 0.688 |

Interpretation:

- Increasing alpha restores attempted refusal.
- It does not restore clean refusal.
- Higher alpha increases benign over-refusal and bad attempts.
- There is no useful alpha sweet spot in this sweep.

## What We Learned

The failure is not just that the original refusal data was too narrow. The
current base setup has a deeper generation-quality problem:

- The non-instruction base is poor at chat-format continuation.
- Small full fine-tunes can teach local response fragments, but often not stable
  answer termination.
- Refusal targets become entangled with chat artifacts and repeated continuations.
- Stronger refusal deltas trade off against benign over-refusal.

The tokenizer uses `<|im_end|>` as EOS, and the chat SFT examples include it, so
the issue is not an obvious missing EOS token. The model simply does not learn
clean termination reliably in this setup.

## Decision

Do not move this v2 setup to Stage 2/3 mechanistic analysis as a clean refusal
target.

Recommended next branch:

1. Start a new Stage 0 instruction-base branch using
   `HuggingFaceTB/SmolLM2-135M-Instruct` as both base model and tokenizer.
2. Retrain arithmetic, polite, and refusal experts from that common instruction
   base.
3. Re-run the same merge and v2 harmful/benign audit.
4. Only if the expert and at least one merge show clean harmful refusal with low
   benign over-refusal, rebuild Stage 1/2/3 caches for that branch.

Fallback if instruction-base merging still fails:

- Treat the SmolLM2 work as a failure-mechanism case study.
- Move the clean-safety-transfer study to a real public merge pair with stronger
  chat behavior.
