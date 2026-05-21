# Candidate Screening Policy

Updated: 2026-05-20

Stage 0 is now a cheap candidate filter, not a place to debug one model deeply.

## Rule

Do not spend major effort on a candidate until it passes a small behavioral
screen.

## Screening Budget

Per candidate:

- one short setup pass;
- one small generation/evaluation pass;
- one simple merge or alpha sweep only if the first pass is promising.

Stop immediately if:

- the base model cannot follow the prompt format cleanly;
- the expert behavior is messy before merging;
- the merge only transfers keywords, artifacts, repetition, or blanket refusal;
- fixing the candidate would require custom training/debugging beyond the cheap
  screen.

## Pass Criteria

A candidate is worth Stage 1/2/SAE work only if:

- the donor expert behavior is clean;
- the merged behavior is clean enough to score automatically and audit manually;
- the behavior is not just keyword matching;
- there is a clear mechanistic question left after raw/PCA/module baselines.

## Current Decision

Abandon the current `HuggingFaceTB/SmolLM2-135M` non-instruction refusal branch
as a clean refusal-transfer target.

Next candidates should be screened quickly:

1. `HuggingFaceTB/SmolLM2-135M-Instruct` as a cheap local instruction-base check.
2. Qwen2.5 0.5B family candidates, especially public fine-tunes/merges with a
   shared base.
3. Existing public MergeKit merges where the recipe and parent checkpoints are
   recoverable.

If none pass quickly, pivot to a simpler non-refusal capability or a real public
merge pair with stronger chat behavior.
