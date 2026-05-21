# SmolLM2 Refusal Direction Ablation

Failure-mode directions are computed from the Stage 3 prompt-final activation cache, then projected out during dynamic greedy generation.

## Baseline

| model | n | clean | attempt | bad attempt | problem | repetition | artifact | unsafe | no refusal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| merge_all_linear | 16 | 0.000 | 0.750 | 0.750 | 0.812 | 0.250 | 0.125 | 0.438 | 0.188 |

## Best Interventions Preserving Attempt

| rep | target | alpha | position | mode | attempt | problem | reduction | repetition | artifact | unsafe | no refusal |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| mlp_out_l20 | failure_repetition | 0.05 | last | constant | 0.688 | 0.750 | 0.062 | 0.188 | 0.188 | 0.375 | 0.250 |
| mlp_out_l20 | failure_repetition | 0.075 | last | constant | 0.688 | 0.750 | 0.062 | 0.125 | 0.250 | 0.375 | 0.250 |
| mlp_out_l20 | failure_repetition | 0.01 | last | constant | 0.750 | 0.812 | 0.000 | 0.250 | 0.125 | 0.438 | 0.188 |
| mlp_out_l20 | failure_repetition | 0.02 | last | constant | 0.750 | 0.812 | 0.000 | 0.250 | 0.125 | 0.438 | 0.188 |

## Best Problem Reduction Overall

| rep | target | alpha | position | mode | attempt | problem | reduction | repetition | artifact | unsafe | no refusal |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| mlp_out_l20 | failure_repetition | 0.2 | last | constant | 0.562 | 0.688 | 0.125 | 0.125 | 0.188 | 0.375 | 0.250 |
| mlp_out_l20 | failure_repetition | 0.25 | last | constant | 0.562 | 0.688 | 0.125 | 0.125 | 0.125 | 0.438 | 0.250 |
| mlp_out_l20 | failure_repetition | 0.05 | last | constant | 0.688 | 0.750 | 0.062 | 0.188 | 0.188 | 0.375 | 0.250 |
| mlp_out_l20 | failure_repetition | 0.075 | last | constant | 0.688 | 0.750 | 0.062 | 0.125 | 0.250 | 0.375 | 0.250 |
| mlp_out_l20 | failure_repetition | 0.1 | last | constant | 0.625 | 0.750 | 0.062 | 0.125 | 0.250 | 0.375 | 0.250 |
| mlp_out_l20 | failure_repetition | 0.15 | last | constant | 0.625 | 0.750 | 0.062 | 0.125 | 0.250 | 0.375 | 0.250 |
| mlp_out_l20 | failure_repetition | 0.01 | last | constant | 0.750 | 0.812 | 0.000 | 0.250 | 0.125 | 0.438 | 0.188 |
| mlp_out_l20 | failure_repetition | 0.02 | last | constant | 0.750 | 0.812 | 0.000 | 0.250 | 0.125 | 0.438 | 0.188 |

## Interpretation

A successful subspace intervention should reduce problem responses while preserving attempted refusal. If the best interventions either do nothing or delete the refusal attempt, the failure direction is predictive but not a clean causal control.
