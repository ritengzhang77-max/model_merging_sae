# SmolLM2 Refusal Direction Ablation

Failure-mode directions are computed from the Stage 3 prompt-final activation cache, then projected out during dynamic greedy generation.

## Baseline

| model | n | clean | attempt | bad attempt | problem | repetition | artifact | unsafe | no refusal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| merge_all_linear | 16 | 0.000 | 0.750 | 0.750 | 0.812 | 0.250 | 0.125 | 0.438 | 0.188 |

## Best Interventions Preserving Attempt

| rep | target | alpha | position | mode | attempt | problem | reduction | repetition | artifact | unsafe | no refusal |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| mlp_out_l25 | failure_unsafe_or_contradictory | 0.1 | last | constant | 0.750 | 0.812 | 0.000 | 0.250 | 0.188 | 0.375 | 0.188 |
| mlp_out_l25 | failure_unsafe_or_contradictory | 0.25 | last | constant | 0.750 | 0.812 | 0.000 | 0.312 | 0.188 | 0.312 | 0.188 |
| mlp_out_l25 | failure_unsafe_or_contradictory | 0.5 | last | constant | 0.750 | 0.812 | 0.000 | 0.250 | 0.188 | 0.375 | 0.188 |
| mlp_out_l25 | failure_unsafe_or_contradictory | 1.0 | last | constant | 0.750 | 0.812 | 0.000 | 0.438 | 0.125 | 0.250 | 0.188 |
| resid_l20 | failure_artifact | 0.1 | last | constant | 0.750 | 0.812 | 0.000 | 0.500 | 0.125 | 0.188 | 0.188 |
| resid_l20 | failure_artifact | 0.25 | last | constant | 0.750 | 0.812 | 0.000 | 0.625 | 0.062 | 0.125 | 0.188 |
| resid_l20 | failure_artifact | 0.5 | last | constant | 0.875 | 0.875 | -0.062 | 0.688 | 0.125 | 0.062 | 0.125 |
| resid_l20 | failure_artifact | 1.0 | last | constant | 1.000 | 1.000 | -0.188 | 0.688 | 0.312 | 0.000 | 0.000 |

## Best Problem Reduction Overall

| rep | target | alpha | position | mode | attempt | problem | reduction | repetition | artifact | unsafe | no refusal |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| mlp_out_l20 | failure_repetition | 0.25 | last | constant | 0.562 | 0.688 | 0.125 | 0.125 | 0.125 | 0.438 | 0.250 |
| mlp_out_l20 | failure_repetition | 0.1 | last | constant | 0.625 | 0.750 | 0.062 | 0.125 | 0.250 | 0.375 | 0.250 |
| mlp_out_l25 | failure_unsafe_or_contradictory | 0.1 | last | constant | 0.750 | 0.812 | 0.000 | 0.250 | 0.188 | 0.375 | 0.188 |
| mlp_out_l25 | failure_unsafe_or_contradictory | 0.25 | last | constant | 0.750 | 0.812 | 0.000 | 0.312 | 0.188 | 0.312 | 0.188 |
| mlp_out_l25 | failure_unsafe_or_contradictory | 0.5 | last | constant | 0.750 | 0.812 | 0.000 | 0.250 | 0.188 | 0.375 | 0.188 |
| mlp_out_l25 | failure_unsafe_or_contradictory | 1.0 | last | constant | 0.750 | 0.812 | 0.000 | 0.438 | 0.125 | 0.250 | 0.188 |
| resid_l20 | failure_artifact | 0.1 | last | constant | 0.750 | 0.812 | 0.000 | 0.500 | 0.125 | 0.188 | 0.188 |
| resid_l20 | failure_artifact | 0.25 | last | constant | 0.750 | 0.812 | 0.000 | 0.625 | 0.062 | 0.125 | 0.188 |
| mlp_out_l20 | failure_repetition | 1.0 | last | constant | 0.312 | 0.812 | 0.000 | 0.000 | 0.188 | 0.625 | 0.188 |
| resid_l20 | failure_artifact | 0.5 | last | constant | 0.875 | 0.875 | -0.062 | 0.688 | 0.125 | 0.062 | 0.125 |
| mlp_out_l20 | failure_repetition | 0.5 | last | constant | 0.500 | 0.938 | -0.125 | 0.062 | 0.250 | 0.625 | 0.062 |
| resid_l20 | failure_artifact | 1.0 | last | constant | 1.000 | 1.000 | -0.188 | 0.688 | 0.312 | 0.000 | 0.000 |

## Interpretation

A successful subspace intervention should reduce problem responses while preserving attempted refusal. If the best interventions either do nothing or delete the refusal attempt, the failure direction is predictive but not a clean causal control.
