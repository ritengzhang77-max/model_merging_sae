# SmolLM2 Refusal Direction Steering

Exploratory causal check: add the refusal-attempt centroid direction to layer-20 MLP outputs during generation.

| model | alpha | n | clean | attempt | bad attempt | problem | repetition | artifact | unsafe/contradictory | no refusal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| merge_arith_polite | 0.0 | 16 | 0.000 | 0.000 | 0.000 | 0.125 | 0.000 | 0.000 | 0.125 | 0.875 |
| merge_arith_polite | 0.25 | 16 | 0.000 | 0.000 | 0.000 | 0.188 | 0.000 | 0.000 | 0.188 | 0.812 |
| merge_arith_polite | 0.5 | 16 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | 0.000 | 0.062 | 0.938 |
| merge_arith_polite | 1.0 | 16 | 0.000 | 0.000 | 0.000 | 0.125 | 0.000 | 0.000 | 0.125 | 0.875 |
| merge_arith_polite | 2.0 | 16 | 0.000 | 0.000 | 0.000 | 0.062 | 0.000 | 0.000 | 0.062 | 0.938 |

## Interpretation

This is a causal sanity check for a raw activation direction, not an SAE result. A useful direction should increase attempted refusal without simply increasing artifacts, repetition, or unsafe continuations.
