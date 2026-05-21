# SmolLM2 Refusal Quality Module Patches

Causal screen: replace selected late modules in a repetitive/refusal recipient with donor modules and measure whether quality failures decrease while attempted refusal is preserved.

## Baselines

| model | kind | n | clean | attempt | bad attempt | problem | repetition | artifact | unsafe | no refusal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| merge_all_linear | recipient_baseline | 16 | 0.000 | 0.750 | 0.750 | 0.812 | 0.250 | 0.125 | 0.438 | 0.188 |
| merge_arith_polite | donor_reference | 16 | 0.000 | 0.000 | 0.000 | 0.125 | 0.000 | 0.000 | 0.125 | 0.875 |
| base | donor_reference | 16 | 0.000 | 0.000 | 0.000 | 0.188 | 0.000 | 0.000 | 0.188 | 0.812 |

## Best Patches Preserving Attempt

| recipient | donor | patch | attempt | problem | problem reduction | repetition | artifact | unsafe | no refusal |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| merge_all_linear | base | 29:block | 0.688 | 0.625 | 0.188 | 0.062 | 0.188 | 0.375 | 0.250 |
| merge_all_linear | merge_arith_polite | 20+25+29:attn | 0.688 | 0.750 | 0.062 | 0.188 | 0.188 | 0.375 | 0.250 |
| merge_all_linear | merge_arith_polite | 25:block | 0.688 | 0.750 | 0.062 | 0.188 | 0.250 | 0.312 | 0.250 |
| merge_all_linear | base | 20+25+29:attn | 0.750 | 0.812 | 0.000 | 0.250 | 0.188 | 0.375 | 0.188 |
| merge_all_linear | base | 20:block | 0.750 | 0.812 | 0.000 | 0.312 | 0.188 | 0.312 | 0.188 |
| merge_all_linear | base | 25:block | 0.750 | 0.812 | 0.000 | 0.375 | 0.188 | 0.250 | 0.188 |

## Best Problem Reduction Overall

| recipient | donor | patch | attempt | problem | problem reduction | repetition | artifact | unsafe | no refusal |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| merge_all_linear | base | 20+25+29:mlp | 0.562 | 0.500 | 0.312 | 0.000 | 0.188 | 0.312 | 0.375 |
| merge_all_linear | base | 20+25+29:block | 0.500 | 0.500 | 0.312 | 0.000 | 0.062 | 0.438 | 0.438 |
| merge_all_linear | base | 29:block | 0.688 | 0.625 | 0.188 | 0.062 | 0.188 | 0.375 | 0.250 |
| merge_all_linear | merge_arith_polite | 20+25+29:attn | 0.688 | 0.750 | 0.062 | 0.188 | 0.188 | 0.375 | 0.250 |
| merge_all_linear | merge_arith_polite | 25:block | 0.688 | 0.750 | 0.062 | 0.188 | 0.250 | 0.312 | 0.250 |
| merge_all_linear | merge_arith_polite | 20:block | 0.562 | 0.750 | 0.062 | 0.188 | 0.125 | 0.438 | 0.250 |
| merge_all_linear | base | 20+25+29:attn | 0.750 | 0.812 | 0.000 | 0.250 | 0.188 | 0.375 | 0.188 |
| merge_all_linear | base | 20:block | 0.750 | 0.812 | 0.000 | 0.312 | 0.188 | 0.312 | 0.188 |
| merge_all_linear | base | 25:block | 0.750 | 0.812 | 0.000 | 0.375 | 0.188 | 0.250 | 0.188 |
| merge_all_linear | merge_arith_polite | 20+25+29:block | 0.250 | 0.812 | 0.000 | 0.000 | 0.188 | 0.625 | 0.188 |
| merge_all_linear | merge_arith_polite | 20+25+29:mlp | 0.250 | 0.812 | 0.000 | 0.000 | 0.188 | 0.625 | 0.188 |
| merge_all_linear | merge_arith_polite | 29:block | 0.375 | 0.875 | -0.062 | 0.125 | 0.188 | 0.562 | 0.125 |

## Interpretation

A useful quality patch should reduce `problem_response_rate` without a large drop in `attempted_refusal_rate`. If the best patches reduce problem rate only by destroying refusal attempts, the quality mechanism is not isolated by that module replacement.
