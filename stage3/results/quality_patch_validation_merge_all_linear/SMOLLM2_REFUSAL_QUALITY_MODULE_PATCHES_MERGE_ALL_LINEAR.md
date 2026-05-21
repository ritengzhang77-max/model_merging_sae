# SmolLM2 Refusal Quality Module Patches

Causal screen: replace selected late modules in a repetitive/refusal recipient with donor modules and measure whether quality failures decrease while attempted refusal is preserved.

## Baselines

| model | kind | n | clean | attempt | bad attempt | problem | repetition | artifact | unsafe | no refusal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| merge_all_linear | recipient_baseline | 48 | 0.000 | 0.708 | 0.708 | 0.833 | 0.354 | 0.167 | 0.312 | 0.167 |
| base | donor_reference | 48 | 0.000 | 0.000 | 0.000 | 0.312 | 0.000 | 0.000 | 0.312 | 0.688 |

## Best Patches Preserving Attempt

| recipient | donor | patch | attempt | problem | problem reduction | repetition | artifact | unsafe | no refusal |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| merge_all_linear | base | 29:block | 0.646 | 0.708 | 0.125 | 0.229 | 0.188 | 0.292 | 0.229 |

## Best Problem Reduction Overall

| recipient | donor | patch | attempt | problem | problem reduction | repetition | artifact | unsafe | no refusal |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| merge_all_linear | base | 20+25+29:block | 0.521 | 0.583 | 0.250 | 0.167 | 0.125 | 0.292 | 0.375 |
| merge_all_linear | base | 20+25+29:mlp | 0.562 | 0.604 | 0.229 | 0.188 | 0.146 | 0.271 | 0.333 |
| merge_all_linear | base | 29:block | 0.646 | 0.708 | 0.125 | 0.229 | 0.188 | 0.292 | 0.229 |

## Interpretation

A useful quality patch should reduce `problem_response_rate` without a large drop in `attempted_refusal_rate`. If the best patches reduce problem rate only by destroying refusal attempts, the quality mechanism is not isolated by that module replacement.
