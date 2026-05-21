# SmolLM2 Refusal Quality Module Patches

Causal screen: replace selected late modules in a repetitive/refusal recipient with donor modules and measure whether quality failures decrease while attempted refusal is preserved.

## Baselines

| model | kind | n | clean | attempt | bad attempt | problem | repetition | artifact | unsafe | no refusal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| merge_arith_refusal | recipient_baseline | 16 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| merge_all_linear | donor_reference | 16 | 0.000 | 0.750 | 0.750 | 0.812 | 0.250 | 0.125 | 0.438 | 0.188 |
| merge_arith_polite | donor_reference | 16 | 0.000 | 0.000 | 0.000 | 0.125 | 0.000 | 0.000 | 0.125 | 0.875 |
| base | donor_reference | 16 | 0.000 | 0.000 | 0.000 | 0.188 | 0.000 | 0.000 | 0.188 | 0.812 |

## Best Patches Preserving Attempt

| recipient | donor | patch | attempt | problem | problem reduction | repetition | artifact | unsafe | no refusal |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| merge_arith_refusal | merge_arith_polite | 20+25+29:block | 0.938 | 0.938 | 0.062 | 0.875 | 0.062 | 0.000 | 0.062 |
| merge_arith_refusal | base | 20+25+29:block | 0.938 | 0.938 | 0.062 | 0.562 | 0.188 | 0.188 | 0.062 |
| merge_arith_refusal | merge_arith_polite | 20+25+29:mlp | 0.938 | 0.938 | 0.062 | 0.875 | 0.062 | 0.000 | 0.062 |
| merge_arith_refusal | base | 20+25+29:mlp | 0.938 | 0.938 | 0.062 | 0.750 | 0.188 | 0.000 | 0.062 |
| merge_arith_refusal | merge_arith_polite | 29:block | 0.938 | 0.938 | 0.062 | 0.875 | 0.062 | 0.000 | 0.062 |
| merge_arith_refusal | base | 29:block | 0.938 | 0.938 | 0.062 | 0.812 | 0.125 | 0.000 | 0.062 |
| merge_arith_refusal | merge_arith_polite | 29:mlp | 0.938 | 0.938 | 0.062 | 0.875 | 0.062 | 0.000 | 0.062 |
| merge_arith_refusal | base | 29:mlp | 0.938 | 0.938 | 0.062 | 0.875 | 0.062 | 0.000 | 0.062 |
| merge_arith_refusal | merge_all_linear | 20+25+29:attn | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| merge_arith_refusal | merge_arith_polite | 20+25+29:attn | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| merge_arith_refusal | base | 20+25+29:attn | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| merge_arith_refusal | merge_all_linear | 20+25+29:block | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |

## Best Problem Reduction Overall

| recipient | donor | patch | attempt | problem | problem reduction | repetition | artifact | unsafe | no refusal |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| merge_arith_refusal | merge_arith_polite | 20+25+29:block | 0.938 | 0.938 | 0.062 | 0.875 | 0.062 | 0.000 | 0.062 |
| merge_arith_refusal | base | 20+25+29:block | 0.938 | 0.938 | 0.062 | 0.562 | 0.188 | 0.188 | 0.062 |
| merge_arith_refusal | merge_arith_polite | 20+25+29:mlp | 0.938 | 0.938 | 0.062 | 0.875 | 0.062 | 0.000 | 0.062 |
| merge_arith_refusal | base | 20+25+29:mlp | 0.938 | 0.938 | 0.062 | 0.750 | 0.188 | 0.000 | 0.062 |
| merge_arith_refusal | merge_arith_polite | 29:block | 0.938 | 0.938 | 0.062 | 0.875 | 0.062 | 0.000 | 0.062 |
| merge_arith_refusal | base | 29:block | 0.938 | 0.938 | 0.062 | 0.812 | 0.125 | 0.000 | 0.062 |
| merge_arith_refusal | merge_arith_polite | 29:mlp | 0.938 | 0.938 | 0.062 | 0.875 | 0.062 | 0.000 | 0.062 |
| merge_arith_refusal | base | 29:mlp | 0.938 | 0.938 | 0.062 | 0.875 | 0.062 | 0.000 | 0.062 |
| merge_arith_refusal | merge_all_linear | 20+25+29:attn | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| merge_arith_refusal | merge_arith_polite | 20+25+29:attn | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| merge_arith_refusal | base | 20+25+29:attn | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| merge_arith_refusal | merge_all_linear | 20+25+29:block | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |

## Interpretation

A useful quality patch should reduce `problem_response_rate` without a large drop in `attempted_refusal_rate`. If the best patches reduce problem rate only by destroying refusal attempts, the quality mechanism is not isolated by that module replacement.
