# SmolLM2 Refusal Activation Patches

Dynamic activation patching during greedy generation. The donor is run on the same current prefix, selected donor activations are inserted into the recipient, and the recipient chooses the next token.

## Baselines

| model | kind | n | clean | attempt | bad attempt | problem | repetition | artifact | unsafe | no refusal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| merge_all_linear | recipient_baseline | 16 | 0.688 | 0.688 | 0.000 | 0.312 | 0.000 | 0.000 | 0.312 | 0.000 |
| merge_arith_polite | donor_reference | 16 | 0.000 | 0.000 | 0.000 | 0.125 | 0.000 | 0.000 | 0.125 | 0.875 |

## Best Patches Preserving Attempt

| donor | patch | position | attempt | problem | reduction | repetition | artifact | unsafe | no refusal |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| merge_arith_polite | 29:block | prompt | 0.688 | 0.312 | 0.000 | 0.000 | 0.000 | 0.312 | 0.000 |
| merge_arith_polite | 20+25+29:mlp | prompt | 0.625 | 0.312 | 0.000 | 0.000 | 0.000 | 0.312 | 0.062 |

## Best Problem Reduction Overall

| donor | patch | position | attempt | problem | reduction | repetition | artifact | unsafe | no refusal |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| merge_arith_polite | 29:block | prompt | 0.688 | 0.312 | 0.000 | 0.000 | 0.000 | 0.312 | 0.000 |
| merge_arith_polite | 20+25+29:mlp | prompt | 0.625 | 0.312 | 0.000 | 0.000 | 0.000 | 0.312 | 0.062 |
| merge_arith_polite | 20+25+29:block | prompt | 0.562 | 0.312 | 0.000 | 0.000 | 0.000 | 0.312 | 0.125 |

## Interpretation

A useful activation patch should reduce problem responses without merely deleting the refusal attempt. If activation patching has the same tradeoff as weight patching, the quality components are still not isolated.
