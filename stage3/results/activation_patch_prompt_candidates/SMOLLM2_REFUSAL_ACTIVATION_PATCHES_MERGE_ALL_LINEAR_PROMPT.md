# SmolLM2 Refusal Activation Patches

Dynamic activation patching during greedy generation. The donor is run on the same current prefix, selected donor activations are inserted into the recipient, and the recipient chooses the next token.

## Baselines

| model | kind | n | clean | attempt | bad attempt | problem | repetition | artifact | unsafe | no refusal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| merge_all_linear | recipient_baseline | 16 | 0.688 | 0.688 | 0.000 | 0.312 | 0.000 | 0.000 | 0.312 | 0.000 |
| base | donor_reference | 16 | 0.000 | 0.000 | 0.000 | 0.188 | 0.000 | 0.000 | 0.188 | 0.812 |

## Best Patches Preserving Attempt

| donor | patch | position | attempt | problem | reduction | repetition | artifact | unsafe | no refusal |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | 29:block | prompt | 0.688 | 0.312 | 0.000 | 0.000 | 0.000 | 0.312 | 0.000 |
| base | 20+25+29:mlp | prompt | 0.625 | 0.438 | -0.125 | 0.125 | 0.062 | 0.250 | 0.125 |
| base | 20+25+29:block | prompt | 0.938 | 0.938 | -0.625 | 0.688 | 0.188 | 0.062 | 0.062 |

## Best Problem Reduction Overall

| donor | patch | position | attempt | problem | reduction | repetition | artifact | unsafe | no refusal |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | 29:block | prompt | 0.688 | 0.312 | 0.000 | 0.000 | 0.000 | 0.312 | 0.000 |
| base | 20+25+29:mlp | prompt | 0.625 | 0.438 | -0.125 | 0.125 | 0.062 | 0.250 | 0.125 |
| base | 20+25+29:block | prompt | 0.938 | 0.938 | -0.625 | 0.688 | 0.188 | 0.062 | 0.062 |

## Interpretation

A useful activation patch should reduce problem responses without merely deleting the refusal attempt. If activation patching has the same tradeoff as weight patching, the quality components are still not isolated.
