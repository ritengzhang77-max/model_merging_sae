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
| none | none | none | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Best Problem Reduction Overall

| donor | patch | position | attempt | problem | reduction | repetition | artifact | unsafe | no refusal |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | 20+25+29:block | last | 0.000 | 0.188 | 0.125 | 0.000 | 0.000 | 0.188 | 0.812 |
| base | 29:block | last | 0.000 | 0.188 | 0.125 | 0.000 | 0.000 | 0.188 | 0.812 |
| base | 20+25+29:mlp | last | 0.000 | 0.375 | -0.062 | 0.000 | 0.000 | 0.375 | 0.625 |

## Interpretation

A useful activation patch should reduce problem responses without merely deleting the refusal attempt. If activation patching has the same tradeoff as weight patching, the quality components are still not isolated.
