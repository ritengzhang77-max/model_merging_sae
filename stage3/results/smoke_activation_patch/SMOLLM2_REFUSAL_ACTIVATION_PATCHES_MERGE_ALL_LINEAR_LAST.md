# SmolLM2 Refusal Activation Patches

Dynamic activation patching during greedy generation. The donor is run on the same current prefix, selected donor activations are inserted into the recipient, and the recipient chooses the next token.

## Baselines

| model | kind | n | clean | attempt | bad attempt | problem | repetition | artifact | unsafe | no refusal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| merge_all_linear | recipient_baseline | 4 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| base | donor_reference | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |

## Best Patches Preserving Attempt

| donor | patch | position | attempt | problem | reduction | repetition | artifact | unsafe | no refusal |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| none | none | none | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Best Problem Reduction Overall

| donor | patch | position | attempt | problem | reduction | repetition | artifact | unsafe | no refusal |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | 29:block | last | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |

## Interpretation

A useful activation patch should reduce problem responses without merely deleting the refusal attempt. If activation patching has the same tradeoff as weight patching, the quality components are still not isolated.
