# SmolLM2 Refusal Alpha Search Summary

Run date: 2026-05-10

Script:

- `stage2/scripts/run_smollm2_refusal_alpha_search.py`

Inputs:

- recipient: `merge_arith_polite`
- donor: `expert_refusal`
- variants:
  - `late_mlp`: refusal MLPs at layers 15, 20, 25, 29
  - `late_mlp_attn`: refusal attention + MLP at layers 15, 20, 25, 29
- alpha values: 0.25, 0.5, 0.75, 1.0, 1.25

Outputs:

- `stage2/results/smollm2_refusal_alpha_search_losses.csv`
- `stage2/results/smollm2_refusal_alpha_search_generation_metrics.csv`
- `stage2/results/smollm2_refusal_alpha_search_generations.jsonl`
- `stage2/results/smollm2_refusal_alpha_search_summary.json`

## Generation Results

| model | alpha | arith | polite | refusal | mean |
|---|---:|---:|---:|---:|---:|
| merge_arith_polite | - | 0.167 | 1.000 | 0.000 | 0.389 |
| late_mlp | 0.25 | 0.167 | 1.000 | 0.000 | 0.389 |
| late_mlp | 0.50 | 0.167 | 1.000 | 0.167 | 0.444 |
| late_mlp | 0.75 | 0.167 | 1.000 | 0.417 | 0.528 |
| late_mlp | 1.00 | 0.167 | 1.000 | 0.750 | 0.639 |
| late_mlp | 1.25 | 0.167 | 1.000 | 0.750 | 0.639 |
| late_mlp_attn | 0.25 | 0.167 | 1.000 | 0.000 | 0.389 |
| late_mlp_attn | 0.50 | 0.167 | 1.000 | 0.167 | 0.444 |
| late_mlp_attn | 0.75 | 0.167 | 1.000 | 0.500 | 0.556 |
| late_mlp_attn | 1.00 | 0.167 | 1.000 | 0.750 | 0.639 |
| late_mlp_attn | 1.25 | 0.167 | 1.000 | 0.750 | 0.639 |
| merge_all_linear | - | 0.583 | 1.000 | 1.000 | 0.861 |

## Loss Results

| model | alpha | arith loss | polite loss | refusal loss | mean loss |
|---|---:|---:|---:|---:|---:|
| merge_arith_polite | - | 1.401 | 0.311 | 3.436 | 1.716 |
| late_mlp | 0.25 | 1.423 | 0.314 | 3.142 | 1.626 |
| late_mlp | 0.50 | 1.448 | 0.319 | 2.828 | 1.532 |
| late_mlp | 0.75 | 1.476 | 0.324 | 2.463 | 1.421 |
| late_mlp | 1.00 | 1.509 | 0.335 | 2.210 | 1.351 |
| late_mlp | 1.25 | 1.550 | 0.353 | 1.975 | 1.293 |
| late_mlp_attn | 0.25 | 1.433 | 0.315 | 3.071 | 1.606 |
| late_mlp_attn | 0.50 | 1.467 | 0.321 | 2.678 | 1.489 |
| late_mlp_attn | 0.75 | 1.512 | 0.330 | 2.207 | 1.350 |
| late_mlp_attn | 1.00 | 1.558 | 0.349 | 1.900 | 1.269 |
| late_mlp_attn | 1.25 | 1.614 | 0.382 | 1.616 | 1.204 |
| merge_all_linear | - | 2.036 | 0.386 | 0.488 | 0.970 |

## Interpretation

Alpha controls a clear tradeoff:

- refusal loss improves monotonically as alpha increases;
- visible refusal generation improves until alpha 1.0, then saturates at 75%;
- arithmetic and polite losses worsen as alpha increases;
- attention helps loss when paired with MLP, but does not improve generation
  beyond the MLP-only alpha 1.0 result.

The cleanest mechanism-aware recipe in this search is:

```text
recipient: merge_arith_polite
donor: expert_refusal
modules: MLPs at layers 15, 20, 25, 29
alpha: 1.0
```

This reaches:

- arithmetic accuracy: 0.167, same as `merge_arith_polite`
- polite accuracy: 1.000, same as `merge_arith_polite`
- refusal accuracy: 0.750, up from 0.000

The loss-best recipe is `late_mlp_attn` at alpha 1.25, but it does not improve
generation accuracy beyond alpha 1.0 and it causes larger arithmetic/polite loss
drift. For now, it is less clean as a merge recipe.

## Current Mechanistic Claim

In this controlled SmolLM2 synthetic setup, refusal can be partially restored to
an arithmetic+polite merge by inserting only a small set of refusal-expert late
MLP modules. The effect is alpha-sensitive and mostly MLP-driven; attention is
auxiliary for loss but not sufficient for visible refusal behavior.

This is stronger than a correlation result: the modules were identified by
patch/ablation, then used to build a targeted micro-merge that changes behavior.

## Next Step

Before moving to SAE/transcoder analysis, run one robustness check:

- evaluate this alpha-1.0 late-MLP recipe over more generated examples;
- add a simple quality label for clean refusal versus messy/refusal-like text;
- compare against `merge_all_linear`.

If the behavior remains stable, the next mechanistic step is to open the
late-layer MLPs and ask which neurons or sparse features carry the refusal
transfer.
