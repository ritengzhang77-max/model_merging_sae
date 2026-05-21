# Stage 2: Capability Inheritance

Stage 2 asks whether a merged model keeps an expert capability by using the
same internal modules as the expert, or whether it preserves benchmark behavior
through a different route.

Current first-pass script:

```bash
HF_HOME=/data/gavin/model_merging/hf_cache \
python3 stage2/scripts/analyze_smollm2_capability_inheritance.py \
  --eval-examples 16 \
  --max-delta-values 80000
```

Fast smoke test:

```bash
HF_HOME=/data/gavin/model_merging/hf_cache \
python3 stage2/scripts/analyze_smollm2_capability_inheritance.py \
  --eval-examples 2 \
  --batch-size 2 \
  --result-dir stage2/results/smoke_capability_inheritance
```

Outputs:

- `stage2/results/smollm2_capability_inheritance_layer_ablation.csv`
- `stage2/results/smollm2_capability_inheritance_profile_correlations.csv`
- `stage2/results/smollm2_capability_inheritance_summary.json`

Module-level follow-up:

```bash
HF_HOME=/data/gavin/model_merging/hf_cache \
python3 stage2/scripts/analyze_smollm2_module_inheritance.py \
  --eval-examples 16 \
  --batch-size 8
```

Outputs:

- `stage2/results/smollm2_module_inheritance_ablation.csv`
- `stage2/results/smollm2_module_inheritance_profile_correlations.csv`
- `stage2/results/smollm2_module_inheritance_top_interventions.csv`

Module sufficiency follow-up:

```bash
HF_HOME=/data/gavin/model_merging/hf_cache \
python3 stage2/scripts/analyze_smollm2_module_sufficiency.py \
  --eval-examples 16 \
  --batch-size 8
```

Outputs:

- `stage2/results/smollm2_module_sufficiency_patches.csv`
- `stage2/results/smollm2_module_sufficiency_top_patches.csv`

Mechanism-aware micro-merge:

```bash
HF_HOME=/data/gavin/model_merging/hf_cache \
python3 stage2/scripts/run_smollm2_mechanism_micro_merge.py \
  --loss-examples 16 \
  --gen-examples 12
```

Outputs:

- `stage2/results/smollm2_micro_merge_losses.csv`
- `stage2/results/smollm2_micro_merge_generation_metrics.csv`
- `stage2/results/smollm2_micro_merge_generations.jsonl`

Refusal alpha search:

```bash
HF_HOME=/data/gavin/model_merging/hf_cache \
python3 stage2/scripts/run_smollm2_refusal_alpha_search.py \
  --loss-examples 16 \
  --gen-examples 12
```

Outputs:

- `stage2/results/smollm2_refusal_alpha_search_losses.csv`
- `stage2/results/smollm2_refusal_alpha_search_generation_metrics.csv`
- `stage2/results/smollm2_refusal_alpha_search_generations.jsonl`

Refusal quality robustness:

```bash
HF_HOME=/data/gavin/model_merging/hf_cache \
python3 stage2/scripts/evaluate_smollm2_refusal_quality.py \
  --examples 48
```

Outputs:

- `stage2/results/smollm2_refusal_quality_summary.csv`
- `stage2/results/smollm2_refusal_quality_records.jsonl`
- `stage2/results/smollm2_refusal_manual_audit.csv`
- `stage2/results/smollm2_refusal_manual_audit_summary.csv`

Interpretation:

- A layer is important for a task/model when replacing that layer with base
  weights increases task loss.
- Expert-merge profile correlation asks whether the merged model is vulnerable
  at the same layers as the source expert.
- This is not yet a full circuit proof. It is a cheap causal screen for where
  inherited mechanisms might live.
