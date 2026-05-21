# Stage 2 Results Provenance

This ledger records module-level and causal-diagnostic artifacts.

## Qwen2.5-1.5B Refusal RQ1/RQ2 Diagnostics

- Date appended: 2026-05-20
- Artifact status: canonical current RQ1/RQ2 public-model evidence
- Summary: `stage2/results/qwen1_5b_refusal_rq1_rq2/QWEN1_5B_REFUSAL_RQ1_RQ2_SUMMARY.md`
- Expanded mid-late summary: `stage2/results/qwen1_5b_refusal_rq1_rq2_expanded_midlate/QWEN1_5B_REFUSAL_RQ1_RQ2_SUMMARY.md`
- Generation check summary: `stage2/results/qwen1_5b_patch_generation_check/QWEN1_5B_PATCH_GENERATION_SUMMARY.md`
- Multi-layer range check summary: `stage2/results/qwen1_5b_mlp_range_patch_generation_check/QWEN1_5B_PATCH_GENERATION_SUMMARY.md`
- Best range validation summary: `stage2/results/qwen1_5b_best_range_patch_generation_check/QWEN1_5B_PATCH_GENERATION_SUMMARY.md`
- MLP localization summary: `stage2/results/qwen1_5b_mlp_localization_generation_check/QWEN1_5B_PATCH_GENERATION_SUMMARY.md`
- Activation target-loss summary: `stage2/results/qwen1_5b_activation_patch_target_loss_all/QWEN1_5B_ACTIVATION_PATCH_TARGET_LOSS_SUMMARY.md`
- Target-position activation summary: `stage2/results/qwen1_5b_activation_patch_target_loss_target/QWEN1_5B_ACTIVATION_PATCH_TARGET_LOSS_SUMMARY.md`
- Dynamic activation generation summary: `stage2/results/qwen1_5b_dynamic_activation_patch_generation/QWEN1_5B_DYNAMIC_ACTIVATION_PATCH_GENERATION_SUMMARY.md`
- Low-dimensional target-loss summary: `stage2/results/qwen1_5b_lowdim_activation_patch_target_loss_nopca/QWEN1_5B_LOWDIM_ACTIVATION_PATCH_TARGET_LOSS_SUMMARY.md`
- Low-dimensional generation summary: `stage2/results/qwen1_5b_lowdim_activation_patch_generation/QWEN1_5B_LOWDIM_ACTIVATION_PATCH_GENERATION_SUMMARY.md`
- Low-dimensional generation smoke summary: `stage2/results/qwen1_5b_lowdim_activation_patch_generation_smoke/QWEN1_5B_LOWDIM_ACTIVATION_PATCH_GENERATION_SUMMARY.md`
- PCA/random target-loss summary: `stage2/results/qwen1_5b_lowdim_activation_patch_target_loss_pca_random/QWEN1_5B_LOWDIM_ACTIVATION_PATCH_TARGET_LOSS_SUMMARY.md`
- PCA/random generation summary: `stage2/results/qwen1_5b_lowdim_activation_patch_generation_pca_random/QWEN1_5B_LOWDIM_ACTIVATION_PATCH_GENERATION_SUMMARY.md`
- PCA/random generation manual audit: `stage2/results/qwen1_5b_lowdim_activation_patch_generation_pca_random/QWEN1_5B_LOWDIM_GENERATION_MANUAL_AUDIT.md`
- Progress presentation package: `presentations/2026-05-20-1550-model-merging-progress/`
- Progress presentation PDF: `presentations/2026-05-20-1550-model-merging-progress/model_merging_progress.pdf`
- Progress presentation source: `presentations/2026-05-20-1550-model-merging-progress/model_merging_progress.tex`
- Main diagnostic script: `stage2/scripts/analyze_qwen1_5b_refusal_rq1_rq2.py`
- Generation-check script: `stage2/scripts/run_qwen1_5b_patch_generation_check.py`
- Activation target-loss script: `stage2/scripts/run_qwen1_5b_activation_patch_target_loss.py`
- Dynamic activation generation script: `stage2/scripts/run_qwen1_5b_dynamic_activation_patch_generation.py`
- Low-dimensional target-loss script: `stage2/scripts/run_qwen1_5b_lowdim_activation_patch_target_loss.py`
- Low-dimensional generation script: `stage2/scripts/run_qwen1_5b_lowdim_activation_patch_generation.py`
- Source Stage 0 behavior: `stage0/results/candidate_screens/qwen1_5b_selected_extended/chat_candidate_screen_metrics.csv`

Commands:

```bash
python3 stage2/scripts/analyze_qwen1_5b_refusal_rq1_rq2.py \
  --device cuda:1 \
  --examples-per-split 8 \
  --batch-size 2 \
  --result-dir stage2/results/qwen1_5b_refusal_rq1_rq2
```

```bash
python3 stage2/scripts/analyze_qwen1_5b_refusal_rq1_rq2.py \
  --device cuda:1 \
  --examples-per-split 8 \
  --batch-size 2 \
  --patch-layers 14,15,16,17,18,19,20,21,22,23,24,25,26,27 \
  --modules mlp,block \
  --result-dir stage2/results/qwen1_5b_refusal_rq1_rq2_expanded_midlate
```

```bash
python3 stage2/scripts/run_qwen1_5b_patch_generation_check.py \
  --device cuda:1 \
  --examples-per-split 8 \
  --max-new-tokens 64 \
  --result-dir stage2/results/qwen1_5b_patch_generation_check
```

```bash
python3 stage2/scripts/run_qwen1_5b_patch_generation_check.py \
  --device cuda:1 \
  --examples-per-split 8 \
  --max-new-tokens 64 \
  --patch-specs '14+15+16+17+18+19:mlp,20+21+22+23+24:mlp,14+15+16+20+21+22+23+24:mlp,14+15+16+17+18+19+20+21+22+23+24:mlp,12+13+14+15+16+17+18+19+20+21+22+23+24:mlp,14+15+16+17+18+19+20+21+22+23+24+25+26+27:mlp,14+15+16+17+18+19+20+21+22+23+24:block' \
  --result-dir stage2/results/qwen1_5b_mlp_range_patch_generation_check
```

```bash
python3 stage2/scripts/run_qwen1_5b_patch_generation_check.py \
  --device cuda:1 \
  --examples-per-split 12 \
  --max-new-tokens 64 \
  --patch-specs '12+13+14+15+16+17+18+19+20+21+22+23+24:mlp,14+15+16+17+18+19+20+21+22+23+24:mlp,14+15+16+17+18+19:mlp,14+15+16+17+18+19+20+21+22+23+24:block' \
  --result-dir stage2/results/qwen1_5b_best_range_patch_generation_check
```

```bash
python3 stage2/scripts/run_qwen1_5b_patch_generation_check.py \
  --device cuda:1 \
  --examples-per-split 12 \
  --max-new-tokens 64 \
  --patch-specs '12+13:mlp,12+13+14+15+16:mlp,12+13+14+15+16+17+18+19:mlp,12+13+14+15+16+17+18+19+20+21:mlp,12+13+14+15+16+17+18+19+20+21+22+23+24:mlp,12+13+20+21+22+23+24:mlp,12+13+14+15+16+20+21+22+23+24:mlp,13+14+15+16+17+18+19+20+21+22+23+24:mlp' \
  --result-dir stage2/results/qwen1_5b_mlp_localization_generation_check
```

```bash
python3 stage2/scripts/run_qwen1_5b_activation_patch_target_loss.py \
  --device cuda:1 \
  --examples-per-split 12 \
  --batch-size 2 \
  --position all \
  --patch-specs '12+13+14+15+16+17+18+19+20+21+22+23+24:mlp,12+13+14+15+16+17+18+19+20+21:mlp,12+13+14+15+16+17+18+19:mlp,13+14+15+16+17+18+19+20+21+22+23+24:mlp' \
  --result-dir stage2/results/qwen1_5b_activation_patch_target_loss_all
```

```bash
python3 stage2/scripts/run_qwen1_5b_activation_patch_target_loss.py \
  --device cuda:1 \
  --examples-per-split 12 \
  --batch-size 2 \
  --position target \
  --patch-specs '12+13+14+15+16+17+18+19+20+21+22+23+24:mlp,12+13+14+15+16+17+18+19+20+21:mlp,12+13+14+15+16+17+18+19:mlp' \
  --result-dir stage2/results/qwen1_5b_activation_patch_target_loss_target
```

```bash
python3 stage2/scripts/run_qwen1_5b_dynamic_activation_patch_generation.py \
  --device cuda:1 \
  --examples-per-split 8 \
  --max-new-tokens 64 \
  --position all \
  --result-dir stage2/results/qwen1_5b_dynamic_activation_patch_generation
```

```bash
python3 stage2/scripts/run_qwen1_5b_lowdim_activation_patch_target_loss.py \
  --device cuda:1 \
  --examples-per-split 8 \
  --batch-size 2 \
  --layers 12,13,14,15,16,17,18,19,20,21,22,23,24 \
  --neuron-ks 16,64,256 \
  --pca-ranks '' \
  --max-pca-rows-per-layer 0 \
  --result-dir stage2/results/qwen1_5b_lowdim_activation_patch_target_loss_nopca
```

```bash
python3 stage2/scripts/run_qwen1_5b_lowdim_activation_patch_generation.py \
  --device cuda:1 \
  --examples-per-split 8 \
  --basis-examples-per-split 8 \
  --batch-size 2 \
  --max-new-tokens 64 \
  --variants mean_delta,top_neuron_k256 \
  --result-dir stage2/results/qwen1_5b_lowdim_activation_patch_generation
```

```bash
python3 stage2/scripts/run_qwen1_5b_lowdim_activation_patch_target_loss.py \
  --device cuda:1 \
  --examples-per-split 8 \
  --batch-size 2 \
  --layers 12,13,14,15,16,17,18,19,20,21,22,23,24 \
  --neuron-ks 16,64,256 \
  --pca-ranks 1,4,16,64 \
  --random-ranks 1,4,16,64 \
  --max-pca-rows-per-layer 512 \
  --pca-oversample 8 \
  --pca-n-iter 1 \
  --random-seed 0 \
  --result-dir stage2/results/qwen1_5b_lowdim_activation_patch_target_loss_pca_random
```

```bash
python3 stage2/scripts/run_qwen1_5b_lowdim_activation_patch_generation.py \
  --device cuda:1 \
  --examples-per-split 8 \
  --basis-examples-per-split 8 \
  --batch-size 2 \
  --max-new-tokens 64 \
  --variants mean_delta,pca_rank16,pca_rank64,random_rank64 \
  --max-pca-rows-per-layer 512 \
  --pca-oversample 8 \
  --pca-n-iter 1 \
  --random-seed 0 \
  --result-dir stage2/results/qwen1_5b_lowdim_activation_patch_generation_pca_random
```

What it shows:

- `nbeerbower/EVA-abliterated-TIES-Qwen2.5-1.5B` has much higher refusal-target
  loss than the base on harmful prompts: `2.213` vs `0.089`.
- Base-vs-abliterated activation drift is prompt-type selective. The largest
  benign-minus-harmful gaps appear around layers 20-24, with layer 22 harmful
  cosine `0.724` and benign cosine `0.899`.
- Static base-to-abliterated module patches give the strongest target-loss
  repair in mid layers, especially layers 14-16 MLP/block. In the expanded
  screen, `16:block` has the best specificity (`0.096`), while `14:mlp` closes
  more harmful-refusal target-loss gap (`0.285`) but also closes more benign
  refusal gap.
- Single-module generation validation does not show causal sufficiency: patches
  `14:mlp`, `14:block`, `16:mlp`, and `16:block` all remain at `0.000` harmful
  clean refusal and `0.000` attempted refusal on 8 harmful prompts.
- Multi-layer MLP replacement is causally sufficient for substantial repair.
  On the 8-prompt screen, `12-24:mlp` restores harmful clean refusal to `0.875`.
  On the 12-prompt validation, `12-24:mlp` restores harmful clean refusal to
  `0.917`, compared with `0.750` for the base and `0.000` for the abliterated
  model.
- The repair remains specific in the current prompt set: benign helpfulness is
  `1.000` and benign over-refusal is `0.000` for the `12-24:mlp` patch.
- Localization shows a smooth range effect: `12-13:mlp` is `0.000`, `12-16:mlp`
  is `0.167`, `12-19:mlp` is `0.500`, `12-21:mlp` is `0.750`, and `12-24:mlp`
  is `0.917` harmful clean refusal on the 12-prompt validation.
- Removing layer 12 weakens the repair: `13-24:mlp` reaches `0.500`, not
  `0.917`.
- Teacher-forced activation patching over all sequence positions closes most of
  the refusal-target loss gap: `12-24:mlp` closes `0.938`, `12-21:mlp` closes
  `0.937`, and `12-19:mlp` closes `0.917`.
- Target-position-only activation patching closes `0.000` of the gap, so the
  repair requires sequence-wide MLP activation replacement.
- Dynamic generation activation patching succeeds on the 8 harmful / 8 benign
  screen: `12-24:mlp` donor activations restore harmful clean refusal to
  `1.000`, with benign helpfulness `1.000` and benign over-refusal `0.000`.
- Low-dimensional target-loss baselines show that a simple mean donor-recipient
  delta direction per layer is surprisingly strong: `mean_delta_rank1` closes
  `0.853` of the harmful-refusal target-loss gap and `0.835` of the benign gap.
- The `top_neuron_k256` baseline is similarly strong at target loss (`0.844`
  harmful gap closed), while a hand-built harmful-vs-benign refusal direction is
  much weaker (`0.371` harmful gap closed).
- Generation validation separates the baselines. On the 8 harmful / 8 benign
  screen, `mean_delta_rank1` matches the base harmful clean-refusal rate
  (`0.875`) and keeps benign helpfulness `1.000` with benign over-refusal
  `0.000`. `top_neuron_k256` is weaker (`0.625` harmful clean refusal and
  `0.875` benign helpfulness).
- Fast randomized PCA improves over mean-delta. At target loss, `pca_rank64`
  closes `0.955` of the harmful gap and `pca_rank16` closes `0.906`; matched
  `random_rank64` closes only `0.230`.
- Automatic generation validation initially marks both `pca_rank16` and
  `pca_rank64` as `1.000` harmful clean refusal, with `random_rank64` at
  `0.000`.
- Manual audit finds the automatic scorer over-credits refusal-prefix answers
  that later drift into unsafe advice. Strict harmful clean-refusal rates are:
  base `1.000`, abliterated `0.000`, `mean_delta_rank1` `0.750`,
  `pca_rank16` `0.625`, `pca_rank64` `1.000`, and `random_rank64` `0.000`.

Current interpretation:

- The abliterated model gives a strong safety-loss case for RQ1/RQ2.
- Mid-layer modules carry some likelihood-level refusal signal, and wide
  mid-to-late MLP restoration can recover greedy refusal behavior.
- Later layers around 20-24 show the largest harmful-specific activation drift,
  but late-only `20-24:mlp` restoration is not sufficient. The current repair
  needs the broader `12-24:mlp` range.
- The immediate mechanistic object is now sharper: the abliterated model appears
  to be missing a low-dimensional donor-recipient MLP delta subspace across
  layers 12-24. The mean-delta direction is the minimal repair, but strict
  manual audit makes `pca_rank64` the current strongest non-sparse behavioral
  baseline.
- SAE or transcoder work must explain, compress, or improve on the PCA delta
  subspace, not just beat full MLP replacement or the mean-delta direction.
- The evaluator itself is now part of RQ0: before larger sweeps, harmful-refusal
  scoring needs contradiction/unsafe-continuation detection.

Caveats:

- Refusal-target loss uses one standard refusal completion and should not be
  treated as behavior by itself.
- Generation validation uses small 8-prompt and 12-prompt harmful/benign
  screens, not a full safety benchmark.
- PCA uses a randomized implementation with 512 activation-delta rows per layer,
  oversample 8, and one power iteration; exact PCA has not been run.
- The current evidence supports localization and failure-mode analysis, not a
  solved repair mechanism.
