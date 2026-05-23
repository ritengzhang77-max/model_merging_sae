# Stage 3: Basis Validation For Clean vs Messy Refusal

Stage 3 tests whether a richer representational basis helps explain the key
Stage 2 finding:

> Late MLP module insertion restores a refusal signal, but the full merge gives
> cleaner refusal behavior.

Stage 3 has since corrected that framing. The behavior is better described as
attempted-refusal transfer with frequent quality failure, not as reliable clean
refusal transfer. The main decision memo is
`results/STAGE3_REFUSAL_FAILURE_FINDINGS.md`.

The Gemma branch now has an actual linear weight-merge bridge:

- Along `abliterated + alpha * (base - abliterated)`, harmful clean refusal
  jumps from `0.000` at alpha `0.25` to `0.667` at alpha `0.50` and `0.917` at
  alpha `0.75`.
- Alpha `0.75` matches the base donor's harmful clean-refusal rate on the
  12-prompt screen while preserving `1.000` benign helpfulness and avoiding the
  base model's one benign over-refusal.
- The same alpha `0.75` result holds better than the sparse pruning branch on
  the fake-ID family: it passes 7/8 harmful variants with 8/8 benign helpfulness,
  while base passes 7/8 but over-refuses once.
- SAE feature trajectories along the linear merge show that L19 feature `16048`
  is not a simple natural safety marker: on the fake-ID family its harmful
  generated activation peaks near alpha `0.50` and drops at safer alpha `0.75`
  / `1.00`. The L12 features `40` and `12075` increase in the safe alpha regime,
  even though they were antagonists in narrow sparse patches.
- A teacher-forced check confirms the split: on fixed safe alpha-`0.75`
  continuations, L12 features increase with model alpha, while L19 `16048` does
  not; on fixed unsafe alpha-`0.00` continuations, L19 `16048` can be high under
  safer model weights.
- A broad fixed-continuation transition search finds stronger natural
  safe-merge candidates than the hand-picked features. The top features are L17
  `4342`, L17 `16011`, L16 `16332`, L18 `10415`, and L18 `11127`; qualitative
  audit shows legal-consequence/refusal-rationale tokens such as `Forgery`,
  `Criminal`, `felony`, `jail`, and `theft`.
- Causal bundle patching is asymmetric but not yet specific: patching the top10
  transition features from alpha `0.75` into alpha `0.25` reduces unsafe
  continuation but does not restore clean refusal, while patching low-alpha
  values into alpha `0.75` drops family clean refusal from 7/8 to 4/8. Matched
  random same-layer ten-feature bundles also drop the alpha `0.75` model to 4/8
  under the same `mix_decode` operator, so the selected top10 bundle is not yet
  validated as specifically necessary.
- This gives the SAE feature-trajectory work a concrete parameter-space merge
  curve to explain.

The first target is deliberately conservative. Before training or trusting
SAEs/transcoders, compare simple bases:

- residual stream activations;
- MLP output activations;
- raw coordinates;
- PCA projections;
- random projections;
- top-variance neuron subsets.

Only if sparse features beat or complement these baselines should we spend more
compute on SAE/transcoder interpretation.

## Active Qwen Residual Track

The active clean target has moved to the public Qwen2.5-1.5B safety-loss case:

- donor/base: `Qwen/Qwen2.5-1.5B-Instruct`;
- recipient: `nbeerbower/EVA-abliterated-TIES-Qwen2.5-1.5B`;
- shared baseline: PCA64 donor-recipient MLP delta on layers `12-24`;
- hard residual repair target: full donor MLP activations in layers `16-23`.

The first frozen benchmark file is:

- `data/qwen1_5b_residual_benchmark/qwen1_5b_residual_benchmark_v0.jsonl`

This benchmark separates donor-solved residual harmful prompts from paired
benign controls. It is intentionally small; it exists to make subsequent
SAE/transcoder validation reproducible before expanding prompt coverage.

Current v0 baseline:

- full donor MLP `16-23` passes all harmful and benign v0 prompts;
- residual PCA/mean baselines do not;
- native MLP-output coordinate top-k only passes the full v0 benchmark at
  `topk1344`, which is `87.5%` of the `1536`-dimensional MLP output space.

This means the first SAE/transcoder target is not merely "beat PCA"; it must
beat or explain a broad coordinate-residual baseline.

First learned SAE smoke:

- per-layer residual SAEs with high train EV (`0.993-0.996`) do not preserve
  the full v0 behavior;
- all tested SAE variants repair one-time-code but fail tracking-script and
  permission-slip;
- this holds even for weakly sparse `d1536`/`d2048` dictionaries.

Generated-trace follow-up:

- training on generated tokens from the successful `PCA64 + full 16-23`
  trajectory improves the family split but still does not pass the full gate;
- generated-token `d512_l1_0.0001` repairs one-time-code and permission-slip
  but still fails tracking-script;
- all-position generated `d512_l1_0.0001` falls back to one-time-code only.

The immediate lesson is that reconstruction quality is not causal completeness,
and distribution/position matching matters. The next sparse attempt should be
family-specific, position-specific, or transcoder-style, not merely a larger
vanilla residual SAE.

## Active GemmaScope MLP SAE Track

The Gemma branch is now the strongest sparse-basis target:

- donor/base: `google/gemma-2-2b-it`;
- recipient: `IlyaGusev/gemma-2-2b-it-abliterated`;
- causal range: post-feedforward MLP update over layers `12-20`;
- full donor `12-20:post_ff` patch: harmful clean refusal `1.000`, benign
  helpfulness `1.000`;
- GemmaScope MLP-SAE decoded `12-20:post_ff` patch: harmful clean refusal
  `1.000`, benign helpfulness `1.000`.

This is a behavioral-completeness pass for a public sparse basis. It is not yet
a feature-level explanation because the intervention uses full decoded
reconstruction. The next Gemma RQ is whether feature subsets can reproduce the
repair beyond broad coordinate baselines such as `top_neuron_k1536`.

First feature-subset result:

- selected GemmaScope MLP-SAE coordinates chosen by harmful donor-recipient
  activation delta beat matched random active-feature controls on heldout
  prompts;
- `mix_decode_delta_abs_k1024` passes heldout prompt slice `4:8` and reaches
  `0.750` harmful clean refusal on slice `8:12`;
- full decoded SAE and all-feature delta repairs pass both heldout slices;
- the result is a promising causal feature signal, but not yet a complete
  mechanistic explanation because the feature budget is still large and one
  heldout family remains unsolved.

Layer-group localization:

- no single 3-layer band among `12-14`, `15-17`, and `18-20` is sufficient;
- `12-17` is weak on both heldout folds, so late layers are necessary;
- late-containing six-layer pairs are much stronger, but prompt-dependent:
  `15-20` fully passes heldout slice `4:8`, while `12-14,18-20` has the
  stronger full-decoded result on slice `8:12`;
- the next target should be late-containing groups and feature identity, not
  another broad all-layer sweep.

Random-seed controls:

- matched random active-feature controls were repeated across five seeds for
  all `12-20`, `15-20`, and `12-14,18-20`;
- random-active k1024 never restored harmful refusal in any tested group/slice;
- random-active k2048 sometimes repaired one or two prompts, but remained below
  top-delta k2048 for all `12-20` and `15-20`;
- the robust sparse-feature claim now belongs mainly to all `12-20` and
  `15-20`; `12-14,18-20` remains too noisy for a strong sparse mechanism claim.

Boundary-vs-content audit:

- top-delta feature event rows were dominated by assistant-boundary/template
  tokens: `2817 / 2880` top absolute-delta rows (`97.8%`) were on tokens such
  as newline, `model`, `<end_of_turn>`, and `<start_of_turn>`;
- selecting features only from content-ish prompt tokens sharply reduced sparse
  repair: all `12-20` k1024 fell from `1.000` to `0.000` on heldout `4:8` and
  from `0.750` to `0.000` on heldout `8:12`;
- content-selected features do activate on meaningful harmful tokens such as
  `keylogger`, `phishing`, `bank`, and `malware`, but they currently do not
  reproduce the all-token sparse repair;
- the live mechanistic hypothesis is now response-boundary refusal-state
  transfer, not a clean prompt-content semantic feature story.

The decisive next Gemma test is position-restricted patching: take the
successful all-token selected features and patch them only at assistant-boundary
positions versus only at content positions versus all positions.

Position-restricted patching:

- static assistant-boundary patching alone does not reproduce the repair:
  all `12-20`, heldout `4:8`, k1024 falls from `1.000` harmful clean refusal
  with all-position patching to `0.000`;
- prompt content patching also fails, even when paired with generated-token
  history: all `12-20`, `4:8`, `contentish_or_generated` remains `0.000`;
- the best reduced mode is `assistant_boundary_or_generated`: patch the
  assistant boundary in the prompt, then maintain the same selected features on
  generated-token history during rollout;
- `assistant_boundary_or_generated` reaches `0.750` on all `12-20` `4:8`,
  matches all-position k1024 on all `12-20` `8:12` at `0.750`, and matches
  late `15-20` `8:12` at `0.500`;
- the current mechanism is therefore better described as autoregressive
  refusal-state trajectory repair, not a static boundary patch and not
  harmful-content semantics.

k2048 and budget follow-up:

- increasing the selected-feature budget to k2048 preserves the same reduced
  mechanism: all-position `12-20` reaches `1.000` harmful clean refusal on
  `4:8` and `0.750` on `8:12`; `assistant_boundary_or_generated` reaches
  `0.750` on both folds;
- `prompt_template_or_generated` matches `assistant_boundary_or_generated`,
  again pointing to chat-template/assistant-start state rather than ordinary
  harmful content tokens;
- `contentish_or_generated` is still not competitive: on k2048 it reaches only
  `0.250` harmful clean refusal with `0.250` unsafe continuation on `4:8`, and
  the `8:12` run failed twice during generation;
- reducing the `assistant_boundary_or_generated` budget to k512 still repairs
  partially: `0.750` harmful clean refusal on `4:8` and `0.500` on `8:12`, both
  with `0.000` unsafe continuation;
- k256 failed twice during generation, so the current practical threshold is
  bracketed between k512 and k1024 rather than established exactly.

Feature-ID causality follow-up:

- a lower-budget sweep inside `assistant_boundary_or_generated` shows the easy
  `4:8` fold reaches `0.750` harmful clean refusal by k256, but the harder
  `8:12` fold stays at `0.500` through k896 and reaches `0.750` only at k1024;
- tail-only rank bands are not sufficient: ranks `897-1024`, `769-1024`, and
  `513-1024` alone all score `0.000` harmful clean refusal on `8:12`;
- the k1024-over-k896 gain localizes to layer 19 on the fake-ID prompt:
  k896 plus only the layer-19 `897-1024` tail recovers the prompt, while the
  same tail from layers `12-18` or `20` does not;
- removing the layer-19 `897-1024` tail from k1024 drops `8:12` from `0.750` to
  `0.500`, while removing any other single-layer tail tested leaves `0.750`;
- splitting that tail further localizes the fake-ID recovery to a single
  feature: layer 19 feature ID `16048`, global rank `1006`;
- adding only feature `16048` to the k896 prefix recovers the fake-ID prompt;
  removing only feature `16048` from k1024 removes that recovery;
- explicit feature-ID validation over all 12 harmful/benign prompts confirms
  that feature `16048` controls exactly the fake-ID recovery, not the broader
  refusal repair;
- cross-basis validation narrows the claim: basis `0:8` naturally includes
  feature `16048` inside k896 and removing it breaks fake-ID recovery, but
  basis `4:8` also ranks the feature inside k896 and still fails fake-ID, so
  the cooperating prefix selected from the basis matters;
- prefix localization shows the cooperating prefix is nonmonotone: with basis
  `0:8`, `k256 + L19 f16048` passes fake-ID, `k384 + L19 f16048` fails,
  `k640 + L19 f16048` passes, and `k768 + L19 f16048` fails;
- the first clear antagonist is an L12 boundary-state band: adding L12 ranks
  `257-384` to `k256 + L19 f16048` breaks fake-ID, while removing that same
  L12 band from failing `k384 + L19 f16048` restores it;
- singleton additions identify L12 rank `274` feature ID `40` and L12 rank
  `295` feature ID `12075` as individually sufficient disruptors, though the
  inverse removals show the larger prefix interaction is redundant and
  nonadditive;
- the feature audit is still mostly assistant-template/boundary and prompt-end
  events, so the lead is a layer-local response-state refinement rather than a
  clean harmful-content semantic feature.
- a decoded-delta-add robustness check does not recover fake-ID for
  `k256 + L19 f16048`, `k384 + L19 f16048`, or their L12 antagonist/removal
  variants, so the strongest claim currently belongs to the `mix_decode`
  coordinate-replacement operator rather than to generic decoded delta addition.
- timing-mask controls show the repair needs prompt-template state plus
  generated-token maintenance: assistant-boundary-only, generated-only, and
  content-ish-or-generated masks fail, while prompt-template-or-generated
  succeeds and bypasses the two singleton L12 antagonist effects seen under
  the narrower assistant-boundary-or-generated mask.
- feature-specific timing shows L19 feature `16048` acts during generated
  tokens in the clean basis `0:4`, k896 test; assistant-boundary/template-only
  patching of this feature fails. Under basis `0:8`, k256 already passes
  fake-ID without feature `16048`, so the feature is causal in some prefixes,
  redundant in others, and insufficient in others.
- the generated-token timing result replicates on all 12 harmful and 12 benign
  prompts: generated-only `f16048` matches boundary-or-generated `f16048`,
  while assistant-boundary-only `f16048` matches the k896 prefix-only baseline.
- a small fake-ID paraphrase family does not replicate the semantic story:
  k896 already passes 6/8 fake-ID variants, and generated-token `f16048` does
  not improve the family pass rate, so feature `16048` should not be labeled a
  broad fake-ID semantic feature.
- the two L12 singleton antagonists split by timing under the narrow trajectory:
  rank `274` / feature `40` disrupts when patched during generation, while
  rank `295` / feature `12075` disrupts when patched at the assistant boundary.
- full-prompt L12 replication shows these antagonist effects also move aggregate
  harmful clean-refusal and unsafe rates, while benign helpfulness remains
  `1.000`.
- broad prompt-template patching bypasses those L12 singleton antagonist
  effects: with k256 `prompt_template_or_generated`, generated-token `f16048`
  repairs fake-ID and the same L12 additions no longer break it.
- mechanism-aware pruning is actionable but scope-sensitive: removing L12 ranks
  `273-352` from failing `k384 + f16048` restores fake-ID and improves the full
  benchmark, while removing the broader `257-384` band hurts the full benchmark.
- same-size removal controls show the pruning effect is structured but not
  unique: removing L12 ranks `1-80` also helps, `81-160` hurts, and several
  other same-size bands are neutral.
- the L12 `1-80` pruning benefit localizes to ranks `1-16` for fake-ID recovery,
  but the narrower removal introduces an unsafe side effect that the wider
  `1-80` removal avoids.
- splitting L12 ranks `1-16` shows another nonadditive bundle: neither ranks
  `1-8` nor `9-16` alone recovers fake-ID, but removing `1-16` together does.
- composing helpful L12 removals is not additive: `1-80` and `273-352` each
  repair the `k384 + f16048` fake-ID failure, but removing both does not improve
  the full benchmark, and `1-16 plus 273-352` loses fake-ID recovery while
  adding one unsafe continuation.
- the L12 pruning repair does not generalize cleanly to the small fake-ID
  paraphrase family: `k384 + f16048` and `k896` each pass 6/8 variants, while
  single L12-band removals drop to 5/8 and the composed removal only returns to
  6/8.
- broad `prompt_template_or_generated` timing also fails to make the fake-ID
  family robust: it leaves k896 at 6/8, drops `k384 + f16048` to 5/8, and the
  only 6/8 pruned condition introduces unsafe continuations.
- signed trajectory logging on the fake-ID prompt shows the key deltas are
  donor-high generated-token trajectory effects. Crucially, the L12 antagonist
  features are also donor-high, so "more donor-like" is not enough; donor-high
  sparse features can be helpful, redundant, or timed antagonists.

Main artifacts:

- `results/GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE_SUBSET_FINDINGS.md`
- `results/GEMMA2_2B_GEMMASCOPE_MLP_SAE_LAYER_GROUP_FINDINGS.md`
- `results/GEMMA2_2B_GEMMASCOPE_MLP_SAE_RANDOM_SEED_FINDINGS.md`
- `results/GEMMA2_2B_GEMMASCOPE_MLP_SAE_BOUNDARY_VS_CONTENT_FINDINGS.md`
- `results/GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE_ID_CAUSALITY_FINDINGS.md`
- `results/GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE16048_PREFIX_LOCALIZATION_FINDINGS.md`
- `results/GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE16048_TIMING_MASK_FINDINGS.md`
- `results/GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE16048_FEATURE_SPECIFIC_TIMING_FINDINGS.md`
- `results/GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE16048_SIGNED_TRAJECTORY_FINDINGS.md`
- `results/GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE16048_FAKE_ID_FAMILY_FINDINGS.md`
- `results/GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE16048_MECHANISM_AWARE_PRUNING_FINDINGS.md`
- `results/GEMMA2_2B_LINEAR_WEIGHT_MERGE_SWEEP_FINDINGS.md`
- `results/GEMMA2_2B_LINEAR_MERGE_SAE_FEATURE_TRAJECTORY_FINDINGS.md`
- `results/GEMMA2_2B_LINEAR_MERGE_SAE_TRANSITION_FEATURE_SEARCH_FINDINGS.md`
- `results/GEMMA2_2B_LINEAR_MERGE_SAE_BUNDLE_PATCH_FINDINGS.md`
- `results/gemma2_2b_gemmascope_mlp_sae_random_seed_controls_v0/GEMMA2_2B_GEMMASCOPE_MLP_SAE_RANDOM_SEED_SUMMARY.md`
- `results/gemma2_2b_gemmascope_mlp_sae_feature_audit_v0/GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE_AUDIT_SUMMARY.md`
- `results/gemma2_2b_gemmascope_mlp_sae_content_token_feature_controls_v0/GEMMA2_2B_GEMMASCOPE_MLP_SAE_RANDOM_SEED_SUMMARY.md`
- `results/gemma2_2b_gemmascope_mlp_sae_content_token_feature_audit_v0/GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE_AUDIT_SUMMARY.md`
- `results/gemma2_2b_gemmascope_mlp_sae_position_restricted_atomic_v0/GEMMA2_2B_GEMMASCOPE_MLP_SAE_POSITION_RESTRICTED_SUMMARY.md`
- `results/gemma2_2b_gemmascope_mlp_sae_position_restricted_k2048_v0/GEMMA2_2B_GEMMASCOPE_MLP_SAE_POSITION_RESTRICTED_SUMMARY.md`
- `results/gemma2_2b_gemmascope_mlp_sae_boundary_generated_budget_v0/GEMMA2_2B_GEMMASCOPE_MLP_SAE_POSITION_RESTRICTED_SUMMARY.md`
- `results/gemma2_2b_gemmascope_mlp_sae_feature_id_threshold_v0/`
- `results/gemma2_2b_gemmascope_mlp_sae_feature_id_l19_tail_blocks_v0/`
- `results/gemma2_2b_gemmascope_mlp_sae_l19_tail_feature_audit_v0/GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE_AUDIT_SUMMARY.md`
- `results/gemma2_2b_gemmascope_mlp_sae_layer_groups_pairs_eval_4_8_v0/GEMMA2_2B_GEMMASCOPE_MLP_SAE_LAYER_GROUP_SUMMARY.md`
- `results/gemma2_2b_gemmascope_mlp_sae_layer_groups_pairs_eval_8_12_v0/GEMMA2_2B_GEMMASCOPE_MLP_SAE_LAYER_GROUP_SUMMARY.md`
- `results/gemma2_2b_gemmascope_mlp_sae_feature_subsets_12_20_heldout_k_sweep_v0/GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE_SUBSET_SUMMARY.md`
- `results/gemma2_2b_gemmascope_mlp_sae_feature_subsets_12_20_heldout_fold2_v0/GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE_SUBSET_SUMMARY.md`
- `scripts/export_gemma2_2b_gemmascope_mlp_sae_feature_audit.py`
- `scripts/aggregate_gemma2_2b_gemmascope_mlp_sae_position_restricted.py`
- `results/gemma2_2b_gemmascope_mlp_sae_validation_12_20_generation/GEMMA2_2B_GEMMASCOPE_MLP_SAE_VALIDATION_SUMMARY.md`
- `results/gemma2_2b_gemmascope_mlp_sae_validation_12_20_generation/GEMMA2_2B_GEMMASCOPE_MLP_SAE_INTERPRETATION.md`
- `scripts/run_gemma2_2b_gemmascope_mlp_sae_feature_subsets.py`
- `scripts/run_gemma2_2b_gemmascope_mlp_sae_layer_groups.py`
- `scripts/run_gemma2_2b_gemmascope_mlp_sae_random_seed_controls.py`
- `scripts/validate_gemma2_2b_gemmascope_mlp_sae.py`
- `scripts/validate_gemma2_2b_gemmascope_transcoders.py`

Primary script:

- `scripts/analyze_smollm2_refusal_basis.py`

Main outputs:

- `results/smollm2_refusal_basis_records.jsonl`
- `results/smollm2_refusal_basis_metrics.csv`
- `results/SMOLLM2_REFUSAL_BASIS_SUMMARY.md`
- `cache/smollm2_refusal_basis_activations.pt`

Audit helper:

- `scripts/prepare_smollm2_refusal_audit.py`
- `scripts/rescore_smollm2_refusal_basis_cache.py`
- `scripts/apply_smollm2_refusal_assistant_audit.py`
- `scripts/analyze_smollm2_refusal_failure_modes.py`
- `scripts/run_smollm2_refusal_direction_steering.py`
- `scripts/run_smollm2_refusal_quality_module_patches.py`
- `scripts/run_smollm2_refusal_activation_patches.py`
- `scripts/run_smollm2_refusal_direction_ablation.py`
- `results/smollm2_refusal_manual_audit_sample.csv`
- `results/SMOLLM2_REFUSAL_MANUAL_AUDIT_GUIDE.md`
- `results/SMOLLM2_REFUSAL_ASSISTANT_AUDIT_SUMMARY.md`
- `results/SMOLLM2_REFUSAL_FAILURE_MODE_SUMMARY.md`
- `results/SMOLLM2_REFUSAL_DIRECTION_STEERING_SUMMARY.md`
- `results/SMOLLM2_REFUSAL_QUALITY_MODULE_PATCHES_*.md`
- `results/SMOLLM2_REFUSAL_ACTIVATION_PATCHES_*.md`
- `results/SMOLLM2_REFUSAL_DIRECTION_ABLATION_*.md`
