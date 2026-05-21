# Stage 3 Refusal Failure Findings

Updated: 2026-05-16

## Main Result

The synthetic SmolLM2 refusal behavior should be treated as an attempted-refusal
mechanism, not as clean safety behavior.

Across all 336 Stage 3 generations, the heuristic assistant-audit labels are:

- clean refusal: 1
- messy refusal by repetition: 120
- messy refusal by artifact/corruption: 30
- contradictory or unsafe continuation: 65
- no refusal: 120

This means clean-refusal analysis is not decision-grade in this toy setup. There
are too few positive examples.

The Stage 3 causal tests now point in the same direction: the strongest simple
directions and late-module patches are useful diagnostics, but they are not
clean repair handles. When an intervention reduces a visible failure mode, it
usually also suppresses attempted refusal or shifts the output into a different
failure mode.

## Model-Level Pattern

The strongest split is:

- `expert_refusal` and `merge_arith_refusal` attempt refusal on all prompts, but
  almost entirely by repetitive looping.
- `merge_all_linear` attempts refusal often, but still mostly fails quality
  checks through repetition, artifacts, or contradictory continuation.
- `alpha_refusal_late_mlp_a1` and `alpha_refusal_late_mlp_attn_a1` partially
  restore refusal attempts, but also create many problem responses.
- `base` and `merge_arith_polite` mostly do not attempt refusal.

## Basis Validation

The most stable target is attempted refusal vs no attempted refusal.

Best simple bases:

- attempted refusal: `mlp_out_l20` raw, balanced accuracy 0.841.
- bad attempted refusal: `mlp_out_l20` raw, balanced accuracy 0.846.
- repetition: `mlp_out_l20` PCA-16/32, balanced accuracy 0.940.
- artifact: `resid_l20` PCA-16/32, balanced accuracy 0.776.
- unsafe/contradictory continuation: `mlp_out_l25` raw, balanced accuracy 0.743.

This is a high bar for SAE/transcoder work. Sparse features must add causal or
interpretive value beyond these raw/PCA baselines.

## Causal Direction Steering

We tested the best attempted-refusal raw direction:

- representation: `mlp_out_l20`
- direction: attempted-refusal centroid minus non-attempt centroid
- recipient: `merge_arith_polite`
- prompt count: 16
- alphas: 0, 0.25, 0.5, 1, 2
- interventions: add direction at last token, and add direction at all token
  positions

Result:

- neither last-token nor all-token steering induced attempted refusal.
- higher alphas mainly changed wording, artifacts, or unsafe continuation.

Interpretation:

The raw `mlp_out_l20` direction is predictive but not sufficient as a simple
additive causal control. This supports the earlier module-level result: refusal
is not just a single linear direction at one MLP output. It likely needs a
multi-component mechanism involving late MLPs, attention/routing, and decoding
quality components.

## Causal Quality Module Patches

We then tested whether late module replacement can reduce quality failures while
preserving attempted refusal.

### Recipient: `merge_arith_refusal`

Setup:

- recipient: `merge_arith_refusal`
- donors: `merge_all_linear`, `merge_arith_polite`, `base`
- examples: 16 refusal prompts
- patch specs: layer 20/25/29 `attn`, `mlp`, `block`, plus 20+25+29 grouped
  variants

Baseline:

- attempted refusal: 1.000
- problem response: 1.000
- repetition: 1.000

Result:

- best attempt-preserving patches reduced problem rate by only 0.062.
- `merge_all_linear` donor patches did essentially nothing.
- `base` or `merge_arith_polite` late MLP/block patches sometimes reduced
  repetition, but mostly shifted failure into artifact, unsafe continuation, or
  no-refusal.

Interpretation:

The fully repetitive refusal behavior in `merge_arith_refusal` is not repaired
by simple late attention/MLP/block replacement. The failure is either distributed
or baked into the refusal expert's output behavior.

### Recipient: `merge_all_linear`

First screen:

- recipient: `merge_all_linear`
- donors: `base`, `merge_arith_polite`
- examples: 16 refusal prompts

Promising candidates:

- `base -> layer 29 block`: preserved attempt approximately and reduced problem
  rate by 0.188.
- `base -> layers 20+25+29 MLP/block`: reduced problem rate by 0.312 but did not
  preserve attempted refusal.

Validation:

- recipient: `merge_all_linear`
- donor: `base`
- examples: 48 refusal prompts
- patches: `29:block`, `20+25+29:mlp`, `20+25+29:block`

Baseline `merge_all_linear`:

- attempted refusal: 0.708
- problem response: 0.833
- repetition: 0.354
- artifact: 0.167
- unsafe/contradictory: 0.312
- no-refusal: 0.167

Validation results:

- `base -> 29:block`: attempted refusal 0.646, problem response 0.708,
  repetition 0.229, no-refusal 0.229.
- `base -> 20+25+29:mlp`: attempted refusal 0.562, problem response 0.604,
  repetition 0.188, no-refusal 0.333.
- `base -> 20+25+29:block`: attempted refusal 0.521, problem response 0.583,
  repetition 0.167, no-refusal 0.375.

Interpretation:

Late base patches can suppress repetition and reduce overall problem rate, but
they also weaken attempted refusal. The strongest quality improvement is partly
a deletion of the refusal behavior, not a clean repair of it.

## Dynamic Activation Patching

We then moved from weight-level module replacement to activation-level patching.

Script:

- `stage3/scripts/run_smollm2_refusal_activation_patches.py`

Method:

- recipient: `merge_all_linear`
- donors: `base`, `merge_arith_polite`
- examples: 16 refusal prompts
- generation: custom greedy loop without KV cache
- at each decoding step:
  - run donor on the current prefix
  - patch selected donor activations into recipient
  - let recipient choose the next token

Important caveat:

- The custom no-cache greedy loop is cleaner than the earlier Hugging Face
  `model.generate` outputs. Therefore these activation-patching numbers should
  be read as within-loop causal comparisons, not as direct replacements for the
  earlier generation metrics.

### Base donor

Candidate patches:

- `29:block`
- `20+25+29:mlp`
- `20+25+29:block`

Position modes:

- `prompt`: patch previous context positions, leave current final token
  unpatched.
- `last`: patch current final token only.
- `all`: patch all current-prefix positions.

Recipient baseline under the custom loop:

- attempted refusal: 0.688
- clean refusal: 0.688
- problem response: 0.312
- no-refusal: 0.000

Results:

- `prompt` + `29:block`: no measurable change.
- `prompt` + `20+25+29:mlp`: worsened problem response from 0.312 to 0.438.
- `prompt` + `20+25+29:block`: strongly worsened quality, with attempted
  refusal 0.938 but problem response 0.938 and repetition 0.688.
- `last` + `29:block` or `20+25+29:block`: deleted refusal attempt entirely,
  increasing no-refusal to 0.812.
- `all` patches behaved like last-token patches: refusal attempt dropped to
  0.000 and no-refusal rose to 0.812.

Interpretation:

- Patching base activations at the current generation position removes refusal.
- Patching prompt/history activations is either inert or harmful.
- This again shows the problem: reducing failure often means deleting the
  refusal mechanism, not repairing its quality.

### `merge_arith_polite` donor

Candidate prompt-position patches:

- `29:block`
- `20+25+29:mlp`
- `20+25+29:block`

Results:

- `29:block`: no measurable change.
- `20+25+29:mlp`: slight reduction in attempted refusal, no problem reduction.
- `20+25+29:block`: larger reduction in attempted refusal, no problem
  reduction.

Interpretation:

- The arith+polite donor does not provide a clean style/quality patch at these
  activation sites.
- Quality repair is not isolated by these coarse activation patch points.

## Dynamic Direction Ablation

We next tested whether the predictive failure-mode directions from the Stage 3
basis analysis are also causal control handles.

Script:

- `stage3/scripts/run_smollm2_refusal_direction_ablation.py`

Method:

- compute failure-mode centroid directions from the Stage 3 prompt-final
  activation cache.
- intervene during generation by removing or subtracting the target direction at
  the current last-token activation.
- evaluate with the stricter assistant-audit labels.
- use Hugging Face `generate` mode for the main refusal-failure comparisons,
  because the custom no-cache loop is cleaner than the earlier generation path
  and can hide the repetition failure.

### Recipient: `merge_arith_refusal`

Baseline under Hugging Face generation:

- attempted refusal: 1.000
- problem response: 1.000
- repetition: 1.000

Positive-projection removal:

- targets: `mlp_out_l20:failure_repetition` and
  `mlp_out_l20:failure_bad_attempt`
- alphas: 0.25, 0.5, 1.0, 2.0
- result: no measurable effect. Attempted refusal, problem response, and
  repetition all stayed at 1.000.

Constant subtraction of `mlp_out_l20:failure_repetition`:

- alpha 0.5 reduced repetition from 1.000 to 0.750, but problem response stayed
  at 1.000 and artifacts appeared.
- alpha 1.0 reduced repetition to 0.250, but artifacts rose to 0.625 and
  problem response remained 0.938.

Interpretation:

- The repetition direction can push the decoder away from literal repetition
  only when the intervention is strong.
- Strong subtraction does not repair refusal quality. It mostly converts
  repetition into artifact/corruption or no-refusal.

### Recipient: `merge_all_linear`

Baseline under Hugging Face generation on the 16-prompt screen:

- attempted refusal: 0.750
- problem response: 0.812
- repetition: 0.250
- artifact: 0.125
- unsafe/contradictory: 0.438
- no-refusal: 0.188

Mixed failure-direction test:

- `mlp_out_l20:failure_repetition`
- `resid_l20:failure_artifact`
- `mlp_out_l25:failure_unsafe_or_contradictory`

Results:

- artifact and unsafe/contradictory directions preserved attempted refusal, but
  gave no problem-rate reduction.
- the repetition direction gave the best overall reduction, but only by 0.125
  at alpha 0.25; attempted refusal dropped from 0.750 to 0.562.
- alpha 0.5 and 1.0 reduced repetition more, but shifted failures into
  unsafe/contradictory continuation or no-refusal.

Fine alpha sweep for `mlp_out_l20:failure_repetition`:

- alpha 0.01 and 0.02: no measurable effect.
- alpha 0.05 and 0.075: preserved attempted refusal approximately, but reduced
  problem response by only 0.062 and raised no-refusal to 0.250.
- alpha 0.20 and 0.25: reduced problem response by 0.125, but attempted refusal
  dropped to 0.562 and no-refusal stayed at 0.250.

Interpretation:

- There is no narrow clean-repair regime in this sweep.
- The predictive repetition direction is causally entangled with attempted
  refusal and with other bad-output modes.
- Direction ablation is therefore useful as a failure-mechanism probe, but not
  as a repair method in the current toy setup.

## Current Scientific Claim

Defensible claim:

> In this controlled SmolLM2 merge, refusal attempts are inherited and
> predictable from mid/late activations, but clean refusal is not reliably
> inherited. Module-level patches and raw activation directions predict the
> signal better than they causally repair quality. Weight-level, activation-level,
> and direction-level interventions reveal the same tradeoff: suppressing
> failures tends to suppress refusal itself or move the output into another
> failure mode.

Do not claim:

- that this setup demonstrates safe refusal transfer.
- that clean safety behavior was merged successfully.
- that a single activation direction causally controls refusal.
- that SAE/transcoder work is justified without beating these baselines.

## Next Step

This is now a decision gate before expensive SAE/transcoder work.

Recommended next move:

- build a better refusal expert and benchmark before treating this as a clean
  refusal-merge interpretability target.

Reason:

- the current refusal expert/merge is scientifically useful as a failure case,
  but too degenerate for a clean-safety mechanism case study.
- SAE/transcoder analysis on this setup would mostly explain repetition,
  artifact, and refusal-deletion tradeoffs.
- that is valuable if the paper is about merge failure mechanisms, but it is
  not the right next step if the target is clean safety/refusal transfer.

Concrete options:

- Option A: continue with this setup as a failure-mechanism case study.
- Option B: retrain a stronger synthetic refusal expert with diverse refusal
  templates, explicit helpful alternatives, neutral prompts, EOS discipline,
  lower learning rate, and early stopping.
- Option C: move to a public real-model merge pair and use the SmolLM2 toy setup
  only as a controlled pilot.

Immediate next actions by option:

- Option A: train/load sparse models only after reframing the paper around merge
  failure, then explain the repetition/artifact/no-refusal tradeoff.
- Option B: build a new refusal data generator, retrain the refusal expert, rerun
  Stage 0 behavioral validation, and only then rerun Stage 2/3.
- Option C: select a real open-source merge pair, reproduce its merge recipe,
  and port the Stage 0-3 diagnostics to that pair.

## 2026-05-19 Option B Check

We tried Option B on the same non-instruction SmolLM2 base.

Summary:

- first v2 refusal expert: harmful clean-refusal 0.062, benign over-refusal
  1.000.
- balanced v2 refusal expert: harmful clean-refusal 0.375, benign over-refusal
  0.188.
- balanced v2 linear merges: harmful clean-refusal 0.000.
- full refusal-delta alpha search: attempted refusal returns at high alpha, but
  clean refusal stays low and benign over-refusal rises.

Decision:

- the current base remains unsuitable for a clean refusal-transfer target.
- the next clean-target branch should restart from
  `HuggingFaceTB/SmolLM2-135M-Instruct` as the common base.
- until that branch succeeds, SAE/transcoder work on the current refusal setup
  should be framed as failure-mechanism analysis, not clean safety-transfer
  interpretation.
