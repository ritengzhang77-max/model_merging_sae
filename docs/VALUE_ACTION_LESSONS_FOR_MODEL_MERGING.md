# Lessons From `value_action` For Model-Merging SAE Work

Created: 2026-05-22

This memo translates the working style of `/home/gavin/value_action` into the
current model-merging project. The key lesson is methodological: do not spend
deep interpretability effort on a weak model pair, weak SAE basis, or weak
behavioral target. First find a setup where the behavior is real, the
activation intervention is causal, and the sparse basis passes honest controls.

## What `value_action` Actually Did Well

The value-action project did not assume SAEs were useful because they were
available. It built an RQ0 gate around the question:

> Are sparse features a better representational basis than raw activations,
> PCA, random projections, or matched-size raw-coordinate controls?

The early result was negative for several off-the-shelf SAEs:

- Llama-3.1-8B-Instruct SAELens, Goodfire, and Llama-Scope SAEs all lost badly
  to raw residual probes on Schwartz value classification.
- Qwen2.5-7B-Instruct SAELens SAEs showed the same pattern.
- `sae_recon ~= sae_full`, so the bottleneck was not probe optimization. The
  sparse transform itself dropped task-relevant information.
- Matched-N MI controls were decisive: a matched number of raw residual
  dimensions beat the SAE active-feature set in every tested row.
- The project then moved to better public sparse bases, especially Keishii
  Llama transcoders, and still required cross-dataset, label-confound, and
  feature-explanation validation before trusting feature stories.

Operationally, `value_action` had four strong habits:

1. Test multiple basis ecosystems before deep feature work.
2. Use matched-N raw/PCA controls, not only full-dimensional baselines.
3. Separate feature discovery pools: metadata-predictive, high-activation,
   natural-domain, and model-decision features.
4. Treat generated explanations as insufficient unless they predict held-out
   natural high-activation records and survive confound checks.

## Direct Translation To Model Merging

For model merging, the analogous gate is:

> Can a sparse/transcoder basis reproduce, predict, or causally explain merge
> behavior better than raw MLP activations, PCA, top coordinates, and module
> patching?

This means we should not go straight from "SAE feature looks safety-related" to
"this explains the merge." A sparse basis is useful only if it passes at least
one of these tests:

- It reconstructs the relevant activation space and preserves behavior under
  activation replacement.
- It recovers the causal patch effect with fewer degrees of freedom than raw
  coordinates or PCA.
- It separates prompt families, token positions, or failure modes better than
  raw/PCA baselines.
- Its discovered features survive benign controls and prompt-template confounds.

## Current Qwen Case Under This Lens

The current active case is:

- donor/base: `Qwen/Qwen2.5-1.5B-Instruct`
- recipient: `nbeerbower/EVA-abliterated-TIES-Qwen2.5-1.5B`
- target: refusal/safety behavior lost by the abliterated merge
- causal upper bound: `PCA64 + full donor MLP 16-23`
- strongest broad coordinate baseline: residual `topk1344` over a 1536-dim MLP
  output space

The first vanilla residual SAE smoke test failed the value-action gate:

- train EV was high (`0.993-0.996`);
- mean L0 ranged from about `146` to `779`;
- all SAE variants repaired only one of the three hard harmful prompts;
- tracking-script and permission-slip still failed;
- broad raw-coordinate `topk1344` and full donor `16-23` passed the v0
  benchmark.

Interpretation:

> High reconstruction EV is not causal completeness. The current vanilla
> residual SAE is not yet a trustworthy feature basis for this merge.

## What We Should Do Next

The next experiments should copy the value-action discipline more directly.

### Track A: Fix The Distribution Before Declaring SAE Failure

The failed SAE was trained on teacher-forced residual rows. The successful
intervention is dynamic generation over specific token positions. This is the
same kind of mismatch that hurt value-action last-token SAE results.

Immediate test:

- collect residual rows from generated-token traces under the successful
  `PCA64 + full 16-23` patch;
- train the same small residual SAEs on those generated-token rows;
- evaluate on the frozen v0 benchmark;
- compare against `topk1344` and full `16-23`.

Pass condition:

- generated-trace SAE repairs tracking and permission-slip, or at least reveals
  a prompt-family split that vanilla teacher-forced SAE missed.

Fail condition:

- generated-trace SAE still repairs only one-time-code while broad coordinates
  pass. Then plain residual SAE is low priority.

### Track B: Search For A Better Model-Pair Plus SAE Ecosystem

The value-action project spent real effort finding a model/basis combination
with enough signal. We should do the same for model merging instead of forcing
Qwen1.5B if its sparse ecosystem is weak.

Candidate requirements:

- same architecture/base lineage so merging is meaningful;
- a public merge/fine-tune pair with a clear behavioral delta;
- public SAEs/transcoders or a feasible local training path;
- behavior that survives generation audit, not only loss;
- a model small enough for repeated activation patching.

Priority ecosystems:

- Gemma/GemmaScope if compatible merge pairs can be found;
- Llama-3.1 with Keishii or Llama-Scope transcoders if compatible public merges
  have a clean behavioral delta;
- smaller Qwen only if public sparse bases or local generated-trace bases pass
  the gate.

### Track C: Use Feature Discovery Pools Only After A Basis Passes

Once a basis passes behavioral completeness, use the ABCD-style discovery split:

- Pool A: prompt/family label probes, e.g. one-time-code vs tracking vs forgery.
- Pool B: high-activation generated-token features during patched refusal.
- Pool C: broader safety/misuse-domain high-activation features.
- Pool D: model-decision or next-token refusal-success probes.

Do not treat Pool A metadata features as mechanisms until Pool D and causal
patching support them.

## Updated Decision Rule

For this project, "ready for feature interpretation" means:

1. the model pair has a real, audited behavioral gap;
2. module or activation patching causally closes the gap;
3. simple baselines are recorded: PCA, top coordinates, random, raw/full patch;
4. a sparse/transcoder basis beats, complements, or mechanistically explains
   those baselines;
5. feature candidates survive benign controls and prompt-family confounds.

The Qwen pair currently passes gates 1-3. It has not passed gate 4.

Therefore the immediate next step is not feature naming. It is a generated-trace
SAE/transcoder basis-validation attempt, plus a parallel candidate search for a
model pair with a stronger public sparse-basis ecosystem.
