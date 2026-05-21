# Model Merging Research Brief

Created: 2026-05-07

This note is the working map for turning the `value_action` style research
program into a model-merging mechanistic-interpretability project.

## Local Paper Cache

Downloaded papers are organized under:

- `papers/00_surveys/`
- `papers/01_foundations/`
- `papers/02_core_methods/`
- `papers/03_llm_applications/`
- `papers/04_interpretability_theory/`

The manifest is `notes/papers_to_download.tsv`.

## Short Answer

Yes: the `value_action` research pattern is a good fit for model merging.
The analogy is strong:

- `value_action` asked whether abstract values and concrete actions share
  internal features.
- A model-merging project can ask whether parent-model capabilities and
  merged-model capabilities share internal circuits/features/directions.
- `value_action` started with RQ0 because SAEs might be the wrong basis.
- Model merging also needs an RQ0: parameter deltas, raw activations, SAE
  features, transcoders, Fisher/Hessian directions, SVD subspaces, and
  gradient directions may each be better or worse bases for explaining merge
  success.

The research gap is real. Many merging papers report that merging works,
and some explain it with loss geometry, task-vector arithmetic, gradient
alignment, subspace overlap, sign conflicts, or representation bias. Very few
give a causal mechanistic account of *which internal components carry each
capability, how those components combine, and why a merge preserves, destroys,
or creates behavior*.

## What Model Merging Is

Model merging means combining two or more trained neural networks into a
single model by manipulating parameters, task vectors, layers, or small
modules, usually without full retraining.

Common setup:

```text
base model:        theta_0
expert i:          theta_i = fine_tune(theta_0, task_i)
task vector i:     tau_i = theta_i - theta_0
merged model:      theta_merge = theta_0 + Merge(tau_1, ..., tau_k)
```

The simplest merge is weight averaging:

```text
theta_merge = sum_i alpha_i * theta_i
```

If all experts share the same base, this is equivalent to adding a weighted
sum of task vectors to the base:

```text
theta_merge = theta_0 + sum_i alpha_i * (theta_i - theta_0)
```

Why people care:

- It can combine capabilities without running an ensemble.
- It can avoid raw-data access.
- It can be much cheaper than multi-task retraining.
- It supports open-source/community model composition.
- It can preserve or restore capabilities after fine-tuning.

## Historical Line

1. **Parameter averaging in distributed/federated learning.**
   FedAvg averages local updates from clients and showed that model averaging
   can work even under unbalanced/non-IID data.

2. **Checkpoint averaging and SWA.**
   SWA and NMT checkpoint averaging showed that averaging nearby checkpoints
   can land in flatter, better-generalizing regions.

3. **Mode connectivity and loss landscapes.**
   Work on mode connectivity found low-loss paths between optima. This gives
   one explanation for why nearby or aligned models can be averaged without
   catastrophic loss.

4. **Permutation symmetry and re-basinning.**
   Independently trained networks may implement similar functions with
   permuted hidden units. Git Re-Basin aligns units before merging, showing
   that "same function" need not mean "same coordinate system."

5. **Model soups.**
   Averaging fine-tuned checkpoints from the same large pretrained model can
   improve accuracy/robustness with no inference cost.

6. **Task vectors and task arithmetic.**
   Fine-tune deltas behave like editable weight-space directions. Adding,
   subtracting, or scaling them can add/remove capabilities.

7. **Interference-aware merging.**
   TIES, DARE, DELLA, Breadcrumbs, and related methods try to remove noisy,
   redundant, sign-conflicting, or low-importance parts of task vectors before
   addition.

8. **Adaptive, representation-aware, and activation-aware merging.**
   AdaMerging learns coefficients. Surgery/ProbSurgery correct representation
   bias. AIM and ACM use activations to decide what to preserve or how to set
   layer-wise coefficients.

9. **LLM community tooling and recipes.**
   MergeKit made LLM merging practical at scale, including linear merges,
   task arithmetic, TIES/DARE, SLERP, passthrough/frankenmerges, and tokenizer
   handling.

10. **Current theory/diagnostics.**
    Recent work studies mergeability via gradient alignment, subspace overlap,
    low-rank task matrices, Fisher/Hessian geometry, activation similarity,
    and representation bias.

## Main Method Families

### Weight Averaging / Soup

Directly average parameters of checkpoints. Best when models are close in the
same basin, share architecture, share initialization, and differ by small
fine-tuning choices.

Pros:

- Simple.
- No training.
- No inference overhead.

Failure mode:

- If models are not aligned in weight space, the average can land in a high-loss
  region or mix incompatible functions.

### Task Arithmetic

Compute task vectors:

```text
tau_i = theta_i - theta_0
```

Then add or subtract them:

```text
theta_merge = theta_0 + lambda * tau_math + lambda * tau_code
theta_unlearn = theta_model - lambda * tau_bad_behavior
```

Interpretation:

- A task vector is a coarse parameter-space direction that moves the base model
  toward a behavior.
- In theory, short fine-tune task vectors approximate gradients; adding them
  approximates a multi-task update if tasks do not interfere too much.

### Fisher / Hessian / Curvature-Aware Merging

Weight parameters according to estimated importance/uncertainty. Fisher merging
views each model's weights as a Gaussian posterior approximation; high-Fisher
parameters are trusted more.

Interpretation:

- Avoid changing parameters that each model is confident are important.
- Gives a geometry-aware alternative to plain averaging.

### TIES, DARE, DELLA, Breadcrumbs

These methods modify task vectors before combining them.

- **TIES**: trim small changes, elect a consensus sign, merge only sign-aligned
  values.
- **DARE**: randomly drop delta parameters and rescale survivors, exploiting
  delta redundancy.
- **DELLA**: magnitude-based dropping; lower-magnitude deltas get higher drop
  probabilities.
- **Breadcrumbs**: keep a sparse mask of useful non-outlier/non-negligible
  task-vector coordinates.

Interpretation:

- Fine-tuning deltas contain signal plus noise/redundancy.
- Merging fails when conflicting deltas destructively interfere.
- Sparsifying and resolving signs can reduce interference.

### SVD / Low-Rank / Task-Matrix Methods

Instead of flattening all parameters, analyze layer matrices:

```text
Delta W_i = W_i - W_0
```

Task Singular Vectors, TSV-style methods, and feature-superposition methods
argue that task deltas are structured and often low-rank. Singular directions
can reveal task-specific subspaces and interference.

Interpretation:

- Capabilities may be carried by low-rank update subspaces.
- Interference can be measured by singular-vector overlap or conflict.

### Adaptive Coefficient Methods

AdaMerging and relatives learn task-wise or layer-wise merge weights, often
without original training labels, using unlabeled calibration or entropy-like
objectives.

Interpretation:

- The right merge coefficient is not globally uniform.
- Different layers/modules carry different amounts of each capability.

### Activation/Representation-Aware Methods

Representation Surgery, AIM, ACM, and related work use activation distributions
to diagnose or guide merging.

Key ideas:

- Merged models can suffer **representation bias**: their internal activations
  differ from expert-model activations even when the weights look close.
- Activation-informed methods preserve or weight components based on internal
  response patterns.
- Layer-wise activation similarity/mutual information can choose where to merge
  more or less aggressively.

## Where People Use Model Merging

### Multi-Task Models

Merge experts trained on separate tasks into one model that handles all tasks.
This is the most standard academic setting.

### LLM Capability Composition

Open-source LLMs are merged to combine instruction following, coding, math,
reasoning, multilingual ability, domain knowledge, or long-context behavior.

### Domain Specialization

Merge a general model with a biomedical/legal/finance/materials expert to gain
domain skill while preserving general language ability.

### Multilingual Transfer

Chat Vector transfers instruction-following/alignment behavior into models
continued-pretrained in other languages.

### Safety Alignment

Merging can accidentally transfer unsafe behavior, but it can also be used to
restore safety by merging with safety-aligned anchors or using safety-aware
merging objectives.

### Efficient Reasoning

Long-to-short reasoning work uses merging to combine long-CoT reasoning models
with shorter/faster response styles.

### Federated / Decentralized / Collaborative AI

The broad idea is older than LLM merging: local models or community checkpoints
are combined without centralizing all data.

### Vision, Multimodal, Diffusion, Robotics

Merging is also used in vision classifiers, vision-language models, diffusion
models, and other architectures, though LLM merging currently dominates the
open-source discussion.

## What People Currently Think Explains It

### 1. Shared Initialization Creates a Common Coordinate System

If all experts start from the same base model, the same parameter index tends
to correspond to roughly the same function. This makes arithmetic meaningful.
If two models were independently trained, hidden-unit permutations and other
symmetries break this coordinate system.

### 2. Fine-Tuning Deltas Are Small and Redundant

DARE reports that many SFT delta parameters can be dropped with little loss,
especially in large models. This suggests fine-tuning often changes routing or
capability expression with small redundant updates rather than rewriting the
whole model.

### 3. Loss Landscapes Are Locally Flat/Connected

If fine-tuned checkpoints live in a connected low-loss region around a base,
averaging them does not leave the good basin. Model soups and SWA rely on this
idea.

### 4. Task Vectors Approximate Gradients

For short fine-tuning, the task vector is close to a scaled negative gradient.
Adding task vectors is then like approximating a multi-task gradient step. This
helps when gradients are compatible and hurts when they conflict.

### 5. Interference Is Often Sign/Subspace/Gradient Conflict

TIES focuses on sign conflict. Demystifying Mergeability argues that gradient
alignment is a core signal of compatibility. TSV-style work studies low-rank
subspace interactions. These all point to the same broad mechanism: merges
fail when the same parameter/subspace needs incompatible changes for different
tasks.

### 6. Merged Models Can Have Representation Bias

Even if output metrics are okay, merged model activations can drift away from
expert activations. Surgery-style work shows that correcting representation
distributions improves multi-task performance.

### 7. Larger / Stronger Base Models Merge More Easily

At scale, experts from stronger base models merge better. One plausible
interpretation is that strong bases already contain many latent capabilities;
fine-tuning mostly activates, routes, or emphasizes them, so deltas are easier
to combine.

## What Is Still Not Mechanistically Understood

The field has partial explanations, but not a full causal circuit story.

Open gaps:

- Which exact layers/modules carry the transferred capability?
- Are task vectors mostly changing feature detectors, feature routing, or
  output readouts?
- When a merge succeeds, did it preserve the expert's internal circuit or
  implement the behavior through a different circuit?
- When a merge fails, is the capability erased, inhibited, rerouted, or present
  but not used?
- Do "emergent" merged capabilities correspond to recombination of parent
  circuits or only benchmark artifacts?
- Which basis is best for explanation: raw weights, task vectors, SVD
  directions, gradients, Fisher directions, activations, SAE features,
  transcoders, or module-level causal patches?
- How does merge behavior vary by layer depth and module type?
- Can mechanistic diagnostics predict merge success before actually merging?

## Existing Interpretability-Adjacent Work

The closest work is not usually "mechanistic interpretability" in the Anthropic
circuit sense, but it is moving in that direction.

- **Deep Model Merging survey** explicitly frames merging as a sibling of
  interpretability via loss geometry and representations.
- **Representation Surgery** diagnoses representation bias and uses internal
  representations to correct merged models.
- **AIM / ACM** use activation information to guide merging.
- **Task Singular Vectors** and **Superpose Task-specific Features** treat
  task deltas as structured matrices/subspaces/features instead of flat
  parameter vectors.
- **Demystifying Mergeability** uses interpretable pairwise metrics such as
  gradient alignment and subspace overlap to predict merge success.
- **Weight Patching** is closest to a causal mechanistic method: patch weights
  from a specialized model into a base model to localize source-side carriers
  of a behavior, and use those scores for mechanism-aware merging.

This leaves a good research opening: combine weight-space merging with
activation/circuit methods to explain merge success and failure causally.

## Proposed Value-Action-Style Research Program

### RQ0 — What Basis Explains Mergeability?

Compare bases for predicting whether a merge preserves a capability:

- raw task-vector coordinates;
- per-layer task matrices;
- SVD singular directions;
- Fisher/Hessian-weighted directions;
- gradient directions;
- residual-stream activations;
- SAE features;
- transcoder features;
- module-level weight-patching scores.

Metrics:

- capability retention;
- interference/loss on other tasks;
- held-out generalization;
- activation similarity to expert;
- causal patch recovery.

Decision rule:

- If sparse features/transcoders do not beat simpler baselines, do not build
  the whole project on them. This mirrors the `value_action` lesson.

### RQ1 — Are Capabilities Represented as Shared Parent-Merge Features?

For each expert capability, identify internal features/circuits in:

- base model;
- expert model;
- merged model.

Ask whether the merged model uses the same features as the parent expert.

### RQ2 — Which Parent Components Survive the Merge?

Define component classes:

- **Inherited component**: active/causal in expert and merged model.
- **Lost component**: active/causal in expert but absent in merged model.
- **Suppressed component**: present in merged model but not routed into output.
- **Interfering component**: from another expert, degrades target behavior.
- **Emergent component**: causal in merged model but not in any individual
  expert under the same prompts.

### RQ3 — Does Feature/Circuit Overlap Predict Behavioral Retention?

For each capability and merge recipe:

- compute parent-merged activation overlap;
- compute causal patch overlap;
- compute task-vector/singular-subspace overlap;
- test whether these predict accuracy/retention.

### RQ4 — What Causes Merge Interference?

Compare failed vs successful merges.

Candidate causes:

- sign conflict;
- gradient anti-alignment;
- singular subspace collision;
- representation bias;
- expert components overwritten at specific layers;
- output head/logit readout conflict;
- safety/refusal circuits inhibiting task circuits.

### RQ5 — Can Mechanistic Patching Rescue Failed Merges?

Use:

- weight patching;
- activation patching;
- targeted task-vector masking;
- layer-wise coefficient changes;
- feature steering.

Goal: show that a predicted component can restore a lost capability with less
damage than global re-merging.

### RQ6 — Layer Depth: Where Does the Merge Happen?

For each task:

- early layers: lexical/domain features?
- middle layers: task abstractions/algorithms?
- late layers: answer format, refusal, style, output routing?

Run layer-wise merges and layer-wise patching to find where capability
transfer and interference concentrate.

### RQ7 — Are Merged Capabilities Actually the Same Algorithms?

Behavior can match while mechanisms differ.

Use:

- activation patching from expert to merged model;
- causal scrubbing/feature ablation;
- attention/MLP circuit comparison;
- prompt counterfactuals.

Ask whether the merged model reuses the parent algorithm or learns/expresses a
different shortcut.

### RQ8 — Do Sparse/Low-Rank Deltas Correspond to Sparse Functional Features?

DARE/DELLA/Breadcrumbs imply many deltas are redundant. Test whether kept
coordinates correspond to:

- high causal effect;
- high activation change;
- high Fisher/gradient importance;
- meaningful SAE/transcoder features;
- singular-vector task subspaces.

### RQ9 — Can We Predict Merge Success Before Merging?

Build a diagnostic model from pre-merge information:

- gradient alignment;
- activation similarity;
- subspace overlap;
- sign conflict rate;
- Fisher overlap;
- weight-patching localization overlap;
- SAE/transcoder feature overlap.

Evaluate on unseen task pairs and unseen merge methods.

### RQ10 — Safety and Alignment Circuits Under Merging

Safety is a natural high-impact case.

Questions:

- Does unsafe behavior transfer through the same components as useful domain
  skill?
- Can safety components be preserved while merging task expertise?
- Are refusal/safety circuits overwritten, bypassed, or suppressed?

### RQ11 — Emergence vs Composition

When a merged model beats all parents on a benchmark:

- Is it composing parent circuits?
- Is it averaging toward a better calibrated/less overfit solution?
- Is it exploiting benchmark artifacts?
- Does it form new internal features or just rebalance existing ones?

## Practical Initial Project

Start small and rigorous.

### Candidate Models

Use small open-weight models first:

- `pythia-410m` / `pythia-1b`;
- `Qwen2.5-0.5B` / `Qwen2.5-1.5B`;
- `TinyLlama-1.1B`;
- small ViT/CLIP models if vision benchmarks are easier.

### Candidate Tasks

Pick tasks with clean behavioral and mechanistic probes:

- arithmetic vs sentiment vs translation style;
- code vs math vs chat style;
- refusal/safety vs helpfulness;
- factual domain QA vs general QA;
- synthetic tasks if clean circuits matter more than LLM realism.

### Phase 0

1. Fine-tune two or three experts from the same base.
2. Merge with weight average, task arithmetic, TIES, DARE.
3. Evaluate held-in and held-out tasks.
4. Collect activations for base, experts, merged models.
5. Run basic diagnostics:
   - task-vector cosine;
   - sign conflict;
   - gradient alignment;
   - activation CKA/cosine;
   - layer-wise logit lens/performance.

### Phase 1

Run RQ0 basis comparison:

- Can simple task-vector statistics predict retention?
- Do activation metrics beat weight metrics?
- Do SVD/Fisher metrics beat raw deltas?
- Do SAE/transcoder features add anything?

### Phase 2

Do causal localization:

- layer-wise weight patching;
- module-wise weight patching;
- activation patching;
- feature ablation/steering if the basis is good.

### Phase 3

Design a mechanism-aware merge:

- merge only causally localized modules;
- layer-wise coefficients based on causal/activation scores;
- avoid components predicted to interfere;
- compare against TIES/DARE/AIM/ACM.

## Risks

- SAEs may again be the wrong basis, especially for weight-space questions.
- Full LLM fine-tuning and activation collection can be expensive.
- Public fine-tunes may differ in tokenizer/chat template/data, producing
  confounds.
- Some "merged capability" gains may be benchmark variance or prompt-template
  effects.
- Mechanistic claims need causal intervention, not only probes/correlations.

## Best Thesis Shape

The strongest project is not "make a new merge algorithm" first.

Better thesis:

> Build a mechanistic diagnostic framework for model merging that predicts and
> explains capability retention, interference, and emergence across merge
> recipes, then use the diagnostics to design a small mechanism-aware merge.

That is directly analogous to `value_action`:

- first validate the basis;
- then identify features/components;
- then test overlap/interference;
- then intervene causally;
- then use the intervention to improve behavior.

