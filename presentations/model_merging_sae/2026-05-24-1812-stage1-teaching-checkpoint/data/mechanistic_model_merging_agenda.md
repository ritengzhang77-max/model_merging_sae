# Mechanistic Model Merging Agenda

Created: 2026-05-07

Citation-count convention: counts below use OpenAlex exact arXiv DOI lookups
on 2026-05-07 unless otherwise noted. OpenAlex undercounts some papers relative
to Semantic Scholar / Google Scholar, so treat counts as approximate influence
markers, not final bibliometric facts.

## Framing

The project should not start by trying to invent another merge recipe. There are
already many recipes. The stronger research direction is:

> Explain and predict why a merge preserves, destroys, or creates capabilities,
> using causal/mechanistic tools, then use those explanations to build a small
> mechanism-aware merge.

This mirrors the `value_action` project:

1. test the representational basis first;
2. identify features/components;
3. compare overlap and interference;
4. intervene causally;
5. use the intervention to improve behavior.

## Stage 0 — Literature/Benchmark Reproduction

### RQ0.0 — Can we reproduce basic merge phenomena on small models?

Question:

- Given one base model and several fine-tuned experts, can we reproduce the
  known pattern: simple averaging sometimes works, task arithmetic sometimes
  works, TIES/DARE-like methods reduce interference, and bad task pairs fail?

Existing work:

- Wortsman et al., "Model soups: averaging weights of multiple fine-tuned models
  improves accuracy without increasing inference time," ICML 2022. OpenAlex:
  205 citations.
- Ilharco et al., "Editing Models with Task Arithmetic," ICLR 2023. OpenAlex:
  31 citations.
- Yadav et al., "TIES-Merging: Resolving Interference When Merging Models,"
  NeurIPS 2023. OpenAlex: 22 citations.
- Yu et al., "Language Models are Super Mario: Absorbing Abilities from
  Homologous Models as a Free Lunch," ICML 2024. OpenAlex: 12 citations.
- Goddard et al., "Arcee's MergeKit: A Toolkit for Merging Large Language
  Models," arXiv preprint, 2024. OpenAlex: 3 citations.

What we do:

- Start with small models: Pythia, Qwen2.5-0.5B/1.5B, TinyLlama, or small ViT.
- Fine-tune 2-4 experts from the same base.
- Run linear merge, task arithmetic, TIES, DARE/DELLA, layer-wise coefficients.
- Establish a benchmark table before making any mechanistic claim.

Deliverable:

- A clean reproduction harness and "merge success/failure dataset."

## Stage 1 — Basis Validation

### RQ0 — What basis best explains/predicts merge success?

Question:

- What representation should we use to study merging mechanistically?

Candidate bases:

- raw parameter deltas;
- task vectors;
- sign-conflict masks;
- Fisher/gradient directions;
- per-layer delta matrices;
- SVD/task singular vectors;
- activations;
- CKA/representation similarity;
- SAE features;
- transcoder features;
- module-wise causal patch scores;
- weight-patching scores.

Existing work:

- Matena and Raffel, "Merging Models with Fisher-Weighted Averaging," NeurIPS
  2022. Citation count not verified in this note.
- Ilharco et al., "Editing Models with Task Arithmetic," ICLR 2023. OpenAlex:
  31 citations.
- Ortiz-Jimenez et al., "Task Arithmetic in the Tangent Space," arXiv preprint,
  2023. Citation count not verified in this note.
- Yang et al., "AdaMerging: Adaptive Model Merging for Multi-Task Learning,"
  ICLR 2024. OpenAlex: 5 citations.
- Yadav et al., "What Matters for Model Merging at Scale?" arXiv preprint, 2024.
  OpenAlex: 0 citations.
- "Demystifying Mergeability," arXiv preprint, 2026. OpenAlex not found at
  lookup time.

Mechanistic gap:

- These works use useful correlates: gradients, weights, coefficients, or
  activations. They generally do not answer whether a behavior is carried by
  the same circuit before and after merging.

What we do:

- Treat this exactly like `value_action` RQ0.
- For each basis, ask: how well does it predict merge retention/interference?
- Do not assume SAEs/transcoders are best. Require them to beat simpler
  baselines before using them as the primary explanatory unit.

Deliverable:

- A basis-comparison table and a justified choice of primary basis.

## Stage 2 — Parent-to-Merged Capability Tracing

### RQ1 — Does the merged model use the same internal components as the expert?

Question:

- If the math expert and merged model both solve GSM-style arithmetic, are they
  using the same internal features/modules/circuits?

Existing work:

- Wortsman et al. show merged checkpoints can improve or preserve behavior, but
  not whether mechanisms are shared.
- Task Arithmetic shows deltas steer behavior, but not the circuit-level path.
- Weight Patching, arXiv preprint, 2026, is closest to causal localization of
  source-model weights. OpenAlex not found at lookup time.

Mechanistic work we can add:

- Activation patching: expert activations -> merged model.
- Weight patching: expert modules/blocks -> base or merge.
- Feature ablation: remove candidate features in expert and merge.
- Compare causal effect maps across base, expert, and merge.

Possible labels:

- inherited component: causal in expert and merge;
- lost component: causal in expert but not merge;
- suppressed component: present in merge but not used;
- emergent component: causal in merge but not expert.

### RQ2 — Which layers/modules carry transferred capability?

Question:

- Are merges mostly preserving early lexical/domain features, middle algorithms,
  or late output/style/refusal routing?

Existing work:

- AdaMerging learns task/layer-wise coefficients but does not causally explain
  why a layer matters.
- Activation-informed methods use layerwise activation information, but are not
  full causal circuit analyses.

Mechanistic work we can add:

- Layer-wise merge ablation.
- Block-wise weight patching.
- MLP vs attention patching.
- Per-layer activation overlap and causal contribution.

Deliverable:

- Capability-by-layer maps for each expert and merge.

## Stage 3 — Interference

### RQ3 — What causes a merge to destroy a capability?

Question:

- When math+code merge loses math, what happened internally?

Existing work:

- Yadav et al., "TIES-Merging," NeurIPS 2023, identifies sign conflicts and
  redundant parameter changes as sources of interference. OpenAlex: 22
  citations.
- Yu et al., "Language Models are Super Mario," ICML 2024, argues many SFT
  delta parameters are redundant and can be dropped/rescaled. OpenAlex: 12
  citations.
- Deep et al., "DELLA-Merging," arXiv preprint, 2024, uses magnitude-based
  random dropping. OpenAlex: 0 citations.
- Davari and Belilovsky, "Model Breadcrumbs," arXiv preprint, 2023. Citation
  count not verified in this note.

Mechanistic gap:

- Sign conflict is a coordinate-level explanation, not a behavioral circuit
  explanation.

Mechanistic work we can add:

- Identify whether failed merges:
  - erase expert features;
  - suppress expert features;
  - reroute late logits;
  - introduce competing features from another expert;
  - shift activations out of the expert manifold;
  - preserve the feature but damage downstream readout.

### RQ4 — Can causal overlap predict retention better than weight statistics?

Question:

- Is parent-merge circuit overlap a stronger predictor than sign conflict,
  cosine similarity, or gradient alignment?

Existing work:

- "Demystifying Mergeability," arXiv preprint, 2026, proposes interpretable
  mergeability predictors such as gradient alignment/subspace metrics.
- "What Matters for Model Merging at Scale?" arXiv preprint, 2024, analyzes
  scaling and practical factors. OpenAlex: 0 citations.

Mechanistic work we can add:

- Compare predictors:
  - task-vector cosine;
  - sign-conflict rate;
  - gradient alignment;
  - SVD subspace overlap;
  - activation similarity;
  - causal patch overlap;
  - feature/circuit overlap.
- Evaluate on held-out task pairs and unseen merge methods.

Deliverable:

- A predictive model of merge success with ablation of predictor families.

## Stage 4 — Representation Shift

### RQ5 — Does merging create representation bias?

Question:

- Does the merged model internally represent inputs more like the base, the
  expert, a mixture, or something new?

Existing work:

- Yang et al., "Representation Surgery for Multi-Task Model Merging," arXiv
  preprint, 2024. OpenAlex: 1 citation.
- Nobari et al., "Activation-Informed Merging of Large Language Models," arXiv
  preprint, 2025. OpenAlex: 0 citations.
- Yao et al., "Activation-Guided Consensus Merging for Large Language Models,"
  arXiv preprint, 2025. OpenAlex: 0 citations.

Mechanistic gap:

- Activation-informed work uses activations to improve merging, but usually does
  not localize causal components or compare algorithms/circuits.

Mechanistic work we can add:

- CKA/cosine/linear probe similarity across base, experts, and merge.
- Activation patching to test whether representation drift is causal.
- Test whether correcting representation drift restores behavior.

### RQ6 — Does the merged model preserve features or only outputs?

Question:

- Does benchmark performance hide a mechanistic change?

Existing work:

- Most merge papers evaluate outputs/benchmarks.

Mechanistic work we can add:

- Same-output / different-mechanism tests.
- Counterfactual prompts.
- Causal interventions on expert-specific features.
- Compare expert and merge under activation patching.

Deliverable:

- A "mechanistic fidelity" score: behavior retained plus circuit retained.

## Stage 5 — Sparse / Low-Rank Structure

### RQ7 — Are useful task deltas sparse or low-rank in a meaningful basis?

Question:

- DARE/DELLA/Breadcrumbs say many deltas are removable. Are the surviving deltas
  actually the causally important ones?

Existing work:

- DARE, DELLA, Breadcrumbs, TIES.
- Task Singular Vectors, arXiv preprint, 2024. OpenAlex exact lookup failed to
  identify a reliable record in this note.
- Superpose Task-specific Features for Model Merging, arXiv preprint, 2025.
  OpenAlex exact lookup was noisy; citation count not trusted in this note.

Mechanistic work we can add:

- Compare kept/pruned delta coordinates against:
  - causal weight-patching scores;
  - activation-change scores;
  - gradient/Fisher importance;
  - SAE/transcoder feature effects;
  - SVD singular directions.
- Ask whether pruning works because it removes noise, preserves sparse causal
  features, or averages out redundant small updates.

### RQ8 — Do task singular vectors correspond to interpretable capabilities?

Question:

- If task deltas are low-rank, do singular directions map to behaviors,
  circuits, or features?

Mechanistic work we can add:

- Project deltas onto top singular directions.
- Patch/ablate singular components.
- Decode associated activation changes.
- Compare singular directions to feature/circuit changes.

Deliverable:

- Interpretable low-rank maps of task deltas.

## Stage 6 — Safety / Alignment As A High-Stakes Case

### RQ9 — What happens to safety circuits under merging?

Question:

- Does merging preserve safety/refusal behavior, erase it, bypass it, or create
  hidden unsafe behavior?

Existing work:

- Model Merging and Safety Alignment, arXiv preprint, 2024. Local PDF:
  `papers/03_llm_applications/2406.14563_model_merging_safety_alignment.pdf`.
- SafeMERGE, arXiv preprint, 2025. Local PDF:
  `papers/03_llm_applications/2503.17239_safemerge.pdf`.

Mechanistic work we can add:

- Find safety/refusal components in aligned and unsafe experts.
- Track those components after merging.
- Test whether unsafe capability survives under the surface.
- Use causal patching/ablation to separate helpful domain skill from unsafe
  behavior.

### RQ10 — Can mechanism-aware merging preserve safety and capability better?

Question:

- Can we merge only the useful domain components while preserving or restoring
  safety components?

Mechanistic work we can add:

- Safety-preserving module masks.
- Feature-level or circuit-level merge constraints.
- Weight patches selected by safety/capability causal scores.

Deliverable:

- A small safety-aware merge experiment with causal explanation.

## Stage 7 — Emergence / Composition

### RQ11 — Are "emergent" merged abilities real circuit composition?

Question:

- Sometimes a merged model appears better than every parent. Is that because it
  composes parent mechanisms, lands in a flatter/better region, or exploits eval
  artifacts?

Existing work:

- Model Soups and DARE report cases where merged models can beat individual
  sources.
- Evolutionary merge recipes search for high-performing mixtures.

Mechanistic work we can add:

- Compare parent vs merge circuits on tasks where merge beats all parents.
- Look for component reuse from multiple parents.
- Test with counterfactual tasks and held-out datasets.
- Separate real mechanism composition from benchmark/prompt artifacts.

### RQ12 — Can we design a mechanistically guided merge recipe?

Question:

- After diagnosing success/failure, can we build a merge method that uses causal
  information rather than only weight statistics?

Existing work:

- AdaMerging learns coefficients.
- AIM/ACM use activations.
- TIES/DARE/DELLA use delta statistics.
- Weight Patching uses causal localization signals.

Mechanistic contribution:

- Merge selection based on causal component scores:
  - keep modules/features that are causal for target ability;
  - avoid components causal for interference;
  - preserve safety/refusal components;
  - choose layer coefficients from causal retention curves.

Evaluation:

- Compare against linear, task arithmetic, TIES, DARE/DELLA, AdaMerging,
  activation-informed methods.

Deliverable:

- A small but defensible mechanism-aware merge method.

## Recommended First Three Experiments

### Experiment A — Small Reproduction Harness

Use one small base model and three experts:

- math;
- code or synthetic algorithm;
- chat/safety/style.

Run:

- linear average;
- task arithmetic;
- TIES;
- DARE;
- layer-wise merge.

Output:

- a table of retained/lost capabilities;
- saved base/expert/merge checkpoints;
- activation cache.

### Experiment B — RQ0 Basis Comparison

For each merge:

- task-vector cosine/sign conflict;
- gradient alignment if data available;
- SVD subspace overlap;
- activation similarity;
- linear probes;
- optional SAE/transcoder features;
- simple causal weight-patching scores.

Target:

- predict retention/interference before doing deep circuit work.

### Experiment C — One Causal Case Study

Pick one clear success and one clear failure.

For each:

- activation patch expert -> merge;
- module-wise weight patch;
- feature/probe ablation if basis is good;
- locate where behavior survives or fails.

Target:

- produce one mechanistic story that is causally tested.

## Thesis-Level Claim We Should Aim For

Not:

> We invented another merge recipe and got +1 point on a benchmark.

Better:

> Merge success is predictable from component-level mechanism overlap, and merge
> failure often corresponds to specific lost/suppressed/interfering internal
> components. Causal localization can guide safer and more reliable model
> merging.

