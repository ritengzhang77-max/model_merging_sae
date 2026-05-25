# Mechanistic Model Merging Project Plan

Created: 2026-05-07

Expanded SAE/transcoder proposal: [MECHANISTIC_MODEL_MERGING_SAE_PROPOSAL.md](MECHANISTIC_MODEL_MERGING_SAE_PROPOSAL.md)

This project asks why model merging works or fails at the level of internal
mechanisms. The goal is not just to invent another merge recipe, but to explain
capability retention, interference, and emergence well enough to predict them
and eventually guide safer merges.

## Working Thesis

Model merging succeeds when the merged checkpoint preserves and correctly routes
the parent experts' causal components. It fails when those components are erased,
suppressed, overwritten, or made incompatible by another expert. Existing merge
methods mostly use weight-space or activation-space heuristics; we want a causal
mechanistic account.

## Stage 0: Reproduction And Dataset Construction

Purpose:

- Build a merge harness.
- Reproduce easy success/failure cases.
- Create a local dataset of base/expert/merged checkpoints and metrics.
- Avoid large LLM downloads until the harness is proven.

Questions:

- Can we produce clear merge success and failure cases?
- Which recipe works best in simple settings: linear average, task arithmetic,
  TIES, DARE, DELLA-like pruning, or layer-wise merging?
- What metadata should every merge experiment save for later mechanistic work?

Output:

- `stage0/results/*.csv`
- `stage0/artifacts/` with small checkpoints
- a model/dataset registry for larger follow-up targets

## Stage 1: RQ0 Basis Validation

RQ0. What basis best explains and predicts merge success?

Candidate bases:

- raw task-vector coordinates;
- sign conflicts;
- gradient alignment;
- Fisher/Hessian-weighted directions;
- per-layer delta matrices;
- SVD/task singular vectors;
- activation similarity;
- linear probes;
- SAE features;
- transcoder features;
- module-wise causal patch scores;
- weight-patching scores.

Value-action lesson:

- Do not assume sparse features are the right basis. Test them against simpler
  baselines first.

## Stage 2: Capability Inheritance

RQ1. When a merged model keeps an expert capability, does it use the same
internal mechanism as the expert?

RQ2. Which layers/modules carry transferred capability?

Planned tools:

- activation patching;
- block-wise weight patching;
- MLP vs attention patching;
- causal feature ablation if the basis supports it.

## Stage 3: Merge Interference

RQ3. When a merge fails, what happened internally?

Candidate failure modes:

- expert feature erased;
- expert feature present but suppressed;
- expert feature present but disconnected from downstream readout;
- another expert introduces an interfering component;
- representation drift moves prompts away from the expert manifold;
- late output/style/safety routing overrides the task circuit.

RQ4. Can causal mechanism overlap predict retention better than weight-space
statistics?

## Stage 4: Representation Drift And Mechanistic Fidelity

RQ5. Does merging create representation bias?

RQ6. Does the merged model preserve mechanisms or only preserve benchmark
outputs?

Metrics:

- behavior retained;
- activation similarity retained;
- causal patch maps retained;
- feature/circuit overlap retained.

## Stage 5: Sparse And Low-Rank Delta Structure

RQ7. Are useful task deltas sparse or low-rank in a meaningful basis?

RQ8. Do task singular vectors correspond to interpretable capabilities?

This connects DARE/TIES/DELLA/Breadcrumbs/TSV-style work to causal
interpretability. The key question is whether kept coordinates or singular
directions are actually causally important.

## Stage 6: Safety And Alignment

RQ9. What happens to safety/refusal circuits under merging?

RQ10. Can mechanism-aware merging preserve safety and capability together?

This is a high-impact case because practical open-source merges can accidentally
weaken safety or transfer unsafe behavior.

## Stage 7: Emergence And Mechanism-Aware Merging

RQ11. When a merged model beats all parents, is that real circuit composition,
better calibration, flatter averaging, or benchmark artifact?

RQ12. Can we design a mechanism-aware merge recipe?

Possible method:

- keep components with positive causal scores for target abilities;
- avoid components with high interference scores;
- preserve safety/refusal components;
- set layer coefficients from causal retention curves.

## Immediate Stage 0 Implementation

Start with a cheap vision/domain-shift setting:

- base: small CNN on MNIST;
- experts: clean, rotated, noisy, inverted, and a deliberately conflicting
  label-permutation expert;
- merges: linear/task arithmetic, TIES, DARE, DELLA-like, layer-wise average;
- metrics: clean/rotated/noisy/inverted accuracy, average accuracy, worst-task
  accuracy, parameter conflict stats.

Then add a small language stage:

- base: `HuggingFaceTB/SmolLM2-135M` or `Qwen/Qwen2.5-0.5B`;
- experts: small LoRA/full fine-tunes on synthetic formatting, arithmetic,
  sentiment/style, and refusal/safety-like tasks;
- merge: LoRA/task-vector variants first, full checkpoints only if needed.

## Paper Reference Convention

When discussing papers, include authors, title, venue/arXiv status, year, and
citation count with source/date when it matters. Citation counts change; they
must be re-checked for formal writing.
