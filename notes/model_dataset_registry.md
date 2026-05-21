# Model And Dataset Registry

Created: 2026-05-07

Validation files:

- `notes/hf_candidate_validation_2026-05-07.tsv`
- `notes/hf_expert_candidate_validation_2026-05-07.tsv`

Hugging Face download/like counts are time-varying. The counts in the TSVs were
queried on 2026-05-07 and should be re-checked before formal writing.

## Stage 0A: Local Vision Domain Merge

Status: running now.

Base/task:

- MNIST from `torchvision`, downloaded to `/data/gavin/model_merging/data/MNIST`.
- Small CNN trained on clean MNIST.
- Experts fine-tuned from the same base:
  - clean;
  - rotate25;
  - noise;
  - invert;
  - labelperm negative-control expert.

Why this is useful:

- Same architecture and same output labels.
- Domain experts specialize in different input transformations.
- The label-permutation expert creates a controlled conflict.
- Fast enough to run repeatedly while developing merge and patching code.

Limitations:

- Not an LLM.
- Small CNN mechanisms are much simpler than transformer mechanisms.
- It validates harness behavior, not the final scientific claim.

## Stage 0B: Small Language Models We Can Fine-Tune Ourselves

Recommended first language base:

- `HuggingFaceTB/SmolLM2-135M`
  - validated: public, ungated, text-generation.
  - paired instruct checkpoint: `HuggingFaceTB/SmolLM2-135M-Instruct`.
  - reason: small enough for full fine-tuning or cheap LoRA; Llama-like enough
    to make later transformer interpretation natural.

Second choice:

- `Qwen/Qwen2.5-0.5B`
  - validated: public, ungated.
  - paired instruct checkpoint: `Qwen/Qwen2.5-0.5B-Instruct`.
  - reason: stronger than 135M models, still feasible on 24GB GPUs with LoRA.

Other feasible bases:

- `HuggingFaceTB/SmolLM2-360M`
- `Qwen/Qwen2.5-1.5B`
- `EleutherAI/pythia-410m`
- `EleutherAI/pythia-1b`
- `openai-community/gpt2`
- `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`

Suggested expert tasks:

- instruction formatting / chat behavior;
- arithmetic templates;
- sentiment/style;
- short code generation;
- refusal/safety-like synthetic prompts.

Preferred training form:

- Start with LoRA experts for speed.
- If LoRA merging is too adapter-specific for the mechanistic question, repeat
  with full fine-tunes on the smallest base.

## Stage 0C: Text Datasets

Validated candidates:

- `stanfordnlp/imdb`: sentiment classification.
- `SetFit/sst2`: small sentiment classification.
- `fancyzhx/ag_news`: topic classification.
- `openai/gsm8k`: grade-school math.
- `google-research-datasets/mbpp`: Python code generation.
- `PKU-Alignment/PKU-SafeRLHF`: safety/helpfulness preference data.
- `tatsu-lab/alpaca`: instruction-tuning data.

Use:

- For generative small LMs, prefer synthetic/task-formatted subsets first so
  labels and evaluation are clean.
- Then use real datasets as robustness checks.

## Stage 0D: Vision Models Beyond MNIST

Validated candidates:

- `google/vit-base-patch16-224-in21k`
- `facebook/deit-tiny-patch16-224`

Validated datasets:

- `ylecun/mnist`
- `zalando-datasets/fashion_mnist`
- `uoft-cs/cifar10`

Use:

- Fine-tune the same base on domain-shifted variants of one label space.
- Avoid merging classifiers with incompatible label heads unless using a
  multi-head setup or CLIP-style zero-shot heads.

## Stage 0E: Large Public LLM Merge Families

These are useful for literature alignment, but should not be downloaded until
we have a specific experiment. They are too large for casual collection.

Validated:

- `mistralai/Mistral-7B-v0.1`
- `mistralai/Mistral-7B-Instruct-v0.2`
- `WizardLMTeam/WizardLM-13B-V1.2`
- `layoric/llama-2-13b-code-alpaca` (PEFT/code expert candidate)
- `meta-llama/Llama-2-13b-hf` (gated/manual access)

Search note:

- `WizardLMTeam/WizardMath-13B-V1.0` was not found under that exact id on
  2026-05-07. Quantized/community variants exist, e.g. TheBloke and
  vanillaOVO variants. We should avoid relying on that exact id until verified.

## Recommended Next Downloads

Do not download large public experts yet.

Next small download, after MNIST Stage 0A:

1. `HuggingFaceTB/SmolLM2-135M`
2. `HuggingFaceTB/SmolLM2-135M-Instruct`
3. small subsets of `gsm8k`, `mbpp`, `sst2`, and synthetic instruction data

Reason:

- This gives us a transformer setting with a known base/instruct pair and room
  to create our own controlled experts.

