# Candidate Screen Summary

Updated: 2026-05-22

This is the new fast Stage 0 candidate filter. The goal is to find a clean
model-merging target quickly, not to rescue weak candidates.

## Policy

See `docs/CANDIDATE_SCREENING_POLICY.md`.

Short version:

- one short setup pass;
- one small generation/evaluation pass;
- one simple merge or alpha sweep only if promising;
- discard candidates that need substantial debugging.

## Screened Candidates

### Qwen Plus/Minus Extended Screen

Models:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2-0.5B-Instruct`
- `djuna-test-lab/Qwen2.5Plus2-0.5B-Instruct`
- `mergekit-community/Qwen2.5Minus2-0.5B-Instruct`
- `mergekit-community/mergekit-slerp-lxmmvuv`

Output:

- `stage0/results/candidate_screens/qwen_plusminus_extended/`

Result:

| model | substrate pass | refusal pass | clean gen | harmful clean | benign helpful | arith | polite |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen2.5-0.5B-Instruct` | true | true | 1.000 | 0.875 | 0.875 | 0.875 | 1.000 |
| `Qwen/Qwen2-0.5B-Instruct` | true | true | 1.000 | 0.875 | 1.000 | 0.500 | 1.000 |
| `djuna-test-lab/Qwen2.5Plus2-0.5B-Instruct` | true | true | 1.000 | 1.000 | 0.875 | 1.000 | 0.750 |
| `mergekit-community/Qwen2.5Minus2-0.5B-Instruct` | true | false | 1.000 | 0.000 | 0.875 | 0.000 | 0.000 |
| `mergekit-community/mergekit-slerp-lxmmvuv` | true | true | 1.000 | 0.750 | 0.875 | 0.875 | 1.000 |

Decision:

- Keep `djuna-test-lab/Qwen2.5Plus2-0.5B-Instruct` as a positive-control
  candidate, not yet as the final composition target.
- Keep `mergekit-community/Qwen2.5Minus2-0.5B-Instruct` as a clean negative
  control.
- Keep `mergekit-community/mergekit-slerp-lxmmvuv` as a secondary positive-ish
  comparison, but not the primary target.

Reason:

- Plus2 and Minus2 have recoverable MergeKit task-arithmetic recipes using
  `Qwen/Qwen2.5-0.5B-Instruct` as the base and `Qwen/Qwen2-0.5B-Instruct` as
  the other parent.
- Plus2 preserves or improves refusal in the extended screen.
- Minus2 keeps clean generation and benign behavior but removes harmful refusal,
  making it an unusually useful contrast case.
- Stage 1 RQ0 follow-up shows Plus2 is extremely close to the Qwen2 parent, so
  this pair is better for task-vector sign analysis than for a final
  multi-expert composition study.

### SmolLM2-135M-Instruct

Model:

- `HuggingFaceTB/SmolLM2-135M-Instruct`

Output:

- `stage0/results/candidate_screens/smollm2_135m_instruct/`

Result:

| substrate pass | refusal pass | clean gen | harmful clean | benign helpful | arith | polite |
|---:|---:|---:|---:|---:|---:|---:|
| true | false | 1.000 | 0.000 | 1.000 | 0.333 | 0.333 |

Decision:

- Keep only as a possible cheap chat substrate.
- Do not prioritize it as the main refusal-transfer target.

Reason:

- It generates clean text, unlike the old non-instruct SmolLM2 base.
- But it gives unsafe answers to several harmful prompts and is weak on simple
  arithmetic/polite screens.

### Qwen2.5-0.5B-Instruct

Model:

- `Qwen/Qwen2.5-0.5B-Instruct`

Source note:

- Hugging Face model card says this is the instruction-tuned 0.5B Qwen2.5 model
  with Qwen2 architecture, 24 layers, hidden size 896, and Apache-2.0 license.

Output:

- `stage0/results/candidate_screens/qwen2_5_0_5b_instruct/`

Result:

| substrate pass | refusal pass | clean gen | harmful clean | benign helpful | arith | polite |
|---:|---:|---:|---:|---:|---:|---:|
| true | true | 1.000 | 1.000 | 0.833 | 0.667 | 1.000 |

Decision:

- This is the current best small base candidate.
- Use it as the anchor for further candidate screening.

Reason:

- It cleanly refused all harmful prompts in the cheap screen.
- It handled benign safety-help prompts mostly well.
- It had much cleaner behavior than SmolLM2.

### Qwen2.5-Coder-0.5B-Instruct

Model:

- `Qwen/Qwen2.5-Coder-0.5B-Instruct`

Source note:

- Qwen2.5-Coder technical report/model family describes this as a code-specific
  Qwen2.5 architecture model continued on large code data.

Output:

- `stage0/results/candidate_screens/qwen_public_candidates/`

Result:

| substrate pass | refusal pass | clean gen | harmful clean | benign helpful | benign over-refusal | arith | polite |
|---:|---:|---:|---:|---:|---:|---:|---:|
| false | false | 1.000 | 1.000 | 0.500 | 0.500 | 0.667 | 0.333 |

Decision:

- Possible code donor, but not a general chat target by itself.

Reason:

- It refuses harmful prompts cleanly.
- It over-refuses some benign prompts and is weak on polite rewriting.

### Qwen2.5-Instruct / Qwen2.5-Coder Linear Interpolation

Script:

- `stage0/scripts/screen_qwen_coder_merge.py`

Formula:

```text
theta = theta_qwen2.5_instruct + alpha * (theta_qwen2.5_coder_instruct - theta_qwen2.5_instruct)
```

Output:

- `stage0/results/candidate_screens/qwen_coder_merge/`

Result:

| model | alpha | pass | clean gen | harmful | benign | arith | polite | code | core mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5_instruct` | 0.00 | true | 1.000 | 1.000 | 1.000 | 0.667 | 1.000 | 1.000 | 0.933 |
| `qwen2.5_coder_instruct` | 1.00 | true | 1.000 | 1.000 | 1.000 | 0.667 | 0.333 | 1.000 | 0.800 |
| `qwen2.5_coder_linear_a0p75` | 0.75 | false | 1.000 | 0.000 | 1.000 | 0.667 | 0.333 | 1.000 | 0.600 |
| `qwen2.5_coder_linear_a0p5` | 0.50 | false | 0.056 | 0.000 | 0.000 | 0.000 | 0.000 | 0.250 | 0.050 |
| `qwen2.5_coder_linear_a0p25` | 0.25 | false | 0.167 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Decision:

- Discard simple base/coder interpolation as a positive merge target.

Reason:

- The endpoints work, but middle alphas break safety/general behavior.
- This is evidence of incompatibility, not a clean successful merge.

### Public Qwen MergeKit Linear Model

Model:

- `mergekit-community/mergekit-linear-qtqmpco`

Source note:

- Model card says this is a linear merge using `Qwen/Qwen2.5-0.5B-Instruct` as
  base and including `Qwen/Qwen2-0.5B-Instruct`.

Output:

- `stage0/results/candidate_screens/qwen_public_candidates/`

Decision:

- Discard.

Reason:

- Cheap screen generated degenerate repeated characters and failed all behavior
  splits.

### Public Qwen Math Fine-Tune

Model:

- `2796gauravc/qwen2.5-0.5b-math`

Source note:

- Model card describes a QLoRA/4-bit math fine-tune from
  `Qwen/Qwen2.5-0.5B-Instruct`; reported GSM8K pass@1 is 5%.

Decision:

- Discard for now.

Reason:

- Loading requires `bitsandbytes` in this environment.
- The reported task performance is weak enough that it is not worth dependency
  debugging during candidate search.

### Public Qwen Merge Candidates Round 3

Models:

- `Youlln/ECE.EIFFEIL.ia-0.5B-SLERP`
- `Sakalti/lakeland`
- `Sakalti/SJT-0.5B`
- `vitus9988/Qwen2.5-0.5B-ko-merge`
- `amadeusai/Amadeus-Verbo-MI-Qwen-2.5-0.5B-PT-BR-Instruct-Experimental`

Source notes:

- `Sakalti/SJT-0.5B` is a TIES merge using `Qwen/Qwen2.5-0.5B` as the
  base, but the model card lists only `Qwen/Qwen2.5-0.5B-Instruct` as an
  included model.
- `vitus9988/Qwen2.5-0.5B-ko-merge` is a TIES merge using
  `Qwen/Qwen2.5-0.5B` as the base and includes
  `Qwen/Qwen2.5-0.5B-Instruct` plus `Qwen2.5-0.5B-Instruct-lora-merge`.
- The Amadeus PT-BR experimental model is a SLERP merge of
  `Qwen/Qwen2.5-0.5B-Instruct` and an Amadeus Portuguese Qwen2.5 instruct
  model.

Output:

- `stage0/results/candidate_screens/qwen_merge_candidates_round3/`
- RQ0 follow-up: `stage1/results/qwen_public_candidates_rq0/`

Result:

| model | substrate pass | refusal pass | clean gen | harmful clean | benign helpful | arith | polite |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Youlln/ECE.EIFFEIL.ia-0.5B-SLERP` | false | false | 1.000 | 0.875 | 0.625 | 1.000 | 1.000 |
| `Sakalti/lakeland` | false | false | 0.031 | 0.000 | 0.000 | 0.125 | 0.000 |
| `Sakalti/SJT-0.5B` | true | true | 1.000 | 0.875 | 1.000 | 1.000 | 0.875 |
| `vitus9988/Qwen2.5-0.5B-ko-merge` | true | true | 1.000 | 0.875 | 0.875 | 0.875 | 0.750 |
| `amadeusai/Amadeus-Verbo-MI-Qwen-2.5-0.5B-PT-BR-Instruct-Experimental` | true | false | 1.000 | 0.125 | 1.000 | 1.000 | 0.750 |

Stage 1 RQ0 follow-up:

| model | sampled delta L2 vs Qwen2.5-Instruct | layer-23 harmful cosine vs Qwen2.5-Instruct | layer-23 benign cosine vs Qwen2.5-Instruct |
|---|---:|---:|---:|
| `Sakalti/SJT-0.5B` | 0.0000165 | 1.000 | 1.000 |
| `vitus9988/Qwen2.5-0.5B-ko-merge` | 0.152 | 0.926 | 0.962 |
| `amadeusai/Amadeus-Verbo-MI-Qwen-2.5-0.5B-PT-BR-Instruct-Experimental` | 1.489 | 0.959 | 0.966 |

Decision:

- Discard `Youlln/ECE.EIFFEIL.ia-0.5B-SLERP` as a primary target because it
  over-refuses benign prompts in this screen.
- Discard `Sakalti/lakeland`; generation quality is broken.
- Keep `Sakalti/SJT-0.5B` only as an anchor sanity check. It behaves well but
  is essentially identical to `Qwen/Qwen2.5-0.5B-Instruct`.
- Promote `vitus9988/Qwen2.5-0.5B-ko-merge` to a focused multilingual/domain
  screen. It is the best current positive candidate from public Qwen merges.
- Keep the Amadeus PT-BR SLERP as a negative contrast for "larger domain/language
  merge that lost English harmful-refusal behavior."

Focused multilingual follow-up:

- Output: `stage0/results/candidate_screens/qwen_multilingual_candidates/`
- The 0.5B Korean merge did not outperform the Qwen2.5-0.5B instruction anchor
  on the small Korean/Portuguese task screen.
- Decision update: do not promote the 0.5B Korean merge to SAE or causal
  tracing. Keep it only as a minor small-delta comparison if needed.

### Public Qwen2.5-1.5B Merge Candidates

Source notes:

- `Qwen/Qwen2.5-1.5B-Instruct` is the Apache-2.0 Qwen2.5 1.5B instruction
  model and is the base anchor for this public-merge branch.
- `bunnycore/Qwen2.5-1.5B-Matrix` is a Model Stock merge using
  `Qwen/Qwen2.5-1.5B-Instruct` as base, with Qwen2.5 Math, Qwen2.5 Coder, and
  an abliterated Qwen2.5 model included.
- `Yuuta208/Qwen2.5-1.5B-Instruct-Qwen2.5-Math-1.5B-Merged-slerp-24` is a
  SLERP merge of `Qwen/Qwen2.5-1.5B-Instruct` and `Qwen/Qwen2.5-Math-1.5B`
  with `t: 0.1`.
- `nbeerbower/EVA-abliterated-TIES-Qwen2.5-1.5B` is a TIES merge using
  `Qwen/Qwen2.5-1.5B` as base and includes an abliterated Qwen2.5-1.5B model
  plus `EVA-UNIT-01/EVA-Qwen2.5-1.5B-v0.0`.

Outputs:

- broad 4-prompt screen:
  `stage0/results/candidate_screens/qwen1_5b_merge_candidates_round1/`
- selected 8-prompt screen:
  `stage0/results/candidate_screens/qwen1_5b_selected_extended/`
- math triage:
  `stage0/results/candidate_screens/qwen1_5b_math_reasoning/`
- RQ0 follow-up:
  `stage1/results/qwen1_5b_selected_rq0/`

Selected 8-prompt behavior:

| model | substrate pass | refusal pass | clean gen | harmful clean | benign helpful | arith | polite |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen2.5-1.5B-Instruct` | true | true | 1.000 | 0.875 | 1.000 | 1.000 | 1.000 |
| `bunnycore/Qwen2.5-1.5B-Matrix` | true | false | 1.000 | 0.375 | 1.000 | 0.750 | 0.000 |
| `Yuuta208/...Merged-slerp-24` | true | true | 1.000 | 0.625 | 1.000 | 0.875 | 0.000 |
| `nbeerbower/EVA-abliterated-TIES-Qwen2.5-1.5B` | true | false | 1.000 | 0.000 | 1.000 | 1.000 | 0.500 |

Focused math triage:

| model | math ok |
|---|---:|
| `Qwen/Qwen2.5-1.5B-Instruct` | 0.800 |
| `bunnycore/Qwen2.5-1.5B-Matrix` | 0.300 |
| `Yuuta208/...Merged-slerp-24` | 0.600 |
| `nbeerbower/EVA-abliterated-TIES-Qwen2.5-1.5B` | 0.900 |

Stage 1 RQ0 follow-up:

| model | sampled delta L2 vs Qwen2.5-1.5B-Instruct | layer-23 harmful cosine vs base | layer-23 benign cosine vs base |
|---|---:|---:|---:|
| `bunnycore/Qwen2.5-1.5B-Matrix` | 121.138 | 0.886 | 0.930 |
| `Yuuta208/...Merged-slerp-24` | 51.991 | 0.927 | 0.959 |
| `nbeerbower/EVA-abliterated-TIES-Qwen2.5-1.5B` | 0.132 | 0.745 | 0.904 |

Decision:

- The 1.5B branch is the best current public-model branch, but not because a
  merge clearly beats the base.
- The strongest scientific case is safety-loss interpretability: the
  abliterated TIES merge has a tiny sampled delta but deletes harmful refusal
  while preserving benign and arithmetic behavior.
- The math SLERP is useful as a partial positive/contrast merge, but it does
  not beat the base on the focused math screen.
- Next step is RQ1/RQ2 diagnostics on the 1.5B selected set, not SAE training.

### Gemma-2-2B Abliterated Candidate

Models:

- `google/gemma-2-2b-it`
- `IlyaGusev/gemma-2-2b-it-abliterated`

Why screened:

- The `value_action` review made model-pair plus SAE-ecosystem quality a first
  class gate.
- Gemma-2-2B has public GemmaScope residual/MLP/attention SAEs and public
  GemmaScope-style transcoders.
- The abliterated checkpoint is non-gated, while this environment has access to
  the official gated Gemma base/instruct checkpoints.

Architecture check:

| model | model type | layers | hidden | intermediate | heads | kv heads | vocab |
|---|---|---:|---:|---:|---:|---:|---:|
| `google/gemma-2-2b-it` | `gemma2` | 26 | 2304 | 9216 | 8 | 4 | 256000 |
| `IlyaGusev/gemma-2-2b-it-abliterated` | `gemma2` | 26 | 2304 | 9216 | 8 | 4 | 256000 |

Output:

- `stage0/results/candidate_screens_gemma2_2b_abliterated_20260522_clean/`
- RQ0 follow-up: `stage1/results/gemma2_2b_abliterated_rq0/`

Result:

| model | substrate pass | refusal pass | clean gen | harmful clean | harmful attempt | benign helpful | benign over-refusal | arith | polite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `google/gemma-2-2b-it` | true | true | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| `IlyaGusev/gemma-2-2b-it-abliterated` | true | false | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.750 |

Decision:

- Promote this to the next candidate branch after the current Qwen residual
  checkpoint.
- Do a fast Gemma RQ0/RQ1 diagnostic before any SAE interpretation:
  harmful-specific activation drift, module patching, and low-rank/top-coordinate
  baselines.
- If Gemma has a clean activation-patch repair, it becomes the best public
  sparse-basis ecosystem target so far because GemmaScope gives us public
  residual/MLP/attention SAEs instead of training every basis locally.

RQ0 activation follow-up:

| layer | harmful cosine | benign cosine | arith cosine | polite cosine | benign-harmful gap |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 | -0.000 |
| 4 | 1.000 | 0.999 | 1.000 | 1.000 | -0.001 |
| 8 | 0.991 | 0.997 | 0.999 | 0.999 | 0.006 |
| 12 | 0.945 | 0.994 | 0.999 | 0.999 | 0.049 |
| 16 | 0.821 | 0.993 | 0.997 | 0.993 | 0.173 |
| 20 | 0.687 | 0.991 | 0.998 | 0.986 | 0.304 |
| 25 | 0.852 | 0.990 | 0.997 | 0.987 | 0.138 |

Reason:

- The behavioral gap is cleaner than the Qwen v0 small benchmark: base refuses
  all sampled harmful prompts, abliterated refuses none, and both preserve
  benign helpfulness.
- The pair is architecture-compatible.
- It directly addresses the concern that Qwen1.5B may have a weak public sparse
  ecosystem.
- The activation drift is strongly harmful-specific in layers 16-20, while
  benign/arithmetic/polite prompts remain near-aligned. This is exactly the
  signature we wanted before spending GemmaScope effort.

## Current Recommendation

Keep the project. The candidate search now has two serious branches:

1. Keep the Qwen Plus/Minus pair as a task-vector sign/safety-loss case.
2. Do not promote the 0.5B Korean/PT-BR branch; it did not show a useful
   advantage over the anchor.
3. Use the Qwen2.5-1.5B selected public set as the current mechanistic branch:
   base, math SLERP, Matrix/model-stock, and abliterated TIES.
4. Add the Gemma-2-2B abliterated pair as the next SAE-ecosystem branch.
5. Frame the near-term research target as merge-induced safety loss and
   activation drift, not as performance improvement from merging.
6. Do not start feature naming until RQ1/RQ2 module and activation
   diagnostics beat or complement simpler baselines.
