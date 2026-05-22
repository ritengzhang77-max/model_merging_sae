# Stage 1 Results Provenance

This ledger records Stage 1 artifacts that may feed the mechanistic model
merging paper plan.

## Qwen Plus/Minus RQ0 Screen

- Date appended: 2026-05-20
- Artifact status: canonical Stage 1 decision evidence
- Summary: `stage1/results/qwen_plusminus_rq0/QWEN_PLUSMINUS_RQ0_SUMMARY.md`
- Generating script: `stage1/scripts/analyze_qwen_plusminus_rq0.py`
- Source Stage 0 behavior: `stage0/results/candidate_screens/qwen_plusminus_extended/chat_candidate_screen_metrics.csv`
- Command:

```bash
HF_HOME=/data/gavin/model_merging/hf_cache python3 stage1/scripts/analyze_qwen_plusminus_rq0.py \
  --device cuda:3 \
  --examples-per-split 8 \
  --batch-size 4 \
  --max-delta-values 120000 \
  --result-dir stage1/results/qwen_plusminus_rq0
```

What it shows:

- `djuna-test-lab/Qwen2.5Plus2-0.5B-Instruct` is extremely close to the
  `Qwen/Qwen2-0.5B-Instruct` parent in activations.
- `mergekit-community/Qwen2.5Minus2-0.5B-Instruct` behaves as a strong
  extrapolated negative control.
- The pair is useful for task-vector sign and refusal-loss analysis, but it is
  not a rich final multi-expert composition target.

## Qwen Public Candidates RQ0 Screen

- Date appended: 2026-05-20
- Artifact status: active Stage 1 decision evidence
- Summary: `stage1/results/qwen_public_candidates_rq0/QWEN_PUBLIC_CANDIDATE_RQ0_SUMMARY.md`
- Generating script: `stage1/scripts/analyze_qwen_candidate_rq0.py`
- Source Stage 0 behavior: `stage0/results/candidate_screens/qwen_merge_candidates_round3/chat_candidate_screen_metrics.csv`
- Command:

```bash
HF_HOME=/data/gavin/model_merging/hf_cache python3 stage1/scripts/analyze_qwen_candidate_rq0.py \
  --device cuda:3 \
  --examples-per-split 8 \
  --batch-size 4 \
  --max-delta-values 120000 \
  --result-dir stage1/results/qwen_public_candidates_rq0
```

What it shows:

- `Sakalti/SJT-0.5B` passes behavior but is effectively a near-copy of
  `Qwen/Qwen2.5-0.5B-Instruct`: sampled delta L2 is `1.65e-05`, and final-layer
  activation cosine is `1.000` against the anchor on harmful and benign prompts.
- `vitus9988/Qwen2.5-0.5B-ko-merge` is the best current public positive
  candidate: it passes the cheap English screen and has a small but nonzero
  displacement from the anchor.
- `amadeusai/Amadeus-Verbo-MI-Qwen-2.5-0.5B-PT-BR-Instruct-Experimental` is a
  useful negative contrast: it is farther from the anchor but fails
  harmful-refusal behavior.

Caveats:

- Delta magnitudes are sampled, not exact full-model distances.
- Activation similarity uses final prompt-token hidden states over 8 prompts per
  split, so it is a triage signal rather than a final mechanistic result.
- The next check should add explicit Korean and Portuguese task prompts before
  moving to SAE or causal tracing.

## Qwen2.5-1.5B Selected Public Merge RQ0 Screen

- Date appended: 2026-05-20
- Artifact status: canonical current public-model decision evidence
- Summary: `stage1/results/qwen1_5b_selected_rq0/QWEN_PUBLIC_CANDIDATE_RQ0_SUMMARY.md`
- Generating script: `stage1/scripts/analyze_qwen_candidate_rq0.py`
- Source Stage 0 behavior: `stage0/results/candidate_screens/qwen1_5b_selected_extended/chat_candidate_screen_metrics.csv`
- Related math screen: `stage0/results/candidate_screens/qwen1_5b_math_reasoning/MATH_REASONING_SUMMARY.md`
- Command:

```bash
HF_HOME=/data/gavin/model_merging/hf_cache python3 stage1/scripts/analyze_qwen_candidate_rq0.py \
  --device cuda:3 \
  --examples-per-split 8 \
  --batch-size 2 \
  --max-delta-values 120000 \
  --stage0-metrics stage0/results/candidate_screens/qwen1_5b_selected_extended/chat_candidate_screen_metrics.csv \
  --models qwen2_5_instruct=Qwen/Qwen2.5-1.5B-Instruct,matrix=bunnycore/Qwen2.5-1.5B-Matrix,math_slerp=Yuuta208/Qwen2.5-1.5B-Instruct-Qwen2.5-Math-1.5B-Merged-slerp-24,abliterated=nbeerbower/EVA-abliterated-TIES-Qwen2.5-1.5B \
  --result-dir stage1/results/qwen1_5b_selected_rq0
```

What it shows:

- `matrix` and `math_slerp` have highly aligned sampled deltas
  (`cosine=0.929`) and both weaken refusal relative to the base.
- `math_slerp` is the best positive-ish merge, but it does not beat the base on
  the focused math screen and has weaker harmful-refusal behavior.
- `abliterated` is the cleanest negative control: sampled delta L2 is only
  `0.132`, but harmful-refusal score is `0.000` while benign and arithmetic
  behavior remain intact in the cheap screen.
- The abliterated model's final-token activation similarity to the base is much
  lower on harmful prompts (`0.745`) than benign prompts (`0.904`), making it a
  good RQ1/RQ2 target for refusal-loss localization.

Caveats:

- Delta values are sampled, not exact full-model distances.
- The current result supports a safety-loss interpretability study, not a claim
  that these public merges improve over the base.
- Do not train SAEs until layer/module diagnostics show a stable target.

## Gemma-2-2B Abliterated RQ0 Screen

- Date appended: 2026-05-22
- Artifact status: active alternate model-pair/SAE-ecosystem decision evidence
- Summary: `stage1/results/gemma2_2b_abliterated_rq0/GEMMA2_2B_ABLITERATED_RQ0_SUMMARY.md`
- Generating script: `stage1/scripts/analyze_gemma2_2b_abliterated_rq0.py`
- Source Stage 0 behavior:
  `stage0/results/candidate_screens_gemma2_2b_abliterated_20260522_clean/chat_candidate_screen_metrics.csv`
- Command:

```bash
python3 stage1/scripts/analyze_gemma2_2b_abliterated_rq0.py \
  --device cuda:3 \
  --examples-per-split 4 \
  --batch-size 2
```

What it shows:

- `google/gemma-2-2b-it` and
  `IlyaGusev/gemma-2-2b-it-abliterated` have matching Gemma2 architecture
  configs: 26 layers, hidden size 2304, intermediate size 9216, 8 attention
  heads, 4 KV heads, and vocab size 256000.
- The cheap behavior screen is clean: base harmful clean refusal `1.000`,
  abliterated harmful clean refusal `0.000`, and both have benign helpfulness
  `1.000`.
- Activation similarity is strongly harmful-specific. At layer 20, base vs
  abliterated cosine is `0.687` on harmful prompts but `0.991` on benign
  prompts, while arithmetic and polite prompts remain near `0.998` and `0.986`.

Decision:

- Promote Gemma-2-2B abliterated to the next alternate branch.
- Run module/activation patching before any GemmaScope feature interpretation.
- If activation patching finds a clean repair target, Gemma becomes the best
  public sparse-basis ecosystem candidate because GemmaScope provides residual,
  MLP, attention, and third-party transcoder bases.
