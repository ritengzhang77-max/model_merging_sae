# Stage 0 Results Provenance

This ledger records Stage 0 artifacts that are likely to feed later paper
tables, figures, or decision memos.

## SmolLM2 Refusal V2 Attempts

- Date appended: 2026-05-19
- Artifact status: canonical decision-gate evidence
- Main decision memo: `stage0/results/SMOLLM2_REFUSAL_V2_DECISION.md`
- Main scripts:
  - `stage0/scripts/run_smollm2_refusal_v2.py`
  - `stage0/scripts/run_smollm2_refusal_v2_alpha_search.py`

Artifacts:

- `stage0/artifacts/smollm2_refusal_v2/`
  - first v2 refusal expert trained on diverse harmful prompts plus benign
    contrast data.
  - key result: expert attempted refusal on all harmful prompts but had only
    0.062 harmful clean-refusal and 1.000 benign over-refusal.
- `stage0/results/refusal_v2/`
  - metrics and generations for the first v2 run.
- `stage0/artifacts/smollm2_refusal_v2_balanced/`
  - balanced v2 refusal expert with higher benign fraction, lower learning
    rate, fewer steps, and shorter decoding.
  - key result: expert harmful clean-refusal improved to 0.375 and benign
    over-refusal dropped to 0.188, but v2 linear merges still had 0.000 harmful
    clean-refusal.
- `stage0/results/refusal_v2_balanced/`
  - metrics and generations for the balanced v2 run.
- `stage0/results/refusal_v2_alpha_search/`
  - full refusal-delta alpha sweep using
    `merge_arith_polite + alpha * (expert_refusal_v2 - base)`.
  - key result: alpha 1.0 reached harmful attempted-refusal 0.938, but harmful
    clean-refusal only 0.125 and benign over-refusal 0.438.

Key conclusion:

- The current `HuggingFaceTB/SmolLM2-135M` non-instruction base remains a poor
  clean-refusal merge target after v2 data improvements.
- The next clean-target attempt should start from
  `HuggingFaceTB/SmolLM2-135M-Instruct`, which is already present in the local
  HF cache.

Caveats:

- Metrics use heuristic string-based harmful/benign quality labels.
- Evaluation prompt count is 16 per harmful/benign split for the main v2 runs.
- Generation artifacts such as `LEGATO`, `%||`, and `pvproperty` are counted as
  quality failures when caught by the current artifact/repetition rules, but the
  heuristic is not a substitute for human audit.

## Qwen Candidate Screens

- Date appended: 2026-05-20
- Artifact status: active Stage 0 candidate selection evidence
- Main summary: `stage0/results/CANDIDATE_SCREEN_SUMMARY.md`
- Main scripts:
  - `stage0/scripts/screen_chat_merge_candidate.py`
  - `stage0/scripts/screen_qwen_coder_merge.py`

Artifacts:

- `stage0/results/candidate_screens/smollm2_135m_instruct/`
  - clean chat substrate, but not a refusal target.
- `stage0/results/candidate_screens/qwen2_5_0_5b_instruct/`
  - `Qwen/Qwen2.5-0.5B-Instruct` passed the cheap harmful/benign/arithmetic/
    polite screen and became the Qwen anchor.
- `stage0/results/candidate_screens/qwen_public_candidates/`
  - public Qwen coder/merge/math candidates.
  - `Qwen/Qwen2.5-Coder-0.5B-Instruct` over-refused benign prompts.
  - `mergekit-community/mergekit-linear-qtqmpco` was degenerate.
  - `2796gauravc/qwen2.5-0.5b-math` required `bitsandbytes`.
- `stage0/results/candidate_screens/qwen_coder_merge/`
  - direct Qwen2.5-Instruct to Qwen2.5-Coder interpolation.
  - endpoints were clean, but middle alphas broke behavior, so this is discarded
    as a positive merge target.
- `stage0/results/candidate_screens/qwen_merge_candidates_round2/`
  - screened six architecture-compatible public Qwen merges.
  - `djuna-test-lab/Qwen2.5Plus2-0.5B-Instruct` and
    `mergekit-community/mergekit-slerp-lxmmvuv` passed.
- `stage0/results/candidate_screens/qwen_plusminus_extended/`
  - extended screen on Qwen2.5 base, Qwen2 parent, Plus2, Minus2, and SLERP.
- `stage0/results/candidate_screens/qwen_merge_candidates_round3/`
  - screened another five architecture-compatible public Qwen merges.
  - `Sakalti/SJT-0.5B` and `vitus9988/Qwen2.5-0.5B-ko-merge` passed the cheap
    behavior screen.
  - `amadeusai/Amadeus-Verbo-MI-Qwen-2.5-0.5B-PT-BR-Instruct-Experimental`
    remained fluent but failed harmful-refusal behavior.
  - `Sakalti/lakeland` failed clean generation.
- `stage0/results/candidate_screens/qwen_multilingual_candidates/`
  - focused Korean/Portuguese task and safety screen for the 0.5B language
    candidates.
  - key result: the Korean merge did not outperform the Qwen2.5-0.5B anchor on
    the small language screen.
- `stage0/results/candidate_screens/qwen1_5b_merge_candidates_round1/`
  - broad 4-prompt screen for Qwen2.5-1.5B public merges.
  - found the math SLERP and Matrix/model-stock candidates; found an abliterated
    TIES merge that drops refusal while preserving benign behavior.
- `stage0/results/candidate_screens/qwen1_5b_selected_extended/`
  - selected 8-prompt behavior screen for Qwen2.5-1.5B base, Matrix, math
    SLERP, and abliterated TIES.
- `stage0/results/candidate_screens/qwen1_5b_math_reasoning/`
  - focused math triage for the same selected 1.5B set.

Key numeric values from `qwen_plusminus_extended`:

- `djuna-test-lab/Qwen2.5Plus2-0.5B-Instruct`: clean generation 1.000, harmful
  clean-refusal 1.000, benign helpful 0.875.
- `mergekit-community/Qwen2.5Minus2-0.5B-Instruct`: clean generation 1.000,
  harmful clean-refusal 0.000, benign helpful 0.875.
- `mergekit-community/mergekit-slerp-lxmmvuv`: clean generation 1.000, harmful
  clean-refusal 0.750, benign helpful 0.875.

Current decision:

- The Qwen Plus/Minus task-arithmetic pair is useful for task-vector sign and
  safety-loss analysis, but not yet the final multi-expert composition target.
- Stage 1 RQ0 follow-up found `plus2` is extremely close to the Qwen2 parent
  checkpoint, while `minus2` is a strong extrapolation away from the Qwen2.5
  base.
- Round-3 follow-up found `Sakalti/SJT-0.5B` is effectively identical to the
  Qwen2.5-Instruct anchor, while `vitus9988/Qwen2.5-0.5B-ko-merge` is a small
  but genuine displacement that preserves the cheap English behavior screen.
- The Amadeus PT-BR SLERP is useful as a negative contrast because it has larger
  displacement but loses harmful-refusal behavior.
- Multilingual follow-up did not support the 0.5B Korean merge as a main
  target.
- The Qwen2.5-1.5B selected set is now the main public-model branch. Its most
  useful contrast is `nbeerbower/EVA-abliterated-TIES-Qwen2.5-1.5B`, which
  keeps benign/arithmetic behavior but loses harmful refusal.
- Do not move to SAE yet.
