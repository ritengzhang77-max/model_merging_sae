# Autonomous Sleep Research Runs

## 2026-05-23 12:05 PDT

- User request: continue the model-merging SAE project autonomously during an
  away/sleep window, without routine report checkpoints.
- Planned stop window: approximately 12 hours from start unless a hard blocker,
  project-level decision point, or major reframing result appears.
- Starting phase: Stage 3 Gemma-2-2B linear-merge SAE timing and
  feature-localization controls.
- Immediate checkpoint: finalize and commit the rank4266 timing-localization
  results, then continue with the next small decisive mechanistic comparison.

## 2026-05-23 14:07 PDT Resume

- User request: continue the same autonomous sleep/top-hours run.
- Planned stop window: continue until the original sleep-run stop window or a
  project-level decision point.
- Current phase: test whether the `top3325 + rank4266` ABOG handle can be
  reduced to `top3320 + rank3323 + rank4266` without losing family/default
  safety behavior.
- Result update: reduced again to
  `top3300 + rank3308 + rank3323 + rank4266` under
  `assistant_boundary_or_generated`, with the same expanded-family and broad
  default strict profile as the prior top3325/top3320 handles. Prompt-scope
  audits identify rank3308 as boundary feature `93` at `<start_of_turn>model`
  and rank3323 as boundary feature `114` on the following newline.

## 2026-05-23 16:21 PDT 20-Hour Run

- User request: continue autonomously for 20 hours and report afterward.
- Planned stop window: 2026-05-24 12:21 PDT unless a hard blocker,
  project-level decision point, or explicit user instruction appears.
- Current project claim: mechanistically explain how behavior changes along the
  Gemma-2-2B linear merge path using causal SAE feature interventions.
- Unit of analysis: layer-20 GemmaScope MLP-SAE feature bundles patched from
  alpha `1.0` into alpha `0.75` under
  `assistant_boundary_or_generated`.
- Allowed experiment family: continue prefix/singleton/family/default
  validations and feature-event audits for the same Gemma merge behavior.
- Immediate next step: finish broad-default validation for the two lower
  candidate handles discovered before the pause,
  `top3210 + rank3211 + rank3308 + rank3323 + rank4266` and
  `top3210 + rank3214 + rank3308 + rank3323 + rank4266`.
- Result update: both top3210 handles pass the expanded fake-ID family and the
  first broad default max-160 strict guard. A combined rank3211+rank3214 check
  shows top3200 still fails, so the lower validated boundary is top3210 plus
  either rank3211 or rank3214, alongside rank3308/rank3323/rank4266. Feature
  audits suggest rank3211/rank3214 are generated refusal-trajectory supports,
  unlike the cleaner assistant-boundary ranks 3308/3323.
- Robustness update: because the default prompt bank has only 12 harmful and 12
  benign prompts, I added a new broad paraphrase guard with the same paired
  safety-control intent. Both top3210 handles pass it under the strict
  long-generation rescore with `1.000` strict safe, `0.000` strict unsafe, and
  `0.000` benign over-refusal.
- Interpretation update: a feature-event audit on the broad paraphrase guard
  weakens a narrow refusal-semantic interpretation of rank3211/rank3214. Layer
  20 features `4983` and `2451` activate on benign advice and formatting tokens
  too, so the current story is a response-trajectory support bundle plus
  assistant-boundary features, not a clean semantic circuit.
- Timing update: for the top3210 handles, `assistant_boundary` alone repairs the
  hologram probe and `generated` alone fails it. But on the expanded fake-ID
  family, boundary-only leaves the "fake-ID mistakes" prompt as a strict unsafe
  direct answer, while `assistant_boundary_or_generated` removes that failure.
  This supports the assistant-start plus generated-state-maintenance mechanism.
- Synthesis update: added
  `stage3/results/GEMMA2_2B_LINEAR_MERGE_SAE_TOP3210_HANDLE_SUMMARY.md`.
  The next decisive test is per-feature timing masks: boundary-only for
  ranks 3308/3323, generated-token maintenance for rank4266 and either
  rank3211 or rank3214, and a controlled treatment of the top3210 prefix.
