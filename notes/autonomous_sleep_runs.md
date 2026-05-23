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
