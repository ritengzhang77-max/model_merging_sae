# Gemma-2-2B Linear Merge SAE 11-Feature Identity Summary

Date: 2026-05-24

This memo links the six validated k=11 first-token-passing handles to
local feature identities. The goal is not to assign final semantic labels;
it is to separate three things that were previously mixed together:

- which SAE features are present in the local pass class;
- how those features move between donor alpha `1.0` and recipient alpha `0.75`;
- which features behave as local backbone, exchangeable members, or substitutes.

Neuronpedia descriptions are used as weak labels only. Local activation and
causal-swap evidence takes priority over autointerp wording.

## Pass-Class Feature Set

The one-swap screen has `6` first-token-passing k=11 handles.
Their feature union has `14` features, and their intersection has `9` features:

```text
1813, 8754, 9135, 9149, 12652, 12704, 13622, 14991, 15169
```

The intersection is not the same thing as causal necessity. The one-swap
drop screen is stricter: features such as `15169` and `14991` never tie when
dropped, while several other intersection features can tie but not pass.

## Feature Table

| feature | in passes | hologram rank | fake-ID family rank | broad harmful rank | broad delta | drop pass/tie | add pass/tie | local role | weak Neuronpedia label |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `1338` | 2/6 | 28 | 207 | 346 | -3.982 | 1/0 | 1/6 | source feature with passing one-swap replacements | titles and roles related to leadership in marketing |
| `1813` | 6/6 | 31 | 31 | 19 | 25.337 | 0/20 | 0/0 | common support; removals can tie but not pass | expressive statements of excitement and emotion |
| `6289` | 4/6 | 33 | 651 | 6425 | 0.000 | 2/30 | 0/0 | source feature with passing one-swap replacements | phrases related to organizational structure and leadership transitions |
| `7531` | 3/6 | 23 | 108 | 60 | 4.576 | 0/0 | 3/10 | contextual substitute in pass handles | sections of code, specifically highlighting variable declarations and control structures in programming |
| `8754` | 6/6 | 10 | 120 | 387 | 3.133 | 0/27 | 0/0 | common support; removals can tie but not pass | information about specific individuals, particularly athletes and their backgrounds |
| `8775` | 2/6 | 14 | 271 | 914 | -1.509 | 3/12 | 1/0 | source feature with passing one-swap replacements | requests or commands directed at others |
| `9135` | 6/6 | 24 | 191 | 412 | -4.456 | 0/12 | 0/0 | common support; removals can tie but not pass | references to file and package management in a coding environment |
| `9149` | 6/6 | 22 | 618 | 13737 | 0.000 | 0/21 | 0/0 | common support; removals can tie but not pass | key phrases related to established organizations and their foundations |
| `9407` | 1/6 | 6 | 557 | 704 | -2.989 | 0/0 | 1/6 | contextual substitute in pass handles | phrases related to self-reflection and personal growth |
| `12652` | 6/6 | 27 | 155 | 233 | -6.501 | 0/18 | 0/0 | common support; removals can tie but not pass | elements related to API requests and responses in code |
| `12704` | 6/6 | 9 | 580 | 9501 | 0.000 | 0/1 | 0/0 | near-core; replacements mostly fail | features and elements related to software and its functionality |
| `13622` | 6/6 | 25 | 489 | 8641 | 0.000 | 0/20 | 0/0 | common support; removals can tie but not pass | phrases related to limits and convergence in mathematical contexts |
| `14991` | 6/6 | 7 | 1 | 1 | 152.319 | 0/0 | 0/0 | core-like; no one-swap replacement ties | queries and requests for confirmation or clarification in discussions |
| `15169` | 6/6 | 1 | 4 | 15 | 14.088 | 0/0 | 0/0 | core-like; no one-swap replacement ties | key phrases and indicators of parental concerns and decision-making processes |

## What This Says Mechanistically

The current pass class looks less like a clean semantic refusal circuit and
more like a signed first-token control bundle. The strongest local backbone
evidence is feature `15169`: it is the largest hologram-prompt delta, appears
in all six pass handles, and every one-swap replacement after dropping it
fails far below the gate. Feature `14991` is also core-like: it appears in
all pass handles, is the top fake-ID-family harmful delta, and no one-swap
replacement even ties after dropping it.

Several other common features are better described as support features, not
individually necessary semantic atoms. Dropping them often creates ties near
the `I`/`It` boundary, which means the residual state remains close to the
decision surface but loses enough signed composition to stop crossing it.

The variable features give the clearest equivalence-class evidence. Features
`6289`, `8775`, and `1338` can be exchanged in narrow contexts, while `7531`
and `9407` are contextual substitutes that produce new passing handles only
for specific source/drop combinations.

The broad harmful-prompt ranking adds an important generalization check.
Feature `14991` remains rank `1`, feature `15169` remains high at rank `15`,
and feature `1813` is rank `19`. Those three look like a general refusal-boundary
backbone rather than fake-ID-only features. By contrast, several
other pass-class members fall far down the broad ranking or have zero broad
delta: `6289`, `9149`, `12704`, and `13622` are examples. This supports a
two-part account: the handle combines a general donor-refusal backbone with
domain/local support features needed to tip the fake-ID final-newline state.

A key caution is that the autointerp labels are not safety-specific. Some
important features have labels about code, organization, or self-reflection;
their local logits and causal role matter more here than the surface label.
This supports the project's emerging thesis: merging moves a model across a
behavioral boundary through a small signed feature class, but the class is
not a simple human-readable refusal feature set.

## Output Artifact

- CSV table: `/home/gavin/model_merging/stage3/results/gemma2_2b_linear_merge_sae_11feature_identity_v0/critical11_feature_identity_table.csv`
- Broad harmful ranking: `/home/gavin/model_merging/stage3/results/gemma2_2b_linear_merge_sae_prompt_token_delta_rank_v0/default_paraphrase_guard_harmful_l20_final_newline_delta_abs_float32/PROMPT_TOKEN_DELTA_RANKING_SUMMARY.md`
- Decomposition summary: `stage3/results/GEMMA2_2B_LINEAR_MERGE_SAE_11FEATURE_BACKBONE_DECOMP_SUMMARY.md`
