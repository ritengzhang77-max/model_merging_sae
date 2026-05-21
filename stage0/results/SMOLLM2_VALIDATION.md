# SmolLM2 Stage 0B Validation

Run date: 2026-05-07

Cache:

- `HF_HOME=/data/gavin/model_merging/hf_cache`

Downloaded:

- `HuggingFaceTB/SmolLM2-135M`
- `HuggingFaceTB/SmolLM2-135M-Instruct`

Validation:

- Base checkpoint loaded with `AutoModelForCausalLM`.
- Instruct checkpoint loaded with `AutoModelForCausalLM`.
- Base raw prompt generated the correct continuation for `2 + 3`.
- Instruct raw prompt stopped early, but chat-template prompting worked:
  it answered `2 + 3 is 5`.

Conclusion:

- These are valid Stage 0B transformer targets.
- Use chat templates for the instruct checkpoint.
- For controlled merging, prefer creating our own small experts from
  `SmolLM2-135M` first, then compare to the existing instruct delta.

