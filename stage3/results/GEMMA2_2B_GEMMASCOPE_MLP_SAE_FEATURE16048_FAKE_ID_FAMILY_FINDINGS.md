# GemmaScope Feature-16048 Fake-ID Family Findings

Date: 2026-05-22

This checkpoint tests whether the L19 feature `16048` generated-token effect
generalizes from the original fake-ID prompt to a small fake-ID paraphrase
family.

## Setup

- Donor/base: `google/gemma-2-2b-it`
- Recipient: `IlyaGusev/gemma-2-2b-it-abliterated`
- Basis: `0:4`, prefix k896, layers `12-20`
- Prompt family:
  `stage3/data/gemma2_feature16048_family_prompts/fake_id_family_v0.jsonl`
- Family size: 8 harmful fake-ID variants and 8 benign ID/safety prompts

## Result

| condition | harmful clean | unsafe | benign helpful | fake-ID-family ok |
|---|---:|---:|---:|---:|
| k896 prefix only | 0.750 | 0.000 | 1.000 | 0.750 |
| k896 + f16048 at assistant boundary | 0.750 | 0.000 | 1.000 | 0.750 |
| k896 + f16048 at generated tokens | 0.750 | 0.125 | 1.000 | 0.750 |
| k896 + f16048 at boundary/generated | 0.750 | 0.125 | 1.000 | 0.750 |

Feature `16048` does not improve the fake-ID family pass rate. The generated
and boundary/generated variants introduce one unsafe continuation on the
hologram/lamination prompt.

## Interpretation

This is a narrowing result. The strong feature-16048 effect on the original
fake-ID prompt is not a broad fake-ID-domain semantic feature. On this small
paraphrase family, the k896 prefix already handles six of eight prompts, and
feature `16048` does not repair the remaining two.

The current best claim is therefore:

- feature `16048` is causally important for a specific generated-token refusal
  trajectory in the original benchmark prompt;
- it is not yet evidence for a reusable fake-ID refusal feature;
- family-level generalization remains unproven and should be tested before any
  paper claim uses semantic labels like "fake-ID feature."

## Artifacts

- Prompt family:
  `stage3/data/gemma2_feature16048_family_prompts/fake_id_family_v0.jsonl`
- Result root:
  `stage3/results/gemma2_2b_gemmascope_mlp_sae_feature16048_family_fake_id_v0/`
