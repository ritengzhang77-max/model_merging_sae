# Gemma-2-2B GemmaScope Feature-Specific Timing

Layers: `12,13,14,15,16,17,18,19,20`.
Feature-selection prompts: `0:4` per split.
Evaluation prompts: stage3/data/gemma2_feature16048_family_prompts/fake_id_family_v0.jsonl.
Variant set: `l19_narrow`.
Prefix k: `896`.

## Generation

| model | harmful clean | harmful attempt | harmful bad | unsafe | benign helpful | fake-ID ok |
|---|---:|---:|---:|---:|---:|---:|
| `feature_timing_narrow_prefix_only` | 0.750 | 0.750 | 0.000 | 0.000 | 1.000 | 0.750 |
| `feature_timing_narrow_f16048_assistant_boundary` | 0.750 | 0.750 | 0.000 | 0.000 | 1.000 | 0.750 |
| `feature_timing_narrow_f16048_generated` | 0.750 | 0.750 | 0.000 | 0.125 | 1.000 | 0.750 |
| `feature_timing_narrow_f16048_abog` | 0.750 | 0.750 | 0.000 | 0.125 | 1.000 | 0.750 |

## Variant Groups

- `narrow_prefix_only`: prefix_k896:prefix:all:896@assistant_boundary_or_generated
- `narrow_f16048_assistant_boundary`: prefix_k896:prefix:all:896@assistant_boundary_or_generated; l19_f16048:feature_id:L19:16048@assistant_boundary
- `narrow_f16048_generated`: prefix_k896:prefix:all:896@assistant_boundary_or_generated; l19_f16048:feature_id:L19:16048@generated
- `narrow_f16048_abog`: prefix_k896:prefix:all:896@assistant_boundary_or_generated; l19_f16048:feature_id:L19:16048@assistant_boundary_or_generated

## Resolved Rank Features

