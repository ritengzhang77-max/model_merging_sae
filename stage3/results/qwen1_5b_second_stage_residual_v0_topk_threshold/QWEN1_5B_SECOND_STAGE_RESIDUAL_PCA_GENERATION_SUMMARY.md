# Qwen2.5-1.5B Second-Stage Residual PCA Generation

Prompt modes: `qwen_residual_benchmark_v0`

| prompt mode | model | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen_residual_benchmark_v0` | `pca64_plus_residual_topk1280_16-23` | 0.667 | 1.000 | 0.333 | 0.000 | 1.000 | 0.000 |
| `qwen_residual_benchmark_v0` | `pca64_plus_residual_topk1536_16-23` | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `qwen_residual_benchmark_v0` | `pca64_plus_residual_topk1792_16-23` | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
