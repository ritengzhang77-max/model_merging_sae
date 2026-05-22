# Qwen2.5-1.5B Second-Stage Residual PCA Generation

Prompt modes: `qwen_residual_benchmark_v0`

| prompt mode | model | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |
|---|---|---:|---:|---:|---:|---:|---:|
| `qwen_residual_benchmark_v0` | `pca64` | 0.000 | 0.333 | 0.333 | 0.333 | 1.000 | 0.000 |
| `qwen_residual_benchmark_v0` | `pca64_plus_residual_topk1024_16-23` | 0.333 | 0.333 | 0.000 | 0.667 | 1.000 | 0.000 |
| `qwen_residual_benchmark_v0` | `pca64_plus_residual_topk2048_16-23` | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| `qwen_residual_benchmark_v0` | `pca64_plus_residual_raw_pca64_16-23` | 0.000 | 0.333 | 0.333 | 0.333 | 1.000 | 0.000 |
| `qwen_residual_benchmark_v0` | `pca64_plus_residual_centered_pca64_16-23` | 0.333 | 0.333 | 0.000 | 0.333 | 1.000 | 0.000 |
| `qwen_residual_benchmark_v0` | `pca64_plus_residual_mean_vec_16-23` | 0.000 | 0.333 | 0.333 | 0.667 | 1.000 | 0.000 |
| `qwen_residual_benchmark_v0` | `pca64_plus_full_16-23` | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
