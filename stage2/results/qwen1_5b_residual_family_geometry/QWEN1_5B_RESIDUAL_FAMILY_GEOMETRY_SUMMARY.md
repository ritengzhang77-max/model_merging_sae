# Qwen2.5-1.5B Residual Family Geometry

Residual is computed after subtracting the PCA64 projection from donor-recipient MLP deltas.

## Coordinate Concentration

| group | residual norm/token | top256 energy | top1024 energy |
|---|---:|---:|---:|
| `benign_account_recovery` | 8.681 | 0.259 | 0.785 |
| `benign_remove_tracking` | 8.547 | 0.262 | 0.786 |
| `one_time_code` | 9.997 | 0.267 | 0.790 |
| `permission_slip` | 8.172 | 0.264 | 0.787 |
| `tracking_original` | 7.893 | 0.267 | 0.788 |

## Pairwise Residual Similarity

| group A | group B | mean cosine | top256 Jaccard | top1024 Jaccard | A energy in B top256 | B energy in A top256 |
|---|---|---:|---:|---:|---:|---:|
| `benign_account_recovery` | `one_time_code` | 0.708 | 0.347 | 0.646 | 0.210 | 0.207 |
| `benign_remove_tracking` | `tracking_original` | 0.943 | 0.571 | 0.759 | 0.244 | 0.251 |
| `one_time_code` | `permission_slip` | 0.738 | 0.377 | 0.664 | 0.210 | 0.220 |
| `one_time_code` | `tracking_original` | 0.698 | 0.341 | 0.651 | 0.204 | 0.218 |
| `permission_slip` | `tracking_original` | 0.835 | 0.441 | 0.684 | 0.226 | 0.233 |

## Reading This

- Low top-coordinate overlap between harmful families supports prompt-family-specific residual mechanisms.
- High concentration with low behavioral repair means coordinate energy alone is not sufficient.
- Benign comparisons are controls for whether the residual coordinates are just prompt-template or topic coordinates.

Top-k settings analyzed: `64, 256, 512, 1024`.
