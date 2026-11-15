# Gate variance threshold calibration

Measured values from `scripts/evaluation/analyze_embeddings.py`, run on
2026-04-02 before the reported training runs. Only the measurements are kept
here; see the paper for how the gate is used and evaluated.

- Embeddings: HistoEncoder (XCiT-S12, 384-d) bags of PANDA slides
- Sample: 200 random slides, 101,335 patch tokens

## Patch-token L2 norm

| Statistic | Value |
| --- | --- |
| Mean | 34.26 |
| Std | 0.66 |
| Min | 31.79 |
| Max | 37.45 |
| Median | 34.29 |
| P1 | 32.73 |
| P99 | 35.66 |
| P99 / mean | 1.05 |

1.44% of tokens exceeded the per-slide μ + 2σ norm cutoff.

## Laplacian spatial variance of the patch

| Statistic | Value |
| --- | --- |
| Mean | 837.04 |
| Std | 731.46 |
| Min | 0.00 |
| Max | 5455.15 |
| Median | 553.00 |
| P1 | 13.85 |
| P5 | 77.54 |
| P10 | 122.19 |
| P25 | 236.97 |

The variance threshold was set to τ = 122.0, the 10th percentile, replacing
the earlier value of 15.0.

## Tokens flagged by both conditions

A token is flagged when its L2 norm exceeds the per-slide μ + 2σ cutoff and
its spatial variance is below τ.

| Variance threshold | Tokens flagged | Share |
| --- | --- | --- |
| 15.0 | 130 | 0.128% |
| 50.0 | 195 | 0.192% |
| 122.0 | ~399 | ~0.4% |

These counts use the standard (mean/std) norm statistic on HistoEncoder
embeddings. All reported experiments use the robust (median/MAD) gate, and
gate firing was not measured during those runs.
