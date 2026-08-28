# v2.3.18 Phase A — Tail-Alpha Decomposition

Input: committed v2.3.17 per-date CSVs (3 seeds, folds 0-5). Alpha/IC below are 3-seed means per date unless noted.

## D1 — Time concentration (per-date alpha, seed-mean)

| fold | n_dates | pos_frac | mean | median | top3_share |
|---:|---:|---:|---:|---:|---:|
| 1 | 25 | 0.40 | -2.75%p | -4.98%p | -97% |
| 2 | 25 | 0.68 | +6.86%p | +7.37%p | 55% |
| 3 | 25 | 0.56 | +2.95%p | +0.38%p | 105% |
| 4 | 25 | 0.52 | +5.95%p | +1.29%p | 77% |
| 5 | 26 | 0.81 | +16.01%p | +12.89%p | 36% |
| pooled | 126 | 0.60 | +5.89%p | +3.38%p | 21% |

## D2 — Seed structure (pairwise Spearman of per-date alpha)

| fold | 42-1 | 42-2 | 1-2 | mean |
|---:|---:|---:|---:|---:|
| 1 | +0.84 | +0.78 | +0.86 | +0.83 |
| 2 | +0.90 | +0.97 | +0.93 | +0.93 |
| 3 | +0.63 | +0.76 | +0.60 | +0.66 |
| 4 | +0.91 | +0.95 | +0.92 | +0.92 |
| 5 | +0.67 | +0.87 | +0.72 | +0.75 |
| pooled | +0.81 | +0.87 | +0.84 | +0.84 |

## D3 — Alpha–IC decoupling (per-date, seed-mean)

| fold | rho(alpha, ic) | n IC<=0 dates | mean alpha on IC<=0 |
|---:|---:|---:|---:|
| 1 | +0.49 | 14 | -8.10%p |
| 2 | +0.56 | 10 | -3.47%p |
| 3 | +0.29 | 16 | +0.28%p |
| 4 | +0.50 | 10 | -5.77%p |
| 5 | +0.50 | 7 | +11.20%p |
| pooled | +0.52 | 57 | -2.16%p |

## D4 — Strongest folds

Fold 5 (2025, binding): pos_frac 0.81, median +12.89%p, top3_share 36%
Fold 0 (2020, NON-BINDING context): pos_frac 0.85, median +20.79%p, top3_share 38%

## D5 — Per-date alpha autocorrelation (overlap-driven dependence)

| fold | lag1 | lag2 | lag3 | lag4 | lag5 | lag6 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | +0.58 | +0.39 | +0.15 | +0.14 | -0.08 | -0.28 |
| 2 | +0.48 | +0.12 | +0.03 | +0.06 | -0.07 | -0.13 |
| 3 | +0.32 | +0.32 | +0.39 | -0.08 | -0.10 | -0.10 |
| 4 | +0.43 | +0.20 | +0.08 | -0.05 | -0.11 | -0.09 |
| 5 | +0.24 | +0.28 | +0.16 | -0.02 | -0.16 | -0.24 |

High positive low-lag autocorrelation is expected (63-td forward windows on a ~10-td grid overlap ~6 deep); any future gate must use dependence-aware statistics per the pre-spec.
