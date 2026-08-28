# v2.3.18 Phase A2 — Beta-Tilt Test

Same committed per-date CSVs; alpha/ic/returns are 3-seed means per date. OLS on dependent (overlapping-window) observations — slopes/intercepts are descriptive; cross-fold consistency is the robustness axis, no p-values.

## A2-1 — Effective beta and decomposition (top5_ret = a + b × univ_ret)

| fold | beta | intercept a | mean univ | mean alpha | tilt (b-1)·E[univ] | residual |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.86 | -2.07%p | +4.86%p | -2.75%p | -0.68%p | -2.07%p |
| 2 | 1.56 | +6.83%p | +0.07%p | +6.86%p | +0.04%p | +6.83%p |
| 3 | 1.57 | +0.01%p | +5.13%p | +2.95%p | +2.95%p | +0.01%p |
| 4 | 2.28 | +1.49%p | +3.48%p | +5.95%p | +4.46%p | +1.49%p |
| 5 | 2.40 | +9.77%p | +4.45%p | +16.01%p | +6.24%p | +9.77%p |
| pooled | 1.63 | +3.60%p | +3.60%p | +5.89%p | +2.28%p | +3.60%p |
| 0 (non-binding) | 1.65 | +19.00%p | +9.91%p | +25.49%p | +6.49%p | +19.00%p |

## A2-2 — Down-market behavior (universe_ret <= 0 dates)

| fold | n_down | alpha on down | alpha on up |
|---:|---:|---:|---:|
| 1 | 5 | -3.53%p | -2.56%p |
| 2 | 12 | +2.47%p | +10.92%p |
| 3 | 6 | +0.04%p | +3.88%p |
| 4 | 5 | -1.53%p | +7.83%p |
| 5 | 5 | +1.39%p | +19.49%p |
| pooled | 33 | +0.35%p | +7.85%p |

## A2-3 — IC–market linkage

| fold | rho(ic, univ_ret) | alpha ic+/uv+ | ic+/uv- | ic-/uv+ | ic-/uv- |
|---:|---:|---:|---:|---:|---:|
| 1 | +0.36 | +4.03%p (10) | +4.43%p (1) | -9.14%p (10) | -5.51%p (4) |
| 2 | +0.59 | +16.52%p (10) | +8.22%p (5) | -7.75%p (3) | -1.64%p (7) |
| 3 | -0.52 | +8.18%p (5) | +7.13%p (4) | +2.34%p (14) | -14.14%p (2) |
| 4 | +0.35 | +15.06%p (13) | +5.41%p (2) | -5.60%p (7) | -6.16%p (3) |
| 5 | +0.81 | +17.78%p (19) | n/a (0) | +35.71%p (2) | +1.39%p (5) |
| pooled | +0.26 | +13.68%p (57) | +7.07%p (12) | -1.38%p (36) | -3.49%p (21) |
