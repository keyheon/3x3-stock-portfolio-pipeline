# R2 verdict (seed 42, binding folds 1-5; anchors per pre-spec)

| fold | A base IC | B resid IC | C nomacro IC | A alpha | B alpha | C alpha |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -0.0085 | -0.0386 | +0.0168 | -3.19%p | -4.58%p | -5.47%p |
| 2 | +0.0088 | +0.0192 | +0.0015 | +6.22%p | +5.03%p | +7.32%p |
| 3 | -0.0882 | +0.0769 | +0.1160 | +4.96%p | +26.41%p | +22.08%p |
| 4 | -0.0054 | -0.0259 | +0.0100 | +6.27%p | +7.13%p | +9.21%p |
| 5 | +0.1199 | +0.1033 | +0.1149 | +15.07%p | +18.01%p | +32.13%p |

## Contrasts (paired per-fold delta IC)

| contrast | f1 | f2 | f3 | f4 | f5 | mean | positive | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B-A resid-base | -0.0301 | +0.0104 | +0.1651 | -0.0205 | -0.0166 | **+0.0217** | 2/5 | **DEAD** |
| C-A nomacro-base | +0.0254 | -0.0073 | +0.2042 | +0.0154 | -0.0049 | **+0.0465** | 3/5 | **MIXED** |
| B-C resid-nomacro | -0.0555 | +0.0178 | -0.0391 | -0.0359 | -0.0117 | **-0.0249** | 1/5 | **DEAD** |

## Arm aggregates (single seed — NOT a gate result)

| arm | mean IC | positive folds | mean alpha |
|---|---:|---:|---:|
| A base (v2.3.17 s42) | +0.0053 | 2/5 | +5.86%p |
| B resid | +0.0270 | 3/5 | +10.40%p |
| C nomacro | +0.0518 | 5/5 | +13.05%p |

Single-seed aggregates are reported for the record only; the v2.3.17 gate thresholds do not apply to a new arm post hoc.
