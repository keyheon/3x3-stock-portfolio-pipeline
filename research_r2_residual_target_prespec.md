# Research Pre-Spec — R2: Residualized-Target Three-Arm Screen

**Status**: pre-specified (anchors fixed before any R2 computation)
**Date**: 2026-09-08
**Track**: R2 — second study of the optional-research branch (R1 closed null). Derived from the v2.3.18 diagnosis (unstable beta tilt; disposition: "beta-adjusted alpha as primary") pushed one level deeper: adjust not just the scoring but the **learning target**.

**Design note (recorded before run)**: a first draft of this pre-spec had two arms (base / resid). Review found a confound: with a per-date-demeaned target, every macro feature (constant across tickers within a date) has exactly zero correlation with the target and is therefore dropped by the corr filter (0.0558). The resid arm would have bundled two treatments — target reshaping and macro removal — with different implications (macro removal → revive the v2.3.3 "disentangle macro" item; target reshaping → market-neutral learning). A third arm isolates them. The two-arm draft was never run.

## Question

Does removing the per-date market-common component from the training target improve time-axis ranking, and if so, is the effect due to the target itself or to the macro removal it implies? Mechanism under test: Y_ret = (market component, non-stationary, unpredictable — momentum IC ≈ 0, 2023 inversion) + (relative component, where all measured skill lives). A per-date constant is rank-invariant, so any effect must come through learning: shared-trunk gradient budget spent on level variance (auxiliary-task interference) and level-driven feature selection.

## Arms (seed 42; binding folds 1–5; N=20; everything else byte-identical to v2.3.17)

- **A base** — raw `Y_ret`, 97 input features. Identical to v2.3.17; reused from committed seed-42 results iff the equivalence check passes.
- **B resid** — target `Y_resid(i,t) = Y_ret(i,t) − mean_{j∈train} Y_ret(j,t)` (per snapshot date, over the fold's train samples), applied before feature selection and before fit/val split. Input 97; macro features expected to drop at selection (logged and verified at runtime). `Y_risk` untouched.
- **C nomacro** — raw `Y_ret`; the 43 macro features (names containing `macro_`, `ff_`, `xasset_`) removed from the input before selection (54 remain; count asserted at runtime).
- Test-side metrics identical in all arms on **raw realized returns** (per-date Spearman IC; top-5 alpha). MAE is scale-incomparable across arms and excluded.

Contrasts: **B−A** (full residual pipeline), **C−A** (macro removal alone), **B−C** (target reshaping net of macro removal).

## Equivalence check and arm-A reuse

Run `--arm base --folds 1 --seed 42` on the executing machine; compare per-date ic/alpha with the committed `results/walkforward_v2317/seed42/fold1_dates.csv`. **Reuse bound**: fold-level |ΔIC| ≤ 0.005. If exceeded, arm A is rerun in full on the executing machine and committed results are not mixed in. The queue should run on the machine that produced v2.3.17 (desktop) to minimize float drift.

## Statistics and anchors (judgment aids; screen, not a gate)

Per contrast, paired per-fold ΔIC over binding folds 1–5:
- **PROMISING** = mean ΔIC ≥ +0.02 AND ΔIC > 0 in ≥ 4/5 folds.
- **DEAD** = mean ΔIC ≤ 0 OR positive folds ≤ 2/5.
- Otherwise **MIXED**.

Seed-noise context: v2.3.17 per-seed binding-mean ICs were +0.0053/+0.0148/+0.0071 (std ≈ 0.005), so +0.02 ≈ 4× that dispersion; a single-seed effect below it is not distinguishable from seed luck. Secondary (reporting only): Δalpha (raw-return alpha, comparable), Δ ic_positive_date_frac, per-fold tables, per-date CSVs committed as evidence, and the runtime feature-count logs (macro survivors per arm).

## Decision map

- B−A PROMISING and C−A PROMISING with B−C ≈ 0 → driver is macro removal → R2-full = 3-seed no-macro study + revival of the "route macro through the regime layer only" design.
- B−A PROMISING and C−A not → driver is target reshaping → R2-full = 3-seed residual study, with beta-precise residualization (per-stock rolling beta; needs daily prices) as the refinement.
- Both DEAD → both cards close; record; proceed to R3 (EDGAR fundamentals feasibility).
- MIXED → record; any continuation requires its own amendment.

Explicitly out of scope for this screen (named now, possible R2-full variants): rank-transformed target, sector-neutral residualization (changes the ranking target itself), shorter horizon (needs cache rebuild), quarterly retraining cadence. The v2.3.17 FAIL and the v2.3.18 disposition remain in force regardless of this screen's outcome; only a 3-seed R2-full could feed any future gate discussion, which would additionally require dependence-aware statistics and beta-adjusted alpha.

**Prior, stated in advance: medium-low** for B−A; low for C−A (v2.3.3 measured macro's cross-sectional harm, but its time-axis role under the NLL architecture is unmeasured). A clean null is a cheap, useful result.
