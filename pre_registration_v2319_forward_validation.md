# Pre-Registration — v2.3.19 Forward Hold-Out Validation

**Status**: pre-registered upon commit; binding before any hold-out snapshot is scored.
**Date**: 2026-09-10
**Purpose**: The project's product is a **pre-registered validation record, an open package, and a manuscript** — not a live portfolio. This study gives the no-macro configuration, which was selected on the 2021–2025 walk-forward folds (R2 Amendment 1: mean IC +0.0513, 5/5, seed std 0.0004) and therefore has no inferential standing on those folds, a test on data that no selection has touched. Its outcome is a **claim the manuscript may or may not make**, decided by rules fixed here.

**Subject**: Trial 52 hyperparameters, raw 63-day target, the 43 macro features removed before selection, N=20 ensemble × seeds {42, 1, 2}. **Secondary tracked configurations** (scored and reported, not gated): the v2.3.17 raw base; the R3 fundamentals arm if R3 returns PROMISING (added by amendment before the first scoring run).

## 1. What this validation can and cannot establish

Every study since v2.3.16 read the same five test years. With ~26 hold-out dates per year and 63-day outcome windows on a 10-day grid (effective independent windows ≈ n/6), a one-year hold-out cannot deliver a conventional significance test for an IC near 0.05. The validation therefore rests on **pre-registration + true out-of-sample**, not on p-values; its thresholds are **decision rules for what the manuscript claims**, labeled as such. This is the honest cost of having selected on the sample.

## 2. Selection-clean boundary — purged

All selection used `backtest_cache.npz` (built 2026-05-30; last snapshot 2026-01-16; last realized outcome ≈ 2026-04-20). The fold-5 results that drove selection depend on market outcomes through that date, so hold-out snapshots are those with **t ≥ 2026-04-21** (first date whose outcome window begins after the last selection-touched outcome). Training snapshots (≤ 2026-01-16) then have no outcome overlap with hold-out windows. The unpurged window (2026-01-17 … 2026-04-20) is scored and reported as *secondary, path-contaminated* context only. Hold-out dates accrue at ≈ 2 per month.

## 3. Frozen model and frozen pipeline

- Train once on every cache sample with snapshot ≤ 2026-01-16 (SNDK excluded), no-macro configuration, feature selection on the full training set per seed; **save weights, selected-feature masks and config** under `results/gate_v2319/model_seed{42,1,2}/`; record the commit hash of the feature-construction code. No retraining during the validation (a retrain is a new pre-registration).
- Hold-out features are built by the same committed code from refreshed data; the universe is **frozen to the 525 cache tickers** (constituent changes not followed; delistings drop out). Snapshot grid unchanged; per-date cross-sections require ≥ 30 tickers.
- Score = ensemble mean of the 60 return-head means per (ticker, date).

## 4. Metrics

- **Primary A — per-date IC**: Spearman across tickers at each hold-out date; quantity = mean over dates; positive-date fraction reported.
- **Primary B — beta-adjusted top-5 alpha**: top-5 by prediction per date; realized 63-day returns; across dates, top5_ret = a + b·universe_ret (equal-weight universe, as in v2.3.18 A2); quantity = intercept **a**; raw alpha and b reported. Estimated at the final n only.
- **Economic-significance bar for B**: a > TC_est, TC re-measured on the frozen model's current top-5 with `transaction_cost_analysis.py` (₩5M × 4-turn cell as in v2.3.17, a modeling convention, not a deployment plan) before the first scoring run and fixed by amendment.
- **Secondary (reporting only)**: momentum-baseline IC (return_180d); the tracked configurations above; ICIR; per-date CSVs committed monthly as evidence.

## 5. Dependence-aware statistics

No per-date t-tests. Reported: n dates; n_eff = number of non-overlapping 63-td windows (≈ n/6); moving-block bootstrap (block = 6 dates) 90% intervals for mean IC and for a — **descriptive**, shown beside the decision rule, never substituted for it.

## 6. Decision rule and timing — fixed before any scoring

- **Verdict**: at the first scoring run with **n ≥ 26 hold-out dates** realized (expected ≈ April–May 2027). Monthly interim scoring is reporting only and triggers nothing. No configuration change during the validation.
- **PASS** = mean IC ≥ 0.05 AND positive-date fraction ≥ 0.60 AND a > TC_est. The IC bar stays at the v2.3.17 level — thresholds are not lowered to fit a smaller sample; 0.60 replaces the fold rule because there are no folds.
- **PARTIAL** = IC passes, a fails.
- **FAIL** = IC fails. Near-miss is FAIL. No post-hoc adjustment.
- **What each outcome licenses the manuscript to say**: PASS — "the no-macro configuration's time-axis ranking signal replicated on a pre-registered, selection-clean forward hold-out (n = …, n_eff ≈ …), with the stated low power." PARTIAL — the ranking claim only; the selection-layer claim withheld. FAIL — "the in-sample improvement from macro removal did not replicate forward"; the negative-result framing of v2.3.17–18 stands as the paper's conclusion. A second scoring at n ≥ 52 (≈ 2028) is pre-declared as a confirmation checkpoint under the same rules, for a follow-up or revision.

## 7. Amendments

Recorded in separate commits; no rule changes after the first scoring run. Implementation deviations discovered while building the freeze/scoring tooling are recorded before that run.
