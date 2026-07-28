# Pre-Registration — v2.3.17 Purged Walk-Forward Re-Validation (Deployment Gate)

**Status**: pre-registered (written before any walk-forward run)
**Date**: 2026-07-28
**Depends on**: v2.3.16 Amendment 1 (IMPROVEMENT; Trial 52 adopted as production, commit 2db6985).
**Gates**: Phase 1 capital deployment and the v2.4.0 version bump are conditional on this document's §4 verdict.

## 1. Purpose

Every validation to date — including the v2.3.16 supersession — used GICS-stratified K-fold, which splits by ticker and mixes time. That answers "can the model rank unseen tickers within seen periods?" Deployment asks a different question: **trained only on the past, does the model rank the future?** The two are known to dissociate in this pipeline (macro features helped Walk-Forward CV while hurting stratified K-fold, v2.3.3 §33), and no time-ordered measurement of the production configuration exists.

This gate is therefore **absolute, not comparative**: it tests whether the adopted production config (Trial 52) shows a deployable time-ordered signal. There is no incumbent arm — the config comparison was settled in v2.3.16.

One structural strength is claimed up front: hyperparameter selection (v2.3.15, 60 trials) and the supersession decision (v2.3.16) both used only the cross-sectional axis. The time axis was never touched by selection, so this gate evaluates the chosen config on an orthogonal validation axis — partially controlling the backtest-selection bias of testing a searched-over configuration on the axis it was searched on (Bailey & López de Prado).

## 2. Pre-registered run specification (locked)

**Scheme: purged expanding walk-forward, calendar-year folds.**

| Fold | Test period | Train period | Status |
|---|---|---|---|
| 0 | 2020 | 2016-07 → (2020 start − 63 trading days) | **Non-binding** (reported, excluded from §4) |
| 1 | 2021 | 2016-07 → (2021 start − 63 td) | Binding |
| 2 | 2022 | 2016-07 → (2022 start − 63 td) | Binding |
| 3 | 2023 | 2016-07 → (2023 start − 63 td) | Binding |
| 4 | 2024 | 2016-07 → (2024 start − 63 td) | Binding |
| 5 | 2025 (+ 2026-01 tail) | 2016-07 → (2025 start − 63 td) | Binding |

- **Purge gap 63 trading days** between train end and test start: the target is the 63-day forward return, so the last unpurged train labels would contain test-period information (label-overlap leakage; López de Prado 2018). Purge is applied to snapshot dates — a train snapshot is admitted only if its date + 63 td precedes the test period start.
- **Expanding window** (not rolling): each fold trains on all history up to the purge boundary, mirroring the deployment procedure (train on everything available, rank the next quarter).
- **Fold 0 non-binding, pre-specified**: its train span is 3.3 years (~40k samples), so a poor result cannot be attributed to regime failure vs data scarcity (confounded measurement). It is run and fully reported (§5) but excluded from the §4 verdict. This is declared here, before any run, with the confound as the stated reason; the COVID-crash regime result is recorded either way rather than hidden.
- **Config**: production Trial 52 exactly as adopted (large [128,64,32], lr 1.5636e-3, wd 4.4856e-5, var_thr 1.0819e-3, corr_thr 5.5843e-2, dropout 0.1146), heteroscedastic dual-head Gaussian NLL, log-space risk target, N = 20 ensemble, epoch cap 20000 / patience 41 — per the standing rule that diagnostics use the production architecture (v2.3.12 §110.2).
- **Feature selection**: computed from each fold's train set only (thresholds as above), consistent with the v2.3.16 per-fold pattern.
- **Universe**: SNDK excluded; cache `results/backtest_cache.npz` (122,240 × 97, 2016-07-21 – 2026-01-16), identical to v2.3.15/16.
- **Seeds**: {42, 1, 2}, pre-fixed, applied to all folds. Data partition (the calendar splits) is deterministic and identical across seeds; seed varies training stochasticity only. Verdict quantities in §4 are 3-seed means.
- **Rebalance frequency modeled**: one training per fold = annual retraining. Live deployment plans quarterly retraining; quarterly simulation (~4× compute) is out of budget. This is a fidelity limitation (§7), not claimed as conservative.

Budget: 3 seeds × 6 folds × 20 NN = 360 NN trainings; expanding windows make early folds cheap. Estimated ~20–28 h total on desktop.

## 3. Metrics (definitions locked)

**Primary A — per-date IC.** For each snapshot date in the test period, Spearman rank correlation between predicted and realized 63-day forward returns across all test-period tickers with a snapshot at that date. Fold IC = mean over dates; run IC = mean over binding folds. This is the Information Coefficient in the Grinold & Kahn sense, computed at the same cadence a live rebalance would face. (Distinct from the v2.3.16 per-ticker-aggregate rank_corr, which collapses time.)

**Primary B — top-5 selection alpha.** For each snapshot date, select the 5 highest-predicted tickers; alpha = mean realized 63-day return of the 5 − mean realized return of all tickers at that date. Fold alpha = mean over dates. This addresses the v2.3.16 Amendment 1 carry (Trial 52 improved ranking while alpha mildly reversed): deployment trades the top 5, not the full ranking.

**Secondary (non-binding, reported)**: ICIR (fold IC / std of per-date ICs — signal stability); alpha vs SPY (Task B infrastructure); time-ordered momentum-baseline IC (early-half signal → late-half realized, no training required — does the NN beat naive momentum on the time axis?); return MAE (continuity with the historical WF-CV, which measured 11.9%p at v2.3.2 under the old architecture — indicative only, not comparable across architectures); all Fold 0 quantities.

## 4. Primary decision rule (mechanical)

Computed on binding folds 1–5, 3-seed means:

| Verdict | Condition | Action |
|---|---|---|
| **PASS** | mean IC ≥ **0.05** AND fold-IC positive in **≥ 4 of 5** binding folds AND mean top-5 alpha > **0.0062** (0.62%p/quarter = TC_est) | Deployment gate open: proceed to Phase 1 capital entry and v2.4.0. |
| **PARTIAL** | IC conditions pass, alpha condition fails | Time-ordered signal exists but does not survive top-5 conversion net of costs. Deployment on hold; diagnose the selection layer (ranking→top-5); re-gate after a documented change. |
| **FAIL** | any IC condition fails | No deployable time-ordered signal. Deployment on hold indefinitely; diagnose the cross-sectional↔time-axis dissociation before further model work. |

**TC_est = 0.62%p/quarter** is measured, not assumed: v2.3.17 rerun of `transaction_cost_analysis.py` on the Trial 52 v2.3.16 selections (`results/tc_analysis_v2316.md`, ₩5M-position × quarterly-rebalance cell; KIS route — commission 0.04%/side, FX 0.10%/side, bid-ask 0.05%/0.20% by cap, cube-root impact; 24 of 25 unique picks above the $10B large-cap threshold). The ₩5M cell is the upper-bound cell across candidate Phase 1 sizings, so the threshold is independent of the final position-size decision.

**Threshold provenance (honest label)**: IC ≥ 0.05 is the practitioner convention for a usable signal in the Grinold & Kahn tradition (0.10+ strong); ≥4/5 sign consistency requires a majority-plus without demanding survival of every single-year regime. Both are literature-anchored defensible choices, **not thresholds validated on this dataset** — no time-ordered IC measurement of this pipeline existed before this study, which is precisely why the gate exists. Cross-sectional values (~0.54) are not comparable and are not the expectation; time-axis ICs are conventionally an order of magnitude smaller.

**No post-hoc adjustment.** A near-miss (e.g., mean IC 0.045 with 5/5 positive folds) is a FAIL under this rule; the recourse is diagnosis and a new pre-registered gate, not threshold revision.

## 5. Secondary corroboration (non-binding)

- Fold 0 (2020) full metrics — the COVID stress result, on the record either way.
- Momentum-baseline IC per fold: if the NN's IC does not exceed naive momentum's on the time axis, a PASS is technically valid but the deployment narrative weakens; flag for the Phase 1 write-up.
- Per-fold alpha dispersion: if the §4 alpha condition passes on the strength of one fold while others are ≤ TC_est, flag as fragile before capital entry.
- ICIR: no threshold, recorded for the v2.4.0 audit.

## 6. Outcome consequences

- **PASS** → record Amendment 1; refresh the regime check (stale since v2.3.2-era checks); run the production pipeline at Trial 52 for a current top-5; Phase 1 entry per the standing decision tree (position sizing is a deployment decision outside this gate); v2.4.0 version bump with the full audit.
- **PARTIAL / FAIL** → record Amendment 1 with the specific failed condition and margins; no capital entry; open the corresponding diagnosis track. v2.3.12→Trial 52 adoption (v2.3.16) is not reversed by this gate — it was a cross-sectional decision and remains valid on that axis.

## 7. Limitations (declared before the run)

- **n = 3 seeds** bounds training-stochasticity variance only; no inferential claim.
- **5 binding folds** = 5 calendar years; the sign-consistency requirement is coarse at this n. One year per regime type is a thin sample of regimes.
- **Annual retraining modeled vs quarterly planned**: fidelity gap, direction of bias unknown (fresher models could rank better or overfit recent noise). Not claimed conservative.
- **Fold 0 exclusion** is pre-specified with a confound rationale, but the designation itself is a judgment call; the mitigations are full reporting (§5) and pre-specification.
- **TC_est** uses current ADV/mcap/FX (not historical), inherits the v2.3.11 model's assumptions, and is measured on cross-sectional-fold picks rather than walk-forward picks (unavailable before this run); dominated by cap-structure-stable components (FX + commission), so pick-set sensitivity is second-order (0.64 → 0.62%p across the v2.3.12 → Trial 52 pick change).
- **Survivorship bias** (Task #8, open): the universe is the current index membership; walk-forward does not cure this. Alpha levels are optimistic to an unquantified degree; the IC sign/level conditions are less exposed than alpha levels.
- **Single-cache provenance**: all folds derive from one cache build; data errors would be shared across folds, not caught by this design.

## 8. Commit discipline

This pre-registration is committed before `walk_forward_v2317.py` is written and before any run. The implementation must match §2–§3; any deviation discovered during implementation is recorded as a rule-preserving amendment *before* the full run (smoke tests excepted). Results are recorded afterward as **Amendment 1** in a separate commit that changes no rules. Per-seed, per-fold metrics for both primaries and all secondaries are force-added as evidence (v2.3.15 best_trials precedent).
