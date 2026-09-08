# Research Pre-Spec — R3-full: Point-in-Time Fundamentals as an Orthogonal Information Axis

**Status**: pre-specified (written before any fundamentals feature is evaluated against returns; before R2 results are known)
**Date**: 2026-09-08
**Track**: R3-full, licensed by the R3 spike (FEASIBLE 18/20, median first-filing lag 30 d).

## Hypothesis

The price/macro sheet carries no next-quarter ranking information on the time axis (v2.3.17; momentum IC ≈ 0). Fundamentals are an information axis **orthogonal to price** — not forward-looking, but new. The test: does adding point-in-time profitability / investment / accrual features to the training sheet raise time-axis per-date IC?

Honest prior, stated in advance: **low-to-medium**. Quality/investment premia exist in the literature (Novy-Marx 2013; Fama & French 2015 RMW/CMA; Sloan 1996) but are themselves regime-dependent, and the corr-filter evidence (below) predicts they enter the model with weak linear signal.

## Features (9; price-free by design — value factors needing market cap are deferred to R3b)

All computed per (ticker, snapshot date t) from filings with **first filing date < t** (strictly before the snapshot; first-filed values only, restatements ignored — no look-ahead). Flow items are trailing-twelve-month (TTM) sums of the 4 most recent quarterly values; quarterly values are direct 80–100-day periods or derived by same-start differencing (YTD − prior YTD; FY − 9M for Q4). Stock items are the latest instant value; "4q-ago" is the instant value whose period end is nearest one year before the latest.

1. `f_roe` = NI_ttm / equity_latest
2. `f_net_margin` = NI_ttm / revenue_ttm
3. `f_asset_growth` = assets_latest / assets_4q_ago − 1
4. `f_accruals` = (NI_ttm − CFO_ttm) / assets_latest
5. `f_rev_growth` = revenue_ttm / revenue_ttm_prior − 1 (prior = the 4 quarters before the TTM window)
6. `f_cfo_assets` = CFO_ttm / assets_latest
7. `f_leverage` = equity_latest / assets_latest
8. `f_ni_growth` = (NI_ttm − NI_ttm_prior) / assets_latest
9. `f_available` = 1 if features 1–8 computable, else 0

Each of 1–8 is **cross-sectionally rank-normalized per snapshot date** to [−0.5, 0.5] (heavy-tailed ratios; uses only same-date information) and missing values set to 0 (median) with `f_available` carrying the missingness. Expected coverage loss ≈ 10% of tickers (spike) plus early-history snapshots lacking 8 quarters.

## Arms (seed 42; binding folds 1–5; N=20; everything else identical to the base pipeline)

- **Base** — determined by a rule fixed now: the R2 arm judged PROMISING by R2's decision map (resid or nomacro) if any; otherwise the v2.3.17 base (raw target, 97 features). Reused from committed results where the R2 reuse rule permits.
- **Fund** — base pipeline + the 9 fundamental features. **Fundamental features bypass the corr filter** (forced in; the var filter still applies) — otherwise weak-linear-corr features would be dropped and the arm would collapse into base, making the experiment vacuous. The number that would have passed the filter anyway is logged.
- Test-side metrics identical (per-date IC on raw realized returns; top-5 alpha). Contrast: **Fund − Base**.

## Anchors (fixed now; screen, not a gate)

Paired per-fold ΔIC over binding folds: **PROMISING** = mean ΔIC ≥ +0.02 AND positive in ≥ 4/5; **DEAD** = mean ≤ 0 OR positive ≤ 2/5; else **MIXED**. Same seed-noise context as R2 (+0.02 ≈ 4× across-seed dispersion of the binding mean). Secondary (reporting only): Δalpha, Δ ic_positive_date_frac, per-fold tables, feature-selection logs, coverage-by-year table from the builder.

## Dataset custody

`r3_build_fundamentals.py` fetches company facts for the 525 cache tickers (raw JSON cached locally, not committed), builds the feature table aligned to the cache's (ticker, date) order, and writes `results/research_r3/fund_features.npz` (committed as evidence) plus a verification report (coverage by year; sample rows showing the period end and availability date actually used — the point-in-time audit trail). The cache `backtest_cache.npz` is never modified.

## Decision map

PROMISING → R3-3seed pre-spec amendment (3 seeds; then and only then any gate discussion, with dependence-aware statistics and beta-adjusted alpha per the v2.3.18 disposition). DEAD → fundamentals-as-input card closes; R3b (value factors) is not automatically pursued — it needs its own case. MIXED → record; continuation by amendment. The v2.3.17 FAIL and v2.3.18 disposition remain in force regardless.

---

## Build amendment (2026-09-08, before any feature-vs-return evaluation)

Build v1 (526 tickers) reached 78.3% availability with 78 tickers below 50%. Three causes, two of them instrument defects: (1) ETFs and foreign filers (20-F/40-F/6-K) have no 10-Q/10-K company facts — legitimately 0, by design; (2) issuers on 12/16-week fiscal calendars (COST, KR, AZO, DPZ, PEP) never produced a Q4 under the 80–100-day quarter window → contiguity failed → all ratios NaN; (3) any single missing concept (regional banks without a total-revenue tag) zeroed all eight ratios. Build v2: quarter windows widened to 70–120 days (direct, derived, and contiguity); ratios computed independently per available inputs; **`f_available` becomes the fraction of the eight ratios computable** (was binary) — rank-normalization and the `filed < t` rule unchanged. Coverage engineering only; no return data was consulted.
