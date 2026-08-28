# Diagnosis Pre-Spec — v2.3.18 Phase A: Tail-Alpha Decomposition

**Status**: pre-specified (written before any Phase A computation; author has not inspected the per-date CSVs beyond the fold-level aggregates already in Amendment 1)
**Date**: 2026-07-30
**Input**: committed v2.3.17 evidence only — `results/walkforward_v2317/seed{42,1,2}/fold{0..5}_dates.csv` (commit 5249d17). No new training.
**Nature**: this is a **diagnosis, not a gate**. Per the Asymmetric Validation Principle this sits between development and validation: questions and interpretive anchors are fixed before computation to prevent post-hoc rationalization, but the anchors are labeled judgment aids, not pass/fail thresholds. The output feeds one decision: whether a **new** pre-registered top-5-alpha gate (Phase B → v2.3.19 candidate) is worth building, or whether the v2.3.17 alpha is dismissed as concentration artifact.

## Question

v2.3.17 found per-date IC ≈ 0 (binding mean +0.0091) while top-5 alpha = +5.81%p (4/5 folds positive, +3.26%p excluding fold 5, ≈9× TC_est). Two hypotheses:

- **H-tail (signal)**: the model identifies a few extreme winners without ranking the cross-section — real but narrow skill; Spearman IC is insensitive to it by construction.
- **H-fluke (artifact)**: the alpha is small-n concentration — a few extreme dates (and, unobservable in Phase A, a few repeated tickers) drive the mean; remove them and nothing remains.

Phase A tests what the committed per-date data can distinguish. Ticker-level concentration requires Phase B (re-run with pick logging).

## D1 — Time concentration (per binding fold + pooled)

Positive-date fraction of alpha; mean vs median per-date alpha; share of the fold's summed alpha contributed by the top-3 alpha dates (note: can exceed 100% when negative dates exist — report raw).

**Anchors**: broad support looks like positive-date fraction ≥ ~0.6 and median > 0 and top-3 share ≤ ~50%. Spiky (fluke-like) looks like median ≈ 0 with top-3 share ≥ ~70%.

## D2 — Seed structure of per-date alpha

Pairwise Spearman correlation of per-date alpha vectors across the three seeds (dates aligned by inner join), per fold and pooled.

**Anchors**: mean pairwise ρ ≥ ~0.7 → the alpha pattern is a property of the data/config (structure), not training stochasticity. ρ ≤ ~0.3 → training-noise-driven; H-fluke strengthened regardless of D1.

## D3 — Alpha–IC decoupling

Per-date Spearman correlation between alpha_d and IC_d, per fold and pooled; plus mean alpha on the subset of dates with IC_d ≤ 0.

**Anchors**: H-tail predicts weak coupling (|ρ| ≤ ~0.3) **and** positive mean alpha even on IC≤0 dates. If alpha exists only where IC is high, the alpha is just the tail expression of ranking skill and dies with it.

## D4 — Fold-5 (2025) and Fold-0 (2020) shape

Same D1 statistics on the strongest folds specifically; fold 0 reported as non-binding context. Is fold 5's +16%p broad or a few dates?

## D5 — Dependence structure (for any future gate's statistics)

Lag-1..6 autocorrelation of per-date alpha within each fold (63-trading-day forward windows on a ~10-trading-day snapshot grid overlap ≈ 6 deep, so high positive autocorrelation is expected). Reported to size the effective sample and to specify, in advance, that any future top-5-alpha gate must use dependence-aware statistics (block bootstrap by year or non-overlapping subsampling) — per-date t-tests are invalid here and will not be used.

## Decision mapping (stated before looking)

- D1 broad + D2 structural + D3 decoupled → H-tail supported at Phase A level → proceed to Phase B (pick-level: ticker concentration, persistence, realized-turnover TC) before any new gate design.
- D2 noise-like, or D1 spiky with median ≈ 0 → H-fluke supported → record and stop; no Phase B, no new gate; the v2.3.17 FAIL stands as the complete characterization.
- Mixed → record honestly; Phase B decision weighed on which components failed.

Anchor values (~0.6, ~0.7, ~0.3, top-3 ~50/70%) are stated conveniences chosen before computation, not validated thresholds; they bind interpretation language, not actions with capital consequences (no capital decision flows from Phase A directly).

---

## Phase A2 — Beta-Tilt Test (appended pre-run, 2026-08-29)

Phase A's surviving characterization (intermittent IC-conditional ranking, asymmetric payoff) has a mundane alternative: the top-5 are simply high-beta/high-vol names, so long-only alpha vs the equal-weight universe is beta premium harvested in a mostly-rising decade, not selection skill. A2 tests this from the same committed CSVs.

**A2-1 — Effective beta and decomposition.** OLS per fold and pooled (binding): top5_ret_d = a + b·univ_ret_d. Since alpha_d = a + (b−1)·univ_ret_d + e_d, mean alpha decomposes into a tilt component (b−1)·E[univ] and a residual (intercept) component a. Observations overlap (63-td windows, ~10-td grid), so estimates are descriptive — cross-fold consistency is the robustness axis; no p-values.

**A2-2 — Down-market behavior.** Mean alpha on univ_ret ≤ 0 dates vs > 0 dates. Pure tilt predicts clearly negative alpha on down dates.

**A2-3 — IC–market linkage.** ρ(IC_d, univ_ret_d) and the 2×2 (IC±, univ±) mean-alpha table: does "intermittent ranking" reduce to "market up"?

**Anchors (stated before computation, judgment aids)**: b ≥ ~1.3 = strong tilt; intercept a ≤ ~TC_est (0.62%p/quarter) pooled and in a majority of binding folds → alpha is substantially risk tilt (H-beta). a ≥ ~2%p robust in ≥3/5 folds → a residual selection component exists beyond tilt.

**Decision mapping**: H-beta supported → v2.3.18 closes as "no deployable selection edge beyond risk tilt"; Phase B optional/low value. Residual component robust → Phase B (pick-level: ticker concentration, persistence, realized-turnover TC) justified before any new gate design. Mixed → record honestly.
