# Research Pre-Spec — R1: Regime Predictability Screen (spectral descriptors vs per-date IC)

**Status**: pre-specified (written before any R1 computation; anchors fixed before looking)
**Date**: 2026-09-08
**Track**: R1 — first study of the optional-research branch opened by the v2.3.18 disposition. Not a gate; no deployment claim can follow from R1 alone. Cheapest-possible falsification test for the "regime-conditional deployment" idea before any training-scale investment (R2).
**Input**: committed v2.3.17 per-date CSVs (`results/walkforward_v2317/seed{42,1,2}/fold{0..5}_dates.csv`; IC_t = 3-seed mean per date) + SPY and ^VIX daily closes fetched once via yfinance and frozen as CSVs under `results/research_r1/` (fetch date recorded; subsequent runs read the frozen CSVs).

## Question

Does a **backward-looking, observable-at-t** spectral description of the market predict the model's same-date ranking performance IC_t? This is the necessary condition for any regime-conditional use of the pipeline: if no observable descriptor separates working dates from non-working dates, the spectral/regime-feature direction is dead regardless of representation choice (FNO, wavelets, or otherwise), because conditioning information that cannot be computed at t cannot be deployed at t.

## Descriptors (computed on trailing 252-trading-day SPY log returns ending at t; 126-day variant is a secondary robustness axis)

New (each with a pre-stated mechanism — no fishing):
1. **VR(10)** — Lo–MacKinlay variance ratio, q=10. Mechanism: VR>1 = trending market; the learned mapping is momentum/growth-tilted (return_180d momentum feature; beta drift toward high-beta names), so **predict IC higher when VR high**.
2. **Spectral slope** — slope of log10(PSD) vs log10(freq), Welch estimate. Mechanism: steeper (more negative) = low-frequency dominance = persistent structure → IC higher.
3. **Spectral entropy** — Shannon entropy of normalized PSD ∈ [0,1]. Mechanism: whiter spectrum = no exploitable structure → IC lower.
4. **High-frequency energy ratio** — fraction of power at periods < 5 days. Mechanism: choppier tape → IC lower.

Controls (information already available to the v2.3.17 model via its feature set; new descriptors must beat these to matter):
C1. 21-day realized vol. C2. VIX level at t. C3. Trailing 126-day SPY return. C4. Vol-of-vol (std of rolling 21-day vol within the window).

## Statistics (per the D5 dependence finding: no p-values; consistency is the robustness axis)

- **Primary**: within-fold Spearman ρ(descriptor_t, IC_t) on binding folds 1–5 (n≈25 each — individually noisy by design), plus the **n-weighted mean of per-fold ρ ("pooled within-fold ρ")** as the anchor quantity. Within-fold is primary because (a) we hold post-hoc knowledge that 2022/2025 were good years — a descriptor that merely separates years re-packages that hindsight; (b) fold-level training-set size differs (expanding), confounding between-fold comparisons. A useful regime signal must discriminate dates **inside** a year.
- Context (non-anchor): plain pooled ρ across all 126 binding dates (confounded, reported for completeness); fold 0 (2020, non-binding) same stats; per-date alpha as target reported alongside IC (secondary target, reporting only).
- **Secondary (reporting only)**: mean within-fold-standardized descriptor on IC>0 vs IC≤0 dates (binary view); 126-day-window variant of the full table; descriptor↔control Spearman collinearity table (is a "promising" descriptor just a control in disguise?).

## Anchors (fixed before computation; judgment aids, not a gate)

Per new descriptor:
- **PROMISING** = |pooled within-fold ρ| ≥ 0.25 AND per-fold sign consistent in ≥ 4/5 binding folds AND |ρ| > |ρ| of the best control (on the same statistic).
- **DEAD** = fails the control comparison OR sign consistency ≤ 3/5.
- Otherwise **MIXED** — record honestly.

Multiple-candidates note: with 4 new descriptors, one clearing |ρ| ≥ 0.25 by chance alone is non-trivial; this is why PROMISING is conjunctive (magnitude AND consistency AND control-beating), and why R1 PROMISING licenses only an R2 pre-spec, never a deployment claim.

## Decision map

- Any descriptor PROMISING → design R2 (regime-conditional study: descriptor as conditioning variable; its own pre-spec; NN budget discussed there). Conditional deployment has its own selection-bias trap — R2's problem, named now.
- All DEAD → record the null; the spectral/regime-descriptor direction closes; remaining (b)-branch cards are forward-looking features and retraining cadence.
- MIXED → record; any continuation requires its own pre-spec.

Honest prior, stated in advance: the controls are close cousins of features the v2.3.17 model already had (VIX, vol structure) and still failed with — the most likely outcome is a clean null obtained cheaply, which is a result, not a failure of R1.
