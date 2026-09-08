# Research Pre-Spec — R3 Spike: SEC EDGAR Point-in-Time Fundamentals Feasibility

**Status**: pre-specified (feasibility criteria fixed before any fetch)
**Date**: 2026-09-08
**Track**: R3 — third card of the optional-research branch. Not a hypothesis test; a **feasibility spike** that decides whether a full R3 study (fundamentals as an orthogonal information axis) is worth building on free data. Runs in parallel with the R2 queue (I/O-bound, no training).

## Why fundamentals, honestly

The v2.3.17/18 arc established that the price-derived feature sheet carries no next-quarter information on the time axis (momentum IC ≈ 0; regime inversion). True forward-looking data (analyst estimate history, guidance) is not free. What is free and official is SEC XBRL "company facts": quarterly financial statement values with **filing dates**, which allows correct point-in-time construction (a value becomes usable only from its `filed` date, never from its period end). This is not forward-looking; it is **orthogonal-to-price** information (value, profitability, investment, accruals — Fama-French/quality families). Caveat carried from v2.3.10: EDGAR-derived *sentiment* was null; that does not decide numeric fundamentals, but tempers expectations.

## Spike design

- **Sample**: 20 fixed tickers spanning sectors (deterministic list in the script; includes banks and utilities whose tag conventions differ — the hard cases, on purpose).
- **Source**: `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json` via the SEC `company_tickers.json` mapping; User-Agent from `config.SEC_USER_AGENT`; ≤ 5 req/s.
- **Concepts** (us-gaap unless noted), each with an ordered list of alternate tags: revenue (Revenues / RevenueFromContractWithCustomerExcludingAssessedTax / SalesRevenueNet), net income (NetIncomeLoss), diluted EPS (EarningsPerShareDiluted), stockholders' equity (StockholdersEquity), total assets (Assets), operating cash flow (NetCashProvidedByUsedInOperatingActivities), shares outstanding (dei:EntityCommonStockSharesOutstanding / us-gaap:CommonStockSharesOutstanding).
- **Per ticker × concept, from 10-Q/10-K entries**: which alternate tag resolved; number of distinct fiscal quarters covered in 2016Q1–2025Q4 (target ≈ 40); first/last period end; **median filing lag** (filed − period end, days); count of annual-only (10-K, fp=FY) entries, which require Q4 derivation (FY − Q1 − Q2 − Q3) — the known XBRL complication; whether `frame` labels are present (helps de-duplication).

## Feasibility criteria (fixed before run)

Core set = {revenue, net income, equity, assets}. A ticker is **covered** if all four core concepts resolve with ≥ 30 distinct quarters in 2016Q1–2025Q4 and median filing lag ≤ 60 days.
- **FEASIBLE** = ≥ 16/20 tickers covered → write the R3-full pre-spec (hypothesis, features, point-in-time join to the snapshot grid, arm design against the R2 winner or base).
- **MARGINAL** = 12–15 covered → feasible only with per-sector tag maps; decide by engineering budget (recorded, not by enthusiasm).
- **NOT FEASIBLE** = < 12 covered → free-data fundamentals card closes; paid data (Compustat/WRDS) is out of scope for this project.

Also recorded regardless of verdict: tag heterogeneity (distinct revenue tags needed across 20), share of annual-only entries, estimated full-build cost (525 tickers ≈ 525 requests ≈ minutes to fetch; engineering = tag normalization + Q4 derivation + point-in-time join).

## What a FEASIBLE verdict does and does not license

It licenses building the dataset and an R3-full pre-spec. It does not imply the features carry time-axis signal — that is R3-full's question, with its own anchors, and the honest prior there is low-to-medium (value/quality premia are themselves regime-dependent; 2023 was a growth year).
