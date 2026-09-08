# R3 Spike — EDGAR Point-in-Time Fundamentals Feasibility (instrument v2)

Lag = first filing per period (v1 median-over-re-reports lag kept in JSON as median_lag_all_reports_v1 to document the comparatives artifact). Quarters = direct quarterly values + Q4 derivable from FY.

Run date: 2026-09-08. Sample: 20 tickers. Criteria: core concepts ['revenue', 'net_income', 'equity', 'assets'] each with >= 30 quarters in 2016-2025 and median filing lag <= 60 days.

| ticker | covered | revenue | net_income | equity | assets | eps | ocf | shares |
|---|---|---|---|---|---|---|---|---|
| AAPL | YES | 38q / 34d | 38q / 34d | 38q / 34d | 38q / 34d | 38q / 34d | 10q / 34d | 40q / 25d |
| MSFT | YES | 40q / 27d | 40q / 27d | 40q / 27d | 40q / 27d | 40q / 27d | 39q / 27d | 40q / 13d |
| NVDA | YES | 40q / 24d | 40q / 24d | 40q / 24d | 40q / 24d | 40q / 24d | 11q / 25d | 40q / 6d |
| AMZN | YES | 40q / 30d | 40q / 30d | 40q / 30d | 40q / 30d | 40q / 30d | 40q / 30d | 40q / 20d |
| TSLA | YES | 40q / 29d | 40q / 29d | 40q / 29d | 40q / 29d | 40q / 29d | 10q / 28d | 40q / 23d |
| NFLX | YES | 40q / 21d | 40q / 21d | 40q / 21d | 40q / 21d | 40q / 21d | 40q / 21d | 40q / 20d |
| JPM | YES | 40q / 34d | 40q / 34d | 40q / 34d | 40q / 34d | 40q / 34d | 10q / 32d | 40q / 33d |
| GS | YES | 40q / 34d | 40q / 34d | 40q / 34d | 40q / 34d | 40q / 34d | 10q / 34d | 40q / 25d |
| XOM | no | 1q / 399d | 1q / 399d | 4q / 444d | 1q / 215d | 1q / 399d | 0q / lag n/a | 4q / 444d |
| CVX | YES | 40q / 36d | 40q / 36d | 40q / 36d | 40q / 36d | 40q / 36d | 10q / 34d | 30q / 34d |
| JNJ | YES | 32q / 29d | 37q / 31d | 38q / 31d | 38q / 31d | 37q / 31d | 10q / 30d | 40q / 6d |
| PFE | no | 29q / 39d | 33q / 39d | 34q / 39d | 34q / 39d | 33q / 39d | 10q / 39d | 40q / 3d |
| LLY | YES | 40q / 31d | 40q / 31d | 40q / 31d | 40q / 31d | 40q / 31d | 10q / 30d | 40q / 4d |
| PG | YES | 40q / 23d | 40q / 23d | 40q / 23d | 40q / 23d | 40q / 23d | 10q / 19d | 31q / 20d |
| KO | YES | 30q / 27d | 35q / 27d | 37q / 27d | 37q / 27d | 35q / 27d | 10q / 27d | 40q / 3d |
| CAT | YES | 40q / 35d | 40q / 35d | 40q / 35d | 40q / 35d | 40q / 35d | 10q / 33d | 40q / 35d |
| HON | YES | 40q / 25d | 40q / 25d | 40q / 25d | 40q / 25d | 40q / 25d | 10q / 24d | 40q / 23d |
| NEE | YES | 40q / 26d | 40q / 26d | 40q / 26d | 40q / 26d | 40q / 26d | 10q / 23d | 40q / 24d |
| DUK | YES | 40q / 38d | 40q / 38d | 40q / 38d | 40q / 38d | 40q / 38d | 10q / 39d | 40q / 33d |
| AMT | YES | 40q / 29d | 40q / 29d | 40q / 29d | 40q / 29d | 40q / 29d | 10q / 29d | 40q / 18d |

**Covered: 18/20 → verdict: FEASIBLE**

- Revenue tags needed across sample: 5 (us-gaap:RegulatedAndUnregulatedOperatingRevenue, us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax, us-gaap:Revenues, us-gaap:RevenuesNetOfInterestExpense, us-gaap:SalesRevenueNet)
- Median filing lag across core concepts: 30 days (p90 of medians 39)
- Annual-only (10-K FY) share of income-statement entries: 19% (Q4 derivation needed for these)
- Full-build fetch estimate: 525 tickers ≈ 525 requests at <=5/s ≈ 2-3 min; engineering = tag normalization + Q4 derivation + point-in-time join.
