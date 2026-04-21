# 3x3 Portfolio Optimization

A deep-learning pipeline I built to help me allocate ~KRW 20M across US equities. It screens a universe of ~84 tickers across 7 sectors, trains an ensemble on roughly 120K cross-sectional samples drawn from S&P 500 + NASDAQ-100 history, adds macro and sentiment features, and outputs a 3x3 (time horizon × risk tier) allocation matrix for 5 selected stocks.

I come from a neuroscience / cognitive engineering / neuroimaging background, not finance, so this repo is also how I learn quantitative investing from first principles. Treat it accordingly: it's a working system that I actually use, but it's a personal project, not a professional product.

## What the pipeline does

1. **Screen the investment universe** (`screener.py`). Auto-discovers seeds from S&P 500 + NASDAQ-100 using GICS industry matching, combines with a small list of niche anchor tickers, and filters to ~84 stocks across 7 sectors (AI Compute, Neuromodulation, CNS Pharma, Digital Health, Space/Aerospace, Solar/Clean Energy, ETF benchmarks).
2. **Collect sentiment** (`sentiment.py`). Four layers: news headlines via FinBERT, SEC EDGAR filings, FDA + ClinicalTrials.gov events, and earnings surprises. Produces 22 sentiment features per ticker.
3. **Train on history** (`historical.py`, `training_universe.py`). Builds ~120K training samples from S&P 500 + NASDAQ-100 with 10 years of per-ticker snapshots, augmented with FRED macro series, Fama-French 5 factors, and cross-asset features (VIX, treasuries, gold, oil, USD). Trains an ensemble of 5 PyTorch networks with Huber loss.
4. **Blend weights data-driven** (`blend_optimizer.py`). Finds the optimal mix of NN predictions and analyst consensus via a multi-window backtest (3m/6m/9m) with regime detection and bounded shrinkage toward a prior.
5. **Select top 5** via a composite score that combines predicted Sharpe, MC Dropout confidence, uncertainty penalty, sentiment boost, and event-risk penalty.
6. **Build the 3x3 allocation matrix** (`models.py`). A small neural network with a differentiable Sinkhorn layer that satisfies row (time horizon) and column (risk tier) marginal constraints, trained end-to-end with a Kahneman-Tversky asymmetric portfolio loss.
7. **(Optional) Stratified K-Fold Portfolio Backtest** (`backtest.py`). Validates whether the model's picks actually outperform on unseen tickers. Ticker-axis K-fold (stratified by GICS sector) plus cross-sector transfer tests.

## Quickstart

```bash
pip install numpy matplotlib yfinance pandas torch transformers scipy
pip install fredapi  # optional, for FRED macro features

# Full pipeline
python run.py --torch --screen --sent

# Portfolio backtest only
python run.py --backtest
# or equivalently:
python backtest.py

# Pipeline + backtest
python run.py --torch --screen --sent --backtest

# Individual modules
python screener.py
python sentiment.py
python training_universe.py
python historical.py       # Walk-Forward CV
```

Before running, get a free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) and put it in `config.py` as `FRED_API_KEY`. Without it you lose ~13 macro features but the pipeline still runs.
Optionally, get a free [Finnhub API key](https://finnhub.io) and set `FINNHUB_API_KEY` in `config.py` too. This adds a secondary news source on top of yfinance for the sentiment layer. Without it you still get news from yfinance, SEC filings, FDA events, and clinical trials, so sentiment works fine — Finnhub is nice-to-have, not required.

## Repository layout

```
run.py                  Entry point
config.py               Hyperparameters
screener.py             7-sector universe screening with auto seed discovery
sentiment.py            Multi-layer sentiment (news/SEC/FDA/trials/earnings)
training_universe.py    S&P 500 + NASDAQ-100 + FRED + Fama-French + cross-asset
historical.py           Training data builder + Walk-Forward CV
blend_optimizer.py      Multi-window backtest + regime detection + shrinkage
backtest.py             Stratified K-Fold portfolio backtest
models.py               Matrix Network + differentiable Sinkhorn
data_auto.py            yfinance data collection
visualize.py            Auto-generated figures
requirements.txt        Dependencies
```

## Why stratified K-fold instead of time-axis backtest

Time-axis splits don't fit this task. My model isn't a time-series forecaster asking "what happens next?" — it's a cross-sectional ranker asking "given current features, which stocks will outperform over the next 3 months?". Training on 2014–2020 and testing on 2024 mixes feature-pattern evaluation with regime change (the AI era didn't even exist before 2023), which confounds the two sources of error.

Stratified K-fold (by GICS sector) holds the time period fixed and splits on the ticker axis: train on ~420 tickers, test on ~107, all spanning the same years. This isolates the question I actually care about — does the model generalize to tickers it hasn't seen? — without conflating it with regime drift.

## Why multi-window blend optimization

A naive single-window optimizer told me to weight NN predictions at 95–100% because analyst rank correlation was around -0.8 in a 3-month window. That number isn't a signal that analysts are bad; it's a structural artifact. Analyst targets are 12-month forward forecasts, but I was measuring them against past 3-month realized returns. Mean reversion means stocks that fell hard recently have the largest apparent upside, which produces a negative Spearman with recent actuals by construction.

`blend_optimizer.py` handles this by running the backtest at 3m / 6m / 9m windows, detecting whether the analyst signal is unstable across windows, and shrinking the optimized weight toward a prior (`w_ret = 0.30`) when instability is detected. Bounded weights (`w_ret ∈ [0.15, 0.60]`) prevent the optimizer from picking extremes when the data is noisy.

## Baseline comparisons

The +15.4%p selection alpha is measured against the test-universe mean, but that number alone doesn't tell you whether the model beats naive strategies. `backtest.py` compares the NN's top-5 picks against three baselines (see `baseline_*` fields in `backtest_results.json`):

| Baseline | Construction | Result (5-fold avg) |
|----------|--------------|--------------------|
| **Random 5** | 1,000 random 5-ticker selections per fold, 95% CI computed from the distribution | Our alpha is outside the 95% CI in all 5 folds (empirical p < 0.0001 in every fold). |
| **SPY / VOO buy-and-hold** | Passive benchmark, where the ETF falls in the fold's test set | Our top-5 beats SPY by +45.2%p (Fold 1) and +10.1%p (Fold 5). The ETF is only in 2 of 5 folds due to the stratified split. |
| **Proper momentum top-5** | For each test ticker, split its snapshots into early (signal) and late (realized) halves; pick top-5 by early-half mean return; measure realized return on the late half. No look-ahead. | NN beats momentum in **5/5 folds** with an average edge of **+6.8%p**. Strongest in Fold 3 (+9.0%p), where momentum alpha is near-zero and simple past-return strategy fails. |

The momentum comparison is the most informative: it tells you the NN's contribution *over and above* a trivial "past winners keep winning" heuristic, which is the first thing any skeptical reviewer would try.

## Ablation study

To characterize what each feature group contributes, I ran the backtest with three configurations. Results for each are stored in `results/backtest_results_{config}.json`:

| Config | Features | Rank Corr | Alpha (5-fold avg) | Cross-sector Transfer |
|--------|----------|-----------|--------------------|-----------------------|
| Full (macro + sentiment) | 97 | +0.465 | +15.4%p | +0.028 |
| No-macro (tech + sentiment) | 54 | **+0.526** | +15.4%p | **+0.219** |
| Tech-only (technical only) | 54 | **+0.526** | +15.4%p | **+0.219** |

Two findings were counter to my initial expectations.

**Macro features hurt cross-sectional ticker ranking.** Removing the 43 macro features (FRED + Fama-French + cross-asset) *improved* rank correlation from +0.465 to +0.526 and cross-sector transfer from +0.028 to +0.219 — the latter roughly 8× higher. My interpretation: in a cross-sectional split, all tickers at a given snapshot share the same macro values, so macro features carry no inter-ticker signal — only noise that the ensemble partially overfits to. Note this is the opposite of what the Walk-Forward CV (time-axis) shows, where macro features reduce return error from 16.6%p to 11.9%p. The two CV schemes measure different things, and for portfolio *selection* (ticker ranking within a time period), the cross-sectional result is the one that matters.

**Sentiment features don't move backtest metrics but do change the selection.** No-macro and tech-only produce identical backtest numbers by construction — sentiment features are computed only for the current 84 stocks at Stage 2 and are absent from the 527-ticker training cache. Their effect shows up in the top-5 picks instead: tech-only selects `BSX, MDT, MSFT, CRM, SYK`, while adding sentiment swaps SYK out for ADSK (ADSK had a positive news composite of +0.149). One swap out of five is a real but modest effect, and it's not measurable through backtest alpha with the current design.

I haven't restructured the pipeline based on these findings. Macro features are still loaded by default because they're useful inside `blend_optimizer.py`'s regime gate for the Walk-Forward CV and weight-shrinkage logic. Isolating them there — rather than concatenating into the per-ticker feature vector — is in the Future work section.

## Known limitations

- **Fold 1 outlier inflates the headline alpha.** SNDK's post-IPO run drives Fold 1 to an unusually high value; the robust estimate across Folds 2–5 is closer to +8.5%p. The aggregate number should be read with this in mind.
- **Hyperparameters set by trial-and-error**, not systematic search. NN architecture, learning rate, epochs, and dropout are all manually chosen. Sensitivity to these choices is not characterized.
- **No transaction cost, slippage, or tax modeling.** All backtest numbers are paper-alpha and will be lower after real-world frictions (typically several %p/year for monthly rebalancing strategies).
- **Survivorship bias in the training universe.** Tickers are sampled from the current S&P 500 + NASDAQ-100 composition, so stocks that were delisted or removed from the index during the 10-year window are underrepresented. This biases the training distribution toward survivors.
- **Composite score coefficients are hand-picked.** `sentiment_weight=0.10`, `uncertainty_penalty=3.0`, `event_risk_penalty=2.0` were not tuned via grid search. The ablation above measures sentiment as a feature group (one swap in top-5); individual coefficient sensitivity is still uncharacterized.
- **No historical fundamentals.** yfinance doesn't expose past PE/ROE/analyst targets, so Stage 1 features are technical + macro only. Fundamentals enter only at Stage 2 via current analyst consensus.
- **Sector concentration.** The selection step has no diversification constraint, so top 5 often cluster in 2 sectors (typically AI Compute + Neuromodulation). I currently accept this because the backtest shows rank-correlation is much stronger within-sector than across — forced diversification would pick lower-scoring stocks.
- **yfinance rate limits.** Heavy S&P 500 batch downloads occasionally cause cross-asset fetches to return truncated history. The tz-safety fix in `training_universe.py` means this degrades gracefully, but ideally the macro loads should happen before the big batch.
- **This is not investment advice.** Predictions carry ±12%p MAE on return and ±9%p on risk. Realized Sharpe will likely be 30–35% lower than predicted Sharpe.

## Output

After a full run, `results/` contains:

- `output.json` — selected tickers, 3x3 matrix, per-stock predictions, rationale, blend weights
- `universe.json` — screener output (84 tickers with sector map)
- `fig1_scatter.png` through `fig8_dashboard.png` — auto-generated figures
- `backtest_results.json` — if `--backtest` was run
- `backtest_cache.npz` — cached training matrix for faster re-runs

## Sector overview

| Sector | Focus | Seed method |
|--------|-------|-------------|
| A: AI Compute | GPU, cloud, AI platforms | auto (GICS industry) |
| B: Neuromodulation | DBS, TMS, BCI, medical devices | 2 anchors + auto |
| C: CNS Pharma | Neurotransmitter-based therapeutics | 1 anchor + auto |
| D: Digital Health | Telemedicine, digital therapeutics | 2 anchors + auto |
| F: Space & Aerospace | Launch, satellites, defense | 6 anchors + auto |
| G: Solar & Clean Energy | Solar, hydrogen, batteries | 10 anchors + auto |
| E: ETF Benchmark | Training benchmarks only (excluded from selection) | fixed 4 |

Anchor tickers are small/niche names that aren't in S&P 500 and therefore can't be auto-discovered. For any large-cap candidate, auto-discovery should find it — if it doesn't, that's usually a GICS classification issue worth investigating rather than patching with an anchor.

## Future work

The current pipeline uses a deep ensemble with MC Dropout, which is an
approximate Bayesian method (Gal & Ghahramani, 2016) — the ensemble
approximates posterior averaging and MC Dropout approximates variational
inference. However, the shrinkage layer in `blend_optimizer.py` and the
composite score use hand-picked coefficients rather than proper posterior
inference.

Planned upgrades:

- **Uncertainty calibration**: verify that predicted standard deviations
  actually match realized errors (calibration plots, temperature scaling
  if needed, following Guo et al., 2017).
- **Heteroscedastic output + aleatoric / epistemic decomposition**:
  output `(mean, log_var)` pairs trained with negative log-likelihood
  loss, following Kendall & Gal (2017). Separates irreducible market
  noise from reducible model uncertainty — useful for portfolio selection.
- **Hierarchical Bayesian structure over sectors**: sector-level priors
  with ticker-level posteriors (analogous to multi-level GLM with
  ROI-level random effects in fMRI analysis). Expected to improve
  cross-sector transfer, which is currently weak (+0.027 rank corr).
- **Disentangle macro from the per-ticker feature matrix.** The ablation above
  showed that macro features hurt cross-sectional rank correlation
  (+0.465 → +0.526 when removed), because all tickers at a given snapshot
  share identical macro values — the ensemble partially overfits to
  time-synchronous signals that carry no inter-ticker information. A cleaner
  design would route macro features through the blend-optimizer's regime gate
  only, rather than concatenating them into each ticker's feature vector.
