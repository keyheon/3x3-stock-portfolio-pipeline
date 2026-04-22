"""
training_universe.py — S&P 500 + NASDAQ-100 training universe.

Training uses the full S&P 500 + NASDAQ-100 (~550 unique tickers).
Investment uses only the 30-50 tickers selected by screener.py.

training universe != investment universe

Usage:
    from training_universe import get_training_tickers
    tickers = get_training_tickers()  # ~550 tickers

    # In historical.py:
    hist_X, hist_Y_ret, hist_Y_risk, meta, _ = build_training_data(
        tickers=tickers, period="10y", snapshot_interval=10
    )
"""
import socket
socket.setdefaulttimeout(120)  # [v2.3.4] fredapi has no built-in timeout; also helps yfinance stalls
import warnings
import time
warnings.filterwarnings('ignore')


# ============================================================
# S&P 500 + NASDAQ-100 TICKER FETCHING
# ============================================================

def get_sp500_tickers():
    """Fetch the full S&P 500 ticker list from Wikipedia."""
    import pandas as pd
    import urllib.request
    from io import StringIO

    headers = {"User-Agent": "Mozilla/5.0 (stock-pipeline/2.0)"}

    try:
        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers
        )
        html = urllib.request.urlopen(req, timeout=15).read().decode()
        tables = pd.read_html(StringIO(html))
        sp500 = tables[0]

        tickers = sp500['Symbol'].tolist()
        # BRK.B → BRK-B (yfinance format)
        tickers = [tk.replace('.', '-') for tk in tickers]

        # Sector mapping (GICS)
        sector_map = {}
        if 'GICS Sector' in sp500.columns:
            for _, row in sp500.iterrows():
                tk = row['Symbol'].replace('.', '-')
                sector_map[tk] = row['GICS Sector']

        print(f"  S&P 500: {len(tickers)} tickers")
        return tickers, sector_map

    except Exception as e:
        print(f"  S&P 500 fetch failed: {e}")
        return [], {}


def get_nasdaq100_tickers():
    """Fetch the full NASDAQ-100 ticker list from Wikipedia."""
    import pandas as pd
    import urllib.request
    from io import StringIO

    headers = {"User-Agent": "Mozilla/5.0 (stock-pipeline/2.0)"}

    try:
        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            headers=headers
        )
        html = urllib.request.urlopen(req, timeout=15).read().decode()
        tables = pd.read_html(StringIO(html))

        tickers = []
        for t in tables:
            if 'Ticker' in t.columns:
                tickers = t['Ticker'].tolist()
                break

        tickers = [tk.replace('.', '-') for tk in tickers if isinstance(tk, str)]
        print(f"  NASDAQ-100: {len(tickers)} tickers")
        return tickers

    except Exception as e:
        print(f"  NASDAQ-100 fetch failed: {e}")
        return []


def get_training_tickers(include_sp500=True, include_nasdaq100=True,
                         include_etf_benchmarks=True, verbose=True):
    """
    Build the full training ticker list.

    S&P 500 (~500) + NASDAQ-100 (~100) = ~550 unique tickers
    + ETF benchmarks (VOO, QQQ, SPY, etc.)

    Returns:
        tickers: list[str] — deduplicated
        gics_sectors: dict — {ticker: GICS sector name}
    """
    if verbose:
        print("\n[Training Universe] Building expanded ticker list...")

    all_tickers = []
    gics_sectors = {}

    if include_sp500:
        sp_tickers, sp_sectors = get_sp500_tickers()
        all_tickers.extend(sp_tickers)
        gics_sectors.update(sp_sectors)

    if include_nasdaq100:
        nq_tickers = get_nasdaq100_tickers()
        all_tickers.extend(nq_tickers)

    if include_etf_benchmarks:
        etfs = ['VOO', 'QQQ', 'SPY', 'SOXX', 'XBI', 'IHI', 'XLV', 'SMH', 'IWM', 'DIA']
        all_tickers.extend(etfs)
        if verbose:
            print(f"  ETF benchmarks: {len(etfs)}")

    # Deduplicate (preserve order)
    seen = set()
    unique = []
    for tk in all_tickers:
        if tk not in seen and isinstance(tk, str) and len(tk) <= 6:
            seen.add(tk)
            unique.append(tk)

    if verbose:
        print(f"\n  Training Universe: {len(unique)} unique tickers")

    return unique, gics_sectors


# ============================================================
# DATA VALIDATION (optional — pre-download check)
# ============================================================

def validate_tickers(tickers, min_history_days=252, batch_size=50, verbose=True):
    """
    Check data availability via yfinance.
    Drop tickers with history < min_history_days.

    Processes many tickers in batches for speed.
    """
    import yfinance as yf

    if verbose:
        print(f"\n[Validation] Checking {len(tickers)} tickers (min {min_history_days} days)...")

    valid = []
    invalid = []

    # Process in batches for efficiency
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_str = ' '.join(batch)

        try:
            # yfinance batch download
            data = yf.download(batch_str, period="10y", group_by='ticker',
                               threads=True, progress=False)

            for tk in batch:
                try:
                    if len(batch) == 1:
                        tk_data = data
                    else:
                        tk_data = data[tk] if tk in data.columns.get_level_values(0) else None

                    if tk_data is not None and len(tk_data.dropna()) >= min_history_days:
                        valid.append(tk)
                    else:
                        invalid.append(tk)
                except:
                    invalid.append(tk)

        except Exception as e:
            if verbose:
                print(f"    Batch {i//batch_size + 1} error: {e}")
            # Fallback: try individually
            for tk in batch:
                try:
                    hist = yf.Ticker(tk).history(period="10y")
                    if len(hist) >= min_history_days:
                        valid.append(tk)
                    else:
                        invalid.append(tk)
                except:
                    invalid.append(tk)

        if verbose:
            pct = (i + len(batch)) / len(tickers) * 100
            print(f"    Progress: {i + len(batch)}/{len(tickers)} ({pct:.0f}%) — "
                  f"{len(valid)} valid, {len(invalid)} invalid")

        time.sleep(0.5)

    if verbose:
        print(f"\n  Valid: {len(valid)} / {len(tickers)} "
              f"({len(valid)/len(tickers)*100:.0f}%)")
        print(f"  Invalid/insufficient: {len(invalid)}")

    return valid, invalid


# ============================================================
# FRED MACRO DATA (free, requires API key)
# ============================================================

FRED_SERIES = {
    # Interest rates
    'DFF': 'fed_funds_rate',          # Federal Funds Effective Rate (daily)
    'DGS10': 'treasury_10y',          # 10-Year Treasury Rate (daily)
    'DGS2': 'treasury_2y',            # 2-Year Treasury Rate (daily)
    'T10Y2Y': 'yield_spread_10y2y',   # 10Y-2Y Spread (daily)

    # Market fear gauge
    'VIXCLS': 'vix',                  # CBOE VIX (daily)

    # Inflation
    'CPIAUCSL': 'cpi',                # CPI (monthly)

    # Unemployment
    'UNRATE': 'unemployment',         # Unemployment Rate (monthly)

    # GDP growth
    'A191RL1Q225SBEA': 'gdp_growth',  # Real GDP Growth (quarterly)

    # USD strength (DXY not in FRED, use Trade Weighted Index instead)
    'DTWEXBGS': 'usd_broad',          # Trade Weighted USD Index (daily)

    # Corporate credit spread
    'BAA10Y': 'credit_spread',        # Baa - 10Y Treasury Spread (daily)
}


def fetch_fred_data(api_key, start_date='2014-01-01', verbose=True):
    """
    Fetch macroeconomic data from the FRED API.
    Free API key: https://fred.stlouisfed.org/docs/api/api_key.html

    Returns:
        macro_df: pandas DataFrame (date index, columns = feature names)
    """
    if not api_key:
        if verbose:
            print("[FRED] No API key provided. Get one free at:")
            print("  https://fred.stlouisfed.org/docs/api/api_key.html")
        return None

    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
    except ImportError:
        if verbose:
            print("[FRED] pip install fredapi")
        return None

    import pandas as pd

    if verbose:
        print(f"\n[FRED] Fetching {len(FRED_SERIES)} macro series...")

    series_data = {}
    for series_id, name in FRED_SERIES.items():
        try:
            data = fred.get_series(series_id, observation_start=start_date)
            series_data[name] = data
            if verbose:
                print(f"    {name} ({series_id}): {len(data)} observations")
        except Exception as e:
            if verbose:
                print(f"    {name} ({series_id}): failed ({e})")
            series_data[name] = pd.Series(dtype=float)

    # Combine into DataFrame
    macro_df = pd.DataFrame(series_data)

    # Forward-fill monthly/quarterly data to daily
    macro_df = macro_df.ffill()

    # Derived features
    if 'treasury_10y' in macro_df.columns and 'treasury_2y' in macro_df.columns:
        macro_df['yield_curve_slope'] = macro_df['treasury_10y'] - macro_df['treasury_2y']

    if 'vix' in macro_df.columns:
        macro_df['vix_ma30'] = macro_df['vix'].rolling(30).mean()
        macro_df['vix_regime'] = (macro_df['vix'] > 25).astype(float)  # 1 = high volatility

    if verbose:
        print(f"\n  FRED DataFrame: {macro_df.shape[0]} days × {macro_df.shape[1]} features")
        print(f"  Date range: {macro_df.index[0]} ~ {macro_df.index[-1]}")

    return macro_df


def get_macro_features_for_date(macro_df, date):
    """
    Return macro features for a given date as a dict.
    Called during snapshot generation in historical.py.
    """
    import pandas as pd

    if macro_df is None or macro_df.empty:
        return {}

    # Use the given date or the most recent prior date
    if isinstance(date, str):
        date = pd.Timestamp(date)

    # tz safety — macro_df is tz-naive, normalize date too
    if macro_df.index.tz is not None and date.tz is None:
        date = date.tz_localize(macro_df.index.tz)
    elif macro_df.index.tz is None and date.tz is not None:
        date = date.tz_localize(None)

    idx = macro_df.index.get_indexer([date], method='ffill')[0]
    if idx < 0:
        return {}

    row = macro_df.iloc[idx]
    return {f'macro_{k}': float(v) if not pd.isna(v) else 0.0
            for k, v in row.items()}


# ============================================================
# FAMA-FRENCH FACTORS (free)
# ============================================================

def fetch_fama_french_factors(start_date='2014-01-01', verbose=True):
    """
    Download 5-factor data from the Kenneth French Data Library.
    Free, no API key required.

    Factors:
      Mkt-RF: Market excess return
      SMB: Small Minus Big (size)
      HML: High Minus Low (value)
      RMW: Robust Minus Weak (profitability)
      CMA: Conservative Minus Aggressive (investment)
      RF: Risk-free rate

    Returns:
        ff_df: pandas DataFrame (date index, 6 columns)
    """
    try:
        import pandas as pd
        import urllib.request
        import zipfile
        import io

        if verbose:
            print("\n[Fama-French] Downloading 5-factor data...")

        url = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
               "ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")

        headers = {"User-Agent": "Mozilla/5.0 (stock-pipeline/2.0)"}
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req, timeout=30)
        zip_data = io.BytesIO(resp.read())

        with zipfile.ZipFile(zip_data) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                lines = f.read().decode('utf-8').split('\n')

        # Parse: skip header rows, find data start
        data_lines = []
        started = False
        for line in lines:
            line = line.strip()
            if not line:
                if started:
                    break
                continue
            parts = line.split(',')
            if len(parts) >= 7 and parts[0].strip().isdigit() and len(parts[0].strip()) == 8:
                started = True
                data_lines.append(parts)

        if not data_lines:
            if verbose:
                print("  Failed to parse data")
            return None

        # Build DataFrame
        dates = [pd.Timestamp(f"{r[0][:4]}-{r[0][4:6]}-{r[0][6:8]}") for r in data_lines]
        cols = ['mkt_excess', 'smb', 'hml', 'rmw', 'cma', 'rf']
        values = []
        for r in data_lines:
            row = [float(r[i].strip()) / 100 for i in range(1, 7)]  # convert from % to decimal
            values.append(row)

        ff_df = pd.DataFrame(values, index=dates, columns=cols)
        ff_df = ff_df[ff_df.index >= start_date]

        if verbose:
            print(f"  Fama-French: {len(ff_df)} trading days × {len(cols)} factors")
            print(f"  Date range: {ff_df.index[0].date()} ~ {ff_df.index[-1].date()}")

        return ff_df

    except Exception as e:
        if verbose:
            print(f"  Fama-French download failed: {e}")
        return None


def get_ff_features_for_date(ff_df, date, lookback=63):
    """
    Return factor averages/cumulatives over the last lookback days up to the given date.
    """
    import pandas as pd

    if ff_df is None or ff_df.empty:
        return {}

    if isinstance(date, str):
        date = pd.Timestamp(date)

    # [v2.3.2] tz safety
    if ff_df.index.tz is not None and date.tz is None:
        date = date.tz_localize(ff_df.index.tz)
    elif ff_df.index.tz is None and date.tz is not None:
        date = date.tz_localize(None)

    # Data up to the given date
    mask = ff_df.index <= date
    recent = ff_df[mask].tail(lookback)

    if len(recent) < 10:
        return {}

    features = {}
    for col in ff_df.columns:
        features[f'ff_{col}_mean'] = float(recent[col].mean())
        features[f'ff_{col}_cum'] = float(recent[col].sum())

    # Factor momentum (last 21 days vs prior 42 days)
    if len(recent) >= 42:
        for col in ['mkt_excess', 'smb', 'hml']:
            recent_21 = recent[col].tail(21).mean()
            prev_42 = recent[col].head(42).mean()
            features[f'ff_{col}_momentum'] = float(recent_21 - prev_42)

    return features


# ============================================================
# CROSS-ASSET DATA (yfinance, free)
# ============================================================

CROSS_ASSET_TICKERS = {
    '^VIX': 'vix_level',
    '^TNX': 'treasury_10y_yield',
    'GC=F': 'gold_price',
    'CL=F': 'oil_price',
    'DX-Y.NYB': 'usd_index',
}


def fetch_cross_asset_data(period='10y', verbose=True):
    """
    Fetch cross-asset data from yfinance.
    VIX, treasury rates, gold, oil, USD index.
    """
    import yfinance as yf
    import pandas as pd

    if verbose:
        print(f"\n[Cross-Asset] Fetching {len(CROSS_ASSET_TICKERS)} assets...")

    all_data = {}
    for ticker, name in CROSS_ASSET_TICKERS.items():
        try:
            hist = yf.Ticker(ticker).history(period=period)
            if len(hist) > 100:
                all_data[name] = hist['Close']
                if verbose:
                    print(f"    {name} ({ticker}): {len(hist)} days")
        except Exception as e:
            if verbose:
                print(f"    {name} ({ticker}): failed ({e})")

    if all_data:
        df = pd.DataFrame(all_data)
        df = df.ffill()
        if verbose:
            print(f"\n  Cross-Asset: {df.shape[0]} days × {df.shape[1]} assets")
        return df
    return None


def get_cross_asset_features(cross_df, date, lookback=63):
    """Return cross-asset features for a given date."""
    import pandas as pd
    import numpy as np

    if cross_df is None or cross_df.empty:
        return {}

    if isinstance(date, str):
        date = pd.Timestamp(date)

    # tz safety — yfinance is tz-aware (UTC), date is typically tz-naive
    # without this, TypeError is caught silently and 15 features are lost
    if cross_df.index.tz is not None and date.tz is None:
        date = date.tz_localize(cross_df.index.tz)
    elif cross_df.index.tz is None and date.tz is not None:
        date = date.tz_localize(None)

    mask = cross_df.index <= date
    recent = cross_df[mask].tail(lookback)

    if len(recent) < 10:
        return {}

    features = {}
    for col in cross_df.columns:
        vals = recent[col].dropna().values
        if len(vals) < 5:
            continue

        features[f'xasset_{col}_level'] = float(vals[-1])
        features[f'xasset_{col}_return_30d'] = float(vals[-1] / vals[-min(21, len(vals))] - 1) * 100
        features[f'xasset_{col}_vol_30d'] = float(np.std(np.diff(vals[-22:]) / vals[-22:-1]) * np.sqrt(252) * 100) if len(vals) >= 22 else 0

    return features


# ============================================================
# CLI: python training_universe.py
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TRAINING UNIVERSE BUILDER")
    print("=" * 70)

    tickers, gics = get_training_tickers(verbose=True)

    # GICS sector distribution
    if gics:
        from collections import Counter
        sector_counts = Counter(gics.values())
        print(f"\nGICS Sector Distribution:")
        for sector, count in sector_counts.most_common():
            print(f"  {sector}: {count}")

    # Estimated training samples
    est_samples = len(tickers) * 250  # ~250 snapshots per ticker (10y / 10days)
    print(f"\nEstimated training samples: ~{est_samples:,}")
    print(f"  (vs current: 5,884 samples — {est_samples/5884:.0f}x increase)")

    # Test Fama-French
    ff = fetch_fama_french_factors(verbose=True)

    # Test cross-asset
    xasset = fetch_cross_asset_data(period='1y', verbose=True)

    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"  Training tickers: {len(tickers)}")
    print(f"  Estimated samples: ~{est_samples:,}")
    print(f"  Fama-French: {'OK' if ff is not None else 'FAILED'}")
    print(f"  Cross-Asset: {'OK' if xasset is not None else 'FAILED'}")
    print(f"  FRED: Needs API key (get at fred.stlouisfed.org)")
    print(f"{'=' * 70}")
