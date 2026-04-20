"""
data_auto.py — automated data collection via yfinance.

Fetches 50+ features per ticker:
  - Price & Technical: 30/60/90/180d returns, volatility, RSI, MACD, Bollinger
  - Fundamentals: PE, PB, PS, EV/EBITDA, PEG, ROE, ROA, margins, growth
  - Analyst: target price, consensus, # of analysts, revision trend
  - Risk: beta, short interest, debt ratios
  - Market: market cap, volume, institutional ownership
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# TICKER CONFIGURATION — can be dynamically set by screener.py
# ============================================================
_DEFAULT_TICKERS = [
    # Sector A1: AI Silicon
    "NVDA", "AVGO", "TSM", "AMD", "MU", "INTC", "ARM", "QCOM", "MRVL",
    # Sector A2: AI Platform / Cloud Infrastructure
    "MSFT", "GOOGL", "AMZN", "META", "ORCL", "CRM", "SNOW", "NOW",
    # Sector A3: AI Consumer / Application
    "AAPL", "TSLA", "PLTR",
    # Sector B: Neuromodulation
    "BWAY", "INSP", "MDT",
    # Sector C: CNS Pharma
    "AXSM",
    # Sector D: Digital Health
    "HIMS", "TDOC",
    # Sector E: ETFs
    "VOO", "QQQ", "SOXX", "XBI",
]

_DEFAULT_SECTORS = {
    "NVDA":"A","AVGO":"A","TSM":"A","AMD":"A","MU":"A","INTC":"A","ARM":"A","QCOM":"A","MRVL":"A",
    "MSFT":"A","GOOGL":"A","AMZN":"A","META":"A","ORCL":"A","CRM":"A","SNOW":"A","NOW":"A",
    "AAPL":"A","TSLA":"A","PLTR":"A",
    "BWAY":"B","INSP":"B","MDT":"B",
    "AXSM":"C",
    "HIMS":"D","TDOC":"D",
    "VOO":"E","QQQ":"E","SOXX":"E","XBI":"E",
}

TICKERS = list(_DEFAULT_TICKERS)
SECTORS = dict(_DEFAULT_SECTORS)


def set_universe(tickers, sector_map):
    """Called by screener.py to dynamically set the universe."""
    global TICKERS, SECTORS
    TICKERS = list(tickers)
    SECTORS.update(sector_map)

# ============================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================

def compute_technical_features(hist):
    """
    Compute technical indicators from price time series.

    Args:
        hist: yfinance .history() DataFrame (columns: Open, High, Low, Close, Volume)

    Returns:
        dict of technical features
    """
    close = hist['Close'].values
    volume = hist['Volume'].values
    
    if len(close) < 30:
        return _empty_technical()
    
    features = {}
    
    # === Returns (momentum) ===
    features['return_5d'] = (close[-1] / close[-5] - 1) * 100 if len(close) >= 5 else 0
    features['return_30d'] = (close[-1] / close[-21] - 1) * 100 if len(close) >= 21 else 0
    features['return_60d'] = (close[-1] / close[-42] - 1) * 100 if len(close) >= 42 else 0
    features['return_90d'] = (close[-1] / close[-63] - 1) * 100 if len(close) >= 63 else 0
    features['return_180d'] = (close[-1] / close[-126] - 1) * 100 if len(close) >= 126 else 0
    features['return_1y'] = (close[-1] / close[0] - 1) * 100 if len(close) >= 200 else 0
    
    # === Volatility ===
    daily_returns = np.diff(close) / close[:-1]
    features['volatility_30d'] = np.std(daily_returns[-21:]) * np.sqrt(252) * 100 if len(daily_returns) >= 21 else 0
    features['volatility_90d'] = np.std(daily_returns[-63:]) * np.sqrt(252) * 100 if len(daily_returns) >= 63 else 0
    
    # === 52-week high/low ===
    high_52w = np.max(close[-252:]) if len(close) >= 252 else np.max(close)
    low_52w = np.min(close[-252:]) if len(close) >= 252 else np.min(close)
    features['pct_from_52w_high'] = (close[-1] / high_52w - 1) * 100
    features['pct_from_52w_low'] = (close[-1] / low_52w - 1) * 100
    features['range_52w_pct'] = (high_52w - low_52w) / low_52w * 100
    
    # === RSI (14-day) ===
    if len(daily_returns) >= 14:
        gains = np.where(daily_returns > 0, daily_returns, 0)
        losses = np.where(daily_returns < 0, -daily_returns, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:]) + 1e-10
        rs = avg_gain / avg_loss
        features['rsi_14'] = 100 - (100 / (1 + rs))
    else:
        features['rsi_14'] = 50
    
    # === Moving Average signals ===
    ma_20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
    ma_50 = np.mean(close[-50:]) if len(close) >= 50 else close[-1]
    ma_200 = np.mean(close[-200:]) if len(close) >= 200 else close[-1]
    features['price_vs_ma20'] = (close[-1] / ma_20 - 1) * 100
    features['price_vs_ma50'] = (close[-1] / ma_50 - 1) * 100
    features['price_vs_ma200'] = (close[-1] / ma_200 - 1) * 100
    features['ma20_vs_ma50'] = (ma_20 / ma_50 - 1) * 100
    
    # === Volume ===
    avg_vol_30 = np.mean(volume[-21:]) if len(volume) >= 21 else np.mean(volume)
    avg_vol_90 = np.mean(volume[-63:]) if len(volume) >= 63 else np.mean(volume)
    features['volume_ratio_30d'] = avg_vol_30 / (avg_vol_90 + 1) * 100
    features['volume_trend'] = (avg_vol_30 / (avg_vol_90 + 1) - 1) * 100
    
    # === Bollinger Band position ===
    if len(close) >= 20:
        bb_mid = np.mean(close[-20:])
        bb_std = np.std(close[-20:])
        features['bollinger_pct'] = (close[-1] - bb_mid) / (2 * bb_std + 1e-10) * 100  # -100 to +100
    else:
        features['bollinger_pct'] = 0
    
    # === Max Drawdown (90d) ===
    if len(close) >= 63:
        window = close[-63:]
        peak = np.maximum.accumulate(window)
        drawdown = (window - peak) / peak
        features['max_drawdown_90d'] = np.min(drawdown) * 100
    else:
        features['max_drawdown_90d'] = 0
        
    # === ADVANCED TECHNICALS ===
    daily_returns = np.diff(close) / close[:-1]
    high = hist['High'].values
    low = hist['Low'].values
    
    # MACD (12, 26, 9)
    e12 = _ema(close, 12)
    e26 = _ema(close, 26)
    if e12 is not None and e26 is not None:
        ml = e12 - e26
        features['macd'] = ml[-1]
        features['macd_signal'] = _ema_val(ml[-30:], 9) if len(ml) >= 30 else 0
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        features['macd_pct'] = features['macd'] / (close[-1] + 1e-10) * 100

    # Stochastic K (14)
    if len(close) >= 14:
        features['stochastic_k'] = (close[-1] - np.min(low[-14:])) / (np.max(high[-14:]) - np.min(low[-14:]) + 1e-10) * 100

    # ADX (14)
    if len(close) >= 28:
        features['adx_14'] = _adx(high, low, close, 14)

    # ATR (14)
    if len(close) >= 15:
        tr = np.maximum(high[-14:] - low[-14:], np.maximum(np.abs(high[-14:] - close[-15:-1]), np.abs(low[-14:] - close[-15:-1])))
        features['atr_14'] = np.mean(tr)
        features['atr_pct'] = features['atr_14'] / close[-1] * 100

    # OBV trend (30d)
    if len(close) >= 31:
        obv = np.cumsum(np.where(np.diff(close[-31:]) > 0, volume[-30:], -volume[-30:]))
        features['obv_trend_30d'] = (obv[-1] - obv[0]) / (np.abs(obv).mean() + 1) * 100

    # Williams %R (14)
    if len(close) >= 14:
        features['williams_r'] = (np.max(high[-14:]) - close[-1]) / (np.max(high[-14:]) - np.min(low[-14:]) + 1e-10) * -100

    # CCI (20)
    if len(close) >= 20:
        tp = (high[-20:] + low[-20:] + close[-20:]) / 3
        features['cci_20'] = (tp[-1] - np.mean(tp)) / (0.015 * np.std(tp) + 1e-10)

    # Donchian Channels (20, 50)
    for p in [20, 50]:
        if len(close) >= p:
            features[f'donchian_high_{p}'] = (close[-1] / np.max(high[-p:]) - 1) * 100
            features[f'donchian_low_{p}'] = (close[-1] / np.min(low[-p:]) - 1) * 100

    # Volume spike count
    if len(volume) >= 63:
        features['volume_spike_30d'] = float(np.sum(volume[-21:] > 2 * np.mean(volume[-63:])))

    # === STATISTICAL ===
    if len(daily_returns) >= 21:
        r = daily_returns[-21:]
        m, s = np.mean(r), np.std(r) + 1e-10
        features['skew_30d'] = np.mean(((r - m) / s) ** 3)
        features['kurt_30d'] = np.mean(((r - m) / s) ** 4) - 3
        if len(r) > 1:
            features['autocorr_30d'] = np.corrcoef(r[:-1], r[1:])[0, 1]
        features['pos_day_pct'] = np.mean(r > 0) * 100
        features['avg_gain'] = np.mean(r[r > 0]) * 100 if np.any(r > 0) else 0
        features['avg_loss'] = np.mean(r[r < 0]) * 100 if np.any(r < 0) else 0
        features['gain_loss_ratio'] = abs(features['avg_gain'] / (features['avg_loss'] + 1e-10))

    if len(daily_returns) >= 63:
        r = daily_returns[-63:]
        m, s = np.mean(r), np.std(r) + 1e-10
        features['skew_90d'] = np.mean(((r - m) / s) ** 3)
        features['kurt_90d'] = np.mean(((r - m) / s) ** 4) - 3
        if len(r) > 1:
            features['autocorr_90d'] = np.corrcoef(r[:-1], r[1:])[0, 1]

    # === MULTI-TIMEFRAME ===
    for d, lab in [(5, '1w'), (10, '2w'), (21, '1m'), (63, '3m'), (126, '6m')]:
        if len(close) >= d + 1:
            features[f'mtf_mom_{lab}'] = (close[-1] / close[-d-1] - 1) * 100
            features[f'mtf_vol_{lab}'] = np.std(daily_returns[-d:]) * np.sqrt(252) * 100 if len(daily_returns) >= d else 0
    return features


def extract_fundamental_features(info, financials=None, balance=None, cashflow=None):
    """
    Extract fundamental metrics from yfinance .info + .financials.

    Args:
        info: ticker.info dict
        financials: ticker.financials DataFrame (optional)
        balance: ticker.balance_sheet DataFrame (optional)
        cashflow: ticker.cashflow DataFrame (optional)

    Returns:
        dict of fundamental features
    """
    f = {}
    
    # Safe getter
    def g(key, default=0):
        val = info.get(key, default)
        return val if val is not None else default
    
    # === Growth ===
    f['revenue_growth'] = g('revenueGrowth', 0) * 100
    f['earnings_growth'] = g('earningsGrowth', 0) * 100
    f['revenue_per_share'] = g('revenuePerShare', 0)
    
    # === Profitability ===
    f['gross_margin'] = g('grossMargins', 0) * 100
    f['operating_margin'] = g('operatingMargins', 0) * 100
    f['net_margin'] = g('profitMargins', 0) * 100
    f['roe'] = g('returnOnEquity', 0) * 100
    f['roa'] = g('returnOnAssets', 0) * 100
    
    # === Valuation ===
    f['forward_pe'] = g('forwardPE', 0)
    f['trailing_pe'] = g('trailingPE', 0)
    f['peg_ratio'] = g('pegRatio', 0)
    f['price_to_book'] = g('priceToBook', 0)
    f['price_to_sales'] = g('priceToSalesTrailing12Months', 0)
    f['ev_ebitda'] = g('enterpriseToEbitda', 0)
    f['ev_revenue'] = g('enterpriseToRevenue', 0)
    
    # === Analyst ===
    f['target_mean_price'] = g('targetMeanPrice', 0)
    f['target_low_price'] = g('targetLowPrice', 0)
    f['target_high_price'] = g('targetHighPrice', 0)
    f['recommendation_mean'] = g('recommendationMean', 3)  # 1=strong buy, 5=sell
    f['num_analysts'] = g('numberOfAnalystOpinions', 0)
    
    current_price = g('currentPrice', g('regularMarketPrice', 1))
    if current_price > 0 and f['target_mean_price'] > 0:
        f['analyst_upside'] = (f['target_mean_price'] / current_price - 1) * 100
    else:
        f['analyst_upside'] = 0
    
    # === Risk ===
    f['beta'] = g('beta', 1.0)
    f['debt_to_equity'] = g('debtToEquity', 0) / 100 if g('debtToEquity', 0) else 0
    f['current_ratio'] = g('currentRatio', 0)
    f['quick_ratio'] = g('quickRatio', 0)
    f['short_ratio'] = g('shortRatio', 0)
    f['short_pct_float'] = g('shortPercentOfFloat', 0) * 100
    
    # === Market Structure ===
    f['market_cap_log'] = np.log10(max(g('marketCap', 1e6), 1e6))
    f['institutional_pct'] = g('heldPercentInstitutions', 0) * 100
    f['insider_pct'] = g('heldPercentInsiders', 0) * 100
    f['float_shares_log'] = np.log10(max(g('floatShares', 1e6), 1e6))
    
    # === Dividends ===
    f['dividend_yield'] = g('dividendYield', 0) * 100 if g('dividendYield') else 0
    f['payout_ratio'] = g('payoutRatio', 0) * 100 if g('payoutRatio') else 0
    
    # === Cash Flow ===
    f['free_cashflow_yield'] = 0
    fcf = g('freeCashflow', 0)
    mcap = g('marketCap', 1)
    if mcap > 0 and fcf:
        f['free_cashflow_yield'] = (fcf / mcap) * 100
    
    return f


def _empty_technical():
    """Return empty technical features dict."""
    keys = [
        'return_5d','return_30d','return_60d','return_90d','return_180d','return_1y',
        'volatility_30d','volatility_90d',
        'pct_from_52w_high','pct_from_52w_low','range_52w_pct',
        'rsi_14','price_vs_ma20','price_vs_ma50','price_vs_ma200','ma20_vs_ma50',
        'volume_ratio_30d','volume_trend','bollinger_pct','max_drawdown_90d',
    ]
    return {k: 0 for k in keys}


# ============================================================
# MAIN FETCH FUNCTION
# ============================================================

def fetch_single_stock(ticker, period="1y"):
    """
    Collect all features for a single ticker.

    Args:
        ticker: str (e.g. "NVDA")
        period: str (e.g. "1y", "2y")

    Returns:
        dict with 'features' (numpy array), 'feature_names' (list), 'meta' (dict)
    """
    import yfinance as yf
    
    tk = yf.Ticker(ticker)
    
    # 1. Price history → technical features
    hist = tk.history(period=period)
    tech = compute_technical_features(hist)
    
    # 2. Fundamentals
    info = tk.info
    fund = extract_fundamental_features(info)
    
    # 3. Is ETF?
    is_etf = 1 if info.get('quoteType') == 'ETF' else 0
    
    # Combine all features
    all_features = {}
    all_features.update(tech)
    all_features.update(fund)
    all_features['is_etf'] = is_etf
    
    # Convert to ordered numpy array
    feature_names = sorted(all_features.keys())
    feature_values = np.array([all_features[k] for k in feature_names], dtype=float)
    
    # Replace NaN/Inf
    feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Target risk: compute directly from price data (avoid info API dependency)
    close = hist['Close'].dropna().values
    daily_ret = np.diff(close) / close[:-1]
    target_risk_val = max(np.std(daily_ret) * np.sqrt(252), 0.01) if len(daily_ret) > 5 else 0.3
    
    meta = {
        'ticker': ticker,
        'name': info.get('shortName', ticker),
        'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
        'sector': SECTORS.get(ticker, 'X'),
        'n_features': len(feature_names),
        'target_risk': target_risk_val,
    }
    
    return {
        'features': feature_values,
        'feature_names': feature_names,
        'meta': meta,
        'raw': all_features,
    }


def fetch_all_stocks(tickers=None, period="1y", verbose=True):
    """
    Collect features for all tickers.

    Args:
        tickers: list of str. If None, uses TICKERS.
        period: str (default "1y")

    Returns:
        dict: {
            'tickers': list,
            'features': np.array (N, D),
            'feature_names': list,
            'meta': dict per ticker,
            'targets': dict per ticker (if available),
        }
    
    Usage:
        from data_auto import fetch_all_stocks
        data = fetch_all_stocks()
        print(f"Shape: {data['features'].shape}")  # (17, ~60)
        print(f"Features: {data['feature_names']}")
    """
    if tickers is None:
        tickers = TICKERS
    
    if verbose:
        print(f"Fetching {len(tickers)} stocks from Yahoo Finance...")
    
    results = {}
    feature_names = None
    
    for i, tk in enumerate(tickers):
        try:
            result = fetch_single_stock(tk, period)
            results[tk] = result
            
            if feature_names is None:
                feature_names = result['feature_names']
            
            if verbose:
                print(f"  [{i+1}/{len(tickers)}] {tk:>6}: {result['meta']['n_features']} features, ${result['meta']['price']:.2f}")
        
        except Exception as e:
            print(f"  [{i+1}/{len(tickers)}] {tk:>6}: ERROR — {e}")
            results[tk] = None
    
    # Build feature matrix — filter out delisted/incomplete tickers
    expected_n = len(feature_names) if feature_names else 0
    valid_tickers = []
    for tk in tickers:
        r = results.get(tk)
        if r is None:
            continue
        if len(r['features']) != expected_n:
            print(f"  {tk:>6}: SKIPPED (features={len(r['features'])}, expected={expected_n})")
            continue
        if r['meta'].get('price', 0) <= 0:
            print(f"  {tk:>6}: SKIPPED (delisted/no price)")
            continue
        valid_tickers.append(tk)
    
    if not valid_tickers:
        raise RuntimeError("No stocks successfully fetched!")
    
    feature_matrix = np.array([results[tk]['features'] for tk in valid_tickers])
    
    # Build targets from analyst data
    targets = {}
    for tk in valid_tickers:
        raw = results[tk]['raw']
        price = results[tk]['meta']['price']
        
        # Target return = analyst upside
        target_ret = raw.get('analyst_upside', 0) / 100
        target_risk = results[tk]['meta']['target_risk']
        targets[tk] = {'return': np.clip(target_ret, -0.5, 2.0),
                       'risk': np.clip(target_risk, 0.01, 1.0)}
    
    if verbose:
        print(f"\n  Total: {len(valid_tickers)} stocks, {len(feature_names)} features each")
        print(f"  Feature matrix shape: {feature_matrix.shape}")
        print(f"  Feature categories:")
        tech_count = sum(1 for n in feature_names if any(k in n for k in ['return_','volatility_','rsi','ma','volume','bollinger','drawdown','52w']))
        fund_count = sum(1 for n in feature_names if any(k in n for k in ['margin','pe','ratio','growth','roe','roa','ev_','price_to','revenue','earnings','peg']))
        analyst_count = sum(1 for n in feature_names if any(k in n for k in ['target','analyst','recommendation','num_analyst']))
        risk_count = sum(1 for n in feature_names if any(k in n for k in ['beta','debt','short','current_ratio','quick']))
        print(f"    Technical: ~{tech_count}, Fundamental: ~{fund_count}, Analyst: ~{analyst_count}, Risk: ~{risk_count}")
    
    return {
        'tickers': valid_tickers,
        'features': feature_matrix,
        'feature_names': feature_names,
        'meta': {tk: results[tk]['meta'] for tk in valid_tickers},
        'targets': targets,
    }


# ============================================================
# CONVENIENCE: Add/remove tickers easily
# ============================================================

def add_tickers(new_tickers, sector="X"):
    """Dynamically add tickers."""
    for tk in new_tickers:
        if tk not in TICKERS:
            TICKERS.append(tk)
            SECTORS[tk] = sector

def set_tickers(ticker_list, sector_map=None):
    """Replace the entire ticker list."""
    global TICKERS
    TICKERS = list(ticker_list)
    if sector_map:
        SECTORS.update(sector_map)


# ============================================================
# CLI: fetch and save when run directly
# ============================================================

if __name__ == "__main__":
    import json
    
    data = fetch_all_stocks()
    
    # Save to JSON
    output = {
        'tickers': data['tickers'],
        'feature_names': data['feature_names'],
        'n_features': len(data['feature_names']),
        'features': data['features'].tolist(),
        'targets': data['targets'],
        'meta': data['meta'],
    }
    
    with open('results/stock_data.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nSaved to results/stock_data.json")
    print(f"Feature count: {len(data['feature_names'])}")
    print(f"\nAll features:")
    for i, name in enumerate(data['feature_names']):
        val_range = f"[{data['features'][:,i].min():.1f} ~ {data['features'][:,i].max():.1f}]"
        print(f"  [{i:>2}] {name:<30} {val_range}")
        
        
# ============================================================
# Helper functions
# ============================================================

def _ema(data, period):
    if len(data) < period: return None
    a = 2 / (period + 1)
    e = np.zeros(len(data)); e[0] = data[0]
    for i in range(1, len(data)): e[i] = a * data[i] + (1 - a) * e[i-1]
    return e

def _ema_val(data, period):
    e = _ema(data, period)
    return e[-1] if e is not None else 0

def _adx(h, l, c, p=14):
    if len(c) < 2*p: return 0
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    pd = np.where((h[1:]-h[:-1])>(l[:-1]-l[1:]), np.maximum(h[1:]-h[:-1],0), 0)
    md = np.where((l[:-1]-l[1:])>(h[1:]-h[:-1]), np.maximum(l[:-1]-l[1:],0), 0)
    atr = np.mean(tr[-p:]); pdi = np.mean(pd[-p:])/(atr+1e-10)*100; mdi = np.mean(md[-p:])/(atr+1e-10)*100
    return abs(pdi-mdi)/(pdi+mdi+1e-10)*100
