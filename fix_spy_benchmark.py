"""
fix_spy_benchmark.py — Recover SPY benchmark for Stage 2 results

The original Task B in stage2_retrain.py failed:
  [Task B] SPY fetch failed: 'NoneType' object is not subscriptable

Likely cause: SPY's first row had None close, or column access issue.
This script re-fetches SPY history and computes 3-month forward returns
aligned to each fold's test date range.

Usage:
  python fix_spy_benchmark.py

Writes to:
  results/stage2/top1_trial58/spy_benchmark.csv (overwrites empty version)
"""
import json
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

SUMMARY_PATH = Path('results/stage2/top1_trial58/summary.json')
OUTPUT_PATH = Path('results/stage2/top1_trial58/spy_benchmark.csv')


def fetch_spy(period='10y'):
    """Fetch SPY history. Returns DataFrame with tz-naive index."""
    print(f"Fetching SPY {period}...")
    spy = yf.Ticker('SPY').history(period=period, auto_adjust=True)
    if spy is None or len(spy) == 0:
        raise RuntimeError("SPY fetch returned empty")
    if spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)
    print(f"  SPY: {len(spy)} rows, {spy.index[0].date()} to {spy.index[-1].date()}")
    return spy


def compute_3m_returns(spy, date_start, date_end):
    """Compute SPY 3-month forward returns averaged over weekly anchor dates
    in [date_start, date_end]. Mirrors stage2_retrain.fetch_spy_benchmark logic
    but with safer indexing."""
    d_start = pd.to_datetime(date_start)
    d_end = pd.to_datetime(date_end)

    # Cap end so 63 trading days forward stays within available data
    spy_last = spy.index[-1]
    d_end = min(d_end, spy_last - pd.Timedelta(days=100))

    returns = []
    for d in pd.date_range(d_start, d_end, freq='W'):
        idx = spy.index.searchsorted(d)
        if idx < 0 or idx >= len(spy) - 63:
            continue
        p_start = spy['Close'].iloc[idx]
        p_end = spy['Close'].iloc[idx + 63]
        if pd.isna(p_start) or pd.isna(p_end) or p_start <= 0:
            continue
        returns.append((p_end / p_start) - 1.0)

    if not returns:
        return None

    arr = np.array(returns, dtype=float)
    return {
        'mean_3m_return': float(arr.mean()),
        'std_3m_return': float(arr.std()),
        'n_obs': int(len(arr)),
    }


def main():
    if not SUMMARY_PATH.exists():
        print(f"ERROR: {SUMMARY_PATH} not found")
        sys.exit(1)

    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    spy = fetch_spy(period='10y')

    # All folds have same date range in stratified k-fold (ticker split)
    # so we compute SPY benchmark once and apply to all folds
    rows = []
    spy_cache = {}

    for fold_info in summary['per_fold']:
        fold_id = fold_info['fold_id']
        # date range comes from per_fold but isn't stored in summary;
        # use the same range as the original empty CSV
        date_start = '2016-07-21'
        date_end = '2026-01-16'

        cache_key = (date_start, date_end)
        if cache_key not in spy_cache:
            spy_cache[cache_key] = compute_3m_returns(spy, date_start, date_end)

        spy_data = spy_cache[cache_key]
        top5_actual = fold_info['selection_alpha'] + summary['aggregate']['all_return_mean']
        # Better: compute top5_actual directly from per_fold if available
        # For now, use the formula that gave us the known top5_return_mean

        # Actually, summary doesn't store per-fold top5_actual directly;
        # recompute as alpha + universe_mean (same all_mean across folds)
        # But this is approximate — better to read from full_ranking.csv

        # Read true top5_actual from fold's full_ranking.csv
        # [v2.3.7 fix] Derive from SUMMARY_PATH.parent so monkey-patching the
        # path works correctly (was hardcoded results/stage2/top1_trial58/)
        full_ranking_path = SUMMARY_PATH.parent / f"fold_{fold_id}" / "full_ranking.csv"
        if full_ranking_path.exists():
            with open(full_ranking_path) as f:
                reader = csv.DictReader(f)
                ranks = list(reader)
            top5 = sorted(ranks, key=lambda r: int(r['pred_rank']))[:5]
            top5_actual = float(np.mean([float(r['actual_ret']) for r in top5]))
        # else fall back to approximation above

        if spy_data is not None:
            alpha_vs_spy = top5_actual - spy_data['mean_3m_return']
        else:
            alpha_vs_spy = None

        rows.append({
            'fold_id': fold_id,
            'fold_date_start': date_start,
            'fold_date_end': date_end,
            'spy_mean_3m_return': spy_data['mean_3m_return'] if spy_data else None,
            'spy_std_3m_return': spy_data['std_3m_return'] if spy_data else None,
            'spy_n_obs': spy_data['n_obs'] if spy_data else None,
            'top5_actual_3m': top5_actual,
            'alpha_vs_spy': alpha_vs_spy,
        })

    # Write CSV
    with open(OUTPUT_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {OUTPUT_PATH}")
    print()
    print(f"{'fold':5s} {'top5_3m':>10s} {'spy_3m':>10s} {'alpha_vs_spy':>14s} {'spy_n_obs':>10s}")
    for r in rows:
        print(f"{r['fold_id']:5d} "
              f"{r['top5_actual_3m']*100:>9.1f}% "
              f"{r['spy_mean_3m_return']*100:>9.1f}% "
              f"{r['alpha_vs_spy']*100:>13.1f}%p "
              f"{r['spy_n_obs']:>10d}")


if __name__ == '__main__':
    main()
