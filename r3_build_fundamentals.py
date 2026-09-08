#!/usr/bin/env python
"""R3 dataset builder — point-in-time fundamentals aligned to the cache grid.

Implements the Features / Dataset custody sections of research_r3_full_prespec.md.
Never modifies backtest_cache.npz; writes results/research_r3/fund_features.npz
(+ audit report). First-filed values only; a value is usable at snapshot t only
if its (first) filing date is strictly before t.

Usage:
  python r3_build_fundamentals.py --tickers AAPL,MSFT,...   # dry run on cached raw
  python r3_build_fundamentals.py                           # all cache tickers (fetches missing raw)
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as config_module

CACHE = 'results/backtest_cache.npz'
OUT = Path('results/research_r3')
RAW = OUT / 'raw'
UA = (os.environ.get('SEC_USER_AGENT')
      or getattr(config_module, 'SEC_USER_AGENT', ''))

CONCEPTS = {
    'revenue': [('us-gaap', 'Revenues'),
                ('us-gaap', 'RevenueFromContractWithCustomerExcludingAssessedTax'),
                ('us-gaap', 'SalesRevenueNet'),
                ('us-gaap', 'RevenuesNetOfInterestExpense'),
                ('us-gaap', 'RegulatedAndUnregulatedOperatingRevenue'),
                ('us-gaap', 'OperatingRevenue')],
    'net_income': [('us-gaap', 'NetIncomeLoss'),
                   ('us-gaap', 'ProfitLoss'),
                   ('us-gaap', 'NetIncomeLossAvailableToCommonStockholdersBasic')],
    'op_cashflow': [('us-gaap', 'NetCashProvidedByUsedInOperatingActivities'),
                    ('us-gaap', 'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations')],
    'equity': [('us-gaap', 'StockholdersEquity'),
               ('us-gaap', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest')],
    'assets': [('us-gaap', 'Assets')],
}
DURATION = ['revenue', 'net_income', 'op_cashflow']
INSTANT = ['equity', 'assets']
FORMS = {'10-Q', '10-K'}
RATIOS = ['f_roe', 'f_net_margin', 'f_asset_growth', 'f_accruals',
          'f_rev_growth', 'f_cfo_assets', 'f_leverage', 'f_ni_growth']
FEATURES = RATIOS + ['f_available']
AUDIT_SAMPLES = [('AAPL', '2021-01'), ('MSFT', '2019-08'), ('JPM', '2023-05'),
                 ('NVDA', '2024-03'), ('DUK', '2022-11')]


# ---------------- SEC access (same header pattern as sentiment.py) ----------------

def sec_get(url):
    import urllib.request
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def load_cik_map():
    f = OUT / 'company_tickers.json'
    if f.exists():
        m = json.loads(f.read_text())
    else:
        m = sec_get('https://www.sec.gov/files/company_tickers.json')
        OUT.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(m))
        time.sleep(0.3)
    return {v['ticker'].upper(): int(v['cik_str']) for v in m.values()}


def load_facts(ticker, cik, fetch=True):
    RAW.mkdir(parents=True, exist_ok=True)
    f = RAW / f'{ticker}.json'
    if f.exists():
        return json.loads(f.read_text())
    if not fetch:
        return None
    for attempt in range(3):
        try:
            facts = sec_get(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json')
            f.write_text(json.dumps(facts))
            time.sleep(0.25)
            return facts
        except Exception as e:
            time.sleep(2.0 * (attempt + 1))
            last = e
    print(f"  [{ticker}] fetch failed: {last}")
    return None


# ---------------- XBRL → quarterly / instant series ----------------

def entries_for(facts, ns, tag):
    node = facts.get('facts', {}).get(ns, {}).get(tag)
    if not node:
        return None
    rows = []
    for unit, arr in node.get('units', {}).items():
        for e in arr:
            if e.get('form') not in FORMS or e.get('val') is None:
                continue
            rows.append({'start': e.get('start'), 'end': e.get('end'),
                         'filed': e.get('filed'), 'val': float(e.get('val'))})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df['end'] = pd.to_datetime(df['end'])
    df['filed'] = pd.to_datetime(df['filed'])
    df['start'] = pd.to_datetime(df['start'], errors='coerce')
    return df


def union_entries(facts, alternates):
    frames = [entries_for(facts, ns, tag) for ns, tag in alternates]
    frames = [f for f in frames if f is not None]
    return pd.concat(frames, ignore_index=True) if frames else None


def quarterly_series(facts, alternates):
    """Quarterly values with availability dates.
    Direct 80-100 day periods, plus same-start differencing (YTD - prior YTD,
    FY - 9M) for companies reporting cumulative values. First-filed only."""
    df = union_entries(facts, alternates)
    if df is None:
        return None
    df = df.dropna(subset=['start']).sort_values('filed')
    df = df.drop_duplicates(['start', 'end'], keep='first')   # first-filed per period
    df['dur'] = (df['end'] - df['start']).dt.days
    out = {}
    direct = df[(df['dur'] >= 70) & (df['dur'] <= 120)]   # 12-16 week quarters
    for _, r in direct.iterrows():
        out[r['end']] = (r['filed'], r['val'], 'direct')
    for start, g in df.groupby('start'):
        g = g.sort_values('end')
        prev = None
        for _, r in g.iterrows():
            if prev is not None:
                gap = (r['end'] - prev['end']).days
                if 70 <= gap <= 120 and r['end'] not in out:
                    out[r['end']] = (max(r['filed'], prev['filed']),
                                     r['val'] - prev['val'], 'derived')
            prev = r
    if not out:
        return None
    q = pd.DataFrame([{'end': e, 'avail': a, 'val': v, 'kind': k}
                      for e, (a, v, k) in out.items()]).sort_values('end')
    return q.reset_index(drop=True)


def instant_series(facts, alternates):
    df = union_entries(facts, alternates)
    if df is None:
        return None
    df = df.sort_values('filed').drop_duplicates(['end'], keep='first')
    return df.rename(columns={'filed': 'avail'})[['end', 'avail', 'val']] \
             .sort_values('end').reset_index(drop=True)


# ---------------- per-snapshot ratios ----------------

def _last_n(q, t, n):
    """Last n quarterly rows available strictly before t, or None if the
    window is not contiguous (~n quarters spanning n*91 ± tolerance days)."""
    if q is None:
        return None
    a = q[q['avail'] < t]
    if len(a) < n:
        return None
    w = a.iloc[-n:]
    span = (w['end'].iloc[-1] - w['end'].iloc[0]).days
    if not ((n - 1) * 70 <= span <= (n - 1) * 120):
        return None
    return w


def ratios_at(t, q_rev, q_ni, q_cfo, i_eq, i_as):
    """Each ratio is computed when its own inputs are available (build
    amendment: v1 required every concept, so one missing tag — e.g. banks
    without a total-revenue tag — zeroed all eight). Returns ratios, audit,
    and the fraction of the eight ratios computable."""
    r = {k: np.nan for k in RATIOS}
    audit = {}
    rev8 = _last_n(q_rev, t, 8)
    ni8 = _last_n(q_ni, t, 8)
    ni4 = ni8 if ni8 is not None else _last_n(q_ni, t, 4)
    cfo4 = _last_n(q_cfo, t, 4)
    eq = i_eq[i_eq['avail'] < t] if i_eq is not None else None
    asst = i_as[i_as['avail'] < t] if i_as is not None else None
    eq_l = eq['val'].iloc[-1] if eq is not None and len(eq) else np.nan
    as_l = asst['val'].iloc[-1] if asst is not None and len(asst) else np.nan
    if not (np.isfinite(as_l) and as_l > 0):
        as_l = np.nan
    as_4q = np.nan
    if asst is not None and len(asst) and np.isfinite(as_l):
        target = asst['end'].iloc[-1] - pd.Timedelta(days=365)
        idx = (asst['end'] - target).abs().idxmin()
        if abs((asst.loc[idx, 'end'] - target).days) <= 60 and asst.loc[idx, 'val'] > 0:
            as_4q = asst.loc[idx, 'val']
    ni_ttm = ni4['val'].iloc[-4:].sum() if ni4 is not None else np.nan
    ni_prior = ni8['val'].iloc[:4].sum() if ni8 is not None else np.nan
    rev_ttm = rev8['val'].iloc[-4:].sum() if rev8 is not None else np.nan
    rev_prior = rev8['val'].iloc[:4].sum() if rev8 is not None else np.nan
    cfo_ttm = cfo4['val'].sum() if cfo4 is not None else np.nan

    if np.isfinite(ni_ttm) and np.isfinite(eq_l) and eq_l > 0:
        r['f_roe'] = ni_ttm / eq_l
    if np.isfinite(ni_ttm) and np.isfinite(rev_ttm) and rev_ttm > 0:
        r['f_net_margin'] = ni_ttm / rev_ttm
    if np.isfinite(as_l) and np.isfinite(as_4q):
        r['f_asset_growth'] = as_l / as_4q - 1
    if np.isfinite(ni_ttm) and np.isfinite(cfo_ttm) and np.isfinite(as_l):
        r['f_accruals'] = (ni_ttm - cfo_ttm) / as_l
    if np.isfinite(rev_ttm) and np.isfinite(rev_prior) and rev_prior > 0:
        r['f_rev_growth'] = rev_ttm / rev_prior - 1
    if np.isfinite(cfo_ttm) and np.isfinite(as_l):
        r['f_cfo_assets'] = cfo_ttm / as_l
    if np.isfinite(eq_l) and np.isfinite(as_l):
        r['f_leverage'] = eq_l / as_l
    if np.isfinite(ni_ttm) and np.isfinite(ni_prior) and np.isfinite(as_l):
        r['f_ni_growth'] = (ni_ttm - ni_prior) / as_l
    if ni4 is not None and asst is not None and len(asst):
        audit = {'ni_latest_end': str(ni4['end'].iloc[-1].date()),
                 'ni_latest_avail': str(ni4['avail'].iloc[-1].date()),
                 'ni_kinds': ''.join('D' if k == 'direct' else 'x' for k in ni4['kind'].iloc[-4:]),
                 'assets_latest_end': str(asst['end'].iloc[-1].date()),
                 'assets_latest_avail': str(asst['avail'].iloc[-1].date())}
    frac = float(np.mean([np.isfinite(r[k]) for k in RATIOS]))
    return r, audit, frac


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tickers', default=None, help='comma list (dry run)')
    ap.add_argument('--cache', default=CACHE)
    ap.add_argument('--no-fetch', action='store_true', help='use cached raw only')
    ap.add_argument('--out', default=str(OUT / 'fund_features.npz'))
    args = ap.parse_args()

    data = np.load(args.cache, allow_pickle=True)
    meta = data['meta']
    s_tk = meta[:, 0].astype(str)
    s_dt = meta[:, 2].astype(str)
    tickers = sorted(set(s_tk))
    if args.tickers:
        tickers = [t for t in args.tickers.split(',') if t in set(s_tk)]
    print(f"[cache] {len(s_tk):,} samples, {len(set(s_tk))} tickers; building for {len(tickers)}")

    if not UA:
        raise SystemExit('SEC_USER_AGENT missing (env or config)')
    cik = load_cik_map()

    n = len(s_tk)
    raw_ratio = np.full((n, len(RATIOS)), np.nan, dtype=np.float64)
    avail = np.zeros(n, dtype=np.float32)
    audits = []
    per_ticker_cov = {}
    t0 = time.time()
    for i, tk in enumerate(tickers):
        c = cik.get(tk)
        facts = load_facts(tk, c, fetch=not args.no_fetch) if c else None
        idx = np.where(s_tk == tk)[0]
        if facts is None:
            per_ticker_cov[tk] = 0.0
            continue
        q_rev = quarterly_series(facts, CONCEPTS['revenue'])
        q_ni = quarterly_series(facts, CONCEPTS['net_income'])
        q_cfo = quarterly_series(facts, CONCEPTS['op_cashflow'])
        i_eq = instant_series(facts, CONCEPTS['equity'])
        i_as = instant_series(facts, CONCEPTS['assets'])
        ok_count = 0
        for j in idx:
            t = pd.Timestamp(str(s_dt[j]))
            r, aud, frac = ratios_at(t, q_rev, q_ni, q_cfo, i_eq, i_as)
            raw_ratio[j] = [r[k] for k in RATIOS]
            avail[j] = frac                      # fraction of the 8 ratios computable
            ok_count += int(frac >= 0.999)
            for atk, amonth in AUDIT_SAMPLES:
                if tk == atk and s_dt[j].startswith(amonth) and aud:
                    audits.append({'ticker': tk, 'snapshot': s_dt[j], **aud,
                                   **{k: round(float(r[k]), 4) for k in RATIOS}})
        per_ticker_cov[tk] = ok_count / len(idx)
        if (i + 1) % 25 == 0 or i == len(tickers) - 1:
            print(f"  {i+1}/{len(tickers)} tickers, {(time.time()-t0)/60:.1f} min")

    # cross-sectional rank-normalization per date; missing -> 0
    F = np.zeros((n, len(FEATURES)), dtype=np.float32)
    dates_u, inv = np.unique(s_dt, return_inverse=True)
    for d in range(len(dates_u)):
        rows = np.where(inv == d)[0]
        for k in range(len(RATIOS)):
            v = raw_ratio[rows, k]
            m = np.isfinite(v)
            if m.sum() >= 5:
                ranks = pd.Series(v[m]).rank(method='average').values
                F[rows[m], k] = (ranks / m.sum()) - 0.5
    F[:, len(RATIOS)] = avail

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, F=F, feat_names=np.array(FEATURES),
                        raw_ratios=raw_ratio.astype(np.float32),
                        ratio_names=np.array(RATIOS), tickers_built=np.array(tickers))

    # audit report
    years = np.array([s[:4] for s in s_dt])
    mask_built = np.isin(s_tk, tickers)
    full = (avail >= 0.999)
    lines = ['# R3 fundamentals build — audit (build v2: widened quarter windows, per-ratio availability)\n',
             f'Built {len(tickers)} tickers; samples {n:,}; mean ratio availability '
             f'{avail[mask_built].mean()*100:.1f}%; all-8-available '
             f'{full[mask_built].mean()*100:.1f}% of built-ticker samples.\n',
             '## Availability by year (built tickers)\n',
             '| year | mean ratio avail | all-8 avail |', '|---|---:|---:|']
    for y in sorted(set(years)):
        m = (years == y) & mask_built
        lines.append(f'| {y} | {avail[m].mean()*100:.1f}% | {full[m].mean()*100:.1f}% |')
    low = sorted([(c, tk) for tk, c in per_ticker_cov.items() if c < 0.5])
    lines += ['', f'Tickers with < 50% all-8 availability: {len(low)} / {len(tickers)}',
              ', '.join(f'{tk}({c*100:.0f}%)' for c, tk in low[:40]) or '(none)', '',
              '## Point-in-time audit samples (verify: avail date < snapshot; latest NI period '
              'is the most recent quarter FILED before the snapshot)\n']
    if audits:
        cols = ['ticker', 'snapshot', 'ni_latest_end', 'ni_latest_avail', 'ni_kinds',
                'assets_latest_end', 'assets_latest_avail'] + RATIOS
        lines.append('| ' + ' | '.join(cols) + ' |')
        lines.append('|' + '---|' * len(cols))
        seen = set()
        for a in audits:
            key = (a['ticker'], a['snapshot'][:7])
            if key in seen:
                continue
            seen.add(key)
            lines.append('| ' + ' | '.join(str(a[c]) for c in cols) + ' |')
    else:
        lines.append('(no audit samples in this ticker set)')
    (OUT / 'fund_build_audit.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f"\nWrote {args.out}\nWrote {OUT/'fund_build_audit.md'}")


if __name__ == '__main__':
    main()
