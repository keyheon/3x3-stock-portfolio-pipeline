#!/usr/bin/env python
"""R3 spike — SEC EDGAR companyfacts point-in-time feasibility (20 tickers).

Implements research_r3_fundamentals_spike_prespec.md. Fetches company
facts JSON per ticker (cached under results/research_r3/raw/, not committed),
resolves concept tags with alternates, scores coverage/filing-lag, and
applies the pre-stated feasibility criteria mechanically.
"""

import json
import os
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import config as config_module

OUT = Path('results/research_r3')
RAW = OUT / 'raw'
UA = (os.environ.get('SEC_USER_AGENT')
      or getattr(config_module, 'SEC_USER_AGENT', ''))   # env overrides config

# 20 fixed tickers across sectors; banks/utilities included on purpose (hard tag cases)
TICKERS = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'TSLA', 'NFLX', 'JPM', 'GS',
           'XOM', 'CVX', 'JNJ', 'PFE', 'LLY', 'PG', 'KO', 'CAT', 'HON',
           'NEE', 'DUK', 'AMT']

# Instrument v2: alternates are UNIONED (v1 used first-match). Bank/utility/
# noncontrolling-interest variants added after the v1 run exposed gaps.
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
    'eps_diluted': [('us-gaap', 'EarningsPerShareDiluted'),
                    ('us-gaap', 'EarningsPerShareBasicAndDiluted')],
    'equity': [('us-gaap', 'StockholdersEquity'),
               ('us-gaap', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest')],
    'assets': [('us-gaap', 'Assets')],
    'op_cashflow': [('us-gaap', 'NetCashProvidedByUsedInOperatingActivities'),
                    ('us-gaap', 'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations')],
    'shares_out': [('dei', 'EntityCommonStockSharesOutstanding'),
                   ('us-gaap', 'CommonStockSharesOutstanding')],
}
DURATION_CONCEPTS = {'revenue', 'net_income', 'eps_diluted', 'op_cashflow'}
CORE = ['revenue', 'net_income', 'equity', 'assets']
WINDOW = ('2016-01-01', '2025-12-31')
MIN_QUARTERS = 30
MAX_LAG_DAYS = 60
FORMS = {'10-Q', '10-K'}


def sec_get(url):
    """Header pattern mirrors the repo's working EDGAR fetch (sentiment.py):
    User-Agent only. Adding Accept-Encoding triggered a WAF 403."""
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


def load_facts(ticker, cik):
    RAW.mkdir(parents=True, exist_ok=True)
    f = RAW / f'{ticker}.json'
    if f.exists():
        return json.loads(f.read_text())
    facts = sec_get(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json')
    f.write_text(json.dumps(facts))
    time.sleep(0.3)
    return facts


def entries_for(facts, ns, tag):
    node = facts.get('facts', {}).get(ns, {}).get(tag)
    if not node:
        return None
    rows = []
    for unit, arr in node.get('units', {}).items():
        for e in arr:
            if e.get('form') not in FORMS:
                continue
            rows.append({'unit': unit, 'start': e.get('start'),
                         'end': e.get('end'), 'val': e.get('val'),
                         'filed': e.get('filed'), 'form': e.get('form'),
                         'fy': e.get('fy'), 'fp': e.get('fp'),
                         'frame': e.get('frame'), 'tag': f'{ns}:{tag}'})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df['end'] = pd.to_datetime(df['end'])
    df['filed'] = pd.to_datetime(df['filed'])
    df['start'] = pd.to_datetime(df['start'], errors='coerce')
    return df


def score_concept(facts, alternates, is_duration):
    """Instrument v2. Union across alternate tags; point-in-time lag = FIRST
    filing date per period (v1 took the median over all re-reports, which is
    dominated by prior-year comparatives ~365d later — an artifact).
    Duration concepts: quarterly values = start..end of 80-100 days; annual
    (FY) values counted as derivable Q4 when Q1-Q3 of that year exist."""
    frames = [entries_for(facts, ns, tag) for ns, tag in alternates]
    frames = [f for f in frames if f is not None]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    w = df[(df['end'] >= WINDOW[0]) & (df['end'] <= WINDOW[1])].copy()
    if len(w) == 0:
        return None
    tags_used = sorted(w['tag'].unique())
    # v1-style lag (all re-reports) kept for the record
    lag_all = (w['filed'] - w['end']).dt.days

    if is_duration:
        dur = (w['end'] - w['start']).dt.days
        q = w[(dur >= 80) & (dur <= 100)]
        fy = w[(dur >= 350) & (dur <= 380)]
    else:
        q = w
        fy = w.iloc[0:0]

    # first availability per period end
    first_q = q.groupby('end')['filed'].min()
    q_quarters = set(first_q.index.to_period('Q').astype(str))
    lag_first = (first_q - first_q.index).dt.days

    derivable = set()
    fy_lags = []
    if len(fy):
        first_fy = fy.groupby('end')['filed'].min()
        for e, f in first_fy.items():
            per = pd.Period(e, freq='Q')
            prior = {str(per - k) for k in (1, 2, 3)}
            if str(per) not in q_quarters and prior <= q_quarters:
                derivable.add(str(per))
                fy_lags.append((f - e).days)
    total = q_quarters | derivable
    all_lags = list(lag_first.values) + fy_lags
    return {
        'tags_used': tags_used,
        'n_entries': int(len(w)),
        'n_quarters_direct': int(len(q_quarters)),
        'n_q4_derivable': int(len(derivable)),
        'n_quarters': int(len(total)),
        'first_end': str(w['end'].min().date()),
        'last_end': str(w['end'].max().date()),
        'median_lag_days': float(np.median(all_lags)) if all_lags else float('nan'),
        'p90_lag_days': float(np.percentile(all_lags, 90)) if all_lags else float('nan'),
        'median_lag_all_reports_v1': float(lag_all.median()),
        'annual_fy_entries': int(len(fy)),
        'frame_present_frac': float(w['frame'].notna().mean()),
    }


def main():
    if not UA:
        raise SystemExit('config.SEC_USER_AGENT is empty — SEC requires a User-Agent')
    OUT.mkdir(parents=True, exist_ok=True)
    cik = load_cik_map()
    per_ticker = {}
    for tk in TICKERS:
        c = cik.get(tk)
        if c is None:
            per_ticker[tk] = {'error': 'no CIK'}
            print(f"[{tk}] no CIK")
            continue
        try:
            facts = load_facts(tk, c)
        except Exception as e:
            per_ticker[tk] = {'error': str(e)}
            print(f"[{tk}] fetch error: {e}")
            continue
        res = {k: score_concept(facts, alts, k in DURATION_CONCEPTS)
               for k, alts in CONCEPTS.items()}
        core_ok = all(res[k] is not None and res[k]['n_quarters'] >= MIN_QUARTERS
                      and np.isfinite(res[k]['median_lag_days'])
                      and res[k]['median_lag_days'] <= MAX_LAG_DAYS for k in CORE)
        per_ticker[tk] = {'cik': c, 'concepts': res, 'covered': bool(core_ok)}
        summary_bits = []
        for k in CORE:
            r = res[k]
            summary_bits.append(f"{k}:{'-' if r is None else str(r['n_quarters'])+'q/'+str(int(r['median_lag_days']))+'d'}")
        print(f"[{tk}] {'COVERED' if core_ok else 'gap'}  " + '  '.join(summary_bits))

    covered = [tk for tk, v in per_ticker.items() if v.get('covered')]
    n_cov = len(covered)
    verdict = ('FEASIBLE' if n_cov >= 16 else
               'MARGINAL' if n_cov >= 12 else 'NOT FEASIBLE')

    rev_tags = sorted({t for v in per_ticker.values() if 'concepts' in v
                       and v['concepts']['revenue']
                       for t in v['concepts']['revenue']['tags_used']})
    lags = [v['concepts'][k]['median_lag_days'] for v in per_ticker.values()
            if 'concepts' in v for k in CORE if v['concepts'][k]
            and np.isfinite(v['concepts'][k]['median_lag_days'])]
    annual_share = []
    for v in per_ticker.values():
        if 'concepts' not in v:
            continue
        for k in ['revenue', 'net_income']:
            r = v['concepts'][k]
            if r and r['n_entries']:
                annual_share.append(r['annual_fy_entries'] / r['n_entries'])

    lines = ['# R3 Spike — EDGAR Point-in-Time Fundamentals Feasibility (instrument v2)\n',
             'Lag = first filing per period (v1 median-over-re-reports lag kept in JSON as '
             'median_lag_all_reports_v1 to document the comparatives artifact). '
             'Quarters = direct quarterly values + Q4 derivable from FY.\n',
             f'Run date: {date.today()}. Sample: {len(TICKERS)} tickers. '
             f'Criteria: core concepts {CORE} each with >= {MIN_QUARTERS} quarters '
             f'in {WINDOW[0][:4]}-{WINDOW[1][:4]} and median filing lag <= {MAX_LAG_DAYS} days.\n',
             '| ticker | covered | ' + ' | '.join(CORE) + ' | eps | ocf | shares |',
             '|---|---|' + '---|' * (len(CORE) + 3)]
    for tk, v in per_ticker.items():
        if 'concepts' not in v:
            lines.append(f"| {tk} | ERR | {v.get('error')} |")
            continue
        cells = []
        for k in CORE + ['eps_diluted', 'op_cashflow', 'shares_out']:
            r = v['concepts'][k]
            if r is None:
                cells.append('—')
            elif not np.isfinite(r['median_lag_days']):
                cells.append(f"{r['n_quarters']}q / lag n/a")
            else:
                cells.append(f"{r['n_quarters']}q / {int(r['median_lag_days'])}d")
        lines.append(f"| {tk} | {'YES' if v['covered'] else 'no'} | " + ' | '.join(cells) + ' |')
    lines += ['',
              f'**Covered: {n_cov}/{len(TICKERS)} → verdict: {verdict}**',
              '',
              f'- Revenue tags needed across sample: {len(rev_tags)} ({", ".join(rev_tags)})',
              f'- Median filing lag across core concepts: {np.median(lags):.0f} days '
              f'(p90 of medians {np.percentile(lags, 90):.0f})' if lags else '- no lag data',
              f'- Annual-only (10-K FY) share of income-statement entries: '
              f'{np.mean(annual_share)*100:.0f}% (Q4 derivation needed for these)'
              if annual_share else '- no annual-share data',
              f'- Full-build fetch estimate: 525 tickers ≈ 525 requests at <=5/s ≈ 2-3 min; '
              f'engineering = tag normalization + Q4 derivation + point-in-time join.']
    report = {'prespec': 'research_r3_fundamentals_spike_prespec.md',
              'run_date': str(date.today()), 'verdict': verdict,
              'n_covered': n_cov, 'covered': covered, 'per_ticker': per_ticker,
              'revenue_tags': rev_tags}
    (OUT / 'spike_summary.json').write_text(json.dumps(report, indent=2, default=str))
    (OUT / 'spike_summary.md').write_text('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))
    print(f"\nWrote {OUT/'spike_summary.json'}\nWrote {OUT/'spike_summary.md'}")


if __name__ == '__main__':
    main()
