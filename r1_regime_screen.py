#!/usr/bin/env python
"""R1 — Regime Predictability Screen.

Computes exactly what research_r1_regime_screen_prespec.md specifies:
backward-looking SPY spectral descriptors (+controls) vs the committed
v2.3.17 per-date IC (3-seed mean). No training. First run fetches SPY
and ^VIX once and freezes them as CSVs; later runs read the frozen CSVs.
"""

import json
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr

WF = Path('results/walkforward_v2317')
OUT = Path('results/research_r1')
SEEDS = [42, 1, 2]
BINDING = [1, 2, 3, 4, 5]
W_PRIMARY, W_SECONDARY = 252, 126
NEW = ['vr10', 'spec_slope', 'spec_entropy', 'hf_ratio']
CTRL = ['rv21', 'vix', 'ret126', 'volofvol']


# ---------- data ----------

def frozen_or_fetch(name, ticker):
    f = OUT / f'{name}_daily_r1.csv'
    if f.exists():
        s = pd.read_csv(f, parse_dates=['date']).set_index('date')['close']
        print(f"[data] {name}: frozen CSV ({len(s)} rows)")
        return s
    import yfinance as yf
    df = yf.download(ticker, start='2014-01-01', auto_adjust=True,
                     progress=False)
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    s = close.dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    OUT.mkdir(parents=True, exist_ok=True)
    s.rename('close').rename_axis('date').to_csv(f)
    print(f"[data] {name}: fetched {len(s)} rows on {_date.today()}, frozen to {f}")
    return s


def load_ic_frames():
    """Per-fold frame: date-indexed, 3-seed mean ic and alpha."""
    frames = {}
    for f in [0] + BINDING:
        dfs = [pd.read_csv(WF / f'seed{s}' / f'fold{f}_dates.csv',
                           parse_dates=['date']).set_index('date')[['ic', 'alpha']]
               .add_suffix(f'_{s}') for s in SEEDS]
        m = dfs[0].join(dfs[1], how='inner').join(dfs[2], how='inner')
        m['ic'] = m[[f'ic_{s}' for s in SEEDS]].mean(axis=1)
        m['alpha'] = m[[f'alpha_{s}' for s in SEEDS]].mean(axis=1)
        frames[f] = m[['ic', 'alpha']]
    return frames


# ---------- descriptors ----------

def descriptors_at(rets_win, vix_at_t):
    x = rets_win.values
    n = len(x)
    out = {}
    # VR(10)
    s10 = pd.Series(x).rolling(10).sum().dropna().values
    out['vr10'] = float(np.var(s10, ddof=1) / (10 * np.var(x, ddof=1)))
    # Welch PSD
    nper = 64 if n >= 200 else 32
    f, p = welch(x, fs=1.0, nperseg=min(nper, n))
    mask = f > 0
    f, p = f[mask], np.maximum(p[mask], 1e-18)
    out['spec_slope'] = float(np.polyfit(np.log10(f), np.log10(p), 1)[0])
    q = p / p.sum()
    out['spec_entropy'] = float(-(q * np.log(q)).sum() / np.log(len(q)))
    out['hf_ratio'] = float(p[f > 0.2].sum() / p.sum())
    # controls
    out['rv21'] = float(np.std(x[-21:], ddof=1))
    out['vix'] = float(vix_at_t)
    out['ret126'] = float(x[-126:].sum()) if n >= 126 else float(x.sum())
    rv = pd.Series(x).rolling(21).std().dropna()
    out['volofvol'] = float(rv.std(ddof=1))
    return out


def build_table(frames, spy, vix, window):
    logret = np.log(spy).diff().dropna()
    rows = []
    for fold, m in frames.items():
        for t, r in m.iterrows():
            pos = logret.index.searchsorted(t, side='right')
            if pos < window:
                continue
            win = logret.iloc[pos - window:pos]
            vpos = vix.index.searchsorted(t, side='right')
            d = descriptors_at(win, vix.iloc[vpos - 1])
            d.update({'fold': fold, 'date': t, 'ic': r['ic'], 'alpha': r['alpha']})
            rows.append(d)
    return pd.DataFrame(rows)


# ---------- stats ----------

def screen(tab, target='ic'):
    """Per-descriptor: per-fold rho (binding), weighted mean, consistency."""
    res = {}
    for d in NEW + CTRL:
        per, ns = {}, {}
        for f in BINDING:
            sub = tab[tab['fold'] == f]
            per[f] = float(spearmanr(sub[d], sub[target])[0])
            ns[f] = len(sub)
        wmean = float(np.average([per[f] for f in BINDING],
                                 weights=[ns[f] for f in BINDING]))
        sign_ok = sum(1 for f in BINDING
                      if np.sign(per[f]) == np.sign(wmean) and per[f] != 0)
        pooled_plain = float(spearmanr(tab[tab['fold'].isin(BINDING)][d],
                                       tab[tab['fold'].isin(BINDING)][target])[0])
        f0 = tab[tab['fold'] == 0]
        res[d] = {'per_fold': per, 'wmean': wmean, 'sign_consistent': sign_ok,
                  'pooled_plain': pooled_plain,
                  'fold0': float(spearmanr(f0[d], f0[target])[0]) if len(f0) else None}
    return res


def verdicts(res):
    best_ctrl = max(abs(res[c]['wmean']) for c in CTRL)
    v = {}
    for d in NEW:
        r = res[d]
        if abs(r['wmean']) <= best_ctrl or r['sign_consistent'] <= 3:
            v[d] = 'DEAD'
        elif abs(r['wmean']) >= 0.25 and r['sign_consistent'] >= 4:
            v[d] = 'PROMISING'
        else:
            v[d] = 'MIXED'
    return v, best_ctrl


def binary_view(tab):
    out = {}
    sub = tab[tab['fold'].isin(BINDING)].copy()
    for d in NEW + CTRL:
        z = sub.groupby('fold')[d].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=1) or 1.0))
        out[d] = {'mean_z_ic_pos': float(z[sub['ic'] > 0].mean()),
                  'mean_z_ic_le0': float(z[sub['ic'] <= 0].mean())}
    return out


def collinearity(tab):
    sub = tab[tab['fold'].isin(BINDING)]
    return {d: {c: float(spearmanr(sub[d], sub[c])[0]) for c in CTRL}
            for d in NEW}


# ---------- report ----------

def table_md(res, title):
    lines = [f'\n## {title}\n',
             '| descriptor | f1 | f2 | f3 | f4 | f5 | wmean | sign | pooled(ctx) | fold0(ctx) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d in NEW + CTRL:
        r = res[d]
        tag = ' (ctrl)' if d in CTRL else ''
        pf = ' | '.join(f"{r['per_fold'][f]:+.2f}" for f in BINDING)
        f0 = f"{r['fold0']:+.2f}" if r['fold0'] is not None else 'n/a'
        lines.append(f"| {d}{tag} | {pf} | **{r['wmean']:+.3f}** | "
                     f"{r['sign_consistent']}/5 | {r['pooled_plain']:+.2f} | {f0} |")
    return lines


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = load_ic_frames()
    spy = frozen_or_fetch('spy', 'SPY')
    vix = frozen_or_fetch('vix', '^VIX')

    tab_p = build_table(frames, spy, vix, W_PRIMARY)
    tab_s = build_table(frames, spy, vix, W_SECONDARY)

    res_ic = screen(tab_p, 'ic')
    v, best_ctrl = verdicts(res_ic)
    res_alpha = screen(tab_p, 'alpha')
    res_126 = screen(tab_s, 'ic')

    lines = ['# R1 — Regime Predictability Screen\n',
             f'Targets: per-date IC (primary), alpha (secondary). '
             f'Windows: {W_PRIMARY}d primary, {W_SECONDARY}d secondary. '
             f'Binding folds 1-5; n≈25/fold (individually noisy by design; '
             f'sign consistency across folds is the robustness axis; no p-values '
             f'per D5 dependence).\n']
    lines += table_md(res_ic, f'IC target, {W_PRIMARY}d window (PRIMARY)')
    lines.append('\n### Anchor verdicts (primary table only)\n')
    lines.append(f'Best control |wmean| = {best_ctrl:.3f}')
    for d in NEW:
        lines.append(f"- **{d}: {v[d]}**  (wmean {res_ic[d]['wmean']:+.3f}, "
                     f"sign {res_ic[d]['sign_consistent']}/5)")
    lines += table_md(res_alpha, 'alpha target (secondary, reporting only)')
    lines += table_md(res_126, f'IC target, {W_SECONDARY}d window (secondary)')

    bv = binary_view(tab_p)
    lines.append('\n## Binary view (within-fold z, IC>0 vs IC<=0; reporting only)\n')
    lines.append('| descriptor | mean z on IC>0 | mean z on IC<=0 |')
    lines.append('|---|---:|---:|')
    for d in NEW + CTRL:
        lines.append(f"| {d} | {bv[d]['mean_z_ic_pos']:+.2f} | "
                     f"{bv[d]['mean_z_ic_le0']:+.2f} |")

    col = collinearity(tab_p)
    lines.append('\n## Descriptor-control collinearity (Spearman)\n')
    lines.append('| new \\ ctrl | ' + ' | '.join(CTRL) + ' |')
    lines.append('|---|' + '---:|' * len(CTRL))
    for d in NEW:
        lines.append(f"| {d} | " + ' | '.join(f"{col[d][c]:+.2f}" for c in CTRL) + ' |')

    report = {'prespec': 'research_r1_regime_screen_prespec.md',
              'n_dates_primary': int(len(tab_p[tab_p['fold'].isin(BINDING)])),
              'ic_252': res_ic, 'verdicts': v, 'best_control_abs': best_ctrl,
              'alpha_252': res_alpha, 'ic_126': res_126,
              'binary_view': bv, 'collinearity': col}
    (OUT / 'r1_screen.json').write_text(json.dumps(report, indent=2))
    (OUT / 'r1_screen.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f"\nWrote {OUT/'r1_screen.json'}\nWrote {OUT/'r1_screen.md'}")


if __name__ == '__main__':
    main()
