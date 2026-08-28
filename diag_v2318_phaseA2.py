#!/usr/bin/env python
"""v2.3.18 Phase A2 — beta-tilt test on the v2.3.17 top-5 alpha.

Computes exactly A2-1..A2-3 of the Phase A2 pre-spec appended to
diagnosis_v2318_phaseA_prespec.md. Same committed CSVs; no training.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path('results/walkforward_v2317')
SEEDS = [42, 1, 2]
BINDING = [1, 2, 3, 4, 5]
OUT_JSON = BASE / 'diagnosis_phaseA2.json'
OUT_MD = BASE / 'diagnosis_phaseA2.md'


def load_fold(seed, fold):
    return pd.read_csv(BASE / f'seed{seed}' / f'fold{fold}_dates.csv'
                       ).set_index('date')


def seed_mean_frame(fold):
    cols = ['alpha', 'ic', 'top5_mean_ret', 'universe_mean_ret']
    dfs = [load_fold(s, fold)[cols].add_suffix(f'_{s}') for s in SEEDS]
    m = dfs[0].join(dfs[1], how='inner').join(dfs[2], how='inner')
    for c in cols:
        m[c] = m[[f'{c}_{s}' for s in SEEDS]].mean(axis=1)
    return m[cols]


def ols(y, x):
    """slope, intercept via least squares."""
    b, a = np.polyfit(x, y, 1)
    return float(b), float(a)


def a2_block(m):
    """All A2 stats for one frame (per fold or pooled)."""
    t5 = m['top5_mean_ret'].values
    uv = m['universe_mean_ret'].values
    al = m['alpha'].values
    ic = m['ic'].values

    b, a = ols(t5, uv)
    tilt_component = (b - 1.0) * uv.mean()
    up = uv > 0
    down = ~up
    r_ic_uv = spearmanr(ic, uv)[0]

    cell = {}
    for ic_pos in (True, False):
        for uv_pos in (True, False):
            mask = ((ic > 0) == ic_pos) & (up == uv_pos)
            key = f"ic{'+' if ic_pos else '-'}_uv{'+' if uv_pos else '-'}"
            cell[key] = {'n': int(mask.sum()),
                         'alpha': float(al[mask].mean()) if mask.sum() else None}

    return {
        'n_dates': int(len(m)),
        'beta': b,
        'intercept': a,
        'mean_univ_ret': float(uv.mean()),
        'mean_alpha': float(al.mean()),
        'tilt_component': float(tilt_component),
        'residual_component': float(al.mean() - tilt_component),
        'n_down_dates': int(down.sum()),
        'alpha_on_down': float(al[down].mean()) if down.sum() else None,
        'alpha_on_up': float(al[up].mean()) if up.sum() else None,
        'rho_ic_univ': float(r_ic_uv),
        'cells': cell,
    }


def fmt_pct(x):
    return f"{x*100:+.2f}%p" if x is not None else 'n/a'


def main():
    frames = {f: seed_mean_frame(f) for f in [0] + BINDING}
    pooled = pd.concat([frames[f] for f in BINDING])

    report = {'prespec': 'diagnosis_v2318_phaseA_prespec.md (Phase A2 section)'}
    lines = ['# v2.3.18 Phase A2 — Beta-Tilt Test\n',
             'Same committed per-date CSVs; alpha/ic/returns are 3-seed '
             'means per date. OLS on dependent (overlapping-window) '
             'observations — slopes/intercepts are descriptive; cross-fold '
             'consistency is the robustness axis, no p-values.\n']

    lines.append('## A2-1 — Effective beta and decomposition '
                 '(top5_ret = a + b × univ_ret)\n')
    lines.append('| fold | beta | intercept a | mean univ | mean alpha | '
                 'tilt (b-1)·E[univ] | residual |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|')
    res = {}
    for f in BINDING + ['pooled']:
        blk = a2_block(pooled if f == 'pooled' else frames[f])
        res[str(f)] = blk
        lines.append(f"| {f} | {blk['beta']:.2f} | {fmt_pct(blk['intercept'])} | "
                     f"{fmt_pct(blk['mean_univ_ret'])} | {fmt_pct(blk['mean_alpha'])} | "
                     f"{fmt_pct(blk['tilt_component'])} | "
                     f"{fmt_pct(blk['residual_component'])} |")
    blk0 = a2_block(frames[0])
    res['fold0_nonbinding'] = blk0
    lines.append(f"| 0 (non-binding) | {blk0['beta']:.2f} | "
                 f"{fmt_pct(blk0['intercept'])} | {fmt_pct(blk0['mean_univ_ret'])} | "
                 f"{fmt_pct(blk0['mean_alpha'])} | {fmt_pct(blk0['tilt_component'])} | "
                 f"{fmt_pct(blk0['residual_component'])} |")

    lines.append('\n## A2-2 — Down-market behavior (universe_ret <= 0 dates)\n')
    lines.append('| fold | n_down | alpha on down | alpha on up |')
    lines.append('|---:|---:|---:|---:|')
    for f in BINDING + ['pooled']:
        blk = res[str(f)]
        lines.append(f"| {f} | {blk['n_down_dates']} | "
                     f"{fmt_pct(blk['alpha_on_down'])} | {fmt_pct(blk['alpha_on_up'])} |")

    lines.append('\n## A2-3 — IC–market linkage\n')
    lines.append('| fold | rho(ic, univ_ret) | alpha ic+/uv+ | ic+/uv- | '
                 'ic-/uv+ | ic-/uv- |')
    lines.append('|---:|---:|---:|---:|---:|---:|')
    for f in BINDING + ['pooled']:
        blk = res[str(f)]
        c = blk['cells']
        lines.append(f"| {f} | {blk['rho_ic_univ']:+.2f} | "
                     f"{fmt_pct(c['ic+_uv+']['alpha'])} ({c['ic+_uv+']['n']}) | "
                     f"{fmt_pct(c['ic+_uv-']['alpha'])} ({c['ic+_uv-']['n']}) | "
                     f"{fmt_pct(c['ic-_uv+']['alpha'])} ({c['ic-_uv+']['n']}) | "
                     f"{fmt_pct(c['ic-_uv-']['alpha'])} ({c['ic-_uv-']['n']}) |")

    OUT_JSON.write_text(json.dumps(report | {'results': res}, indent=2))
    OUT_MD.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")


if __name__ == '__main__':
    main()
