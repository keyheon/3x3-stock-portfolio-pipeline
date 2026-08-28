#!/usr/bin/env python
"""v2.3.18 Phase A — tail-alpha decomposition from committed v2.3.17 CSVs.

Computes exactly D1-D5 of diagnosis_v2318_phaseA_prespec.md. No training.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path('results/walkforward_v2317')
SEEDS = [42, 1, 2]
BINDING = [1, 2, 3, 4, 5]
OUT_JSON = BASE / 'diagnosis_phaseA.json'
OUT_MD = BASE / 'diagnosis_phaseA.md'


def load_fold(seed, fold):
    df = pd.read_csv(BASE / f'seed{seed}' / f'fold{fold}_dates.csv')
    return df.set_index('date')


def d1_stats(alpha):
    """Time-concentration stats for one alpha vector."""
    a = np.asarray(alpha, dtype=float)
    total = a.sum()
    top3 = np.sort(a)[::-1][:3].sum()
    return {
        'n_dates': int(len(a)),
        'pos_frac': float((a > 0).mean()),
        'mean': float(a.mean()),
        'median': float(np.median(a)),
        'top3_share_of_sum': float(top3 / total) if total != 0 else None,
    }


def main():
    # seed-mean per-date alpha/ic per fold (aligned by date, inner join)
    fold_frames = {}
    for f in [0] + BINDING:
        dfs = [load_fold(s, f)[['alpha', 'ic']].rename(
                   columns={'alpha': f'a{s}', 'ic': f'i{s}'}) for s in SEEDS]
        m = dfs[0].join(dfs[1], how='inner').join(dfs[2], how='inner')
        m['alpha_mean'] = m[[f'a{s}' for s in SEEDS]].mean(axis=1)
        m['ic_mean'] = m[[f'i{s}' for s in SEEDS]].mean(axis=1)
        fold_frames[f] = m

    report = {'prespec': 'diagnosis_v2318_phaseA_prespec.md'}
    lines = ['# v2.3.18 Phase A — Tail-Alpha Decomposition\n',
             'Input: committed v2.3.17 per-date CSVs (3 seeds, folds 0-5). '
             'Alpha/IC below are 3-seed means per date unless noted.\n']

    # D1 — time concentration, per binding fold + pooled
    d1 = {}
    lines.append('## D1 — Time concentration (per-date alpha, seed-mean)\n')
    lines.append('| fold | n_dates | pos_frac | mean | median | top3_share |')
    lines.append('|---:|---:|---:|---:|---:|---:|')
    pooled = []
    for f in BINDING:
        st = d1_stats(fold_frames[f]['alpha_mean'])
        d1[f] = st
        pooled.append(fold_frames[f]['alpha_mean'].values)
        t3 = f"{st['top3_share_of_sum']*100:.0f}%" if st['top3_share_of_sum'] is not None else 'n/a'
        lines.append(f"| {f} | {st['n_dates']} | {st['pos_frac']:.2f} | "
                     f"{st['mean']*100:+.2f}%p | {st['median']*100:+.2f}%p | {t3} |")
    st = d1_stats(np.concatenate(pooled))
    d1['pooled'] = st
    t3 = f"{st['top3_share_of_sum']*100:.0f}%"
    lines.append(f"| pooled | {st['n_dates']} | {st['pos_frac']:.2f} | "
                 f"{st['mean']*100:+.2f}%p | {st['median']*100:+.2f}%p | {t3} |")
    report['D1'] = d1

    # D2 — seed structure: pairwise Spearman of per-date alpha across seeds
    d2 = {}
    lines.append('\n## D2 — Seed structure (pairwise Spearman of per-date alpha)\n')
    lines.append('| fold | 42-1 | 42-2 | 1-2 | mean |')
    lines.append('|---:|---:|---:|---:|---:|')
    pairs = [(42, 1), (42, 2), (1, 2)]
    pooled_cols = {s: [] for s in SEEDS}
    for f in BINDING:
        m = fold_frames[f]
        rhos = [spearmanr(m[f'a{a}'], m[f'a{b}'])[0] for a, b in pairs]
        d2[f] = {'pairwise': rhos, 'mean': float(np.mean(rhos))}
        for s in SEEDS:
            pooled_cols[s].append(m[f'a{s}'].values)
        lines.append(f"| {f} | " + " | ".join(f"{r:+.2f}" for r in rhos)
                     + f" | {np.mean(rhos):+.2f} |")
    pc = {s: np.concatenate(pooled_cols[s]) for s in SEEDS}
    rhos = [spearmanr(pc[a], pc[b])[0] for a, b in pairs]
    d2['pooled'] = {'pairwise': rhos, 'mean': float(np.mean(rhos))}
    lines.append(f"| pooled | " + " | ".join(f"{r:+.2f}" for r in rhos)
                 + f" | {np.mean(rhos):+.2f} |")
    report['D2'] = d2

    # D3 — alpha-IC decoupling
    d3 = {}
    lines.append('\n## D3 — Alpha–IC decoupling (per-date, seed-mean)\n')
    lines.append('| fold | rho(alpha, ic) | n IC<=0 dates | mean alpha on IC<=0 |')
    lines.append('|---:|---:|---:|---:|')
    all_a, all_i = [], []
    for f in BINDING:
        m = fold_frames[f]
        rho = spearmanr(m['alpha_mean'], m['ic_mean'])[0]
        sub = m[m['ic_mean'] <= 0]
        d3[f] = {'rho': float(rho), 'n_ic_le0': int(len(sub)),
                 'alpha_on_ic_le0': float(sub['alpha_mean'].mean())
                 if len(sub) else None}
        all_a.append(m['alpha_mean'].values)
        all_i.append(m['ic_mean'].values)
        av = (f"{d3[f]['alpha_on_ic_le0']*100:+.2f}%p"
              if d3[f]['alpha_on_ic_le0'] is not None else 'n/a')
        lines.append(f"| {f} | {rho:+.2f} | {len(sub)} | {av} |")
    aa, ii = np.concatenate(all_a), np.concatenate(all_i)
    rho = spearmanr(aa, ii)[0]
    sub_mask = ii <= 0
    d3['pooled'] = {'rho': float(rho), 'n_ic_le0': int(sub_mask.sum()),
                    'alpha_on_ic_le0': float(aa[sub_mask].mean())}
    lines.append(f"| pooled | {rho:+.2f} | {int(sub_mask.sum())} | "
                 f"{aa[sub_mask].mean()*100:+.2f}%p |")
    report['D3'] = d3

    # D4 — fold 5 and fold 0 shape (D1 stats; fold 0 non-binding context)
    d4 = {'fold5': d1[5], 'fold0_nonbinding': d1_stats(fold_frames[0]['alpha_mean'])}
    report['D4'] = d4
    st0 = d4['fold0_nonbinding']
    lines.append('\n## D4 — Strongest folds\n')
    lines.append(f"Fold 5 (2025, binding): pos_frac {d1[5]['pos_frac']:.2f}, "
                 f"median {d1[5]['median']*100:+.2f}%p, "
                 f"top3_share {d1[5]['top3_share_of_sum']*100:.0f}%")
    lines.append(f"Fold 0 (2020, NON-BINDING context): pos_frac {st0['pos_frac']:.2f}, "
                 f"median {st0['median']*100:+.2f}%p, "
                 f"top3_share {st0['top3_share_of_sum']*100:.0f}%")

    # D5 — autocorrelation of per-date alpha (lags 1-6), per binding fold
    d5 = {}
    lines.append('\n## D5 — Per-date alpha autocorrelation (overlap-driven dependence)\n')
    lines.append('| fold | lag1 | lag2 | lag3 | lag4 | lag5 | lag6 |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|')
    for f in BINDING:
        a = fold_frames[f]['alpha_mean'].values
        a = a - a.mean()
        denom = (a * a).sum()
        acs = [float((a[k:] * a[:-k]).sum() / denom) for k in range(1, 7)]
        d5[f] = acs
        lines.append(f"| {f} | " + " | ".join(f"{v:+.2f}" for v in acs) + " |")
    report['D5'] = d5
    lines.append('\nHigh positive low-lag autocorrelation is expected (63-td '
                 'forward windows on a ~10-td grid overlap ~6 deep); any future '
                 'gate must use dependence-aware statistics per the pre-spec.')

    OUT_JSON.write_text(json.dumps(report, indent=2))
    OUT_MD.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")


if __name__ == '__main__':
    main()
