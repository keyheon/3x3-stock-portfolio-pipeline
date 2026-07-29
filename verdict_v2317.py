#!/usr/bin/env python
"""v2.3.17 section-4 verdict — mechanical, from the three seed summaries.

Reads results/walkforward_v2317/seed{42,1,2}/summary.json, applies the
pre-registered rule (mean IC >= 0.05 AND fold-IC positive >= 4/5 AND
mean top-5 alpha > TC_est 0.62%p/quarter) on 3-seed means over binding
folds 1-5, prints the verdict, and writes verdict.txt next to the data.
"""

import json
import sys
from pathlib import Path

import numpy as np

SEEDS = [42, 1, 2]
BASE = Path('results/walkforward_v2317')
IC_THR = 0.05
POS_THR = 4
TC_EST = 0.0062
BINDING_FOLDS = [1, 2, 3, 4, 5]


def main():
    S = {}
    for s in SEEDS:
        p = BASE / f'seed{s}' / 'summary.json'
        if not p.exists():
            sys.exit(f"Missing {p} — run all three seeds first.")
        S[s] = json.loads(p.read_text())

    def fold_metric(s, fid, key):
        return next(fr[key] for fr in S[s]['per_fold']
                    if fr['fold_id'] == fid)

    ic_m = {f: [fold_metric(s, f, 'ic') for s in SEEDS]
            for f in BINDING_FOLDS}
    al_m = {f: [fold_metric(s, f, 'alpha') for s in SEEDS]
            for f in BINDING_FOLDS}

    fold_ic_mean = {f: float(np.mean(ic_m[f])) for f in BINDING_FOLDS}
    mean_ic = float(np.mean(list(fold_ic_mean.values())))
    n_pos = sum(1 for f in BINDING_FOLDS if fold_ic_mean[f] > 0)
    mean_alpha = float(np.mean([np.mean(al_m[f]) for f in BINDING_FOLDS]))

    c1 = mean_ic >= IC_THR
    c2 = n_pos >= POS_THR
    c3 = mean_alpha > TC_EST
    if c1 and c2:
        verdict = 'PASS' if c3 else 'PARTIAL'
    else:
        verdict = 'FAIL'

    lines = []
    out = lines.append
    out('=== v2.3.17 §4 verdict (3-seed means, binding folds 1-5) ===\n')
    out(f'{"fold":>5} {"year":>5} {"IC s42":>8} {"IC s1":>8} {"IC s2":>8} '
        f'{"IC mean":>8} | {"alpha mean":>10}')
    for f in BINDING_FOLDS:
        out(f'{f:>5} {2020 + f:>5} '
            + ' '.join(f'{v:>+8.4f}' for v in ic_m[f])
            + f' {fold_ic_mean[f]:>+8.4f} | '
            + f'{np.mean(al_m[f]) * 100:>+9.2f}%p')
    out('')
    out(f'mean IC          = {mean_ic:+.4f}   (threshold >= {IC_THR})'
        f'      -> {"PASS" if c1 else "FAIL"}')
    out(f'positive folds   = {n_pos}/5      (threshold >= {POS_THR}/5)'
        f'       -> {"PASS" if c2 else "FAIL"}')
    out(f'mean top-5 alpha = {mean_alpha * 100:+.2f}%p '
        f'(threshold > {TC_EST * 100:.2f}%p) -> {"PASS" if c3 else "FAIL"}')
    out(f'\n>>> VERDICT: {verdict}')

    out('\n--- Secondary (non-binding) ---')
    f0_ic = float(np.mean([fold_metric(s, 0, 'ic') for s in SEEDS]))
    f0_al = float(np.mean([fold_metric(s, 0, 'alpha') for s in SEEDS]))
    out(f'Fold 0 (2020, non-binding): IC {f0_ic:+.4f}, '
        f'alpha {f0_al * 100:+.2f}%p')
    mom_vals = [np.mean([fold_metric(s, f, 'momentum_ic') for s in SEEDS])
                for f in BINDING_FOLDS]
    out(f'Momentum baseline IC (binding mean): '
        f'{float(np.mean(mom_vals)):+.4f}  vs model {mean_ic:+.4f}')
    try:
        spy = float(np.mean(
            [np.mean([fold_metric(s, f, 'alpha_vs_spy') for s in SEEDS])
             for f in BINDING_FOLDS]))
        out(f'Alpha vs SPY (binding mean): {spy * 100:+.2f}%p')
    except Exception:
        out('Alpha vs SPY: not available')
    icir = float(np.mean(
        [np.mean([fold_metric(s, f, 'icir') for s in SEEDS])
         for f in BINDING_FOLDS]))
    out(f'ICIR (binding mean): {icir:+.2f}')
    al_wo5 = float(np.mean([np.mean(al_m[f]) for f in [1, 2, 3, 4]]))
    out(f'Alpha excl. fold 5: {al_wo5 * 100:+.2f}%p  '
        f'(fragility check, §5)')

    text = '\n'.join(lines)
    print(text)
    (BASE / 'verdict.txt').write_text(text + '\n')
    print(f'\nWrote {BASE}/verdict.txt')


if __name__ == '__main__':
    main()
