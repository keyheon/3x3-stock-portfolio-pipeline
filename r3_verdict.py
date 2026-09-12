#!/usr/bin/env python
"""R3-full verdict per research_r3_full_prespec.md anchors (seed 42; primary raw base, secondary no-macro base)."""
import json
import numpy as np
from pathlib import Path

def load(p):
    return {f['fold_id']: f for f in json.load(open(p))['per_fold']}

F = [1, 2, 3, 4, 5]
pairs = [
    ('PRIMARY   fund-on-raw  vs raw base (v2.3.17 s42)',
     'results/research_r3/wf/raw_fund_seed42/summary.json', 'results/walkforward_v2317/seed42/summary.json'),
    ('SECONDARY fund-on-nomacro vs nomacro base (R2 s42)',
     'results/research_r3/wf/raw_nomacro_fund_seed42/summary.json', 'results/research_r2/arm_nomacro_seed42/summary.json'),
]
L = ['# R3-full verdict (seed 42)\n']
for name, fp, bp in pairs:
    X, Y = load(fp), load(bp)
    d = [X[f]['ic'] - Y[f]['ic'] for f in F]
    m, pos = float(np.mean(d)), int(sum(x > 0 for x in d))
    v = 'PROMISING' if (m >= 0.02 and pos >= 4) else ('DEAD' if (m <= 0 or pos <= 2) else 'MIXED')
    L += [f'## {name}\n', '| fold | base IC | +fund IC | dIC | base alpha | +fund alpha |', '|---:|---:|---:|---:|---:|---:|']
    for f in F:
        L.append(f"| {f} | {Y[f]['ic']:+.4f} | {X[f]['ic']:+.4f} | {d[f-1]:+.4f} | "
                 f"{Y[f]['alpha']*100:+.2f}%p | {X[f]['alpha']*100:+.2f}%p |")
    nf = [X[f].get('n_fund_selected') for f in F]; wp = [X[f].get('n_fund_would_pass_corr') for f in F]
    L += ['', f'**mean dIC = {m:+.4f}, positive folds = {pos}/5 -> {v}**  '
          f'(fund selected per fold {nf}; would pass corr filter {wp})', '']
Path('results/research_r3/wf/verdict.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
