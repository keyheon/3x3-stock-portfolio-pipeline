#!/usr/bin/env python
"""R2 Amendment 1 verdict: 3-seed no-macro vs 3-seed v2.3.17 base (anchors fixed pre-run)."""
import json
import numpy as np
from pathlib import Path

SEEDS = [42, 1, 2]
F = [1, 2, 3, 4, 5]

def load(p):
    return {f['fold_id']: f for f in json.load(open(p))['per_fold']}

A = {s: load(f'results/walkforward_v2317/seed{s}/summary.json') for s in SEEDS}
C = {s: load(f'results/research_r2/arm_nomacro_seed{s}/summary.json') for s in SEEDS}

def mean_ic(D, f): return float(np.mean([D[s][f]['ic'] for s in SEEDS]))
def mean_al(D, f): return float(np.mean([D[s][f]['alpha'] for s in SEEDS]))

L = ['# R2 Amendment 1 verdict — 3-seed no-macro vs 3-seed base\n',
     '| fold | base IC (s42/s1/s2) | mean | nomacro IC (s42/s1/s2) | mean | dIC |', '|---:|---|---:|---|---:|---:|']
d = []
for f in F:
    a = [A[s][f]['ic'] for s in SEEDS]; c = [C[s][f]['ic'] for s in SEEDS]
    dd = float(np.mean(c) - np.mean(a)); d.append(dd)
    L.append(f"| {f} | {'/'.join(f'{x:+.4f}' for x in a)} | {np.mean(a):+.4f} | "
             f"{'/'.join(f'{x:+.4f}' for x in c)} | {np.mean(c):+.4f} | **{dd:+.4f}** |")
m, pos = float(np.mean(d)), int(sum(x > 0 for x in d))
v = 'PROMISING' if (m >= 0.02 and pos >= 4) else ('DEAD' if (m <= 0 or pos <= 2) else 'MIXED')
L += ['', f'**C−A on 3-seed means: mean dIC = {m:+.4f}, positive folds = {pos}/5 → {v}**', '',
      '## No-macro 3-seed aggregate (for the record; not a gate result)\n']
ic3 = [mean_ic(C, f) for f in F]; al3 = [mean_al(C, f) for f in F]
per_seed = [float(np.mean([C[s][f]['ic'] for f in F])) for s in SEEDS]
L += [f"mean IC (3-seed means, binding) = {np.mean(ic3):+.4f}; positive folds = {sum(x > 0 for x in ic3)}/5; "
      f"mean alpha = {np.mean(al3)*100:+.2f}%p",
      f"per-seed binding-mean IC: {' / '.join(f'{x:+.4f}' for x in per_seed)} (std {np.std(per_seed):.4f})",
      f"base 3-seed reference: mean IC {np.mean([mean_ic(A, f) for f in F]):+.4f}, alpha {np.mean([mean_al(A, f) for f in F])*100:+.2f}%p"]
Path('results/research_r2/verdict_3seed.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
