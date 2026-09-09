#!/usr/bin/env python
"""R2 mechanical verdict per research_r2_residual_target_prespec.md anchors."""
import json
import numpy as np
from pathlib import Path

def load(p):
    return {f['fold_id']: f for f in json.load(open(p))['per_fold']}

A = load('results/walkforward_v2317/seed42/summary.json')
B = load('results/research_r2/arm_resid_seed42/summary.json')
C = load('results/research_r2/arm_nomacro_seed42/summary.json')
F = [1, 2, 3, 4, 5]

def anchors(d):
    m, pos = float(np.mean(d)), int(sum(x > 0 for x in d))
    v = 'PROMISING' if (m >= 0.02 and pos >= 4) else ('DEAD' if (m <= 0 or pos <= 2) else 'MIXED')
    return m, pos, v

L = ['# R2 verdict (seed 42, binding folds 1-5; anchors per pre-spec)\n',
     '| fold | A base IC | B resid IC | C nomacro IC | A alpha | B alpha | C alpha |', '|---:|---:|---:|---:|---:|---:|---:|']
for f in F:
    L.append(f"| {f} | {A[f]['ic']:+.4f} | {B[f]['ic']:+.4f} | {C[f]['ic']:+.4f} | "
             f"{A[f]['alpha']*100:+.2f}%p | {B[f]['alpha']*100:+.2f}%p | {C[f]['alpha']*100:+.2f}%p |")
L += ['', '## Contrasts (paired per-fold delta IC)\n', '| contrast | ' + ' | '.join(f'f{f}' for f in F) + ' | mean | positive | verdict |', '|---|' + '---:|' * 8]
for name, X, Y in [('B-A resid-base', B, A), ('C-A nomacro-base', C, A), ('B-C resid-nomacro', B, C)]:
    d = [X[f]['ic'] - Y[f]['ic'] for f in F]
    m, pos, v = anchors(d)
    L.append(f"| {name} | " + ' | '.join(f'{x:+.4f}' for x in d) + f" | **{m:+.4f}** | {pos}/5 | **{v}** |")
L += ['', '## Arm aggregates (single seed — NOT a gate result)\n', '| arm | mean IC | positive folds | mean alpha |', '|---|---:|---:|---:|']
for name, X in [('A base (v2.3.17 s42)', A), ('B resid', B), ('C nomacro', C)]:
    ics = [X[f]['ic'] for f in F]; als = [X[f]['alpha'] for f in F]
    L.append(f"| {name} | {np.mean(ics):+.4f} | {sum(x > 0 for x in ics)}/5 | {np.mean(als)*100:+.2f}%p |")
L.append('\nSingle-seed aggregates are reported for the record only; the v2.3.17 gate thresholds do not apply to a new arm post hoc.')
Path('results/research_r2/verdict.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
