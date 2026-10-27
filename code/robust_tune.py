"""Robust tuning protocol used for every controller.

Minimise the nominal full-run RMS error (switching-level model, seed 0) subject to bounded operation
(RMS < 8 V) at the four corner plants L/L0, C/C0 in {0.7, 1.3}. The three best robust candidates are
re-evaluated over five noise seeds; the best mean is kept.  -> ../results/final_params.json
"""
import json
import itertools
from multiprocessing import Pool
import numpy as np
from sim import NOMINAL, W0, VPK
from experiments import job, CTRL_OF

SA0 = 0.5 * W0 ** 2 * VPK * 50
P2 = 2 * np.pi
CORNERS = [(0.7, 0.9), (0.7, 1.1), (1.3, 0.9), (1.3, 1.1)]
THR = 8.0


def grid(**axes):
    names = list(axes)
    return [dict(zip(names, v)) for v in itertools.product(*axes.values())]


def candidates():
    C = {}
    C['proposed'] = []
    for kw in grid(sigma_a=[SA0 * s for s in (1, 3)], alpha=[0.1, 0.2, 0.35], r_eta=[3e-7, 3e-6],
                   qv=[0.01, 0.05], N=[5, 10],
                   hb=[((), 0.0), ((1, -5, 7), 0.2), ((1, -5, 7), 0.5), ((1, -5, 7), 1.0), ((1, -5, 7, -11, 13), 0.5)]):
        kw['harm'], kw['beta_h'] = kw.pop('hb')
        kw.update(gamma_u=1.0, M=24)
        C['proposed'].append(kw)
    C['nominal'] = [dict(kw, N=5, gamma_u=1.0, M=24) for kw in
                    grid(sigma_a=[SA0 * s for s in (1, 3, 10)], r_eta=[3e-7, 3e-6, 3e-5], qv=[0.005, 0.05])]
    C['ladrc'] = grid(wc=[P2 * f for f in (800, 1000, 1200, 1500, 2000, 2500)], wo=[P2 * f for f in (2000, 3000, 4000, 5000, 6000)])
    C['pr'] = [dict(Kc=a, Kpv=b, Kr=(k1, k5, k5)) for a, b, (k1, k5) in
               itertools.product([5.0, 10.0, 15.0, 20.0, 25.0], [0.05, 0.1, 0.15, 0.2], [(5.0, 5.0), (10.0, 5.0), (5.0, 2.0)])]
    C['faspd'] = grid(sigma_a=[SA0 * s for s in (0.3, 1, 3)], wc=[P2 * f for f in (800, 1000, 1500, 2000, 2500)], zeta=[0.5, 0.7, 1.0])
    C['fcs40'] = grid(lam=[0.0, 0.3, 1.0, 3.0], obs_poles=[(0.5, 0.55, 0.6), (0.6, 0.65, 0.7), (0.7, 0.75, 0.8), (0.8, 0.83, 0.86)])
    C['fcs20'] = [dict(kw, _Ts=50e-6) for kw in
                  grid(lam=[0.3, 1.0, 3.0, 10.0], obs_poles=[(0.3, 0.35, 0.4), (0.4, 0.45, 0.5), (0.5, 0.55, 0.6)])]
    return C


if __name__ == '__main__':
    import sys
    C = candidates()
    if len(sys.argv) > 1:
        C = {k: v for k, v in C.items() if k in sys.argv[1:]}
    J = []
    for c, kws in C.items():
        for i, kw in enumerate(kws):
            J.append(('rt', c, 'sw', kw, {'tag': f'{i}|nom'}, 0))
            for lr, cr in CORNERS:
                J.append(('rt', c, 'sw', kw, {'true': dict(Lf=lr * NOMINAL['L0'], Cf=cr * NOMINAL['C0']),
                                               'tag': f'{i}|{lr},{cr}'}, 0))
    print('jobs', len(J))
    with Pool(30) as p:
        R = p.map(job, J, chunksize=2)
    json.dump([dict(r, kw={k: (list(v) if isinstance(v, tuple) else v) for k, v in r['kw'].items()}) for r in R],
              open('../results/robust_stage1_' + '_'.join(sorted(C)) + '.json', 'w'))
    stage1 = {}
    for c, kws in C.items():
        rows = []
        for i, kw in enumerate(kws):
            rs = {r['sc']['tag'].split('|')[1]: r['m'] for r in R if r['name'] == c and r['sc']['tag'].split('|')[0] == str(i)}
            nom = rs['nom'].get('rms', np.inf) if 'rms' in rs['nom'] else np.inf
            worst = max((m.get('rms', np.inf) if 'rms' in m else np.inf) for k, m in rs.items() if k != 'nom')
            rows.append((nom, worst, i))
        rob = sorted([r for r in rows if r[1] < THR])
        if not rob:
            continue
        print(c, 'robust', len(rob), '/', len(rows), 'best', [(round(a, 2), round(b, 2)) for a, b, _ in rob[:3]],
              'unconstrained best', round(min(rows)[0], 2))
        stage1[c] = [C[c][i] for _, _, i in rob[:3]]
    J2 = [('rt2', c, 'sw', kw, {'tag': str(j)}, s) for c, kws in stage1.items() for j, kw in enumerate(kws) for s in range(5)]
    with Pool(30) as p:
        R2 = p.map(job, J2, chunksize=1)
    final = {}
    for c, kws in stage1.items():
        best = None
        for j, kw in enumerate(kws):
            v = [r['m']['rms'] for r in R2 if r['name'] == c and r['sc']['tag'] == str(j) and 'rms' in r['m']]
            if len(v) == 5 and (best is None or np.mean(v) < best[0]):
                best = (float(np.mean(v)), float(np.std(v)), kw)
        if best is None:
            continue
        final[c] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in best[2].items()}
        print(c, round(best[0], 3), '+-', round(best[1], 3), best[2])
    import os
    fn = '../results/final_params_robust.json'
    prev = json.load(open(fn)) if os.path.exists(fn) else {}
    prev.update(final)
    json.dump(prev, open(fn, 'w'), indent=1)
