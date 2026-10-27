"""Worst-case RMS over the practical L/C box for the remaining controllers (same grid as exp_all lc_map)."""
import json
from multiprocessing import Pool
from experiments import job, P
from sim import NOMINAL

if __name__ == '__main__':
    J = []
    for lr in (0.7, 0.85, 1.0, 1.15, 1.3):
        for cr in (0.9, 1.0, 1.1):
            for c in ('fcs20', 'nominal'):
                J.append(('lc_extra', c, 'sw', P[c], {'true': dict(Lf=lr * NOMINAL['L0'], Cf=cr * NOMINAL['C0']),
                                                      'tag': f'{lr},{cr}'}, 0))
    with Pool(30) as p:
        R = p.map(job, J)
    out = {}
    for c in ('fcs20', 'nominal'):
        v = [r['m'].get('rms', float('inf')) if 'rms' in r['m'] else float('inf') for r in R if r['name'] == c]
        out[c] = max(v)
    print(out)
    S = json.load(open('../results/summary.json'))
    S['lc_box_worst'].update(out)
    json.dump(S, open('../results/summary.json', 'w'), indent=1)
