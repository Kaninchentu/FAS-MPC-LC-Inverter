"""Worst-case RMS over the L/C grid for the deadline-safe variant (same grid as lc_map)."""
import json
from multiprocessing import Pool
from experiments import job, P
from sim import NOMINAL

if __name__ == '__main__':
    J = [('lc_fb', 'propfb', 'sw', P['propfb'], {'true': dict(Lf=l * NOMINAL['L0'], Cf=c * NOMINAL['C0'])}, 0)
         for l in (0.7, 0.85, 1.0, 1.15, 1.3) for c in (0.9, 1.0, 1.1)]
    with Pool(15) as p:
        R = p.map(job, J)
    w = max(r['m'].get('rms', float('inf')) for r in R)
    S = json.load(open('../results/summary.json'))
    S['lc_box_worst']['propfb'] = w
    json.dump(S, open('../results/summary.json', 'w'), indent=1)
    print('propfb worst', w)
