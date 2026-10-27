"""Spectral radius of the linear closed loop (limit inactive, linear load) on a dense parameter grid.

This is a numerical verification on grid points, not a proof for the continuum between them.
python box_cert.py -> ../results/box_cert.json
"""
import json
from multiprocessing import Pool
import numpy as np
from analysis import build, KW

LR = np.round(np.arange(0.7, 1.3001, 0.01), 3)
CR = np.round(np.arange(0.9, 1.1001, 0.01), 3)
GS = (-0.033, 0.0, 1 / 30, 0.067, 0.1)


def rho(args):
    lr, cr, g, delay = args
    A, *_ = build(KW, delay=delay, Lr=lr, Cr=cr, G=g)
    ev = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
    return lr, cr, g, delay, float(ev[0]), float(ev[ev < 0.99][0]) if np.any(ev < 0.99) else float('nan')


if __name__ == '__main__':
    jobs = [(l, c, g, d) for d in (0, 1) for l in LR for c in CR for g in GS]
    with Pool(30) as p:
        R = p.map(rho, jobs, chunksize=20)
    out = {}
    for d in (0, 1):
        rr = [r for r in R if r[3] == d]
        w = max(rr, key=lambda r: r[4])
        out[f'delay{d}'] = dict(n_points=len(rr), max_rho=w[4], argmax=dict(Lr=w[0], Cr=w[1], G=w[2]),
                                max_fast_rho=max(r[5] for r in rr), all_stable=all(r[4] < 1 for r in rr))
        print('delay', d, out[f'delay{d}'])
    json.dump(out, open('../results/box_cert.json', 'w'), indent=1)
