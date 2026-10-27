"""Robust tuning scan for the proposed controller based on the closed-loop spectral radius (constraint inactive)."""
import itertools, json
from multiprocessing import Pool
import numpy as np
from analysis import build
from sim import W0, VPK
SA0 = 0.5 * W0 ** 2 * VPK * 50
BOX = [(l, c) for l in (0.7, 0.85, 1.0, 1.15, 1.3) for c in (0.9, 1.0, 1.1)]


def worst(kw):
    r = []
    for l, c in BOX:
        A, *_ = build(dict(kw, N=kw.get('N', 5), gamma_u=1.0, M=24), Lr=l, Cr=c)
        r.append(max(abs(np.linalg.eigvals(A))))
    return dict(kw=kw, rho_max=float(max(r)), rho_nom=float(r[BOX.index((1.0, 1.0))]))


if __name__ == '__main__':
    G = [dict(sigma_a=SA0 * a, alpha=al, r_eta=re, qv=q, s_eta=se, kappa_f=kf, N=n)
         for a, al, re, q, se, kf, n in itertools.product((0.1, 0.3, 1, 3), (0.1, 0.2, 0.35, 0.5, 0.7),
                                                         (3e-7, 3e-6, 3e-5, 3e-4), (0.001, 0.01, 0.05),
                                                         (1e-5, 1e-4), (5.0,), (5, 10))]
    with Pool(30) as p:
        out = p.map(worst, G, chunksize=4)
    json.dump([dict(o, kw={k: v for k, v in o['kw'].items()}) for o in out], open('../results/lti_scan.json', 'w'))
    ok = [o for o in out if o['rho_max'] < 0.995]
    print('robust (rho_max<0.995):', len(ok), '/', len(out))
    for o in sorted(ok, key=lambda o: o['rho_max'])[:5]:
        print(round(o['rho_max'], 4), {k: (round(v / SA0, 2) if k == 'sigma_a' else v) for k, v in o['kw'].items()})
