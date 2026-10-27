"""Tube check for the small-gain condition and closed-form check of kappa.

E_y  = max ||y - v_ref|| / Vpk,  E_v = max ||ydot - vdot_ref|| / (w Vpk),  ydot = (i_L - i_load) / C_f,
evaluated at the sampling instants (the sampled-data representation used by the small-gain test) and, for
reference, on the 5-us record (which contains the PWM ripple of the capacitor current).
Windows: CPL step transient (20-25 ms), steady CPL with linear load (25-50 ms, the setting of the small-gain test),
linear load only (5-20 ms), and the whole 120-ms test.
kappa: sup_{|v|=1} ||d(J_g(y) v)/dy|| = 2 / (|y|^2 + eps)^{3/2}, so kappa = 2 (1 + eps/Vmin^2)^{-3/2}.
python tube_check.py -> ../results/tube_check.json
"""
import json
from multiprocessing import Pool
import numpy as np
from sim import simulate, default_scenario, NOMINAL, TRUE, VPK, W0
from small_gain import kappa3, kappa3_closed

P = json.load(open('../results/final_params.json'))['proposed']
P = {k: (tuple(v) if isinstance(v, list) else v) for k, v in P.items()}
PLANTS = [(0.9, 1.1), (0.7, 0.9), (0.7, 1.1), (1.3, 0.9), (1.3, 1.1)]
WIN = dict(base=(0.005, 0.02), cpl_step=(0.02, 0.025), cpl_steady=(0.025, 0.05), all=(0.0, 0.12))


def run(a):
    Lr, Cr, seed = a
    sc = default_scenario(seed=seed)
    sc['true'].update(Lf=Lr * NOMINAL['L0'], Cf=Cr * NOMINAL['C0'])
    r = simulate('proposed', 'sw', sc=sc, ctrl_kw=P)
    t, rec, Cf = r['t'], r['rec'], sc['true']['Cf']
    ey = np.linalg.norm(rec[:, :2] - VPK * np.stack([np.cos(W0 * t), np.sin(W0 * t)], 1), axis=1) / VPK
    yd = (rec[:, 2:4] - rec[:, 4:6]) / Cf
    ev = np.linalg.norm(yd - VPK * W0 * np.stack([-np.sin(W0 * t), np.cos(W0 * t)], 1), axis=1) / (W0 * VPK)
    step = int(round(r['Ts'] / (t[1] - t[0])))
    smp = np.zeros(len(t), bool)
    smp[step - 1::step] = True
    out = dict(Lr=Lr, Cr=Cr, seed=seed)
    for wn, (a0, a1) in WIN.items():
        w = (t >= a0) & (t < a1)
        out[wn] = dict(Ey_samp=float(ey[w & smp].max()), Ev_samp=float(ev[w & smp].max()),
                       Ey_cont=float(ey[w].max()), Ev_cont=float(ev[w].max()))
    return out


if __name__ == '__main__':
    eps = NOMINAL['eps']
    Vmin = 0.8 * VPK
    k_num = kappa3(Vmin, eps) * Vmin ** 3
    k_cf = kappa3_closed(Vmin, eps) * Vmin ** 3
    with Pool(10) as p:
        R = p.map(run, [(a, b, s) for a, b in PLANTS for s in range(5)])
    res = dict(kappa_numeric=k_num, kappa_closed_form=k_cf, runs=R)
    for wn in WIN:
        res[f'max_{wn}'] = {q: max(r[wn][q] for r in R) for q in R[0][wn]}
        res[f'max_{wn}_default'] = {q: max(r[wn][q] for r in R if r['Lr'] == 0.9) for q in R[0][wn]}
    json.dump(res, open('../results/tube_check.json', 'w'), indent=1)
    print('kappa numeric', k_num, 'closed form', k_cf)
    for wn in WIN:
        print(wn, 'all plants', {q: round(v, 3) for q, v in res[f'max_{wn}'].items()},
              'default', {q: round(v, 3) for q, v in res[f'max_{wn}_default'].items()})
