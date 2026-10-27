"""Incremental small-gain test for the CPL nonlinearity (limit inactive).

With P0 = 0 the implemented controller is linear for linear loads, so the only nonlinearity is the true CPL
acceleration  psi(y, yd) = -(2P/3) [ r_f/(L_f C_f) g(y) + (1/C_f) J_g(y) yd ],  g(y) = y / (|y|^2 + eps).
It is bounded on the tube  D = {|y - v_ref(t)| <= dy, |yd - dv_ref(t)| <= dv}  (convex at each t), on which
|y| >= Vmin = Vpk - dy and |yd| <= ydmax = w Vpk + dv:
  ||dpsi/dyd|| <= (2P/3) / (C_f (Vmin^2 + eps))                                    = l_v
  ||dpsi/dy||  <= (2P/3) [ r_f / (L_f C_f (Vmin^2 + eps)) + kappa3 ydmax / C_f ]   = l_y
with kappa3 = sup_{|y| >= Vmin, |v| = 1} ||d(J_g(y) v)/dy|| = 2 / (Vmin^2 + eps)^{3/2} (closed form, checked against
dense sampling). The loop is incrementally
l2-stable if gamma_y l_y + gamma_v l_v < 1, where gamma_y, gamma_v are the l2 gains of the linear closed loop
from a matched acceleration (held over each sample) to y and yd.
python small_gain.py -> ../results/small_gain.json
"""
import json
from multiprocessing import Pool
import numpy as np
from analysis import build, KW
from sim import NOMINAL, TRUE, W0, VPK


def kappa3_closed(Vmin, eps):
    """sup over |y| >= Vmin and unit v of ||d(J_g(y) v)/dy||: the norm equals 2 / (|y|^2 + eps)^{3/2} (valid for
    |y|^2 >= eps / 3), which decreases in |y|."""
    return 2.0 / (Vmin ** 2 + eps) ** 1.5


def kappa3(Vmin, eps):
    """Numerical cross-check of kappa3_closed (exact derivative, dense sampling of |y| and of the angle)."""
    best = 0.0
    for r in np.linspace(Vmin, 3 * Vmin, 60):
        for th in np.linspace(0, np.pi, 721):  # angle between y and v (symmetry)
            y = np.array([r, 0.0])
            v = np.array([np.cos(th), np.sin(th)])
            S = r * r + eps
            yv = y @ v
            # d/dy [ v/S - 2 y (y.v)/S^2 ]
            Dm = (-2.0 * np.outer(v, y) / S ** 2 - 2.0 * (yv * np.eye(2) + np.outer(y, v)) / S ** 2
                  + 8.0 * yv * np.outer(y, y) / S ** 3)
            best = max(best, np.linalg.norm(Dm, 2))
    return best


def gains(args):
    Lr, Cr = args
    A, Ed, En, idx, C = build(KW, Lr=Lr, Cr=Cr, G=NOMINAL['G0'])
    n = A.shape[0]
    Cy = np.zeros((2, n)); Cy[:, idx['x'].start:idx['x'].start + 2] = np.eye(2)
    Cv = np.zeros((2, n)); Cv[:, idx['x'].start + 2:idx['x'].start + 4] = np.eye(2)
    th = np.unique(np.concatenate([np.logspace(-6, np.log10(np.pi), 6000), np.linspace(0, np.pi, 6000)]))
    I = np.eye(n)
    gy = gv = 0.0
    for w in th:
        R = np.linalg.solve(np.exp(1j * w) * I - A, Ed)
        gy = max(gy, np.linalg.svd(Cy @ R, compute_uv=False)[0])
        gv = max(gv, np.linalg.svd(Cv @ R, compute_uv=False)[0])
    return Lr, Cr, gy, gv


if __name__ == '__main__':
    eps = NOMINAL['eps']
    out = {}
    plants = [(1.0, 1.0), (0.9, 1.1)] + [(l, c) for l in (0.7, 1.0, 1.3) for c in (0.9, 1.1)]
    with Pool(len(plants)) as p:
        G = p.map(gains, plants)
    out['gains'] = [dict(Lr=a, Cr=b, gamma_y=gy, gamma_v=gv) for a, b, gy, gv in G]
    # (0.1, 2.0) contains the sampled errors observed in steady CPL operation at all plants (tube_check.py)
    for dy_rel, dv_rel in ((0.1, 2.0), (0.2, 2.0), (0.2, 0.2), (0.1, 0.1)):
        Vmin = (1 - dy_rel) * VPK
        ydmax = (1 + dv_rel) * W0 * VPK
        k3 = kappa3_closed(Vmin, eps)
        out[f'kappa_check,dy={dy_rel}'] = dict(closed=k3 * Vmin ** 3, sampled=kappa3(Vmin, eps) * Vmin ** 3)
        for Pcpl in (600.0, 1200.0, 1800.0, 2400.0):
            worst = 0.0
            rows = []
            for a, b, gy, gv in G:
                Lf, Cf, rf = a * NOMINAL['L0'], b * NOMINAL['C0'], TRUE['rf']
                c = 2 * Pcpl / 3
                lv = c / (Cf * (Vmin ** 2 + eps))
                ly = c * (rf / (Lf * Cf * (Vmin ** 2 + eps)) + k3 * ydmax / Cf)
                idx_ = gy * ly + gv * lv
                rows.append(dict(Lr=a, Cr=b, l_y=ly, l_v=lv, index=idx_))
                worst = max(worst, idx_)
            key = f'P={Pcpl:.0f},dy={dy_rel},dv={dv_rel}'
            out[key] = dict(kappa3_times_Vmin3=k3 * Vmin ** 3, worst_index=worst,
                            default_index=[r['index'] for r in rows if r['Lr'] == 0.9 and r['Cr'] == 1.1][0], rows=rows)
            print(key, 'kappa3*Vmin^3 =', round(k3 * Vmin ** 3, 3), ' default index', round(out[key]['default_index'], 3),
                  ' worst over plants', round(worst, 3))
    json.dump(out, open('../results/small_gain.json', 'w'), indent=1)
