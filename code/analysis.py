"""Section IV stability test and implementation checks.

1) Closed-loop LTI model (constraint inactive, linear load): plant + steady-state KF + DOB (with
   internal model) + unconstrained MPC (+ one-step delay compensation). Spectral radius, robustness
   region over (L/L0, C/C0, G), l2 gains for the CPL small-gain test, noise floor, internal-model check.
2) Implemented QP (inscribed polygon) versus the disk-constrained problem (8) on recorded
   closed-loop snapshots.
python analysis.py -> ../results/analysis.json
"""
import json
import numpy as np
import scipy.linalg as sla
from scipy.optimize import minimize
from sim import (FASMPC, NOMINAL, TRUE, W0, VPK, dbl_int, simulate, phi_hat, SQ3, default_scenario)

P = json.load(open('../results/final_params.json'))
KW = {k: (tuple(v) if isinstance(v, list) else v) for k, v in P['proposed'].items()}
TS = 50e-6


def mpc_gain(C):
    """eta_n(0:2) = -Kz z0 + Ke eta_prev_n   (regulation, unconstrained)."""
    Hinv = np.linalg.inv(C.Hm)
    Fz = C.As * C.Su.T @ C.W @ C.Sx           # f = Fz z0 - s e1 eta_prev
    E1 = np.zeros((C.Hm.shape[0], 2))
    E1[:2] = np.eye(2)
    return (Hinv @ Fz)[:2], (Hinv @ (C.s_eta * E1))[:2]


def build(kw=KW, Ts=TS, delay=1, Lr=1.0, Cr=1.0, G=None, rf=None, gain=1.0, extra_delay=0):
    """Closed-loop matrices x+ = A x + Ed d + En nu  (two axes)."""
    nom = NOMINAL
    C = FASMPC(Ts, delay, nom, use_dob=True, **kw)
    L, Cf = Lr * nom['L0'], Cr * nom['C0']
    r = nom['r0'] if rf is None else rf
    G = nom['G0'] if G is None else G
    a0, a1, b = (1 + r * G) / (L * Cf), r / L + G / Cf, gain / (L * Cf)
    I2, O2 = np.eye(2), np.zeros((2, 2))
    Ap = np.block([[O2, I2], [-a0 * I2, -a1 * I2]])
    Bp = np.vstack([O2, b * I2])
    Dp = np.vstack([O2, I2])
    M = sla.expm(np.block([[Ap, Bp, Dp], [np.zeros((4, 8))]]) * Ts)
    F, gu, gd = M[:4, :4], M[:4, 4:6], M[:4, 6:8]
    Ad, Bd = dbl_int(Ts)
    Hm = np.hstack([I2, O2])
    Pp = sla.solve_discrete_are(Ad.T, Hm.T, C.kf.Q, C.kf.R)
    K = Pp @ Hm.T @ np.linalg.inv(Hm @ Pp @ Hm.T + C.kf.R)
    Bp_ = C.Bp
    Kz, Ke = mpc_gain(C)
    LC0 = nom['L0'] * nom['C0']
    ah0 = (1 + nom['r0'] * nom['G0']) / LC0
    ah1 = nom['r0'] / nom['L0'] + nom['G0'] / nom['C0']
    nh = len(C.harm)
    Rh = C.Rh
    nd = 1 + extra_delay  # samples between computation and application (delay=1 case)
    # state: x4 | zh4 | d0 2 | dh 2nh | etap2 | mu_prev2 | [mu queue 2*nd | u queue 2*nd]
    idx = {}
    n = 0
    for name, sz in [('x', 4), ('zh', 4), ('d0', 2), ('dh', 2 * nh), ('etap', 2), ('mup', 2)]:
        idx[name] = slice(n, n + sz)
        n += sz
    if delay:
        idx['muq'] = slice(n, n + 2 * nd)
        n += 2 * nd
        idx['uq'] = slice(n, n + 2 * nd)
        n += 2 * nd

    def step(s, d, nu):
        x, zh, d0 = s[idx['x']], s[idx['zh']], s[idx['d0']]
        dh = [s[idx['dh']][2 * j:2 * j + 2] for j in range(nh)]
        etap, mup = s[idx['etap']], s[idx['mup']]
        dprev = d0 + sum(dh) if nh else d0
        ym = x[:2] + nu
        zp = Ad @ zh + Bd @ (mup + dprev)
        eps = K @ (ym - Hm @ zp)
        zh1 = zp + eps
        ed = Bp_ @ eps
        d0n = d0 + C.alpha * ed
        dhn = [Rh[j] @ (dh[j] + C.alpha * C.beta_h * ed) for j in range(nh)]
        dk = d0n + sum(dhn) if nh else d0n
        out = np.zeros(n)
        if delay:
            muq = [s[idx['muq']][2 * j:2 * j + 2] for j in range(nd)]
            uq = [s[idx['uq']][2 * j:2 * j + 2] for j in range(nd)]
            z0 = Ad @ zh1 + Bd @ (muq[0] + dk)
            dst = d0n + sum(R @ c for R, c in zip(Rh, dhn)) if nh else dk
            u_apply = uq[0]
        else:
            z0, dst = zh1, dk
        eta_n = -Kz @ z0 + Ke @ etap
        mu_new = C.As * eta_n - dst
        u_new = LC0 * (mu_new + ah0 * z0[:2] + ah1 * z0[2:])
        if not delay:
            u_apply = u_new
        out[idx['x']] = F @ x + gu @ u_apply + gd @ d
        out[idx['zh']] = zh1
        out[idx['d0']] = d0n
        if nh:
            out[idx['dh']] = np.concatenate(dhn)
        out[idx['etap']] = eta_n
        if delay:
            out[idx['mup']] = muq[0]
            out[idx['muq']] = np.concatenate(muq[1:] + [mu_new])
            out[idx['uq']] = np.concatenate(uq[1:] + [u_new])
        else:
            out[idx['mup']] = mu_new
        return out

    z = np.zeros(n)
    A = np.stack([step(e, np.zeros(2), np.zeros(2)) for e in np.eye(n)], 1)
    Ed = np.stack([step(z, e, np.zeros(2)) for e in np.eye(2)], 1)
    En = np.stack([step(z, np.zeros(2), e) for e in np.eye(2)], 1)
    return A, Ed, En, idx, C


def hinf(A, B, Cm, nw=4000):
    best = 0.0
    I = np.eye(A.shape[0])
    for th in np.linspace(0, np.pi, nw):
        Gm = Cm @ np.linalg.solve(np.exp(1j * th) * I - A, B)
        best = max(best, np.linalg.svd(Gm, compute_uv=False)[0])
    return best


def stability_test():
    out = {}
    A, Ed, En, idx, C = build()
    rho = max(abs(np.linalg.eigvals(A)))
    out['rho_nominal_delay'] = float(rho)
    A0, *_ = build(delay=0)
    out['rho_nominal_nodelay'] = float(max(abs(np.linalg.eigvals(A0))))
    n = A.shape[0]
    Cy = np.zeros((2, n)); Cy[:, idx['x'].start:idx['x'].start + 2] = np.eye(2)
    Cv = np.zeros((2, n)); Cv[:, idx['x'].start + 2:idx['x'].start + 4] = np.eye(2)
    gy, gv = hinf(A, Ed, Cy), hinf(A, Ed, Cv)
    out['l2_gain_d_to_y'] = float(gy)
    out['l2_gain_d_to_ydot'] = float(gv)
    # CPL incremental Lipschitz bounds on |y| >= Vmin (true CPL P vs nominal P0)
    Pmax, P0, Vmin = 2400.0, NOMINAL['P0'], 0.8 * VPK
    dP = max(Pmax - P0, P0)
    Cf, Lf, rf = TRUE['Cf'], TRUE['Lf'], TRUE['rf']
    lv = (2 / 3) * dP / Vmin ** 2 / Cf
    ly = (2 / 3) * dP * (rf / (Lf * Cf) / Vmin ** 2 + 6 * W0 * VPK / Vmin ** 3 / Cf)
    out['cpl_lip'] = dict(l_y=ly, l_v=lv, Pmax=Pmax, Vmin=Vmin)
    out['small_gain'] = float(gy * ly + gv * lv)
    # noise floor (constraint inactive): steady-state covariance
    sv = default_scenario()['sigma_v']
    S = sla.solve_discrete_lyapunov(A, sv ** 2 * En @ En.T)
    out['noise_rms_y'] = float(np.sqrt(np.trace(Cy @ S @ Cy.T)))
    # internal model: steady-state response to constant and rotating (h w0) disturbances
    def resp(h):
        th = h * W0 * TS
        Gm = Cy @ np.linalg.solve(np.exp(1j * th) * np.eye(n) - A, Ed @ np.array([1, -1j]) / np.sqrt(2))
        return float(np.linalg.norm(Gm))
    out['gain_dist_to_y'] = {str(h): resp(h) for h in (0, 1, -5, 7, -11, 13)}  # sequence-matched direction

    def resp_full(h):  # largest singular value of the full 2x2 transfer matrix at the same frequency
        th = h * W0 * TS
        Gm = Cy @ np.linalg.solve(np.exp(1j * th) * np.eye(n) - A, Ed)
        return float(np.linalg.svd(Gm, compute_uv=False)[0])
    out['gain_full_to_y'] = {str(h): resp_full(h) for h in (0, 1, -5, 7)}
    # robustness region
    grid = {}
    for Lr in (0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0):
        for Cr in (0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0):
            Am, *_ = build(Lr=Lr, Cr=Cr)
            grid[f'{Lr},{Cr}'] = float(max(abs(np.linalg.eigvals(Am))))
    out['rho_LC'] = grid
    gG = {}
    for Gv in (-0.1, -0.05, -0.033, 0.0, 1 / 30, 0.1, 0.2, 0.5):
        Am, *_ = build(G=Gv)
        gG[str(Gv)] = float(max(abs(np.linalg.eigvals(Am))))
    out['rho_G'] = gG
    gm = None
    for g in np.linspace(0.05, 5, 100):
        Am, *_ = build(gain=g)
        if max(abs(np.linalg.eigvals(Am))) >= 1 and gm is None and g > 1:
            gm = float(g)
    lo = None
    for g in np.linspace(1, 0.01, 100):
        Am, *_ = build(gain=g)
        if max(abs(np.linalg.eigvals(Am))) >= 1:
            lo = float(g)
            break
    out['gain_margin'] = dict(upper=gm, lower=lo)
    dm = None
    for ed in range(1, 6):
        Am, *_ = build(extra_delay=ed)
        if max(abs(np.linalg.eigvals(Am))) >= 1:
            dm = ed
            break
    out['extra_delay_unstable_at'] = dm
    return out


def nlp_check():
    """Implemented polygon QP versus the disk-constrained problem (8).

    With the linear nominal drift (P0 = 0), u_{i|k} is affine in eta, so (8) is a convex SOCP whose global
    optimum is computed with Clarabel; the only remaining approximation of the QP is the inscribed polygon.
    Recorded closed-loop snapshots of a 300-V-sag run are used.
    """
    import cvxpy as cp
    import sim
    orig = sim.CTRL['proposed']
    sim.CTRL['proposed'] = lambda Ts, d, nom, **kw: _rec(FASMPC(Ts, d, nom, use_dob=True, **kw))
    r = simulate('proposed', 'sw', ctrl_kw=KW, keep_ctrl=True, sc=default_scenario(Vdc_sag=300.0))
    sim.CTRL['proposed'] = orig
    C = r['C']
    sn = C.snaps
    N, LC, As, M = C.N, C.LC, C.As, C.M
    act = [i for i, s_ in enumerate(sn) if np.max(C.Acon @ s_['eta'] - s_['ub']) > -1e-3 * s_['R']]
    sel = np.unique(np.concatenate([np.linspace(50, len(sn) - 1, 300).astype(int), np.array(act, dtype=int)]))
    du0, gap, stage_ratio, failed = [], [], [], 0
    for i in sel:
        s_ = sn[i]
        z0, mur, dd, f, eq, R = s_['z0'], s_['mur'], s_['d'], s_['f'], s_['eta'], s_['R']
        mur_s = mur.reshape(-1)
        Gs, hs = [], []
        for k in range(N):
            E = np.zeros((2, 2 * N)); E[:, 2 * k:2 * k + 2] = np.eye(2)
            if k == 0:
                Gs.append(LC * As * E)
                hs.append(LC * (mur[0] - dd[0] - phi_hat(z0, C.nom)))
            else:
                Su_k = C.Su[(k - 1) * 4:k * 4]
                zk0 = C.Sx[(k - 1) * 4:k * 4] @ z0 + Su_k @ mur_s
                Gs.append(LC * (As * E + C.Kphi @ (As * Su_k)))
                hs.append(LC * (mur[k] - dd[k] + C.Kphi @ zk0))
        stage_ratio.append(max(np.linalg.norm(G @ eq + h) for G, h in zip(Gs, hs)) / R)
        x = cp.Variable(2 * N)
        cons = [cp.norm(G @ x + h) <= R for G, h in zip(Gs, hs)]
        prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(C.Hm)) + f @ x), cons)
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:  # noqa
            failed += 1
            continue
        if prob.status not in ('optimal', 'optimal_inaccurate'):
            failed += 1
            continue
        xs = x.value
        du0.append(float(np.linalg.norm(Gs[0] @ (eq - xs))))
        Jq, Js = 0.5 * eq @ C.Hm @ eq + f @ eq, 0.5 * xs @ C.Hm @ xs + f @ xs
        Ju = -0.5 * f @ np.linalg.solve(C.Hm, f)  # unconstrained optimum
        gap.append(float((Jq - Js) / max(Js - Ju, 1e-12)) if Js - Ju > 1e-9 else 0.0)
    du0, gap, stage_ratio = map(np.array, (du0, gap, stage_ratio))
    return dict(n_attempted=int(len(sel)), n_failed=int(failed), n=int(len(du0)), n_active=int(len(act)),
                solver='SOCP (8) with Clarabel, global optimum (convex for P0 = 0)',
                du0_mean=float(du0.mean()), du0_p95=float(np.percentile(du0, 95)), du0_max=float(du0.max()),
                cost_increase_ratio_mean=float(gap.mean()), cost_increase_ratio_p95=float(np.percentile(gap, 95)),
                max_stage_ratio=float(stage_ratio.max()), polygon_conservatism=float(1 - np.cos(np.pi / KW['M'])))


def _rec(C):
    C.record = True
    C.snaps = []
    return C


if __name__ == '__main__':
    out = stability_test()
    print(json.dumps({k: v for k, v in out.items() if k not in ('rho_LC',)}, indent=1))
    print('rho_LC', {k: round(v, 4) for k, v in out['rho_LC'].items()})
    out['nlp'] = nlp_check()
    print(out['nlp'])
    json.dump(out, open('../results/analysis.json', 'w'), indent=1)
