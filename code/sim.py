"""Controllers, scenario runner and metrics for the voltage-only FAS-MPC study.

Controllers
  proposed : FAS-MPC + Kalman filter + posterior-residual DOB (voltage only)
  nominal  : FAS-MPC + Kalman filter, no DOB (ablation)
  faspd    : FAS-PD + Kalman filter (ablation)
  ladrc    : linear ADRC with discrete ESO, bandwidth parameterisation (voltage only)
  pr       : multi-resonant PR voltage loop + capacitor-current active damping
             (needs a capacitor-current sensor)
  fcs      : current-sensorless FCS-MPC with Luenberger observer (Zheng et al. 2020)
"""
import time
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import osqp
from plant import run_period, SQ3

# --------------------------------------------------------------------------- parameters
F0 = 50.0
W0 = 2 * np.pi * F0
VPK = 110.0 * np.sqrt(2.0)

# P0 = 0: the nominal drift has no CPL term, so the controller is linear for linear loads (Proposition 2);
# with P0 = 600 W the RMS errors differ by less than 0.25 V (Supplementary Material).
NOMINAL = dict(L0=2.0e-3, C0=30e-6, r0=0.10, G0=1 / 30.0, P0=0.0, eps=(0.1 * VPK) ** 2)
TRUE = dict(Lf=1.8e-3, Cf=33e-6, rf=0.15, G=1 / 30.0)
RECT = dict(Ls=0.5e-3, Rs=0.05, Cd=470e-6, Rd=45.0)
SW = dict(td=1e-6, Von=1.0, vdc_rip=0.01, f_rip=300.0, adc_lsb=800.0 / 4096)


def default_scenario(**kw):
    sc = dict(Tsim=0.12, t_cpl=0.02, P_cpl=1200.0, t_nl=0.05, t_sag=0.08, t_rec=0.10,
              Vdc_nom=400.0, Vdc_sag=340.0, sigma_v=0.5, sigma_i=0.05, seed=0,
              true=dict(TRUE), rect=dict(RECT), nominal=dict(NOMINAL), sw=dict(SW))
    sc.update(kw)
    return sc


def vref(t):
    return VPK * np.array([np.cos(W0 * t), np.sin(W0 * t)])


def vref_d(t):
    return VPK * W0 * np.array([-np.sin(W0 * t), np.cos(W0 * t)])


def vref_dd(t):
    return -W0 ** 2 * vref(t)


def zref(t):
    return np.concatenate([vref(t), vref_d(t)])


def dbl_int(Ts):
    I2 = np.eye(2)
    Ad = np.block([[I2, Ts * I2], [0 * I2, I2]])
    Bd = np.vstack([0.5 * Ts ** 2 * I2, Ts * I2])
    return Ad, Bd


def phi_hat(z, nom):
    """Nominal FAS drift  Phi_hat(y, ydot)."""
    L0, C0, r0, G0, P0, eps = (nom[k] for k in ('L0', 'C0', 'r0', 'G0', 'P0', 'eps'))
    y = z[:2]
    yd = z[2:]
    S = y @ y + eps
    i0 = G0 * y + (2 / 3) * P0 * y / S
    J0 = G0 * np.eye(2) + (2 / 3) * P0 * (S * np.eye(2) - 2 * np.outer(y, y)) / S ** 2
    return -y / (L0 * C0) - r0 / L0 * yd - r0 / (L0 * C0) * i0 - (J0 @ yd) / C0


CLIP_TOL = 1e-3  # relative excess counted as clipping (QP tolerance is about 1e-6)


def project_disk(u, R, stats):
    """Project onto the modulation disk of radius R; record clipping and the largest raw-command ratio."""
    n = np.linalg.norm(u)
    ratio = n / R
    stats['proj'] += int(ratio > 1 + CLIP_TOL)
    stats['exc_max'] = max(stats.get('exc_max', 0.0), ratio)
    return u * R / n if n > R else u


# --------------------------------------------------------------------------- KF shared by FAS controllers
class FASKF:
    def __init__(self, Ts, sigma_a, sigma_v, z0):
        self.Ad, self.Bd = dbl_int(Ts)
        self.Q = sigma_a ** 2 * self.Bd @ self.Bd.T + 1e-12 * np.eye(4)
        self.R = sigma_v ** 2 * np.eye(2)
        self.H = np.hstack([np.eye(2), np.zeros((2, 2))])
        self.z = z0.copy()
        self.P = np.diag([sigma_v ** 2] * 2 + [(0.05 * W0 * VPK) ** 2] * 2)
        self.first = True

    def update(self, y, mu_prev):
        if self.first:
            zp, Pp = self.z, self.P
            self.first = False
        else:
            zp = self.Ad @ self.z + self.Bd @ mu_prev
            Pp = self.Ad @ self.P @ self.Ad.T + self.Q
        S = self.H @ Pp @ self.H.T + self.R
        K = Pp @ self.H.T @ np.linalg.inv(S)
        self.z = zp + K @ (y - self.H @ zp)
        self.P = (np.eye(4) - K @ self.H) @ Pp
        return self.z - zp  # posterior residual eps_k


# --------------------------------------------------------------------------- proposed / nominal FAS-MPC
class FASMPC:
    name = 'proposed'

    def __init__(self, Ts, delay, nom, use_dob=True, N=10, qy=1.0, qv=0.05, r_eta=2e-5, s_eta=1e-5,
                 kappa_f=5.0, M=16, gamma_u=0.95, alpha=0.3, sigma_a=None, sigma_v=1.0,
                 freeze='shift', dob_in_kf=True, theta_w=0.0, harm=(), beta_h=0.05, qp_mode='qp', t0=0.0, **_):
        self.Ts, self.delay, self.nom, self.use_dob = Ts, delay, nom, use_dob
        self.dob_in_kf = dob_in_kf and use_dob
        # qp_mode: 'qp' solves the QP (its solution equals the unconstrained candidate when that is feasible);
        # 'fallback' never solves it: unconstrained law followed by clipping to the modulation disk
        self.qp_mode = qp_mode
        c, s_ = np.cos(theta_w * Ts), np.sin(theta_w * Ts)
        self.Th = np.array([[c, -s_], [s_, c]])  # disturbance internal model (rotation); I for theta_w = 0
        # harmonic internal model: components rotating at h*w0 (h<0: negative sequence)
        self.harm = tuple(harm)
        self.Rh = [np.array([[np.cos(h * W0 * Ts), -np.sin(h * W0 * Ts)], [np.sin(h * W0 * Ts), np.cos(h * W0 * Ts)]])
                   for h in self.harm]
        self.dh_comp = [np.zeros(2) for _ in self.harm]
        self.dh0 = np.zeros(2)
        self.beta_h = beta_h
        self.N, self.M, self.gamma_u, self.alpha = N, M, gamma_u, alpha
        self.freeze = freeze
        self.Ad, self.Bd = dbl_int(Ts)
        self.Bp = np.linalg.pinv(self.Bd)
        if sigma_a is None:
            sigma_a = 0.5 * W0 ** 2 * VPK * 50  # default KF acceleration std
        self.kf = FASKF(Ts, sigma_a, sigma_v, zref(t0))
        self.dhat = np.zeros(2)
        self.LC = nom['L0'] * nom['C0']
        self.Ys, self.Yds, self.As = VPK, W0 * VPK, W0 ** 2 * VPK
        # ---- condensed prediction Z = Sx z0 + Su mu  (Z = [z_1..z_N])
        nx, nu = 4, 2
        Sx = np.zeros((nx * N, nx))
        Su = np.zeros((nx * N, nu * N))
        Ap = np.eye(4)
        for i in range(N):
            Ap = self.Ad @ Ap
            Sx[i * nx:(i + 1) * nx] = Ap
            for j in range(i + 1):
                Su[i * nx:(i + 1) * nx, j * nu:(j + 1) * nu] = np.linalg.matrix_power(self.Ad, i - j) @ self.Bd
        self.Sx, self.Su = Sx, Su
        wz = np.array([qy / self.Ys ** 2] * 2 + [qv / self.Yds ** 2] * 2)
        W = np.tile(wz, N)
        W[-4:] *= (1.0 + kappa_f)  # stage term at i=N-1 plus terminal penalty J_f
        self.W = np.diag(W)
        D = np.eye(nu * N) - np.eye(nu * N, k=-nu)
        self.D = D
        As = self.As
        Hm = As ** 2 * Su.T @ self.W @ Su + r_eta * np.eye(nu * N) + s_eta * D.T @ D
        self.Hm = 0.5 * (Hm + Hm.T)
        self.Hinv = np.linalg.inv(self.Hm)
        self.s_eta = s_eta
        th = 2 * np.pi * np.arange(M) / M
        self.Cpoly = np.stack([np.cos(th), np.sin(th)], 1)
        self.Acon = np.kron(np.eye(N), self.LC * As * self.Cpoly)
        # with P0 = 0 the nominal drift is linear, Phi_hat(z) = -Kphi z, so u_{i|k} is affine in eta and the
        # polygon constraints are exact at every stage (no frozen drift)
        self.exact_lin = nom.get('P0', 0.0) == 0.0
        if self.exact_lin:
            a0 = (1 + nom['r0'] * nom['G0']) / self.LC
            a1 = nom['r0'] / nom['L0'] + nom['G0'] / nom['C0']
            self.Kphi = np.hstack([a0 * np.eye(2), a1 * np.eye(2)])
            Acon = self.Acon.copy()
            for i in range(1, N):
                rows = slice(i * M, (i + 1) * M)
                # z_{i|k} = Sx[i-1] z0 + Su[i-1] (mu_r + As eta)
                Acon[rows] += self.LC * self.Cpoly @ self.Kphi @ (As * Su[(i - 1) * nx:i * nx])
            self.Acon = Acon
        self.solver = osqp.OSQP()
        self.solver.setup(P=sp.csc_matrix(np.triu(self.Hm)), q=np.zeros(nu * N), A=sp.csc_matrix(self.Acon),
                          l=-np.inf * np.ones(M * N), u=np.ones(M * N), verbose=False, warm_starting=True,
                          eps_abs=1e-6, eps_rel=1e-6, max_iter=4000, polishing=False)
        self.eta_prev_n = np.zeros(2)
        self.Zprev = None
        self.mu_appl = {0: vref_dd(t0)}  # virtual input applied over [j, j+1)
        self.u_init = self.LC * (vref_dd(t0) - phi_hat(zref(t0), nom))
        self.stats = dict(active=0, proj=0, solve_time=[], iters=[], n=0)
        self.last_pred = None

    def _refs(self, t_start):
        N, Ts = self.N, self.Ts
        zr = np.array([zref(t_start + i * Ts) for i in range(N + 1)])
        mur = np.array([self.Bp @ (zr[i + 1] - self.Ad @ zr[i]) for i in range(N)])
        return zr, mur

    def solve_mpc(self, z0, t_start, vdc, dstart=None):
        N = self.N
        zr, mur = self._refs(t_start)
        mur_s = mur.reshape(-1)
        Zr = zr[1:].reshape(-1)
        f = self.As * self.Su.T @ self.W @ (self.Sx @ z0 + self.Su @ mur_s - Zr)
        f[:2] -= self.s_eta * self.eta_prev_n
        dh = dstart if (self.use_dob and dstart is not None) else np.zeros(2)
        # drift frozen along the shifted previous prediction; exact at i = 0
        if self.freeze == 'shift' and self.Zprev is not None:
            zbar = [z0] + [self.Zprev[i + 1] for i in range(N - 1)]
        else:
            zbar = [z0] + [zr[i] for i in range(1, N)]
        Rp = self.gamma_u * vdc / SQ3 * np.cos(np.pi / self.M)
        ub = np.empty(self.M * N)
        comps = [c.copy() for c in self._dstart_comps] if (self.harm and self.use_dob) else None
        dpre = dh.copy()
        dlist = []
        for i in range(N):
            dlist.append(dpre.copy())
            if self.exact_lin and i >= 1:  # affine part of Phi_hat(z_{i|k}) independent of eta
                zi0 = self.Sx[(i - 1) * 4:i * 4] @ z0 + self.Su[(i - 1) * 4:i * 4] @ mur_s
                off = self.LC * self.Cpoly @ (mur[i] - dpre + self.Kphi @ zi0)
            else:
                off = self.LC * self.Cpoly @ (mur[i] - dpre - phi_hat(zbar[i], self.nom))
            if comps is not None:
                comps = [R @ c for R, c in zip(self.Rh, comps)]
                dpre = self._dstart_d0 + sum(comps)
            else:
                dpre = self.Th @ dpre if self.dob_in_kf else dpre
            ub[i * self.M:(i + 1) * self.M] = Rp - off
        # the unconstrained minimizer solves the QP iff it satisfies all N*M inequalities
        eta_unc = -self.Hinv @ f
        unc_ok = bool(np.all(self.Acon @ eta_unc <= ub + 1e-9 * Rp))
        self.stats['unc_ok'] = self.stats.get('unc_ok', 0) + int(unc_ok)
        if self.qp_mode == 'fallback':
            # deadline-safe variant: no QP; the unconstrained first move is applied and the command is
            # clipped to the modulation disk in step()
            eta_n = eta_unc
            mu = mur_s + self.As * eta_n
            Z = (self.Sx @ z0 + self.Su @ mu).reshape(N, 4)
            self.Zprev = np.vstack([z0, Z])
            self.eta_prev_n = eta_n[:2]
            self.stats['active'] += int(not unc_ok)
            return mur[0] + self.As * eta_n[:2], eta_n
        self.solver.update(q=f, u=ub)
        t1 = time.perf_counter()
        res = self.solver.solve()
        self.stats['solve_time'].append(time.perf_counter() - t1)
        self.stats['iters'].append(res.info.iter)
        self.stats.setdefault('qp_solve', []).append(res.info.solve_time)
        self.stats.setdefault('qp_run', []).append(res.info.run_time)
        eta_n = res.x if res.x is not None and np.all(np.isfinite(res.x)) else np.zeros(2 * N)
        if getattr(self, 'record', False):
            self.snaps.append(dict(z0=z0.copy(), mur=mur.copy(), d=np.array(dlist), f=f.copy(), eta=eta_n.copy(),
                                   R=self.gamma_u * vdc / SQ3, ub=ub.copy()))
        mu = mur_s + self.As * eta_n
        Z = (self.Sx @ z0 + self.Su @ mu).reshape(N, 4)
        self.Zprev = np.vstack([z0, Z])
        cons = self.Acon[:self.M, :2] @ eta_n[:2] - ub[:self.M]
        if np.max(cons) > -1e-3 * Rp:
            self.stats['active'] += 1
        self.eta_prev_n = eta_n[:2]
        return mur[0] + self.As * eta_n[:2], eta_n

    def step(self, k, t, y, vdc, aux=None):
        mu_prev = self.mu_appl.get(k - 1, np.zeros(2))
        # with dob_in_kf the KF predicts with mu_{k-1} + dhat_{k-1} (disturbance estimate over [k-1, k))
        eps = self.kf.update(y, mu_prev + (self.dhat if self.dob_in_kf else 0.0))
        if self.use_dob and k > 0:
            if self.dob_in_kf and self.harm:
                ed = self.Bp @ eps
                self.dh0 = self.dh0 + self.alpha * ed
                for j, R in enumerate(self.Rh):
                    self.dh_comp[j] = R @ (self.dh_comp[j] + self.alpha * self.beta_h * ed)
                self.dhat = self.dh0 + sum(self.dh_comp)
            elif self.dob_in_kf:
                self.dhat = self.Th @ (self.dhat + self.alpha * (self.Bp @ eps))
            else:
                self.dhat = (1 - self.alpha) * self.dhat + self.alpha * (self.Bp @ eps)
        zh = self.kf.z
        dk = self.dhat if self.use_dob else np.zeros(2)
        self._dstart_d0 = self.dh0.copy()
        self._dstart_comps = [c.copy() for c in self.dh_comp]
        if self.delay:
            # one-step-ahead prediction with the committed input (and disturbance estimate if used in the KF)
            z0 = self.Ad @ zh + self.Bd @ (self.mu_appl.get(k, np.zeros(2)) + (dk if self.dob_in_kf else 0.0))
            if self.harm and self.use_dob:
                self._dstart_comps = [R @ c for R, c in zip(self.Rh, self.dh_comp)]
                dk = self._dstart_d0 + sum(self._dstart_comps)
            else:
                dk = self.Th @ dk if self.dob_in_kf else dk
            t_start = t + self.Ts
        else:
            z0 = zh
            t_start = t
        mubar0, _ = self.solve_mpc(z0, t_start, vdc, dk)
        mu = mubar0 - dk
        ph = phi_hat(z0, self.nom)
        u = self.LC * (mu - ph)
        u = project_disk(u, vdc / SQ3, self.stats)
        self.stats['n'] += 1
        self.mu_appl[k + self.delay] = u / self.LC + ph
        self.mu_appl.pop(k - 2, None)
        return dict(u=u, dhat=self.dhat.copy(), zhat=zh.copy())


class FASPD(FASMPC):
    name = 'faspd'

    def __init__(self, Ts, delay, nom, wc=2 * np.pi * 600, zeta=0.9, sigma_a=None, sigma_v=1.0, t0=0.0, **_):
        self.Ts, self.delay, self.nom = Ts, delay, nom
        self.Ad, self.Bd = dbl_int(Ts)
        if sigma_a is None:
            sigma_a = 0.5 * W0 ** 2 * VPK * 50
        self.kf = FASKF(Ts, sigma_a, sigma_v, zref(t0))
        self.LC = nom['L0'] * nom['C0']
        self.Kp, self.Kd = wc ** 2, 2 * zeta * wc
        self.mu_appl = {0: vref_dd(t0)}
        self.u_init = self.LC * (vref_dd(t0) - phi_hat(zref(t0), nom))
        self.stats = dict(active=0, proj=0, solve_time=[], iters=[], n=0)

    def step(self, k, t, y, vdc, aux=None):
        self.kf.update(y, self.mu_appl.get(k - 1, np.zeros(2)))
        zh = self.kf.z
        if self.delay:
            z0 = self.Ad @ zh + self.Bd @ self.mu_appl.get(k, np.zeros(2))
            ta = t + self.Ts
        else:
            z0, ta = zh, t
        t1 = time.perf_counter()
        mu = vref_dd(ta) - self.Kp * (z0[:2] - vref(ta)) - self.Kd * (z0[2:] - vref_d(ta))
        ph = phi_hat(z0, self.nom)
        u = project_disk(self.LC * (mu - ph), vdc / SQ3, self.stats)
        self.stats['solve_time'].append(time.perf_counter() - t1)
        self.stats['n'] += 1
        self.mu_appl[k + self.delay] = u / self.LC + ph
        self.mu_appl.pop(k - 2, None)
        return dict(u=u)


# --------------------------------------------------------------------------- LADRC
class LADRC:
    """Model-assisted linear ADRC: ESO on ydd = b0 u - a0 y - a1 yd + f (nominal LC + linear load known)."""
    name = 'ladrc'

    def __init__(self, Ts, delay, nom, wc=2 * np.pi * 600, wo=2 * np.pi * 2500, model_assisted=True, t0=0.0, **_):
        self.Ts, self.delay = Ts, delay
        L0, C0, r0, G0 = nom['L0'], nom['C0'], nom['r0'], nom['G0']
        self.b0 = 1.0 / (L0 * C0)
        a0 = (1 + r0 * G0) / (L0 * C0) if model_assisted else 0.0
        a1 = (r0 / L0 + G0 / C0) if model_assisted else 0.0
        Ac = np.array([[0, 1, 0], [-a0, -a1, 1], [0, 0, 0.0]])
        Bc = np.array([[0], [self.b0], [0.0]])
        M = sla.expm(np.block([[Ac, Bc], [np.zeros((1, 4))]]) * Ts)
        A, B = M[:3, :3], M[:3, 3]
        C = np.array([1.0, 0, 0])
        Cp = C @ A
        O = np.vstack([Cp, Cp @ A, Cp @ A @ A])
        beta = np.exp(-wo * Ts)
        phiA = (A - beta * np.eye(3)) @ (A - beta * np.eye(3)) @ (A - beta * np.eye(3))
        self.Lg = phiA @ np.linalg.solve(O, np.array([0, 0, 1.0]))
        self.A, self.B, self.a0, self.a1 = A, B, a0, a1
        r0_ = zref(t0)
        self.x = np.array([[r0_[0], r0_[2], 0.0], [r0_[1], r0_[3], 0.0]])  # per axis [y, yd, f]
        self.kp, self.kd = wc ** 2, 2 * wc
        self.u_appl = {}
        self.u_init = (vref_dd(t0) + a0 * vref(t0) + a1 * vref_d(t0)) / self.b0
        self.stats = dict(active=0, proj=0, solve_time=[], iters=[], n=0)
        self.first = True

    def step(self, k, t, y, vdc, aux=None):
        up = self.u_appl.get(k - 1, self.u_init)
        t1 = time.perf_counter()
        for a in range(2):
            xp = self.x[a] if self.first else self.A @ self.x[a] + self.B * up[a]
            self.x[a] = xp + self.Lg * (y[a] - xp[0])
        self.first = False
        if self.delay:
            ucur = self.u_appl.get(k, self.u_init)
            xs = np.array([self.A @ self.x[a] + self.B * ucur[a] for a in range(2)])
            ta = t + self.Ts
        else:
            xs, ta = self.x, t
        r, rd, rdd = vref(ta), vref_d(ta), vref_dd(ta)
        u = (rdd + self.kp * (r - xs[:, 0]) + self.kd * (rd - xs[:, 1]) + self.a0 * xs[:, 0] + self.a1 * xs[:, 1]
             - xs[:, 2]) / self.b0
        u = project_disk(u, vdc / SQ3, self.stats)
        self.stats['solve_time'].append(time.perf_counter() - t1)
        self.stats['n'] += 1
        self.u_appl[k + self.delay] = u
        self.u_appl.pop(k - 2, None)
        return dict(u=u)


# --------------------------------------------------------------------------- multi-resonant PR + capacitor-current AD
class PRAD:
    name = 'pr'

    def __init__(self, Ts, delay, nom, Kc=6.0, Kpv=0.12, Kr=(60.0, 20.0, 20.0), harm=(1, 5, 7), wb=5.0,
                 t0=0.0, **_):
        self.Ts, self.C0, self.Kc, self.Kpv = Ts, nom['C0'], Kc, Kpv
        self.res = []
        for h, kr in zip(harm, Kr):
            wh = h * W0
            Ac = np.array([[0, 1], [-wh ** 2, -2 * wb]])
            Bc = np.array([[0], [1.0]])
            Cc = np.array([[0, 2 * kr * wb]])
            M = sla.expm(np.block([[Ac, Bc], [np.zeros((1, 3))]]) * Ts)
            self.res.append((M[:2, :2], M[:2, 2], Cc[0]))
        self.xr = [np.zeros((2, 2)) for _ in self.res]  # [axis] states
        self.stats = dict(active=0, proj=0, solve_time=[], iters=[], n=0)

    def step(self, k, t, y, vdc, aux=None):
        t1 = time.perf_counter()
        e = vref(t) - y
        ic_ref = self.C0 * vref_d(t) + self.Kpv * e
        for (Ad, Bd, Cc), xr in zip(self.res, self.xr):
            for a in range(2):
                ic_ref[a] += Cc @ xr[a]
                xr[a] = Ad @ xr[a] + Bd * e[a]
        u = y + self.Kc * (ic_ref - aux['ic'])
        u = project_disk(u, vdc / SQ3, self.stats)
        self.stats['solve_time'].append(time.perf_counter() - t1)
        self.stats['n'] += 1
        return dict(u=u)


# --------------------------------------------------------------------------- current-sensorless FCS-MPC
SW_STATES = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]])


class FCSMPC:
    name = 'fcs'

    def __init__(self, Ts, delay, nom, lam=1.0, obs_poles=(0.6, 0.65, 0.7), t0=0.0, **_):
        L0, C0, r0 = nom['L0'], nom['C0'], nom['r0']
        self.Ts, self.C0 = Ts, C0
        Ac = np.array([[-r0 / L0, -1 / L0, 0], [1 / C0, 0, -1 / C0], [0, 0, 0]])
        Bc = np.array([[1 / L0], [0], [0]])
        M = sla.expm(np.block([[Ac, Bc], [np.zeros((1, 4))]]) * Ts)
        self.A, self.B = M[:3, :3], M[:3, 3]
        C = np.array([0, 1.0, 0])
        Cp = C @ self.A
        O = np.vstack([Cp, Cp @ self.A, Cp @ self.A @ self.A])
        phiA = np.eye(3)
        for p in obs_poles:
            phiA = phiA @ (self.A - p * np.eye(3))
        self.Lg = phiA @ np.linalg.solve(O, np.array([0, 0, 1.0]))
        r = zref(t0)
        G0 = nom['G0']
        self.x = np.array([[G0 * r[0] + C0 * r[2], r[0], G0 * r[0]], [G0 * r[1] + C0 * r[3], r[1], G0 * r[1]]])
        self.lam = lam
        self.s_appl = {}
        self.prev_s = np.array([0, 0, 0])
        self.stats = dict(active=0, proj=0, solve_time=[], iters=[], n=0)
        self.first = True

    @staticmethod
    def vec(s, vdc):
        v = (s - 0.5) * vdc
        return np.array([(2 / 3) * (v[0] - 0.5 * v[1] - 0.5 * v[2]), (v[1] - v[2]) / SQ3])

    def step(self, k, t, y, vdc, aux=None):
        t1 = time.perf_counter()
        sp_ = self.s_appl.get(k - 1, self.prev_s)
        up = self.vec(sp_, vdc)
        for a in range(2):
            xp = self.x[a] if self.first else self.A @ self.x[a] + self.B * up[a]
            self.x[a] = xp + self.Lg * (y[a] - xp[1])
        self.first = False
        sc = self.s_appl.get(k, self.prev_s)
        uc = self.vec(sc, vdc)
        x1 = np.array([self.A @ self.x[a] + self.B * uc[a] for a in range(2)])
        t2 = t + 2 * self.Ts
        vr, icr = vref(t2), self.C0 * vref_d(t2)
        best, bs = np.inf, sc
        for j in range(8):
            s = SW_STATES[j]
            u = self.vec(s, vdc)
            x2 = np.array([self.A @ x1[a] + self.B * u[a] for a in range(2)])
            ev = vr - x2[:, 1]
            ei = icr - (x2[:, 0] - x2[:, 2])
            g = ev @ ev + self.lam * ei @ ei + 1e-6 * np.sum(np.abs(s - sc))
            if g < best:
                best, bs = g, s
        self.stats['solve_time'].append(time.perf_counter() - t1)
        self.stats['n'] += 1
        self.s_appl[k + 1] = bs
        self.s_appl.pop(k - 2, None)
        return dict(s=bs)


CTRL = dict(proposed=lambda Ts, d, nom, **kw: FASMPC(Ts, d, nom, use_dob=True, **kw),
            nominal=lambda Ts, d, nom, **kw: FASMPC(Ts, d, nom, use_dob=False, **kw),
            faspd=lambda Ts, d, nom, **kw: FASPD(Ts, d, nom, **kw),
            ladrc=lambda Ts, d, nom, **kw: LADRC(Ts, d, nom, **kw),
            pr=lambda Ts, d, nom, **kw: PRAD(Ts, d, nom, **kw),
            fcs=lambda Ts, d, nom, **kw: FCSMPC(Ts, d, nom, **kw))


# --------------------------------------------------------------------------- runner
def duties(u, vdc):
    va = u[0]
    vb = -0.5 * u[0] + 0.5 * SQ3 * u[1]
    vc = -0.5 * u[0] - 0.5 * SQ3 * u[1]
    v = np.array([va, vb, vc])
    v0 = -0.5 * (v.max() + v.min())
    return np.clip(0.5 + (v + v0) / vdc, 0.0, 1.0)


def simulate(ctrl, model='avg', sc=None, ctrl_kw=None, Ts=None, h=0.1e-6, rec_dt=5e-6, keep_ctrl=False,
             pwm='double'):
    """pwm='double': 10 kHz carrier, sampling/update at carrier peak and valley (Ts = 50 us);
    pwm='single': one update per carrier period (Ts = carrier period)."""
    sc = default_scenario() if sc is None else sc
    ctrl_kw = {} if ctrl_kw is None else dict(ctrl_kw)
    if Ts is None:
        Ts = 25e-6 if ctrl == 'fcs' else (50e-6 if pwm == 'double' else 100e-6)
    switching = (model == 'sw') or ctrl == 'fcs'
    delay = 1 if switching else 0  # one-sample computation delay in the switching-level model
    tr, rc, swp, nom = sc['true'], sc['rect'], sc['sw'], sc['nominal']
    rng = np.random.default_rng(sc['seed'])
    nsub = int(round(Ts / h))
    rec_every = int(round(rec_dt / h))
    K = int(round(sc['Tsim'] / Ts))
    nrec = K * nsub // rec_every
    rec = np.zeros((nrec, 6))
    C = CTRL[ctrl](Ts, delay, nom, **ctrl_kw)
    # initial steady state with the base linear load
    x = np.array([tr['G'] * VPK, tr['Cf'] * W0 * VPK, VPK, 0.0])
    isr = np.zeros(3)
    st = np.zeros(3, dtype=np.int64)
    vd = 0.95 * SQ3 * VPK
    prev_gate = np.zeros(3, dtype=np.int64)
    u_pending = getattr(C, 'u_init', vref(0.0))
    s_pending = np.array([0, 0, 0])
    log = dict(u=np.zeros((K, 2)), dhat=np.zeros((K, 2)), zhat=np.zeros((K, 4)), y=np.zeros((K, 2)),
               vdc=np.zeros(K), edges=0, uavg=np.zeros((K, 2)))
    ri = 0
    wall = 0.0
    for k in range(K):
        t = k * Ts
        vdc_true = sc['Vdc_sag'] if sc['t_sag'] <= t < sc['t_rec'] else sc['Vdc_nom']
        y = x[2:4] + sc['sigma_v'] * rng.standard_normal(2)
        if switching:
            vdc_meas = vdc_true * (1 + swp['vdc_rip'] * np.sin(2 * np.pi * swp['f_rip'] * t))
            y = np.round(y / swp['adc_lsb']) * swp['adc_lsb']
        else:
            vdc_meas = vdc_true
        Pc = sc['P_cpl'] if t >= sc['t_cpl'] else 0.0
        S = x[2] ** 2 + x[3] ** 2 + nom['eps']
        il = (tr['G'] + (2 / 3) * Pc / S) * x[2:4] + np.array([isr[0], (isr[1] - isr[2]) / SQ3])
        aux = dict(ic=x[0:2] - il + sc['sigma_i'] * rng.standard_normal(2))
        t1 = time.perf_counter()
        out = C.step(k, t, y, vdc_meas, aux)
        wall += time.perf_counter() - t1
        rect_on = t >= sc['t_nl']
        if ctrl == 'fcs':
            s_apply = out['s'] if delay == 0 else s_pending
            s_pending = out['s']
            a_on = np.zeros(3)
            b_on = np.where(s_apply == 1, Ts + 1e-12, 0.0)
            ua0 = ua1 = 0.0
        else:
            u_apply = out['u'] if delay == 0 else u_pending
            C.stats.setdefault('app_ratio', []).append(np.linalg.norm(u_apply) / (vdc_meas / SQ3))
            u_pending = out['u']
            log['u'][k] = u_apply
            if switching and pwm == 'double':
                # half carrier period: rising half (k even) ends 'on', falling half (k odd) starts 'on'
                d = duties(u_apply, vdc_meas)
                if k % 2 == 0:
                    a_on = (1 - d) * Ts
                    b_on = np.full(3, Ts + 1e-12)
                    a_on = np.where(d <= 0.0, Ts + 1e-12, a_on)
                else:
                    a_on = np.zeros(3)
                    b_on = d * Ts
            elif switching:
                d = duties(u_apply, vdc_meas)
                a_on = (1 - d) * Ts / 2
                b_on = (1 + d) * Ts / 2
                b_on = np.where(d >= 1.0, Ts + 1e-12, b_on)
            else:
                a_on = b_on = np.zeros(3)
            ua0, ua1 = u_apply
        vd, uavg0, uavg1, ed = run_period(x, isr, st, vd, nsub, h, tr['Lf'], tr['Cf'], tr['rf'], switching,
                                          ua0, ua1, a_on, b_on, prev_gate, swp['td'], vdc_true,
                                          swp['vdc_rip'] if switching else 0.0, swp['f_rip'], t, swp['Von'],
                                          tr['G'], Pc, nom['eps'], rect_on, rc['Ls'], rc['Rs'], rc['Cd'], rc['Rd'],
                                          rec_every, rec, ri)
        ri += nsub // rec_every
        log['edges'] += ed
        log['uavg'][k] = (uavg0, uavg1)
        if ctrl == 'fcs':
            prev_gate = s_apply.astype(np.int64)
        elif switching:
            prev_gate = ((b_on > Ts) & (a_on < Ts)).astype(np.int64)
        log['vdc'][k] = vdc_true
        log['y'][k] = y
        if 'dhat' in out:
            log['dhat'][k] = out['dhat']
            log['zhat'][k] = out['zhat']
        if not np.all(np.isfinite(x)) or np.abs(x).max() > 1e4:
            log['diverged'] = True
            break
    trec = (np.arange(nrec) + 1) * rec_dt
    res = dict(t=trec, rec=rec, log=log, Ts=Ts, stats=C.stats, wall=wall / K, sc=sc, ctrl=ctrl, model=model)
    res['metrics'] = metrics(res)
    if keep_ctrl:
        res['C'] = C
    return res


# --------------------------------------------------------------------------- metrics
def thd(v, dt, f0=F0, hmax=50):
    n = int(round(1 / (f0 * dt)))
    v = v[-n:]
    X = np.abs(np.fft.rfft(v)) * 2 / n
    V1 = X[1]
    return 100 * np.sqrt(np.sum(X[2:hmax + 1] ** 2)) / V1


def metrics(res):
    sc = res['sc']
    t, rec = res['t'], res['rec']
    if res['log'].get('diverged'):
        return dict(diverged=True)
    e = rec[:, :2] - np.stack([VPK * np.cos(W0 * t), VPK * np.sin(W0 * t)], 1)
    en = np.linalg.norm(e, axis=1)

    def rms(a, b):
        m = (t >= a) & (t < b)
        return float(np.sqrt(np.mean(en[m] ** 2)))

    dt = t[1] - t[0]
    m = {}
    m['rms'] = rms(0, sc['Tsim'])
    m['base'] = rms(0.005, sc['t_cpl'])
    m['cpl'] = rms(sc['t_cpl'], sc['t_nl'])
    m['nl'] = rms(sc['t_nl'], sc['t_sag'])
    m['sag'] = rms(sc['t_sag'], sc['t_rec'])
    m['final'] = rms(sc['t_rec'], sc['Tsim'])
    mk = (t >= sc['t_cpl']) & (t < sc['t_cpl'] + 0.01)
    m['peak_cpl'] = float(en[mk].max())
    # recovery: last time in the CPL window the error exceeds 5 % of Vpk (before the NL event)
    w = (t >= sc['t_cpl']) & (t < sc['t_nl'])
    above = np.where(en[w] > 0.05 * VPK)[0]
    m['t_rec_cpl_ms'] = float((t[w][above[-1]] - sc['t_cpl']) * 1e3) if len(above) else 0.0
    idx_nl = (t >= sc['t_sag'] - 0.02) & (t < sc['t_sag'])
    idx_fin = (t >= sc['Tsim'] - 0.02) & (t < sc['Tsim'])
    m['thd_nl'] = float(thd(rec[idx_nl, 0], dt))
    m['thd_final'] = float(thd(rec[idx_fin, 0], dt))
    u = res['log']['u']
    vdc = res['log']['vdc']
    nz = vdc > 0
    m['max_mod'] = float(np.max(np.linalg.norm(u[nz], axis=1) / (vdc[nz] / SQ3))) if np.any(u) else float('nan')
    st = res['stats']
    m['active_pct'] = 100.0 * st['active'] / max(st['n'], 1)
    m['proj_pct'] = 100.0 * st['proj'] / max(st['n'], 1)
    m['exc_max'] = float(st.get('exc_max', float('nan')))
    ar = np.array(st.get('app_ratio', [np.nan]))
    m['app_exc_pct'] = 100.0 * float(np.mean(ar > 1 + CLIP_TOL)) if 'app_ratio' in st else float('nan')
    m['app_exc_max'] = float(np.nanmax(ar))
    m['unc_ok_pct'] = 100.0 * st['unc_ok'] / max(st['n'], 1) if 'unc_ok' in st else float('nan')
    for key in ('qp_solve', 'qp_run'):
        v = 1e6 * np.array(st.get(key, [np.nan]))
        m[key + '_us_med'], m[key + '_us_p99'], m[key + '_us_max'] = (float(np.median(v)), float(np.percentile(v, 99)),
                                                                       float(np.max(v)))
    m['solve_ms_med'] = 1e3 * float(np.median(st['solve_time'])) if st['solve_time'] else 0.0
    m['solve_ms_p99'] = 1e3 * float(np.percentile(st['solve_time'], 99)) if st['solve_time'] else 0.0
    m['iters_med'] = float(np.median(st['iters'])) if st['iters'] else 0.0
    m['fsw_khz'] = res['log']['edges'] / 3 / sc['Tsim'] / 1e3
    m['wall_ms'] = 1e3 * res['wall']
    return m
