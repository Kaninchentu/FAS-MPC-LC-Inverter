"""Plant models for the standalone LC-filtered three-phase VSI.

Two fidelity levels share the same LC filter, load and rectifier models:
  * averaged : the inverter voltage equals the commanded alpha-beta voltage (ZOH),
  * switching: two-level three-leg bridge with centred PWM / direct switching
               states, dead time, device on-state drops and dc-link ripple.
The nonlinear load is a physical three-phase diode bridge with ac-side
inductance and an RC dc side, simulated as a hybrid system.
All states are in amplitude-invariant alpha-beta coordinates except the
rectifier ac currents (abc).
"""
import numpy as np
import numba as nb

SQ3 = np.sqrt(3.0)


@nb.njit(cache=True)
def _abc_from_ab(a, b):
    return a, -0.5 * a + 0.5 * SQ3 * b, -0.5 * a - 0.5 * SQ3 * b


@nb.njit(cache=True)
def _rectifier_step(vx, isr, st, vd, h, Ls, Rs, Cd, Rd):
    """One forward-Euler step of the diode bridge (hybrid model).

    vx: phase voltages at the bridge input (abc, w.r.t. capacitor star point)
    isr: ac currents into the bridge (abc), st: conduction state (+1 to P, -1 to N, 0 off)
    returns new vd; isr and st are updated in place.
    """
    nP = 0
    nN = 0
    for x in range(3):
        if st[x] == 1:
            nP += 1
        elif st[x] == -1:
            nN += 1
    if nP == 0 or nN == 0:
        # bridge blocked: start conduction when the line-line voltage exceeds vd
        for x in range(3):
            st[x] = 0
            isr[x] = 0.0
        imax = 0
        imin = 0
        for x in range(3):
            if vx[x] > vx[imax]:
                imax = x
            if vx[x] < vx[imin]:
                imin = x
        if vx[imax] - vx[imin] > vd:
            st[imax] = 1
            st[imin] = -1
            nP = 1
            nN = 1
        else:
            return vd + h * (-vd / Rd) / Cd
    # node potential of the negative rail (common mode floats, sum di/dt = 0)
    vN = 0.0
    for it in range(3):
        s = 0.0
        nP = 0
        nN = 0
        for x in range(3):
            if st[x] != 0:
                s += vx[x] - Rs * isr[x]
                if st[x] == 1:
                    nP += 1
                else:
                    nN += 1
        vN = (s - nP * vd) / (nP + nN)
        vP = vN + vd
        changed = False
        for x in range(3):
            if st[x] == 0:
                if vx[x] > vP:
                    st[x] = 1
                    changed = True
                elif vx[x] < vN:
                    st[x] = -1
                    changed = True
        if not changed:
            break
    vP = vN + vd
    iP = 0.0
    for x in range(3):
        if st[x] == 1:
            iP += isr[x]
            isr[x] += h * (vx[x] - Rs * isr[x] - vP) / Ls
        elif st[x] == -1:
            isr[x] += h * (vx[x] - Rs * isr[x] - vN) / Ls
    vd_new = vd + h * (iP - vd / Rd) / Cd
    # diode turn-off at current reversal
    off = False
    for x in range(3):
        if (st[x] == 1 and isr[x] <= 0.0) or (st[x] == -1 and isr[x] >= 0.0):
            st[x] = 0
            isr[x] = 0.0
            off = True
    if off:
        # restore sum(i)=0 after clipping
        s = isr[0] + isr[1] + isr[2]
        non = 0
        for x in range(3):
            if st[x] != 0:
                non += 1
        if non > 0:
            for x in range(3):
                if st[x] != 0:
                    isr[x] -= s / non
    return vd_new


@nb.njit(cache=True)
def _load_current(v0, v1, G, Pcpl, eps, isr):
    S = v0 * v0 + v1 * v1 + eps
    c = G + (2.0 / 3.0) * Pcpl / S
    ira = isr[0]
    irb = (isr[1] - isr[2]) / SQ3
    return c * v0 + ira, c * v1 + irb


@nb.njit(cache=True)
def run_period(x, isr, st, vd, nsub, h, Lf, Cf, rf,
               switching, uab0, uab1, a_on, b_on, prev_gate, td, vdc0, vdc_rip, f_rip, t0,
               Von, G, Pcpl, eps, rect_on, Ls, Rs, Cd, Rd, rec_every, rec_buf, rec_idx):
    """Integrate the plant over one control/PWM period.

    x = [i_alpha, i_beta, v_alpha, v_beta] (in place). For switching=True the leg
    gate signals are 'upper on' for a_on[x] <= tau < b_on[x]; prev_gate holds the
    gate at the end of the previous period (for dead time). Returns
    (vd, avg_u_alpha, avg_u_beta, n_rising_edges).
    """
    ua_sum = 0.0
    ub_sum = 0.0
    edges = 0
    g_last = np.empty(3, dtype=np.int64)
    for q in range(3):
        g_last[q] = prev_gate[q]
    vabc = np.empty(3)
    vleg = np.empty(3)
    ri = rec_idx
    for n in range(nsub):
        tau = n * h
        t = t0 + tau
        if switching:
            vdc = vdc0 * (1.0 + vdc_rip * np.sin(2.0 * np.pi * f_rip * t))
            ia, ib, ic = _abc_from_ab(x[0], x[1])
            iabc = (ia, ib, ic)
            for q in range(3):
                g_now = 1 if (tau >= a_on[q] and tau < b_on[q]) else 0
                if g_now == 1 and g_last[q] == 0:
                    edges += 1
                g_last[q] = g_now
                tp = tau - td
                if tp < 0.0:
                    g_past = prev_gate[q]
                else:
                    g_past = 1 if (tp >= a_on[q] and tp < b_on[q]) else 0
                iq = iabc[q]
                sgn = 1.0 if iq >= 0.0 else -1.0
                if g_now == 1 and g_past == 1:
                    vleg[q] = 0.5 * vdc - Von * sgn
                elif g_now == 0 and g_past == 0:
                    vleg[q] = -0.5 * vdc - Von * sgn
                else:  # dead time: freewheeling diode decided by current sign
                    vleg[q] = -0.5 * vdc * sgn - Von * sgn
            ua = (2.0 / 3.0) * (vleg[0] - 0.5 * vleg[1] - 0.5 * vleg[2])
            ub = (vleg[1] - vleg[2]) / SQ3
        else:
            ua = uab0
            ub = uab1
        ua_sum += ua
        ub_sum += ub
        # rectifier
        if rect_on:
            va, vb, vc = _abc_from_ab(x[2], x[3])
            vabc[0] = va
            vabc[1] = vb
            vabc[2] = vc
            vd = _rectifier_step(vabc, isr, st, vd, h, Ls, Rs, Cd, Rd)
        il0, il1 = _load_current(x[2], x[3], G, Pcpl, eps, isr)
        # semi-implicit Euler for the LC filter
        x[0] += h * (ua - x[2] - rf * x[0]) / Lf
        x[1] += h * (ub - x[3] - rf * x[1]) / Lf
        x[2] += h * (x[0] - il0) / Cf
        x[3] += h * (x[1] - il1) / Cf
        if (n + 1) % rec_every == 0:
            rec_buf[ri, 0] = x[2]
            rec_buf[ri, 1] = x[3]
            rec_buf[ri, 2] = x[0]
            rec_buf[ri, 3] = x[1]
            rec_buf[ri, 4] = il0
            rec_buf[ri, 5] = il1
            ri += 1
    return vd, ua_sum / nsub, ub_sum / nsub, edges
