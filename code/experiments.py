"""All experiments reported in the manuscript.

python experiments.py           -> ../results/exp_all.json  and  ../results/waveforms.npz
"""
import json
import time
from multiprocessing import Pool
import numpy as np
from sim import simulate, default_scenario, NOMINAL

P = json.load(open('../results/final_params.json'))
P.pop('selection_rule', None)
P = {c: {k: (tuple(v) if isinstance(v, list) else v) for k, v in kw.items()} for c, kw in P.items()}
P['fcs40'] = dict(P.pop('fcs'))
KEYS = ['rms', 'base', 'cpl', 'nl', 'sag', 'final', 'peak_cpl', 't_rec_cpl_ms', 'thd_nl', 'thd_final',
        'max_mod', 'active_pct', 'proj_pct', 'solve_ms_med', 'solve_ms_p99', 'iters_med', 'fsw_khz', 'wall_ms',
        'exc_max', 'app_exc_pct', 'app_exc_max', 'unc_ok_pct', 'qp_solve_us_med', 'qp_solve_us_p99',
        'qp_solve_us_max', 'qp_run_us_med', 'qp_run_us_p99', 'qp_run_us_max']
CTRL_OF = dict(fcs20='fcs', fcs40='fcs', propnoim='proposed', propfb='proposed')
P['propfb'] = dict(P['proposed'], qp_mode='fallback')  # deadline-safe variant: no QP, clipping only


def job(a):
    exp, name, model, kw, sc_kw, seed = a
    kw = dict(kw)
    Ts = kw.pop('_Ts', None)
    sc = default_scenario(seed=seed)
    for k, v in sc_kw.items():
        if isinstance(v, dict):
            sc[k].update(v)
        else:
            sc[k] = v
    try:
        m = simulate(CTRL_OF.get(name, name), model, sc=sc, ctrl_kw=kw, Ts=Ts)['metrics']
        m = {k: m.get(k) for k in KEYS} if not m.get('diverged') else dict(diverged=True)
    except Exception as e:  # noqa
        m = dict(error=repr(e))
    return dict(exp=exp, name=name, model=model, kw={k: v for k, v in kw.items()}, sc=sc_kw, seed=seed, m=m)


def jobs_all():
    J = []
    allc = ['proposed', 'propnoim', 'nominal', 'faspd', 'ladrc', 'pr', 'fcs20', 'fcs40']
    for c in allc:  # E1 main switching-level comparison
        for s in range(10):
            J.append(('main_sw', c, 'sw', P[c], {}, s))
    for c in ['proposed', 'propnoim', 'nominal', 'faspd', 'ladrc', 'pr']:  # E2 averaged model
        for s in range(5):
            J.append(('main_avg', c, 'avg', P[c], {}, s))
    for s in range(10):  # E1b deadline-safe fallback
        J.append(('main_sw', 'propfb', 'sw', P['propfb'], {}, s))
    for vs in (340.0, 320.0, 300.0):  # E3 deep sag
        for c in ['proposed', 'propfb', 'faspd', 'ladrc', 'pr', 'fcs40']:
            for s in range(3):
                J.append((f'sag_{vs:.0f}', c, 'sw', P[c], {'Vdc_sag': vs}, s))
    base = P['proposed']
    sens = dict(N=[3, 5, 10, 20], alpha=[0.05, 0.1, 0.2, 0.35],
                sigma_a=[base['sigma_a'] * f for f in (0.1, 0.3, 1.0, 3.0)],
                beta_h=[0.0, 0.2, 0.5, 1.0], r_eta=[3e-8, 3e-7, 3e-6, 3e-5],
                qv=[0.01, 0.05, 0.2, 1.0], kappa_f=[0.0, 1.0, 5.0, 20.0],
                gamma_u=[0.9, 0.95, 1.0], M=[8, 12, 24, 48])
    for p, vals in sens.items():  # E4 one-at-a-time sensitivity (proposed)
        for v in vals:
            kw = dict(base)
            if p == 'beta_h' and v == 0.0:
                kw['harm'] = ()
            kw[p] = v
            for s in range(3):
                J.append((f'sens_{p}', 'proposed', 'sw', kw, {}, s))
    for lr in (0.6, 0.7, 0.85, 1.0, 1.15, 1.3, 1.4):  # E5 L/C mismatch map
        for cr in (0.8, 0.9, 1.0, 1.1, 1.2):
            tr = dict(Lf=lr * NOMINAL['L0'], Cf=cr * NOMINAL['C0'])
            for c in ['proposed', 'propnoim', 'faspd', 'ladrc', 'pr', 'fcs40']:
                J.append(('lc_map', c, 'sw', P[c], {'true': tr, 'tag': f'{lr},{cr}'}, 0))
    rob = dict(sigma_v=[0.2, 0.5, 1.0, 2.0], td=[0.0, 0.5e-6, 1e-6, 2e-6], P_cpl=[600.0, 1200.0, 1800.0, 2400.0])
    for p, vals in rob.items():
        for v in vals:
            sc_kw = {'sw': {'td': v}} if p == 'td' else {p: v}
            for c in ['proposed', 'ladrc', 'pr', 'fcs40']:
                for s in range(2):
                    J.append((f'rob_{p}', c, 'sw', P[c], dict(sc_kw, tag=str(v)), s))
    return J


def longrun_job(name):
    """1-s run (events as in the main test, then 0.9 s of steady operation) for slow-mode convergence."""
    from sim import thd, VPK, W0
    r = simulate(CTRL_OF.get(name, name), 'sw', sc=default_scenario(Tsim=1.0), ctrl_kw=P[name])
    t, rec = r['t'], r['rec']
    e = np.linalg.norm(rec[:, :2] - np.stack([VPK * np.cos(W0 * t), VPK * np.sin(W0 * t)], 1), axis=1)
    dt = t[1] - t[0]
    n = int(round(0.02 / dt))
    cyc = []
    for i in range(len(t) // n):
        sl = slice(i * n, (i + 1) * n)
        cyc.append(dict(t_end=float(t[sl][-1]), rms=float(np.sqrt(np.mean(e[sl] ** 2))), thd=float(thd(rec[sl, 0], dt))))
    dh = r['log']['dhat']
    k20 = int(round(0.02 / r['Ts']))
    dcyc = [float(np.sqrt(np.mean(np.sum(dh[i * k20:(i + 1) * k20] ** 2, 1)))) for i in range(len(dh) // k20)]
    return dict(name=name, cycles=cyc, dhat_rms=dcyc, metrics={k: r['metrics'].get(k) for k in KEYS})


def waveforms():
    out = {}
    for c in ['proposed', 'propnoim', 'nominal', 'faspd', 'ladrc', 'pr', 'fcs40']:
        kw = dict(P[c])
        Ts = kw.pop('_Ts', None)
        r = simulate(CTRL_OF.get(c, c), 'sw', ctrl_kw=kw, Ts=Ts)
        out[f'{c}_t'] = r['t'][::2]
        out[f'{c}_rec'] = r['rec'][::2].astype(np.float32)
        out[f'{c}_u'] = r['log']['u'].astype(np.float32)
        out[f'{c}_vdc'] = r['log']['vdc'].astype(np.float32)
        out[f'{c}_dhat'] = r['log']['dhat'].astype(np.float32)
        out[f'{c}_zhat'] = r['log']['zhat'].astype(np.float32)
        out[f'{c}_y'] = r['log']['y'].astype(np.float32)
        out[f'{c}_Ts'] = r['Ts']
    ra = simulate('proposed', 'avg', ctrl_kw=P['proposed'])
    out['proposed_avg_t'] = ra['t'][::2]
    out['proposed_avg_rec'] = ra['rec'][::2].astype(np.float32)
    np.savez_compressed('../results/waveforms.npz', **out)


if __name__ == '__main__':
    t0 = time.time()
    J = jobs_all()
    print('jobs', len(J))
    with Pool(30) as p:
        res = p.map(job, J, chunksize=1)
    for r in res:  # json-friendly
        r['kw'] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in r['kw'].items()}
    json.dump(dict(params={k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
                           for k, v in P.items()}, results=res), open('../results/exp_all.json', 'w'), indent=0)
    print('sim done', round(time.time() - t0), 's; errors:', sum(1 for r in res if 'error' in r['m']),
          'diverged:', sum(1 for r in res if r['m'].get('diverged')))
    waveforms()
    print('waveforms done', round(time.time() - t0), 's')
    with Pool(3) as p:
        lr = p.map(longrun_job, ['proposed', 'propnoim', 'ladrc'])
    json.dump(lr, open('../results/longrun.json', 'w'), indent=0)
    print('long runs done', round(time.time() - t0), 's')
