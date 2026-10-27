"""Parallel parameter sweeps: python sweep.py <name>  (results -> ../results/sweep_<name>.json)."""
import sys
import json
import itertools
from multiprocessing import Pool
import numpy as np
from sim import simulate, default_scenario, W0, VPK

KEYS = ['rms', 'base', 'cpl', 'nl', 'sag', 'final', 'peak_cpl', 't_rec_cpl_ms', 'thd_nl', 'thd_final',
        'max_mod', 'active_pct', 'proj_pct', 'solve_ms_med', 'fsw_khz']


def job(args):
    ctrl, model, kw, sc_kw, tag = args
    sc = default_scenario()
    for k, v in sc_kw.items():
        if isinstance(v, dict):
            sc[k].update(v)
        else:
            sc[k] = v
    try:
        m = simulate(ctrl, model, sc=sc, ctrl_kw=kw)['metrics']
    except Exception as e:  # noqa
        m = dict(error=str(e))
    return dict(ctrl=ctrl, model=model, kw=kw, sc=sc_kw, tag=tag,
                m={k: m.get(k) for k in KEYS} if 'error' not in m and not m.get('diverged') else m)


def grid(**axes):
    names = list(axes)
    return [dict(zip(names, vals)) for vals in itertools.product(*axes.values())]


def run(jobs, name, procs=30):
    with Pool(procs) as p:
        out = p.map(job, jobs, chunksize=1)
    with open(f'../results/sweep_{name}.json', 'w') as f:
        json.dump(out, f, indent=1)
    return out


SA0 = 0.5 * W0 ** 2 * VPK * 50

if __name__ == '__main__':
    name = sys.argv[1]
    jobs = []
    if name == 'tune1':
        for kw in grid(sigma_a=[SA0 * s for s in (0.1, 0.3, 1, 3, 10)], alpha=[0.1, 0.3, 0.6, 1.0],
                       r_eta=[2e-6, 2e-5, 2e-4]):
            jobs.append(('proposed', 'avg', kw, {}, ''))
    elif name == 'tune2':
        for kw in grid(sigma_a=[SA0 * s for s in (0.03, 0.1, 0.3, 1, 3)], alpha=[0.05, 0.1, 0.2, 0.4, 0.7],
                       r_eta=[2e-6, 2e-5], dob_in_kf=[True]):
            jobs.append(('proposed', 'avg', kw, {}, ''))
    elif name == 'tune3':
        for kw in grid(sigma_a=[SA0 * s for s in (0.5, 1, 2)], alpha=[0.5, 0.7, 0.9],
                       r_eta=[3e-7, 1e-6, 3e-6], qv=[0.0, 0.005, 0.05], N=[5, 10]):
            jobs.append(('proposed', 'avg', kw, {}, ''))
    elif name == 'tune_pr2':
        for kw in grid(Kc=[10.0, 15.0, 20.0], Kpv=[0.05, 0.1, 0.2], k1=[1.0, 5.0], k57=[0.5, 2.0, 5.0]):
            k1, k57 = kw.pop('k1'), kw.pop('k57')
            kw['Kr'] = (k1, k57, k57)
            jobs.append(('pr', 'avg', kw, {}, ''))
    elif name == 'tune_ladrc2':
        for kw in grid(wc=[2 * np.pi * f for f in (450, 600, 800, 1000, 1200)], wo=[2 * np.pi * f for f in (2500, 3000, 3500, 4000, 5000)]):
            jobs.append(('ladrc', 'avg', kw, {}, ''))
    elif name == 'tune_pd2':
        for kw in grid(sigma_a=[SA0 * s for s in (0.03, 0.1, 0.3)], wc=[2 * np.pi * f for f in (600, 800, 1000, 1200, 1500)], zeta=[0.7, 0.9, 1.2]):
            jobs.append(('faspd', 'avg', kw, {}, ''))
    elif name == 'tune4':
        for kw in grid(sigma_a=[SA0 * s for s in (0.3, 1, 2, 4)], alpha=[0.3, 0.5, 0.7, 0.9],
                       r_eta=[3e-7, 3e-6], qv=[0.0, 0.005], N=[10], gamma_u=[1.0], M=[24]):
            jobs.append(('proposed', 'avg', kw, {}, ''))
    elif name == 'tune5':
        for kw in grid(sigma_a=[SA0 * s for s in (1, 2, 4, 8)], alpha=[0.9, 1.2, 1.5],
                       r_eta=[3e-7], qv=[0.0, 0.005], kappa_f=[1.0, 5.0, 20.0], gamma_u=[1.0], M=[24]):
            jobs.append(('proposed', 'avg', kw, {}, ''))
    elif name == 'swtune':
        P = 2 * np.pi
        for kw in grid(sigma_a=[SA0 * s for s in (1, 3, 10)], alpha=[0.2, 0.4, 0.7],
                       r_eta=[3e-7, 3e-6], qv=[0.005, 0.05], hb=[((), 0.0), ((1, -5, 7), 0.02), ((1, -5, 7), 0.05)],
                       gamma_u=[1.0], M=[24]):
            kw['harm'], kw['beta_h'] = kw.pop('hb')
            jobs.append(('proposed', 'sw', kw, {}, ''))
        for kw in grid(sigma_a=[SA0 * s for s in (1, 3, 10)], r_eta=[3e-7, 3e-6, 3e-5], qv=[0.005, 0.05],
                       gamma_u=[1.0], M=[24]):
            jobs.append(('nominal', 'sw', kw, {}, ''))
        for kw in grid(wc=[P * f for f in (600, 800, 1000, 1200, 1500, 2000)], wo=[P * f for f in (3000, 4000, 5000, 6000, 8000)]):
            jobs.append(('ladrc', 'sw', kw, {}, ''))
        for kw in grid(Kc=[6.0, 10.0, 15.0, 20.0], Kpv=[0.1, 0.2, 0.3], kr=[(5.0, 5.0), (10.0, 5.0), (10.0, 10.0), (20.0, 10.0)]):
            k1, k57 = kw.pop('kr')
            kw['Kr'] = (k1, k57, k57)
            jobs.append(('pr', 'sw', kw, {}, ''))
        for kw in grid(sigma_a=[SA0 * s for s in (0.3, 1, 3, 10)], wc=[P * f for f in (800, 1000, 1200, 1500, 2000)], zeta=[0.7, 1.0]):
            jobs.append(('faspd', 'sw', kw, {}, ''))
        for kw in grid(lam=[0.0, 1.0, 3.0, 10.0],
                       obs_poles=[(0.4, 0.45, 0.5), (0.5, 0.55, 0.6), (0.6, 0.65, 0.7)]):
            jobs.append(('fcs', 'sw', kw, {}, ''))
    elif name == 'swtune_p2':
        for kw in grid(sigma_a=[SA0 * s for s in (2, 3, 5)], alpha=[0.7, 1.0, 1.3],
                       r_eta=[3e-8, 3e-7], qv=[0.0, 0.001, 0.005], beta_h=[0.02, 0.05], N=[5, 10, 20],
                       harm=[(1, -5, 7)], gamma_u=[1.0], M=[24]):
            jobs.append(('proposed', 'sw', kw, {}, ''))
    elif name == 'swtune_b2':
        P = 2 * np.pi
        for kw in grid(wc=[P * f for f in (1500, 2000, 2500, 3000, 3500)], wo=[P * f for f in (2000, 2500, 3000, 4000, 5000)]):
            jobs.append(('ladrc', 'sw', kw, {}, ''))
        for kw in grid(Kc=[15.0, 20.0, 25.0, 30.0, 35.0], Kpv=[0.15, 0.2, 0.3, 0.4], kr=[(5.0, 5.0), (10.0, 5.0), (10.0, 10.0)]):
            k1, k57 = kw.pop('kr')
            kw['Kr'] = (k1, k57, k57)
            jobs.append(('pr', 'sw', kw, {}, ''))
        for kw in grid(sigma_a=[SA0 * s for s in (1, 3, 10)], wc=[P * f for f in (2000, 2500, 3000, 3500)], zeta=[0.5, 0.7, 1.0]):
            jobs.append(('faspd', 'sw', kw, {}, ''))
        for kw in grid(lam=[0.0, 0.3, 1.0, 2.0],
                       obs_poles=[(0.6, 0.65, 0.7), (0.7, 0.75, 0.8), (0.8, 0.83, 0.86)]):
            jobs.append(('fcs', 'sw', kw, {}, ''))
    elif name == 'tune_pd':
        for kw in grid(sigma_a=[SA0 * s for s in (0.1, 0.3, 1, 3, 10)], wc=[2 * np.pi * f for f in (300, 600, 1000, 1500)]):
            jobs.append(('faspd', 'avg', kw, {}, ''))
    elif name == 'tune_ladrc':
        for kw in grid(wc=[2 * np.pi * f for f in (300, 600, 1000, 1500)], wo=[2 * np.pi * f for f in (1000, 2000, 3000, 4000)]):
            jobs.append(('ladrc', 'avg', kw, {}, ''))
    elif name == 'tune_pr':
        for kw in grid(Kc=[3.0, 6.0, 10.0, 15.0], Kpv=[0.02, 0.05, 0.1, 0.2], ks=[0.2, 1.0, 5.0]):
            ks = kw.pop('ks')
            kw['Kr'] = (ks * 1.0, ks * 0.5, ks * 0.5)
            jobs.append(('pr', 'avg', kw, {}, f'ks={ks}'))
    out = run(jobs, name)
    ok = [o for o in out if 'rms' in o['m'] and o['m']['rms'] is not None]
    ok.sort(key=lambda o: o['m']['rms'])
    if name in ('swtune', 'swtune_b2'):
        for c in ['proposed', 'nominal', 'faspd', 'ladrc', 'pr', 'fcs']:
            oc = [o for o in ok if o['ctrl'] == c]
            for o in oc[:3]:
                print(c, {k: (round(v / SA0, 3) if k == 'sigma_a' else (round(v / 2 / np.pi) if k in ('wc', 'wo') else v)) for k, v in o['kw'].items()},
                      {k: round(o['m'][k], 2) for k in ('rms', 'base', 'cpl', 'nl', 'sag', 'final', 'thd_nl', 'active_pct', 'proj_pct', 'fsw_khz')})
            print(c, 'failed', sum(1 for o in out if o['ctrl'] == c) - len(oc))
        sys.exit()
    for o in ok[:12]:
        print(o['tag'], {k: (round(v, 6) if isinstance(v, float) else v) for k, v in o['kw'].items()},
              {k: round(o['m'][k], 3) for k in ('rms', 'base', 'cpl', 'nl', 'sag', 'thd_nl', 'max_mod')})
    print('failed:', len(out) - len(ok))
