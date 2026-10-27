"""Aggregate exp_all.json into the numbers used in the manuscript -> ../results/summary.json"""
import json
from collections import defaultdict
import numpy as np

D = json.load(open('../results/exp_all.json'))
d = D['results']
OUT = {}


def agg(exp):
    rows = defaultdict(list)
    for r in d:
        if r['exp'] == exp:
            rows[r['name']].append(r['m'])
    return {c: {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms])))
                for k in ms[0] if isinstance(ms[0][k], (int, float))} for c, ms in rows.items()}


for exp in ('main_sw', 'main_avg', 'sag_340', 'sag_320', 'sag_300'):
    OUT[exp] = agg(exp)
lc = defaultdict(dict)
for r in d:
    if r['exp'] == 'lc_map':
        lc[r['name']][r['sc']['tag']] = r['m'].get('rms', float('nan')) if 'rms' in r['m'] else float('nan')
OUT['lc_map'] = lc
OUT['lc_box_worst'] = {c: max(x for t, x in v.items() if 0.69 <= float(t.split(',')[0]) <= 1.31 and 0.89 <= float(t.split(',')[1]) <= 1.11)
                       for c, v in lc.items()}
sens = defaultdict(list)
for r in d:
    if r['exp'].startswith('sens_'):
        p = r['exp'][5:]
        v = 0.0 if (p == 'beta_h' and r['kw'].get('harm') == []) else r['kw'].get(p)
        sens[f'{p}={v}'].append(r['m']['rms'])
OUT['sens'] = {k: float(np.mean(v)) for k, v in sens.items()}
rob = defaultdict(list)
for r in d:
    if r['exp'].startswith('rob_'):
        rob[f"{r['exp'][4:]}={r['sc']['tag']}|{r['name']}"].append(r['m']['rms'])
OUT['rob'] = {k: float(np.mean(v)) for k, v in rob.items()}
json.dump(OUT, open('../results/summary.json', 'w'), indent=1)

if __name__ == '__main__':
    for exp in ('main_sw', 'main_avg'):
        print('==', exp)
        for c, m in OUT[exp].items():
            print(f'{c:9s}', ' '.join(f"{k}={m[k][0]:.2f}" for k in ('rms', 'base', 'cpl', 'nl', 'sag', 'final', 'peak_cpl',
                                                                       't_rec_cpl_ms', 'thd_nl', 'thd_final', 'proj_pct', 'active_pct',
                                                                       'fsw_khz', 'solve_ms_med', 'solve_ms_p99')), f"sd={m['rms'][1]:.2f}")
    print('box worst', {c: round(v, 2) for c, v in OUT['lc_box_worst'].items()})
    for exp in ('sag_340', 'sag_320', 'sag_300'):
        print(exp, {c: (round(m['rms'][0], 2), round(m['sag'][0], 2)) for c, m in OUT[exp].items()})
