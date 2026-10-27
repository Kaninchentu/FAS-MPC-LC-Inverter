"""Pick final controller parameters: top grid-search configurations re-evaluated over 5 noise seeds."""
import json
import numpy as np
from sweep import run, SA0

N_TOP, SEEDS = 6, range(5)


def top(fname, ctrl, n):
    d = json.load(open(f'../results/sweep_{fname}.json'))
    d = [o for o in d if o['ctrl'] == ctrl and 'rms' in o['m'] and o['m']['rms'] is not None]
    d.sort(key=lambda o: o['m']['rms'])
    return [o['kw'] for o in d[:n]]


def tup(kw):
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in kw.items()}


if __name__ == '__main__':
    cands = {'proposed': top('swtune_p2', 'proposed', N_TOP),
             'nominal': top('swtune', 'nominal', 3),
             'faspd': top('swtune_b2', 'faspd', 3) + top('swtune', 'faspd', 2),
             'ladrc': top('swtune_b2', 'ladrc', 3) + top('swtune', 'ladrc', 2),
             'pr': top('swtune_b2', 'pr', 3) + top('swtune', 'pr', 2),
             'fcs': top('swtune_b2', 'fcs', 3)}
    jobs = []
    for c, kws in cands.items():
        for i, kw in enumerate(kws):
            for s in SEEDS:
                jobs.append((c, 'sw', tup(kw), {'seed': s}, str(i)))
    out = run(jobs, 'select')
    final = {}
    for c, kws in cands.items():
        best = None
        for i, kw in enumerate(kws):
            r = [o['m']['rms'] for o in out if o['ctrl'] == c and o['tag'] == str(i) and 'rms' in o['m']]
            if len(r) < len(SEEDS):
                continue
            mu = float(np.mean(r))
            if best is None or mu < best[0]:
                best = (mu, float(np.std(r)), kw)
        final[c] = best[2]
        print(c, round(best[0], 3), '+-', round(best[1], 3), best[2])
    json.dump(final, open('../results/final_params.json', 'w'), indent=1)
