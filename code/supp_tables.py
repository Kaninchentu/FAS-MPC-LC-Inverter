"""Generate the LaTeX tables of the Supplementary Material from the result files."""
import json
import numpy as np

S = json.load(open('../results/summary.json'))
A = json.load(open('../results/analysis.json'))
G = json.load(open('../results/small_gain.json'))
B = json.load(open('../results/box_cert.json'))
T = json.load(open('../results/timing.json'))
LR = json.load(open('../results/longrun.json'))
TC = json.load(open('../results/tube_check.json'))
NAME = dict(proposed='Proposed', propfb='Proposed, no QP', propnoim='B1: w/o harmonic models', nominal='B2: w/o DOB', faspd='B3: FAS-PD',
            ladrc='B4: MA-LADRC', pr='B5: PR+AD', fcs20='B6: FCS-MPC (20 kHz)', fcs40='B6: FCS-MPC (40 kHz)')
out = []


def tab(caption, label, cols, rows, size='\\footnotesize'):
    out.append('\\begin{table}[!ht]\\centering\\caption{%s}\\label{%s}%s\n\\begin{tabular}{%s}\\toprule' % (caption, label, size, cols))
    for r in rows:
        out.append(r)
    out.append('\\bottomrule\\end{tabular}\\end{table}\n')


# S-I full metrics, switching level and averaged
for exp, cap in (('main_sw', 'Switching-level metrics (mean $\\pm$ standard deviation over ten noise realizations)'),
                 ('main_avg', 'Averaged-model metrics (mean over five noise realizations)')):
    M = S[exp]
    keys = ['rms', 'base', 'cpl', 'nl', 'sag', 'final', 'peak_cpl', 't_rec_cpl_ms', 'thd_nl', 'thd_final', 'proj_pct', 'active_pct']
    hdr = 'Controller & RMS & Base & CPL & NL & Sag & Final & Peak & $t_{\\mathrm{rec}}$ & THD$_{\\mathrm{NL}}$ & THD$_{\\mathrm{fin}}$ & Clip & Active\\\\ & V & V & V & V & V & V & V & ms & \\% & \\% & \\% & \\%\\\\\\midrule'
    rows = [hdr]
    for c in ['proposed', 'propfb', 'propnoim', 'nominal', 'faspd', 'ladrc', 'pr', 'fcs20', 'fcs40']:
        if c not in M:
            continue
        m = M[c]
        cells = []
        for k in keys:
            v, sd = m[k]
            if k == 'rms' and exp == 'main_sw':
                cells.append(f'{v:.2f}$\\pm${sd:.2f}')
            elif k == 't_rec_cpl_ms' and v > 25:
                cells.append('n.r.')
            elif k in ('proj_pct', 'active_pct') and c.startswith('fcs'):
                cells.append('--')
            else:
                cells.append(f'{v:.2f}')
        rows.append(NAME[c] + ' & ' + ' & '.join(cells) + '\\\\')
    tab(cap, 'tab:s_' + exp, 'l' + 'c' * len(keys), rows, '\\scriptsize\\setlength{\\tabcolsep}{1.4pt}')

# S-II sensitivity
sens = S['sens']
groups = [('N', '$N$'), ('alpha', '$\\alpha$'), ('beta_h', '$\\beta_h$'), ('sigma_a', '$\\sigma_a$ (V/s$^2$)'),
          ('r_eta', '$r_\\eta$'), ('qv', '$q_v$'), ('kappa_f', '$\\kappa_f$'), ('gamma_u', '$\\gamma_u$'), ('M', '$M$')]
rows = ['Parameter & Tested values $\\to$ RMS error (V)\\\\\\midrule']
for p, lab in groups:
    items = sorted([(float(k.split('=')[1]), v) for k, v in sens.items() if k.split('=')[0] == p])
    def fmt(x):
        return f'{x:.3g}' if (abs(x) >= 1e-2 or x == 0) else f'{x:.0e}'
    rows.append(lab + ' & ' + '; '.join(f'{fmt(x)}$\\to${v:.2f}' for x, v in items) + '\\\\')
tab('One-at-a-time sensitivity of the proposed controller (switching level, three realizations; $\\beta_h=0$ means $\\mathcal H=\\emptyset$)',
    'tab:s_sens', 'lp{0.78\\linewidth}', rows)

# S-III robustness to noise, dead time, CPL power, sag depth
rob = S['rob']
rows = ['Test & Value & Proposed & MA-LADRC & PR+AD & FCS-MPC (40 kHz)\\\\\\midrule']
for p, lab, unit, scale in (('sigma_v', '$\\sigma_v$', 'V', 1), ('td', '$t_d$', '$\\mu$s', 1e6), ('P_cpl', '$P_{\\mathrm{CPL}}$', 'kW', 1e-3)):
    vals = sorted({float(k.split('|')[0].split('=')[1]) for k in rob if k.startswith(p + '=')})
    for v in vals:
        cells = [f"{rob[f'{p}={k}|{c}']:.2f}" for c in ('proposed', 'ladrc', 'pr', 'fcs40')
                 for k in [next(kk.split('|')[0].split('=')[1] for kk in rob if kk.startswith(p + '=') and float(kk.split('|')[0].split('=')[1]) == v)]]
        rows.append(f'{lab} & {v * scale:g} {unit} & ' + ' & '.join(cells) + '\\\\')
for vs in ('340', '320', '300'):
    m = S[f'sag_{vs}']
    rows.append(f'Sag depth & {vs} V & ' + ' & '.join(f"{m[c]['rms'][0]:.2f}" for c in ('proposed', 'ladrc', 'pr', 'fcs40')) + '\\\\')
tab('Robustness tests (RMS error in V; noise, dead time, and CPL: two realizations; sag: three realizations). FAS-PD under the sag: '
    + ', '.join(f"{S[f'sag_{v}']['faspd']['rms'][0]:.2f}" for v in ('340', '320', '300')) + ' V.',
    'tab:s_rob', 'llcccc', rows)

# S-IV L/C map
lc = S['lc_map']
L = (0.6, 0.7, 0.85, 1.0, 1.15, 1.3, 1.4)
C = (0.8, 0.9, 1.0, 1.1, 1.2)
for c in ('proposed', 'faspd', 'ladrc', 'pr', 'fcs40'):
    rows = ['$L_f/L_0$ $\\backslash$ $C_f/C_0$ & ' + ' & '.join(f'{x:g}' for x in C) + '\\\\\\midrule']
    for l in L:
        cells = []
        for cc in C:
            v = lc[c].get(f'{l},{cc}', float('nan'))
            cells.append(f'\\textbf{{{v:.1f}}}' if v > 10 else f'{v:.1f}')
        rows.append(f'{l:g} & ' + ' & '.join(cells) + '\\\\')
    tab(f'RMS error (V) over the parameter grid: {NAME[c]} (bold: unbounded/unstable operation, RMS $>10$~V)', 'tab:s_lc_' + c,
        'cccccc', rows, '\\scriptsize')

# S-V stability test, small-gain test, QP check, timing, long runs
rows = ['Quantity & Value\\\\\\midrule',
        f"Dense grid: points per case, all Schur stable & {B['delay1']['n_points']}, {B['delay1']['all_stable'] and B['delay0']['all_stable']}\\\\",
        f"Largest sampled $\\rho(A_{{\\mathrm{{cl}}}})$ with / without delay compensation & {B['delay1']['max_rho']:.5f} / {B['delay0']['max_rho']:.5f}\\\\",
        f"Input-gain range preserving stability (default plant) & [0.01, {A['gain_margin']['upper'] - 0.05:.2f}]\\\\",
        f"Extra delay causing instability & {A['extra_delay_unstable_at']} samples\\\\"]
for h in ('0', '1', '-5', '7'):
    rows.append(f"Gain to $y$ at $h={h}$: matched sequence / full $2\\times2$ & {A['gain_dist_to_y'][h]:.1e} / {A['gain_full_to_y'][h]:.1e}\\\\")
g0 = [g for g in G['gains'] if g['Lr'] == 0.9 and g['Cr'] == 1.1][0]
rows.append(f"$\\ell_2$ gains $\\gamma_y$, $\\gamma_v$ (default plant) & {g0['gamma_y']:.2e}, {g0['gamma_v']:.2e}\\\\")
kc = G['kappa_check,dy=0.2']
rows.append(f"$V_{{\\min}}^3\\sup_{{\\|v\\|=1}}\\|\\partial_y(J_gv)\\|$ for $\\delta_y=0.2V_{{\\mathrm{{pk}}}}$: closed form / sampled & "
            f"{kc['closed']:.6f} / {kc['sampled']:.6f}\\\\")
for w, lab in (('cpl_steady', 'steady CPL, 25--50 ms'), ('cpl_step', 'CPL step, 20--25 ms')):
    a, d = TC[f'max_{w}'], TC[f'max_{w}_default']
    rows.append(f"Sampled $E_y$ / $E_v$, {lab}: all five plants (default plant) & "
                f"{a['Ey_samp']:.3f} / {a['Ev_samp']:.2f} ({d['Ey_samp']:.3f} / {d['Ev_samp']:.2f})\\\\")
for key in ('P=600,dy=0.1,dv=2.0', 'P=1200,dy=0.1,dv=2.0', 'P=1800,dy=0.1,dv=2.0', 'P=2400,dy=0.1,dv=2.0',
            'P=1200,dy=0.2,dv=2.0'):
    P_, dy, dv = key.split(',')
    rows.append(f"Small-gain index, {P_.split('=')[1]} W, $(\\delta_y,\\delta_v)=({dy.split('=')[1]}V_{{\\mathrm{{pk}}}},"
                f"{float(dv.split('=')[1]):.0f}\\omega V_{{\\mathrm{{pk}}}})$: default / worst plant & "
                f"{G[key]['default_index']:.2f} / {G[key]['worst_index']:.2f}\\\\")
n = A['nlp']
rows += [f"QP vs.\\ SOCP (8): samples (active polygon), failures & {n['n']} ({n['n_active']}), {n['n_failed']}\\\\",
         f"$\\|\\Delta u_0\\|$ mean / 95th pct.\\ / max & {n['du0_mean']:.2f} / {n['du0_p95']:.2f} / {n['du0_max']:.2f} V\\\\",
         f"Largest predicted QP command over the horizon / $\\bar U_k$ & {n['max_stage_ratio']:.7f}\\\\"]
for vs in ('340.0', '300.0'):
    t = T[vs]
    rows.append(f"QP time, {vs[:3]}-V sag (median / 99th / max, C level) & {t['qp_solve_med']:.0f} / {t['qp_solve_p99']:.0f} / {t['qp_solve_max']:.0f} $\\mu$s\\\\")
    rows.append(f"Samples with feasible unconstrained law, {vs[:3]}-V sag & {t['unc_ok']:.1f}\\%\\\\")
for r in LR:
    c = r['cycles']
    rows.append(f"1-s run, {NAME.get(r['name'], r['name'])}: RMS / THD of last cycle & {c[-1]['rms']:.2f} V / {c[-1]['thd']:.2f}\\%\\\\")
tab('Stability test, small-gain test, QP check, computation time, and long runs (CPU: Intel Core i9-14900, single process)',
    'tab:s_cert', 'lc', rows)

open('../supp/supp_tables.tex', 'w', encoding='utf-8').write('\n'.join(out))
print('written', len(out))
