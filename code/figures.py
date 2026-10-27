"""Figures for the manuscript -> ../paper/fig/*.pdf"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sim import VPK, W0, SQ3

OUT = '../paper/fig/'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.linewidth': 0.6, 'grid.linewidth': 0.4, 'grid.color': '#d9d8d4', 'lines.linewidth': 1.0,
                     'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'pdf.fonttype': 42})
# categorical color order
COL = dict(proposed='#2a78d6', ladrc='#eb6834', faspd='#1baf7a', pr='#eda100', fcs40='#e87ba4', propnoim='#6250d6')
LS = dict(proposed='-', ladrc='--', faspd='-.', pr=':', fcs40=(0, (5, 1, 1, 1)), propnoim=(0, (3, 1)))
LAB = dict(proposed='Proposed', ladrc='MA-LADRC', faspd='FAS-PD', pr='PR+AD', fcs40='FCS-MPC (40 kHz)',
           propnoim='Proposed w/o harmonic IM')
EV = [(20, 'CPL'), (50, 'NL'), (80, 'Sag'), (100, 'Rec.')]


def events(ax, label=False, ymax=1.0):
    """Event markers; ymax (axes fraction) keeps the lines clear of in-axes legends and labels."""
    for t, s in EV:
        ax.axvline(t, ymax=ymax, color='#9b9a95', lw=0.5, ls=':', zorder=0)
        if label:
            ax.text(t + 0.8, 1.02, s, transform=ax.get_xaxis_transform(), fontsize=6.5, color='#52514e', va='bottom')


def mov_rms(e, n):
    c = np.cumsum(np.insert(e ** 2, 0, 0))
    r = np.sqrt((c[n:] - c[:-n]) / n)
    return np.concatenate([np.full(n - 1, np.nan), r])


def fig_time():
    W = np.load('../results/waveforms.npz')
    figv, axv = plt.subplots(1, 1, figsize=(3.45, 1.2))
    t = W['proposed_t'] * 1e3
    rec = W['proposed_rec']
    ax = axv
    ax.plot(t, VPK * np.cos(W0 * t / 1e3), color='#9b9a95', lw=0.8, ls='--', label=r'$v_{\mathrm{ref},\alpha}$')
    ax.plot(t, rec[:, 0], color=COL['proposed'], lw=0.7, label=r'$v_{c,\alpha}$ (proposed)')
    events(ax, label=True)
    ax.set_ylabel('Voltage (V)')
    ax.set_xlim(0, 120)
    ax.set_ylim(-175, 175)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.42), ncol=2, frameon=False, borderaxespad=0.0, handlelength=1.6)
    ax.grid(True, axis='y')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel('Time (ms)', labelpad=1)
    figv.savefig(OUT + 'figS_voltage.pdf', bbox_inches='tight', pad_inches=0.01)
    fig, axs = plt.subplots(3, 1, figsize=(3.45, 2.8), gridspec_kw=dict(height_ratios=[1.15, 0.8, 0.95], hspace=0.62))
    ax = axs[0]
    for c in ['proposed', 'faspd', 'ladrc', 'pr', 'fcs40']:
        tt = W[f'{c}_t'] * 1e3
        r = W[f'{c}_rec']
        e = np.linalg.norm(r[:, :2] - np.stack([VPK * np.cos(W0 * tt / 1e3), VPK * np.sin(W0 * tt / 1e3)], 1), axis=1)
        n = int(round(1e-3 / ((tt[1] - tt[0]) * 1e-3)))
        top = c == 'proposed'  # proposed drawn on the top layer, above all baselines
        ax.plot(tt, mov_rms(e, n), color=COL[c], ls=LS[c], lw=1.2 if top else 0.9, label=LAB[c],
                zorder=10 if top else 2)
    events(ax, label=True, ymax=0.62)
    ax.set_ylabel(r'$\|e_v\|$ RMS (V)')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 25)  # headroom so the legend sits above all curves
    ax.set_yticks([0, 10, 20])
    ax.legend(loc='upper left', ncol=3, frameon=False, borderaxespad=0.1, columnspacing=0.8, handlelength=2.0, fontsize=6.3)
    ax.text(0.0, 1.03, '(a)', transform=ax.transAxes, va='bottom', ha='left', fontsize=7.5)
    ax = axs[1]
    u, vdc = W['proposed_u'], W['proposed_vdc']
    tk = np.arange(len(u)) * float(W['proposed_Ts']) * 1e3
    vdc_meas = vdc * (1 + 0.01 * np.sin(2 * np.pi * 300 * tk / 1e3))
    vdc_used = np.roll(vdc_meas, 1)  # dc-link sample used when the command was computed (one-step delay)
    vdc_used[0] = vdc_meas[0]
    m = np.linalg.norm(u, axis=1) / (vdc_used / SQ3)
    nw = int(round(1e-3 / float(W['proposed_Ts'])))
    mmax = np.array([m[max(0, i - nw + 1):i + 1].max() for i in range(len(m))])
    ax.plot(tk, m, color='#9fc2ee', lw=0.4, label='per sample')
    ax.plot(tk, mmax, color=COL['proposed'], lw=0.9, label='1-ms max')
    ax.axhline(1.0, color='#52514e', lw=0.6)
    ax.text(119, 1.03, 'linear-modulation limit', fontsize=6.2, color='#52514e', va='bottom', ha='right')
    events(ax, ymax=(1.0 - 0.45) / (1.19 - 0.45))  # stop at the limit line, below the label band
    ax.set_ylabel(r'$\|u_{\mathrm{cmd}}\|/(V_{\mathrm{dc}}/\sqrt{3})$', fontsize=7)
    ax.set_xlabel('Time (ms)', labelpad=1)
    ax.set_xlim(0, 120)
    ax.set_ylim(0.45, 1.19)  # band above the limit holds the legend and the limit label
    ax.set_yticks([0.5, 0.75, 1.0])
    ax.legend(loc='upper left', ncol=2, frameon=False, borderaxespad=0.1, fontsize=6.3, handlelength=1.6)
    ax.text(0.0, 1.03, '(b)', transform=ax.transAxes, va='bottom', ha='left', fontsize=7.5)
    ax = axs[2]
    for c in ['propnoim', 'proposed']:
        tt = W[f'{c}_t'] * 1e3
        r = W[f'{c}_rec']
        m = (tt >= 60) & (tt < 80)
        ea = r[m, 0] - VPK * np.cos(W0 * tt[m] / 1e3)
        ax.plot(tt[m], ea, color=COL[c], ls=LS[c] if c == 'proposed' else '-', lw=0.9 if c == 'proposed' else 0.7,
                label=LAB[c], alpha=1.0 if c == 'proposed' else 0.85, zorder=10 if c == 'proposed' else 2)
    ax.set_xlim(60, 80)
    ax.set_ylabel(r'$e_{v,\alpha}$ (V)')
    ax.set_xlabel('Time (ms), rectifier load', labelpad=1)
    ax.legend(loc='upper center', ncol=2, frameon=False, borderaxespad=0.1, fontsize=6.3)
    ax.set_ylim(-25, 40)  # headroom so the legend sits above both curves
    ax.set_yticks([-20, 0, 20])
    ax.text(0.0, 1.03, '(c)', transform=ax.transAxes, va='bottom', ha='left', fontsize=7.5)
    for ax in axs:
        ax.grid(True, axis='y')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.savefig(OUT + 'fig_switching.pdf', bbox_inches='tight', pad_inches=0.01)
    fig.savefig(OUT + 'fig_switching.png', dpi=220, bbox_inches='tight', pad_inches=0.01)


def fig_sens():
    S = json.load(open('../results/summary.json'))
    sens = S['sens']
    design = S['main_sw']['proposed']['rms'][0]
    groups = [('N', r'$N$', [3, 5, 10, 20]), ('alpha', r'$\alpha$', [0.05, 0.1, 0.2, 0.35]),
              ('beta_h', r'$\beta_h$', [0.0, 0.2, 0.5, 1.0]), ('sigma_a', r'$\sigma_a$', None),
              ('r_eta', r'$r_\eta$', None), ('qv', r'$q_v$', [0.01, 0.05, 0.2, 1.0]), ('kappa_f', r'$\kappa_f$', [0.0, 1.0, 5.0, 20.0]),
              ('gamma_u', r'$\gamma_u$', [0.9, 0.95, 1.0]), ('M', r'$M$', [8, 12, 24, 48])]
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.52), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.6))
    ax = axs[0]
    for j, (p, lab, _) in enumerate(groups):
        vals = [(k.split('=')[1], v) for k, v in sens.items() if k.split('=')[0] == p]
        rms = [v for _, v in vals]
        lo, hi = min(rms), max(rms)
        y = len(groups) - 1 - j
        ax.plot([lo, hi], [y, y], color=COL['proposed'], lw=3.2, solid_capstyle='round')
        ax.plot([design], [y], marker='|', color='#0b0b0b', ms=7, mew=1.0)
        ax.text(hi + 0.15, y, f'{hi:.2f}', va='center', fontsize=6.8, color='#52514e')
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[1] for g in groups][::-1])
    ax.set_xlabel('RMS of $\\|e_v\\|$ (V)', labelpad=1)
    ax.set_xlim(3.0, 9.6)
    ax.set_xticks([4, 6, 8])
    ax.tick_params(labelsize=7.5)
    ax.grid(True, axis='x')
    ax.text(0.02, 1.02, '(a) one-at-a-time ranges', transform=ax.transAxes, fontsize=7.5)
    ax = axs[1]
    order = ['proposed', 'faspd', 'ladrc', 'pr', 'fcs40']
    short = dict(proposed='Prop.', faspd='FAS-PD', ladrc='LADRC', pr='PR+AD', fcs40='FCS40')
    nom = [S['main_sw'][c]['rms'][0] for c in order]
    worst = [S['lc_box_worst'][c] for c in order]
    x = np.arange(len(order))
    ax.bar(x - 0.19, nom, 0.36, color=COL['proposed'], label='default plant', edgecolor='white', linewidth=0.8)
    ax.bar(x + 0.19, worst, 0.36, color=COL['ladrc'], label='worst over box', edgecolor='white', linewidth=0.8,
           hatch='////')
    ax.set_xticks(x)
    ax.set_xticklabels([short[c] for c in order], rotation=40, ha='right')
    ax.set_ylabel('RMS of $\\|e_v\\|$ (V)', labelpad=1)
    ax.set_ylim(0, 10.5)  # headroom so the legend sits above the tallest bar
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.tick_params(labelsize=7.5)
    ax.grid(True, axis='y')
    ax.legend(loc='upper left', frameon=False, fontsize=6.6, borderaxespad=0.1, handlelength=1.2)
    ax.text(0.02, 1.02, '(b) $L/L_0\\pm30\\%$, $C/C_0\\pm10\\%$', transform=ax.transAxes, fontsize=7.5)
    for a in axs:
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    fig.savefig(OUT + 'fig_sensitivity.pdf', bbox_inches='tight', pad_inches=0.01)
    fig.savefig(OUT + 'fig_sensitivity.png', dpi=220, bbox_inches='tight', pad_inches=0.01)


def fig_longrun():
    """Supplementary figure: per-cycle voltage-error RMS and THD over 1-s runs."""
    L = json.load(open('../results/longrun.json'))
    fig, axs = plt.subplots(2, 1, figsize=(3.45, 2.3), sharex=True, gridspec_kw=dict(hspace=0.25))
    names = dict(proposed='Proposed', propnoim='Proposed w/o harmonic IM', ladrc='MA-LADRC')
    for r in L:
        c = r['name']
        t = [x['t_end'] * 1e3 for x in r['cycles']]
        kw = dict(color=COL[c], ls=LS[c], lw=1.2 if c == 'proposed' else 0.9, zorder=10 if c == 'proposed' else 2,
                  label=names[c])
        axs[0].plot(t, [x['rms'] for x in r['cycles']], **kw)
        axs[1].plot(t, [x['thd'] for x in r['cycles']], **kw)
    axs[0].set_ylabel('RMS (V)', fontsize=7.5)
    axs[1].set_ylabel('THD (%)', fontsize=7.5)
    axs[0].text(0.0, 1.03, r'(a) per-cycle RMS of $\|e_v\|$', transform=axs[0].transAxes, fontsize=7.5)
    axs[1].text(0.0, 1.03, r'(b) per-cycle THD of $v_{c,\alpha}$', transform=axs[1].transAxes, fontsize=7.5)
    axs[1].set_xlabel('Time (ms)', labelpad=1)
    axs[0].set_ylim(0, 10)
    axs[1].set_ylim(0, 7)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False, fontsize=6.3,
               handlelength=1.8, columnspacing=0.9)
    for a in axs:
        a.grid(True, axis='y')
        a.set_xlim(0, 1000)
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
    fig.savefig(OUT + 'figS_longrun.pdf', bbox_inches='tight', pad_inches=0.01)
    fig.savefig(OUT + 'figS_longrun.png', dpi=220, bbox_inches='tight', pad_inches=0.01)


if __name__ == '__main__':
    import os
    os.makedirs(OUT, exist_ok=True)
    fig_time()
    fig_sens()
    fig_longrun()
