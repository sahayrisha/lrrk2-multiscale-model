"""
Network-scale figures for network_v3.py.

Produces two figures:
  Fig5_v3_network_synergy.png    -- regime-invariance sweep + synergy bars
  Fig6_v3_network_wilcoxon.png   -- paired Wilcoxon forest plot 

Run directly:
    python3 generate_figures_network.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Uncomment if running from a subfolder structure where pd_model_v3.py 
# is not in the same directory as this script: 
# import sys, os 
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model')) 

import network_v3 as N

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False})


def fig5_network_synergy(out_path='Fig5_v3_network_synergy.png'):
    """Left: synergy bars at the calibrated transmission scale, on the
    incidence-rate endpoint (the network-scale analogue of
    Fig2_v3_synergy.png's left panel).
    Right: CI vs transmission-scale sweep across all three candidate
    endpoints (attack rate, cumulative hazard, incidence rate) -- shows
    that only incidence rate gives a stable synergy verdict across the
    full sweep; attack rate and hazard both drift because they saturate."""
    BS = N.calibrate_beta()
    A = {k: N.attack_rate(f, BS) for k, f in N.EXO_FLUX.items()}
    H = {k: N.cumulative_hazard(v) for k, v in A.items()}
    I = {k: N.incidence_rate(f, BS) for k, f in N.EXO_FLUX.items()}
    ci_inc = N.combo_index(I)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.5))

    bars = ['Rab35i\nalone', 'LAMP-2A\nalone', 'Bliss\nexpected', 'Combined\nobserved']
    vals = [1 - ci_inc['f_rab'], 1 - ci_inc['f_lamp'], 1 - ci_inc['expected'],
            1 - ci_inc['f_comb']]
    ax[0].bar(range(4), np.array(vals) * 100,
              color=['#DD8452', '#55A868', '#BBBBBB', '#8172B3'], edgecolor='k', lw=0.5)
    ax[0].axhline((1 - ci_inc['expected']) * 100, ls='--', c='k', lw=0.8)
    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels(bars, fontsize=8)
    ax[0].set_ylabel('reduction in incidence rate (%)')
    ax[0].set_title(f"Network synergy\nCI = {ci_inc['CI']:.3f}, excess = "
                    f"{ci_inc['excess_pp']:+.1f} pp", fontsize=9)

    print("Running regime-invariance sweep (this takes a minute)...")
    multipliers = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    ci_att, ci_haz, ci_inc_sweep = [], [], []
    for m in multipliers:
        bs = BS * m
        a = {k: N.attack_rate(f, bs, n_runs=80, seed0=5000) for k, f in N.EXO_FLUX.items()}
        h = {k: N.cumulative_hazard(v) for k, v in a.items()}
        i_ = {k: N.incidence_rate(f, bs, n_runs=80, seed0=5000) for k, f in N.EXO_FLUX.items()}
        ci_att.append(N.combo_index(a)['CI'])
        ci_haz.append(N.combo_index(h)['CI'])
        ci_inc_sweep.append(N.combo_index(i_)['CI'])

    bs_values = [BS * m for m in multipliers]
    ax[1].semilogx(bs_values, ci_att, 'o-', c='#C44E52', label='attack rate (bounded)')
    ax[1].semilogx(bs_values, ci_haz, '^-', c='#DD8452', label='cumulative hazard')
    ax[1].semilogx(bs_values, ci_inc_sweep, 's-', c='#4C72B0', label='incidence rate (adopted)')
    ax[1].axhline(0.9, ls=':', c='k', lw=0.8)
    ax[1].axhspan(0.9, 1.05, color='#DDDDDD', alpha=0.5)
    ax[1].axvline(BS, ls='--', c='gray', lw=0.8)
    ax[1].text(BS * 1.1, ax[1].get_ylim()[1] * 0.95, 'calibrated', fontsize=7, color='gray')
    ax[1].set_xlabel('transmission scale (beta_scale)')
    ax[1].set_ylabel('Combination Index')
    ax[1].set_title('Only incidence rate is stable\nacross transmission regimes', fontsize=9)
    ax[1].legend(fontsize=7, loc='lower left')

    fig.suptitle('Figure 5. Network-level synergy and endpoint regime-invariance',
                 fontsize=10, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path


def fig6_wilcoxon_forest(out_path='Fig6_v3_network_wilcoxon.png'):
    """Forest plot of the paired Wilcoxon test (common random numbers)
    comparing each intervention arm's incidence rate against control.
    Points are the bootstrapped mean difference, error bars the 95%
    bootstrap CI, with Holm-corrected p-values annotated. This test was
    previously computed in network_v3.py's __main__ (control vs each arm)
    but only printed as text -- plotted here for the first time."""
    BS = N.calibrate_beta()
    I = {k: N.incidence_rate(f, BS) for k, f in N.EXO_FLUX.items()}

    arms = ['wildtype', 'rab35i', 'lamp2a', 'combined']
    arm_labels = {'wildtype': 'WT', 'rab35i': 'Rab35i', 'lamp2a': 'LAMP-2A',
                  'combined': 'Combined'}

    raw_p = [stats.wilcoxon(I['control'], I[a], alternative='two-sided').pvalue
             for a in arms]
    adj_p = N.holm(np.asarray(raw_p))

    diffs, los, his = [], [], []
    for a in arms:
        d, lo, hi = N.bootstrap_ci(I['control'], I[a])
        diffs.append(d)
        los.append(d - lo)
        his.append(hi - d)

    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = np.arange(len(arms))
    ax.errorbar(diffs, y_pos, xerr=[los, his], fmt='o', color='#4C72B0',
               ecolor='#4C72B0', capsize=4, markersize=7)
    ax.axvline(0, ls='--', c='gray', lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'control vs {arm_labels[a]}' for a in arms])
    ax.invert_yaxis()
    ax.set_xlabel('incidence rate difference (arm - control), 95% bootstrap CI')
    ax.set_title('Figure 6. Paired Wilcoxon test on incidence rate\n'
                 '(common random numbers, Holm-corrected)', fontsize=10)

    for i, (d, p) in enumerate(zip(diffs, adj_p)):
        x_text = max(his[i], abs(los[i])) * 1.15 + d
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(x_text, i, f'p={p:.2e} {sig}', va='center', fontsize=8)

    fig.text(0.5, -0.04,
             'P-values from paired Wilcoxon over 200 simulation runs using common random\n'
             'numbers (shared stochastic seeds); very small p-values reflect simulation\n'
             'consistency, not real-world statistical certainty at that magnitude.',
             ha='center', fontsize=6.5, style='italic', color='#555555')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path


if __name__ == '__main__':
    paths = [fig5_network_synergy(), fig6_wilcoxon_forest()]
    print("\nFigures written:")
    for p in paths:
        print(f"  -> {p}")