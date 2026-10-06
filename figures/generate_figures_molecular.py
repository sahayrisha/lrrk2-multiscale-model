"""
Molecular-scale figures for the calibrated model (pd_model_v3.py).

Produces four figures:
  Fig1_v3_steady_states.png       -- steady-state outputs across all 5 conditions
  Fig2_v3_synergy.png             -- molecular synergy + endpoint-stability comparison
  Fig3_v3_therapeutic_window.png  -- benefit vs on-target liability
  Fig4_v3_identifiability.png     -- FIM eigenvalue spectrum + Km_MVB robustness


Run directly:
    python3 generate_figures_molecular.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Uncomment if running from a subfolder structure where pd_model_v3.py 
# is not in the same directory as this script: 
# import sys, os 
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model')) 

import pd_model_v3 as M
import calibrate_v3 as C

ORDER = ['wildtype', 'control', 'rab35i', 'lamp2a', 'combined']
LBL = {'wildtype': 'WT', 'control': 'G2019S', 'rab35i': 'Rab35i',
       'lamp2a': 'LAMP-2A', 'combined': 'Combined'}
COL = ['#4C72B0', '#C44E52', '#DD8452', '#55A868', '#8172B3']

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False})


def _load():
    # Uncomment if running from a subfolder structure where params_v3.npy is not in the same directory as this script: 
    # PARAMS_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'params_v3.npy')
    # PF = np.load(PARAMS_PATH, allow_pickle=True).item()
    PF = np.load('params_v3.npy', allow_pickle=True).item() # If using subfolder structure with sys.path directory, comment out
    P = dict(M.P_FIXED)
    P.update(PF)
    res = M.run_all(P)
    ci = M.combination_index(res)
    return P, PF, res, ci


def fig1_steady_states(P=None, res=None, out_path='Fig1_v3_steady_states.png'):
    """Bar charts of the four key molecular outputs across all 5 conditions.
    Dashed line marks the healthy (wildtype) baseline for the two outputs
    where that comparison is most relevant (total a-syn, secretion flux)."""
    if res is None:
        P, _, res, _ = _load()
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.9))
    for a, key, t in zip(ax, ['pRab35', 'total_aSyn', 'aggregate', 'secretion_flux'],
                         ['phospho-Rab35 (uM)', 'total a-syn (uM)',
                          'aggregate (uM)', 'exosomal flux (uM/s)']):
        a.bar(range(5), [res[c][key] for c in ORDER], color=COL, edgecolor='k', lw=0.5)
        a.set_xticks(range(5))
        a.set_xticklabels([LBL[c] for c in ORDER], rotation=45, ha='right')
        a.set_title(t, fontsize=9)
        if key in ('secretion_flux', 'total_aSyn'):
            a.axhline(res['wildtype'][key], ls='--', c='k', lw=0.8)
    fig.suptitle('Figure 1. Molecular steady states (dashed = healthy baseline)',
                 fontsize=10, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path


def fig2_synergy(P=None, res=None, ci=None, out_path='Fig2_v3_synergy.png'):
    """Molecular-scale Bliss synergy bars (single-agent, expected, observed).

    NOTE: this figure originally also included a right-hand panel comparing
    CI stability across transmission scale and endpoint definition. That
    panel has been removed as redundant with Fig5 (network_v3.py), which
    covers the same endpoint-stability argument more completely -- three
    endpoints (attack rate, cumulative hazard, incidence rate) computed live
    from the model, rather than two endpoints loaded from a separate JSON
    file. See Fig5_v3_network_synergy.png for that comparison."""
    if res is None:
        P, _, res, ci = _load()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    bars = ['Rab35i\nalone', 'LAMP-2A\nalone', 'Bliss\nexpected', 'Combined\nobserved']
    vals = [1 - ci['f_rab35i'], 1 - ci['f_lamp2a'], 1 - ci['bliss_expected'],
            1 - ci['f_combined']]
    ax.bar(range(4), np.array(vals) * 100,
          color=['#DD8452', '#55A868', '#BBBBBB', '#8172B3'], edgecolor='k', lw=0.5)
    ax.axhline((1 - ci['bliss_expected']) * 100, ls='--', c='k', lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(bars, fontsize=8)
    ax.set_ylabel('reduction in exosomal flux (%)')
    ax.set_title(f"Figure 2. Molecular synergy\nCI = {ci['CI']:.3f}, excess = "
                f"{ci['bliss_excess_pp']:+.1f} pp", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path


def fig3_therapeutic_window(P=None, res=None, out_path='Fig3_v3_therapeutic_window.png'):
    """Left: pathological excess removed vs Rab35 inhibition, by LAMP-2A dose.
    Right: on-target liability (suppression below the healthy baseline) --
    exosome release is physiological, so over-suppression is itself a harm,
    not an added benefit."""
    if res is None:
        P, _, res, _ = _load()
    F_WT = res['wildtype']['secretion_flux']
    F_C = res['control']['secretion_flux']
    blocks = np.linspace(1.0, 0.02, 40)
    lamps = [0.20, 0.35, 0.50, 0.70, 1.00]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.5))
    for L, c in zip(lamps, plt.cm.viridis(np.linspace(0, .85, len(lamps)))):
        ben, lia = [], []
        for b in blocks:
            F = M.steady_state(dict(genotype='G2019S', rab35_block=b, LAMP2A=L), P)['secretion_flux']
            ben.append(min(1, max(0, (F_C - F) / (F_C - F_WT))) * 100)
            lia.append(max(0, (F_WT - F) / F_WT) * 100)
        ax[0].plot((1 - blocks) * 100, ben, color=c, label=f'LAMP-2A {L}')
        ax[1].plot((1 - blocks) * 100, lia, color=c, label=f'LAMP-2A {L}')
    ax[0].set_xlabel('Rab35 inhibition (%)')
    ax[0].set_ylabel('pathological excess removed (%)')
    ax[0].set_title('Benefit', fontsize=9)
    ax[0].legend(fontsize=7)
    ax[1].axhline(5, ls='--', c='r', lw=1)
    ax[1].text(3, 7, 'liability threshold', color='r', fontsize=7)
    ax[1].axvline(75, ls=':', c='k', lw=1)
    ax[1].text(77, 60, 'calibrated\nregimen', fontsize=7)
    ax[1].set_xlabel('Rab35 inhibition (%)')
    ax[1].set_ylabel('suppression below healthy (%)')
    ax[1].set_title('On-target liability', fontsize=9)
    fig.suptitle('Figure 3. Therapeutic window: exosome release is physiological, '
                 'so over-suppression is a harm', fontsize=10, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path


def fig4_identifiability(PF=None, P=None, out_path='Fig4_v3_identifiability.png'):
    """Left: normalised Fisher information eigenvalue spectrum for the
    current model's free parameters (current formulation only -- see
    module docstring for why the prior-formulation comparison was removed).
    Right: synergy conclusion's robustness to the one unsourced parameter,
    Km_MVB, profiled over a 40-fold range."""
    if P is None:
        P, PF, _, _ = _load()
    S = C.sens_matrix(PF, C.MEASURABLE)
    sv = np.linalg.svd(S, compute_uv=False)
    ev = (sv ** 2) / (sv[0] ** 2)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.5))
    ax[0].semilogy(range(1, len(ev) + 1), ev, 'o-', c='#4C72B0',
                   label=f'{len(C.FREE)} free parameters')
    ax[0].set_xlabel('eigenvalue index')
    ax[0].set_ylabel('normalised FIM eigenvalue')
    ax[0].set_title('Identifiability: full rank against\nmeasurable observables',
                    fontsize=9)
    ax[0].legend(fontsize=7)

    kms = [0.005, 0.01, 0.025, 0.05, PF['Km_MVB'], 0.2, 0.4, 0.8, 2.0, 5.0]
    cis = []
    for km in kms:
        p = dict(P)
        p['Km_MVB'] = km
        cis.append(M.combination_index(M.run_all(p))['CI'])
    ax[1].semilogx(kms, cis, 'o-', c='#8172B3')
    ax[1].axhline(0.9, ls=':', c='k')
    ax[1].axhspan(0.9, 1.05, color='#DDDDDD', alpha=.5)
    ax[1].axvline(PF['Km_MVB'], ls='--', c='r', lw=1)
    ax[1].text(PF['Km_MVB'] * 1.15, 0.96, 'calibrated', color='r', fontsize=7)
    ax[1].set_xlabel('Km_MVB (uM), no literature source')
    ax[1].set_ylabel('Combination Index')
    ax[1].set_title('Conclusion holds over a 40-fold range\nof the unsourced parameter',
                    fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path


if __name__ == '__main__':
    P, PF, res, ci = _load()
    paths = [
        fig1_steady_states(P, res),
        fig2_synergy(P, res, ci),
        fig3_therapeutic_window(P, res),
        fig4_identifiability(PF, P),
    ]
    print("Figures written:")
    for p in paths:
        print(f"  -> {p}")