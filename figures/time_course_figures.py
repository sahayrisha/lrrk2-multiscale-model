"""

Time-course figure generation for pd_model_v3.py.

Run directly to regenerate all three figures from the calibrated parameters
in params_v3.npy (falls back to nominal parameters with a warning if that
file is not present):
 
    python3 time_course_figures.py
 
Figures produced:
  1. pd_v3_secretion_flux_timecourse.png
     secretion_flux(t) for all 5 CONDITIONS, log-y axis.
 
  2. pd_v3_inhibitor_washout.png
     pRab35(t) decay under simulated near-total LRRK2 kinase inhibition,
     shown against Fan et al. 2018's reported Rab10 (1-2 min) and Ser1292
     (80-160 min) near-complete-dephosphorylation windows for reference.
 
  3. pd_v3_genotype_gap_timecourse.png
     G2019S/WT fold-gap in total_aSyn over time -- shows whether/how fast
     the genotype difference emerges (see T7 discussion in pd_model_v3.py).
"""

import os
import numpy as np

# Uncomment if running from a subfolder structure where pd_model_v3.py 
# is not in the same directory as this script: 
# import sys, os 
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model')) 

import pd_model_v3 as M

def generate_time_course_plots(p=None, out_dir='.'):
    """Generates three figures and saves them as PNG files:

      1. pd_v3_secretion_flux_timecourse.png
         secretion_flux(t) for all 5 CONDITIONS, on log-y (since flux spans
         orders of magnitude across conditions once genotype x LAMP2A x
         rab35_block are combined)

      2. pd_v3_inhibitor_washout.png
         pRab35(t) decay under simulated near-total LRRK2 kinase inhibition
         starting from the WT steady state -- the same setup used in
         validate_fan2018_timecourse.py, shown against Fan et al. 2018's
         reported Rab10 (1-2 min) and Ser1292 (80-160 min) windows for
         visual reference.

      3. pd_v3_genotype_gap_timecourse.png
         G2019S/WT fold-gap in total_aSyn over time, from
         genotype_gap_over_time() -- shows whether/how fast the genotype
         difference emerges.

    Returns the list of file paths written.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os

    p = p or full_params()
    written = []

    # --- Figure 1: secretion flux time-course, all 5 conditions ----------
    t_span = (0.0, 50000.0)
    tcs = M.run_all_time_courses(p, t_span=t_span, n_points=2000)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'wildtype': 'tab:blue', 'control': 'tab:orange',
              'rab35i': 'tab:green', 'lamp2a': 'tab:red',
              'combined': 'tab:purple'}
    for k, tc in tcs.items():
        ax.plot(tc['t'], tc['secretion_flux'], label=k, color=colors.get(k))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('secretion_flux (uM/s)')
    ax.set_yscale('log')
    ax.set_title('pd_model_v3: secretion flux time-course, all conditions')
    ax.legend()
    fig.tight_layout()
    path1 = os.path.join(out_dir, 'pd_v3_secretion_flux_timecourse.png')
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    written.append(path1)

    # --- Figure 2: inhibitor washout, pRab35 decay vs Fan et al. windows -
    cond_wt = dict(genotype='WT', rab35_block=1.0, LAMP2A=0.40)
    ss_wt = M.steady_state(cond_wt, p)
    y0_prestim = np.array([ss_wt['pRab35'], ss_wt['pRab10'], ss_wt['aSyn'],
                            ss_wt['aggregate'], ss_wt['MVB'], ss_wt['exosomal']])
    p_inhibited = dict(p)
    p_inhibited['V_lrrk2'] = p['V_lrrk2'] * 0.001
    t_span_inhib = (0.0, 3.0 * 60.0 * 60.0)
    tc_inhib = M.time_course(cond_wt, p=p_inhibited, t_span=t_span_inhib,
                            n_points=5000, y0=y0_prestim)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tc_inhib['t'] / 60.0, tc_inhib['pRab35'], color='tab:blue',
            label='model pRab35 (this study)')
    ax.axvspan(1.0, 2.0, color='tab:green', alpha=0.25,
               label='Fan et al. 2018: Rab10 near-complete window (1-2 min)')
    ax.axvspan(80.0, 160.0, color='tab:red', alpha=0.15,
               label='Fan et al. 2018: Ser1292 near-complete window (80-160 min)')
    ax.set_xlabel('time after simulated kinase inhibition (min)')
    ax.set_ylabel('pRab35 (uM)')
    ax.set_xscale('log')
    ax.set_title('pd_model_v3: LRRK2-inhibitor washout vs Fan et al. 2018 windows')
    ax.legend(fontsize=8)
    fig.tight_layout()
    path2 = os.path.join(out_dir, 'pd_v3_inhibitor_washout.png')
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    written.append(path2)

    # --- Figure 3: genotype gap over time ---------------------------------
    t, gap, tc_wt_g, tc_gs_g = M.genotype_gap_over_time(p, t_span=t_span,
                                                        n_points=2000)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, gap, color='tab:purple')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1,
               label='no genotype difference (gap = 1.0x)')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('G2019S / WT fold-gap in total_aSyn')
    ax.set_title('pd_model_v3: genotype gap emergence over time')
    ax.legend()
    fig.tight_layout()
    path3 = os.path.join(out_dir, 'pd_v3_genotype_gap_timecourse.png')
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    written.append(path3)

    return written


if __name__ == '__main__':
    if os.path.exists('params_v3.npy'):
        # Uncomment if running from a subfolder structure where params_v3.npy is not in the same directory as this script: 
        # PARAMS_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'params_v3.npy')
        # pf = np.load(PARAMS_PATH, allow_pickle=True).item()
        pf = np.load('params_v3.npy', allow_pickle=True).item()
        p = dict(M.P_FIXED)
        p.update(pf)
        print("Parameter set: CALIBRATED (params_v3.npy)")
    else:
        p = M.full_params()
        print("WARNING: params_v3.npy not found -- using NOMINAL parameters. "
              "Figures will not reflect the reported calibrated results.")
 
    try:
        out_paths = generate_time_course_plots(p, out_dir='.')
        print("\nFigures written:")
        for pth in out_paths:
            print(f"  -> {pth}")
    except ImportError:
        print("matplotlib not available -- install with:")
        print("  pip install matplotlib --break-system-packages")
 
