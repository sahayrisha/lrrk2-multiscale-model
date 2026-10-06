"""
profile_and_uq.py

Uncertainty quantification for pd_model_v3.py.

Does two things:

  PART 1  Profile likelihood for each of the 12 free parameters.
          For each parameter, it is held fixed at a grid of values while
          all 11 others are re-optimised. The resulting curve of best-
          achievable cost gives a confidence interval, and a flat curve
          identifies a parameter the data cannot constrain at all.

  PART 2  Ensemble uncertainty on predictions. Every parameter set that
          fits within tolerance is kept, and the spread of the combination
          index and of the validation checks across that ensemble is
          reported. This is the interval that belongs in a write-up.

Outputs, written to ./uq_output/ :
    profile_results.csv     one row per parameter: best fit, lower and
                            upper 95% bound, verdict
    profile_curves.png      the 12 profile curves with the threshold drawn
    ensemble_results.csv    one row per accepted parameter set
    ensemble_summary.txt    prediction intervals and validation pass rates

Run with:
    python profile_and_uq.py            full run, roughly 10 to 20 minutes
    python profile_and_uq.py --quick    coarse run, roughly 1 to 2 minutes

READ THIS BEFORE TRUSTING ANY NUMBER
------------------------------------
The script needs to know what "a good fit" means. That is the objective
function, and it lives in the CALIBRATION TARGETS block below. Right now it
falls back to a self-consistent placeholder built from the model's own
nominal output, which makes the pipeline runnable but the numbers
meaningless. Replace the placeholder with the real T1 to T6 targets before
reporting anything. The script prints a loud warning whenever it is running
on the placeholder.
"""

import os
import sys
import time
import numpy as np
from scipy.optimize import minimize

# Uncomment if running from a subfolder structure where pd_model_v3.py 
# is not in the same directory as this script: 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model')) 

import pd_model_v3 as M

QUICK = '--quick' in sys.argv
OUT_DIR = 'uq_output'

# Grid resolution and ensemble size. Lower in quick mode.
N_GRID = 7 if QUICK else 13
N_ENSEMBLE = 150 if QUICK else 600
MAXITER = 150 if QUICK else 400

# Chi-square threshold for a 95% interval on one parameter at a time.
DELTA_CHI2 = 3.84

# How far either side of the best fit each profile scans, in log10 units.
# 1.0 means one order of magnitude down and one up.
PROFILE_SPAN_DEX = 1.0

FREE = list(M.P_FREE_NOMINAL.keys())

# Fixed parameters that carry real literature uncertainty and should be
# sampled in the ensemble rather than treated as exact. Relative sigma.
FIXED_PRIORS = {
    'Km_clear': 0.30,
    'Km_CMA': 0.30,
}


# ===========================================================================
# CALIBRATION TARGETS  <-- THIS IS THE BLOCK YOU EDIT
# ===========================================================================
# Each entry is one checkpoint: a name, a function that pulls the predicted
# value out of the model result dict, the measured target value, and the
# assumed relative standard deviation of that measurement.
#
# Example of a real entry, once you have the T1 to T6 numbers to hand:
#
#     ('T1  GS/WT secretion fold change',
#      lambda r: r['control']['secretion_flux'] / r['wildtype']['secretion_flux'],
#      2.10,       # measured value
#      0.25),      # 25 percent relative uncertainty on that measurement
#
# Add one tuple per checkpoint. Delete the placeholder block below once you
# have at least as many checkpoints as you have free parameters.

TARGETS = [   # <-- put your real T1 to T6 entries here
    ('T1  GS/WT secretion fold change',
     lambda r: r['control']['secretion_flux'] / r['wildtype']['secretion_flux'],
     3.00,
     0.33),

    ('T2  GS/WT pRab35 fold change',
     lambda r: r['control']['pRab35'] / r['wildtype']['pRab35'],
     2.25,
     0.33),

    ('T3  WT aggregate fraction',
     lambda r: r['wildtype']['aggregate_fraction'],
     0.15,
     0.33),

    ('T4  GS/WT total alpha-syn fold change',
     lambda r: r['control']['total_aSyn'] / r['wildtype']['total_aSyn'],
     1.85,
     0.35),
]          

USING_PLACEHOLDER = len(TARGETS) == 0

if USING_PLACEHOLDER:
    _ref = M.run_all(M.full_params())

    def _mk(name, fn, sigma=0.15):
        return (name, fn, fn(_ref), sigma)

    TARGETS = [
    ('T1  GS/WT secretion fold change',
     lambda r: r['control']['secretion_flux'] / r['wildtype']['secretion_flux'],
     2.10,
     0.25),

    ('T2  rab35i / control flux',
     lambda r: r['rab35i']['secretion_flux'] / r['control']['secretion_flux'],
     0.82,
     0.20),

    # ... one entry per checkpoint, through T6
]



def cost(p):
    """Sum of squared standardised residuals. Lower is better. Returns a
    large finite number if the model fails to solve, so the optimiser can
    walk away from bad regions instead of crashing."""
    try:
        r = M.run_all(p, fast=True)
    except Exception:
        return 1e6
    if r is None:
        return 1e6
    total = 0.0
    for _name, fn, target, sigma in TARGETS:
        try:
            pred = fn(r)
        except Exception:
            return 1e6
        if not np.isfinite(pred):
            return 1e6
        total += ((pred - target) / (sigma * abs(target))) ** 2

    # --- T5/T6 soft constraints, matching calibrate_v3.py's original cost
    # function. These are NOT point-value measurements (no single reported
    # number + error bar), so they don't belong in TARGETS -- they are
    # plausibility/inequality penalties that shape the fit the same way the
    # original calibration did. Without these, the optimizer here is free
    # to wander into regions the original calibration deliberately avoided,
    # producing a "best fit" inconsistent with the reported CI=0.80 result.
    try: 
        ci_dict = M.combination_index(r)
        comb_wt = r['combined']['secretion_flux'] / r['wildtype']['secretion_flux']

        # T5: combined-treatment flux should not collapse below 30% of WT
        if comb_wt < 0.30:
            total += 8.0 * (0.30 - comb_wt) ** 2

        # T6: neither single agent should be a near-total knockout (>75%
        # potency) nor negligibly weak (<25% potency) on its own
        for f in (ci_dict['f_rab35i'], ci_dict['f_lamp2a']):
            if not (0.35 <= f <= 0.75):
                total += 6.0 * min((f - 0.35) ** 2, (f - 0.85) ** 2) + 0.8

        # Symmetry term: discourages one arm's effect from dominating the
        # other by an implausible margin. Included for exact consistency
        # with the original calibration's optimization landscape.
        total += 0.30 * np.log(ci_dict['f_rab35i'] / ci_dict['f_lamp2a']) ** 2
    except Exception:
        return 1e6
    return total



# ===========================================================================
# Optimiser plumbing. Parameters are handled in log space because they are
# all strictly positive and span several orders of magnitude.
# ===========================================================================

def vec_to_params(x, names, frozen):
    p = dict(frozen)
    for name, val in zip(names, x):
        p[name] = float(np.exp(val))
    return p


def fit(frozen, free_names, x0, maxiter=MAXITER):
    """Minimise cost over free_names, holding everything in frozen fixed."""
    def f(x):
        return cost(vec_to_params(x, free_names, frozen))
    res = minimize(f, x0, method='Nelder-Mead',
                   options=dict(maxiter=maxiter, xatol=1e-4, fatol=1e-8,
                                disp=False))
    return res.x, res.fun


def best_fit(start=None):
    if start is None:
        try:
            # Uncomment if running from a subfolder structure where params_v3.npy is not in the same directory as this script: 
            # PARAMS_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'params_v3.npy')
            # calibrated = np.load(PARAMS_PATH, allow_pickle=True).item()
            calibrated = np.load('params_v3.npy', allow_pickle=True).item() # If using subfolder structure with sys.path directory, comment out
            base = dict(M.P_FIXED)
            base.update(calibrated)
            print("  (seeding optimizer from params_v3.npy, not nominal values)")
        except FileNotFoundError:
            print("  WARNING: params_v3.npy not found -- seeding from nominal "
                  "values instead. Results will not match the reported "
                  "calibration.")
            base = M.full_params()
    else:
        base = start

    frozen = dict(M.P_FIXED)
    x0 = np.array([np.log(base[k]) for k in FREE])
    x, c = fit(frozen, FREE, x0)
    return vec_to_params(x, FREE, frozen), c
   


# ===========================================================================
# PART 1  Profile likelihood
# ===========================================================================

def profile_one(param, p_best, c_best):
    """Scan one parameter across a grid, refitting the other 11 at each
    point. Returns (grid_values, costs)."""
    others = [k for k in FREE if k != param]
    centre = np.log10(p_best[param])
    grid = np.logspace(centre - PROFILE_SPAN_DEX,
                       centre + PROFILE_SPAN_DEX, N_GRID)
    costs = []
    x_warm = np.array([np.log(p_best[k]) for k in others])
    for val in grid:
        frozen = dict(M.P_FIXED)
        frozen[param] = float(val)
        x_warm, c = fit(frozen, others, x_warm, maxiter=MAXITER)
        costs.append(c)
    return grid, np.array(costs)


def interval_from_profile(grid, costs, c_best):
    """Where the profile crosses c_best + DELTA_CHI2, by linear
    interpolation in log10(parameter)."""
    thresh = c_best + DELTA_CHI2
    lg = np.log10(grid)
    below = costs <= thresh
    if not below.any():
        return np.nan, np.nan, 'no point below threshold (refit failed)'
    lo_i, hi_i = np.argmax(below), len(below) - 1 - np.argmax(below[::-1])

    def cross(i, j):
        if costs[j] == costs[i]:
            return lg[j]
        return lg[i] + (thresh - costs[i]) * (lg[j] - lg[i]) / (costs[j] - costs[i])

    lo = 10 ** cross(lo_i - 1, lo_i) if lo_i > 0 else np.nan
    hi = 10 ** cross(hi_i + 1, hi_i) if hi_i < len(grid) - 1 else np.nan

    rise = costs.max() - c_best
    if rise < 0.5 * DELTA_CHI2:
        verdict = 'FLAT: not identifiable from this data'
    elif np.isnan(lo) and np.isnan(hi):
        verdict = 'OPEN both sides: widen PROFILE_SPAN_DEX'
    elif np.isnan(lo) or np.isnan(hi):
        verdict = 'OPEN one side: only bounded in one direction'
    else:
        verdict = 'identifiable'
    return lo, hi, verdict


# ===========================================================================
# PART 2  Ensemble on predictions
# ===========================================================================

def flux(genotype, p, rab35_block=1.0, kinase_scale=1.0, LAMP2A=0.20):
    q = dict(p)
    q['V_lrrk2'] = q['V_lrrk2'] * kinase_scale
    r = M.steady_state(dict(genotype=genotype, rab35_block=rab35_block,
                            LAMP2A=LAMP2A), q, fast=True)
    return r['secretion_flux'] if r else np.nan


def predictions(p):
    r = M.run_all(p, fast=True)
    if r is None:
        return None
    out = dict(CI=M.combination_index(r)['CI'],
               bliss_excess_pp=M.combination_index(r)['bliss_excess_pp'],
               control_total_aSyn=r['control']['total_aSyn'],
               control_agg_fraction=r['control']['aggregate_fraction'])
    g0 = flux('G2019S', p) / flux('WT', p)
    gk = flux('G2019S', p, kinase_scale=0.01) / flux('WT', p, kinase_scale=0.01)
    gr = flux('G2019S', p, rab35_block=0.02) / flux('WT', p, rab35_block=0.02)
    if not all(np.isfinite([g0, gk, gr])):
        return None

    def collapse(a, b):
        return 1.0 - abs(np.log(b)) / max(abs(np.log(a)), 1e-12)

    out['V1_pass'] = float(g0 > 1.0)
    out['V2_pass'] = float(collapse(g0, gk) > 0.8)
    out['V3_pass'] = float(collapse(g0, gr) > 0.8)
    return out


def run_ensemble(p_best, c_best, rng):
    accepted, rows = 0, []
    thresh = c_best + DELTA_CHI2
    for _ in range(N_ENSEMBLE):
        p = dict(p_best)
        for k in FREE:
            p[k] = p_best[k] * np.exp(rng.normal(0, 0.15))
        for k, sig in FIXED_PRIORS.items():
            p[k] = M.P_FIXED[k] * np.exp(rng.normal(0, sig))
        if cost(p) > thresh:
            continue
        pred = predictions(p)
        if pred is None:
            continue
        accepted += 1
        rows.append({**{k: p[k] for k in FREE},
                     **{k: p[k] for k in FIXED_PRIORS},
                     **pred})
    return rows, accepted


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    if USING_PLACEHOLDER:
        print('!' * 74)
        print('WARNING: running on PLACEHOLDER calibration targets.')
        print('Every number below is a plumbing test, not a result.')
        print('Edit the TARGETS block near the top before reporting anything.')
        print('!' * 74)
        print()

    print(f'Mode: {"QUICK" if QUICK else "FULL"}   '
          f'grid={N_GRID}  ensemble={N_ENSEMBLE}\n')
    print(f'Checkpoints: {len(TARGETS)}   free parameters: {len(FREE)}')
    if len(TARGETS) < len(FREE):
        print(f'  NOTE: fewer checkpoints than free parameters, so some '
              f'parameters cannot be identified even in principle.')
    print()

    print('Finding the best fit...')
    p_best, c_best = best_fit()
    print(f'  best cost = {c_best:.4f}\n')

    # ---- Part 1 --------------------------------------------------------
    print('PART 1  Profile likelihood')
    print(f'  {"parameter":<14}{"best":>11}{"lower 95%":>12}{"upper 95%":>12}'
          f'   verdict')
    profile_rows, curves = [], {}
    for i, param in enumerate(FREE, 1):
        grid, costs = profile_one(param, p_best, c_best)
        lo, hi, verdict = interval_from_profile(grid, costs, c_best)
        curves[param] = (grid, costs)
        profile_rows.append(dict(parameter=param, best=p_best[param],
                                 lower95=lo, upper95=hi, verdict=verdict))
        print(f'  {param:<14}{p_best[param]:>11.4g}{lo:>12.4g}{hi:>12.4g}'
              f'   {verdict}   [{i}/{len(FREE)}]')

    with open(os.path.join(OUT_DIR, 'profile_results.csv'), 'w') as f:
        f.write('parameter,best,lower95,upper95,verdict\n')
        for r in profile_rows:
            f.write(f'{r["parameter"]},{r["best"]:.6g},{r["lower95"]:.6g},'
                    f'{r["upper95"]:.6g},"{r["verdict"]}"\n')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 3, figsize=(13, 12))
        for ax, param in zip(axes.ravel(), FREE):
            g, c = curves[param]
            ax.plot(g, c, 'o-', ms=3)
            ax.axhline(c_best + DELTA_CHI2, color='r', ls='--', lw=1)
            ax.axvline(p_best[param], color='k', ls=':', lw=1)
            ax.set_xscale('log')
            ax.set_title(param, fontsize=9)
            ax.tick_params(labelsize=7)
        fig.suptitle('Profile likelihood. Red line is the 95% threshold. '
                     'A flat curve means the parameter is not identifiable.',
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, 'profile_curves.png'), dpi=140)
        plt.close(fig)
        print(f'\n  wrote {OUT_DIR}/profile_curves.png')
    except ImportError:
        print('\n  matplotlib not installed, skipping the plot')

    # ---- Part 2 --------------------------------------------------------
    print('\nPART 2  Ensemble on predictions')
    rng = np.random.default_rng(0)
    rows, accepted = run_ensemble(p_best, c_best, rng)
    print(f'  accepted {accepted} of {N_ENSEMBLE} sampled sets')

    if accepted < 20:
        print('  TOO FEW ACCEPTED to quote an interval. Loosen the sampling '
              'width (0.15 in run_ensemble) or check the objective.')
    else:
        keys = list(rows[0].keys())
        with open(os.path.join(OUT_DIR, 'ensemble_results.csv'), 'w') as f:
            f.write(','.join(keys) + '\n')
            for r in rows:
                f.write(','.join(f'{r[k]:.6g}' for k in keys) + '\n')

        lines = []
        lines.append(f'Accepted sets: {accepted} of {N_ENSEMBLE}')
        lines.append(f'Best cost: {c_best:.4f}   threshold: '
                     f'{c_best + DELTA_CHI2:.4f}')
        lines.append('')
        lines.append(f'{"prediction":<24}{"median":>12}{"5%":>12}{"95%":>12}')
        for k in ['CI', 'bliss_excess_pp', 'control_total_aSyn',
                  'control_agg_fraction']:
            v = np.array([r[k] for r in rows])
            lines.append(f'{k:<24}{np.median(v):>12.4f}'
                         f'{np.percentile(v, 5):>12.4f}'
                         f'{np.percentile(v, 95):>12.4f}')
        ci = np.array([r['CI'] for r in rows])
        lines.append('')
        lines.append(f'P(CI < 0.90), i.e. synergistic by this study\'s '
                     f'threshold: {np.mean(ci < 0.90)*100:.1f}%')
        lines.append('')
        lines.append('Validation pass rates across the ensemble:')
        for k, label in [('V1_pass', 'V1 genotype direction'),
                         ('V2_pass', 'V2 kinase-inhibition collapse'),
                         ('V3_pass', 'V3 RAB35-blockade collapse')]:
            rate = np.mean([r[k] for r in rows]) * 100
            note = ''
            if rate == 0.0:
                note = '   <- fails everywhere: structural, not parametric'
            if rate == 100.0:
                note = '   <- passes everywhere: robust'
            lines.append(f'  {label:<34}{rate:6.1f}%{note}')
        lines.append('')
        lines.append('Parameter spread across accepted sets:')
        lines.append(f'  {"parameter":<14}{"median":>12}{"5%":>12}{"95%":>12}')
        for k in FREE + list(FIXED_PRIORS):
            v = np.array([r[k] for r in rows])
            lines.append(f'  {k:<14}{np.median(v):>12.4g}'
                         f'{np.percentile(v, 5):>12.4g}'
                         f'{np.percentile(v, 95):>12.4g}')
        text = '\n'.join(lines)
        print('\n' + text)
        with open(os.path.join(OUT_DIR, 'ensemble_summary.txt'), 'w') as f:
            f.write(text + '\n')

    print(f'\nDone in {time.time() - t_start:.0f} s. '
          f'All output is in ./{OUT_DIR}/')
    if USING_PLACEHOLDER:
        print('\nREMINDER: placeholder targets were used. '
              'These numbers are not results.')


if __name__ == '__main__':
    main()
