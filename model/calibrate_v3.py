"""
Calibrates the free parameter set and verifies it is structurally
identifiable from measurable observables.
 
The partition into FREE and FIXED
parameters (see pd_model_v3.py) is only legitimate if the FREE set is full
rank. This script checks that directly, rather than assuming it.
 
Validation targets:
  T1  exosomal secretion flux, control/wildtype    2.0 - 4.0   Emmanouilidou (2010)
  T2  pRab35 fold change, control/wildtype         1.5 - 3.0   knock-in Rab phospho data
  T3  aggregate fraction in wildtype               0.10 - 0.20 Fauvet (2012)
  T4  total alpha-syn, control/wildtype            1.2 - 2.5   Volpicelli-Daley (2016)
  T5  combined flux / wildtype flux                >= 0.30     baseline sanity
  T6  single-agent potency, both arms              25% - 65%   non-degeneracy
"""

import numpy as np
from scipy.optimize import differential_evolution
import pd_model_v3 as M

FREE = M.FREE_NAMES
BOUNDS = {
    'V_lrrk2':     (0.02, 0.12),
    'g2019s_kin':  (1.40, 2.20),    # constrained to cellular pRab literature
    'Km_lrrk2':    (0.30, 3.00),
    'k_dephos':    (0.03, 0.20),
    'k_prod_WT':   (0.001, 0.010),
    'g2019s_prod': (1.05, 1.60),    # [21] conservative literature band
    'Vmax_CMA':    (0.003, 0.15),
    'k_nuc':       (0.05, 2.00),
    'Vmax_clear':  (0.02, 0.60),
    'Km_MVB':      (0.02, 0.50),
    'k_load':      (0.01, 1.00),
    'Km_load':     (0.10, 2.00),
}
B = [BOUNDS[k] for k in FREE]
TGT = dict(flux=2.8, prab=2.0, agg=0.15, asyn=1.5)


def metrics(pf, fast=False):
    p = dict(M.P_FIXED); p.update(pf)
    res = M.run_all(p, fast=fast)
    if res is None:
        return None, None
    wt, ct = res['wildtype'], res['control']
    if wt['secretion_flux'] <= 0 or wt['total_aSyn'] <= 0 or wt['pRab35'] <= 0:
        return None, None
    return res, dict(
        flux=ct['secretion_flux'] / wt['secretion_flux'],
        prab=ct['pRab35'] / wt['pRab35'],
        agg=wt['aggregate_fraction'],
        asyn=ct['total_aSyn'] / wt['total_aSyn'],
        comb_wt=res['combined']['secretion_flux'] / wt['secretion_flux'])


def cost(x):
    try:
        res, m = metrics(dict(zip(FREE, x)), fast=True)
    except Exception:
        return 1e6
    if m is None:
        return 1e6
    c = 0.0
    c += 9.0 * np.log(m['flux'] / TGT['flux']) ** 2
    c += 3.0 * np.log(m['prab'] / TGT['prab']) ** 2
    c += 4.0 * np.log(m['agg'] / TGT['agg']) ** 2
    c += 3.0 * np.log(m['asyn'] / TGT['asyn']) ** 2
    if m['comb_wt'] < 0.30:
        c += 8.0 * (0.30 - m['comb_wt']) ** 2
    ci = M.combination_index(res)
    for f in (ci['f_rab35i'], ci['f_lamp2a']):
        if not (0.35 <= f <= 0.75):
            c += 6.0 * min((f - 0.35) ** 2, (f - 0.85) ** 2) + 0.8
    c += 0.30 * np.log(ci['f_rab35i'] / ci['f_lamp2a']) ** 2
    return c


MEASURABLE = ['secretion_flux', 'pRab35', 'total_aSyn']
ALL_OBS = ['secretion_flux', 'pRab35', 'aSyn', 'aggregate', 'MVB']


def sens_matrix(pf, obs, eps=1e-4):
    p0 = dict(M.P_FIXED); p0.update(pf)
    r0 = M.run_all(p0)
    base = np.array([np.log(max(r0[c][o], 1e-300)) for c in M.CONDITIONS for o in obs])
    S = np.zeros((len(base), len(FREE)))
    for j, k in enumerate(FREE):
        pp = dict(pf); pp[k] = pf[k] * (1 + eps)
        p = dict(M.P_FIXED); p.update(pp)
        r = M.run_all(p)
        v = np.array([np.log(max(r[c][o], 1e-300)) for c in M.CONDITIONS for o in obs])
        S[:, j] = (v - base) / np.log(1 + eps)
    return S


def rank_of(S):
    s = np.linalg.svd(S, compute_uv=False)
    return int(np.sum(s > max(S.shape) * np.finfo(float).eps * s[0])), s


# ---------------------------------------------------------------------------
# TIMESCALE CORRECTION (post-calibration, literature-anchored)
# ---------------------------------------------------------------------------
# T1-T6 above are all ratios/fold-changes between steady-state conditions --
# none of them constrain any absolute rate. Differential_evolution therefore
# picks k_dephos and V_lrrk2 to fit those ratios, but the resulting absolute
# pRab35 dephosphorylation TIMESCALE is an unconstrained side effect of the
# fit, not a validated prediction.
#
# Fan et al. 2018 (Biochem. J. 473:2671) constrains that timescale: reported
# near-complete Rab10 dephosphorylation after LRRK2 inhibition falls in
# 1-10 min (inhibitor-dependent: 1-2 min for GSK2578215A/HG-10-102-01,
# 5-10 min for MLi-2), well separated from the much slower Ser935 (40-80 min)
# and Ser1292 (80-160 min) autophosphorylation windows.
#
# Rescaling BOTH k_dephos and V_lrrk2 by the same scalar s leaves every
# steady state (and therefore T1-T6 and the combination index) exactly
# invariant -- s cancels out of the steady-state balance equation
#   Vmax * Rab / (Km + Rab) = k_dephos * pRab35
# and only affects the transient decay rate. This has been verified
# numerically (see verify_timescale_correction.py): every steady-state
# field, the full combination-index dict, and all five T1-T6 checkpoints
# match to floating-point precision before/after rescaling.
#
# TARGET_DECAY_TIME_S is the free choice in this correction: any value in
# 60-600 s is equally literature-supportable. 300 s (the midpoint) is used
# here absent a reason to target a specific inhibitor's kinetics. Change
# this constant -- not k_dephos/V_lrrk2 directly -- if a different target
# is preferred, and re-run verify_timescale_correction.py to re-confirm the
# zero-cost property against the new value.
TARGET_DECAY_TIME_S = 300.0   # 5 min; corrected Fan et al. Rab10 window is 60-600 s
NEAR_COMPLETE_FRACTION = 0.10  # matches validate_fan2018_timecourse.py's definition


def apply_timescale_correction(pf, target_decay_s=TARGET_DECAY_TIME_S,
                                frac_remaining=NEAR_COMPLETE_FRACTION):
    """Rescale k_dephos and V_lrrk2 by a literature-anchored scalar s so that
    the model's near-total-inhibition pRab35 decay reaches frac_remaining of
    its initial value at target_decay_s. Returns a NEW dict; does not mutate
    pf. All other free parameters are left untouched -- Fan et al.'s data
    speaks only to phosphorylation/dephosphorylation kinetics, not
    nucleation, clearance, or export."""
    k_dephos_old = pf['k_dephos']
    t_old = np.log(1.0 / frac_remaining) / k_dephos_old
    s = t_old / target_decay_s
    pf_corrected = dict(pf)
    pf_corrected['k_dephos'] = k_dephos_old * s
    pf_corrected['V_lrrk2'] = pf['V_lrrk2'] * s
    return pf_corrected, s


if __name__ == '__main__':
    r = differential_evolution(cost, B, seed=11, maxiter=150, popsize=20,
                               tol=1e-11, polish=True, disp=False)
    pf = dict(zip(FREE, r.x))
    res, m = metrics(pf)

    print("=" * 74)
    print("CALIBRATED FREE PARAMETERS (v3)")
    print("=" * 74)
    for k in FREE:
        print(f"  {k:<14}{pf[k]:.6g}")
    print(f"\n  final cost = {r.fun:.6g}")

    print("\n" + "=" * 74)
    print("VALIDATION CHECKPOINTS")
    print("=" * 74)
    checks = [("T1 exosomal flux control/WT", m['flux'], 2.0, 4.0),
              ("T2 pRab35 fold control/WT",   m['prab'], 1.5, 3.0),
              ("T3 aggregate fraction WT",    m['agg'],  0.10, 0.20),
              ("T4 total a-syn control/WT",   m['asyn'], 1.2, 2.5)]
    allpass = True
    for n, v, lo, hi in checks:
        ok = lo <= v <= hi
        allpass &= ok
        print(f"  {n:<30}{v:8.3f}  [{lo}, {hi}]   {'PASS' if ok else 'FAIL'}")
    ok5 = m['comb_wt'] >= 0.30; allpass &= ok5
    print(f"  {'T5 combined flux / WT flux':<30}{m['comb_wt']:8.3f}  >= 0.30      "
          f"{'PASS' if ok5 else 'FAIL'}")
    ci = M.combination_index(res)
    ok6 = all(0.35 <= f <= 0.75 for f in (ci['f_rab35i'], ci['f_lamp2a']))
    allpass &= ok6
    print(f"  {'T6 single-agent potency':<30}{1-ci['f_rab35i']:.3f}/"
          f"{1-ci['f_lamp2a']:.3f}  25%-65%     {'PASS' if ok6 else 'FAIL'}")
    print(f"\n  ALL CHECKPOINTS: {'PASS' if allpass else 'FAIL'}")

    print("\n" + "=" * 74)
    print("STEADY STATES")
    print("=" * 74)
    print(f"{'condition':<10}{'pRab35':>9}{'aSyn':>9}{'aggr':>9}{'MVB':>9}{'flux':>13}")
    for k, v in res.items():
        print(f"{k:<10}{v['pRab35']:>9.4f}{v['aSyn']:>9.4f}{v['aggregate']:>9.4f}"
              f"{v['MVB']:>9.4f}{v['secretion_flux']:>13.5e}")

    print("\n  SYNERGY (molecular endpoint: exosomal secretion flux)")
    for k, v in ci.items():
        print(f"    {k:<18}{v: .6f}")

    print("\n" + "=" * 74)
    print("STRUCTURAL IDENTIFIABILITY OF THE FREE SET  [item 23]")
    print("=" * 74)
    for tag, obs in [("all species (incl. unmeasurable)", ALL_OBS),
                     ("MEASURABLE observables only", MEASURABLE)]:
        S = sens_matrix(pf, obs)
        rk, sv = rank_of(S)
        v = "FULL RANK" if rk == len(FREE) else f"DEFICIENT by {len(FREE)-rk}"
        print(f"  {tag:<38} {S.shape[0]:>3} obs x {len(FREE)} par -> rank {rk}/{len(FREE)}  {v}")
    S = sens_matrix(pf, MEASURABLE)
    rk, sv = rank_of(S)
    ev = sv ** 2
    print(f"\n  v2 comparison: 17 free parameters, rank 12  (5 unidentifiable)")
    print(f"  v3 result:     {len(FREE)} free parameters, rank {rk}")
    print(f"\n  FIM eigenvalue span: {np.log10(ev[0]/ev[-1]):.1f} decades "
          f"(v2 was 38.0)")
    print(f"  Degrees of freedom: {len(FREE)} free parameters vs 6 quantitative "
          f"constraints (v2 was 17 vs 5)")

    print("\n" + "=" * 74)
    print("TIMESCALE CORRECTION (Fan et al. 2018 -- see comment block above)")
    print("=" * 74)
    pf_raw = dict(pf)   # keep the uncorrected fit around for the printout
    pf, s = apply_timescale_correction(pf_raw)
    print(f"  k_dephos: raw fit = {pf_raw['k_dephos']:.6f} /s   "
          f"-> corrected = {pf['k_dephos']:.6f} /s")
    print(f"  V_lrrk2:  raw fit = {pf_raw['V_lrrk2']:.6f} /s   "
          f"-> corrected = {pf['V_lrrk2']:.6f} /s")
    print(f"  scalar s = {s:.6f}  (target decay {TARGET_DECAY_TIME_S:.0f} s "
          f"to {NEAR_COMPLETE_FRACTION*100:.0f}% remaining)")
    print("  NOTE: T1-T6 / combination index are unaffected by this step by")
    print("  construction -- see verify_timescale_correction.py for the")
    print("  numerical proof. Re-run that script if TARGET_DECAY_TIME_S")
    print("  above is ever changed.")

    np.save('params_v3.npy', pf, allow_pickle=True)
    print("\n  saved -> params_v3.npy  (k_dephos/V_lrrk2 are TIMESCALE-CORRECTED,")
    print("  not the raw differential_evolution output -- see above)")
