"""
PD alpha-synuclein propagation model.
 
Design rationale for the parameter structure below:
 
  PARAMETERISED to an identifiable set. Parameters here are partitioned into FREE (calibrated,
  verified full rank) and FIXED (pinned to literature, or structurally
  unconstrainable nuisance), to avoid fitting degenerate combinations:
 
         - Km_CMA PINNED to Dice (2007). Only Vmax_CMA/Km_CMA is identifiable
           in the sub-saturating regime; pinning the denominator makes the
           numerator identifiable.
         - k_deg_MVB and k_MVB BOTH PINNED, and pinned equal to each other.
           MVB enters only the loading flux and only linearly, so the sole
           identifiable quantity is the product k_load x k_MVB / k_deg_MVB.
           Setting k_MVB = k_deg_MVB normalises MVB to fractional occupancy
           in [0,1] and lets k_load carry the export magnitude.
         - k_secrete PINNED as a declared nuisance. At steady state secretion
           flux equals loading flux, so k_secrete sets only the standing
           exosomal concentration and is invisible to any flux measurement
           (a null direction in the sensitivity matrix). It is retained for
           mechanistic completeness and explicitly excluded from calibration.
 
  alpha-synuclein PRODUCTION IS GENOTYPE-DEPENDENT. LRRK2 G2019S is reported
  to raise alpha-syn expression and impair its degradation (Volpicelli-Daley
  et al. 2016 Neurobiol Dis 88:186); production is modeled as genotype-
  dependent to represent this documented disease mechanism (see g2019s_prod).
 
  Km_MVB has no direct literature source and is subjected to a dedicated
  one-dimensional profile (see therapeutic_window.py) reporting the range
  over which conclusions hold.
 
State vector:
    0 pRab35   phosphorylated Rab35            (uM)
    1 pRab10   phosphorylated Rab10            (uM)
    2 aSyn     soluble alpha-synuclein         (uM)
    3 aggr     seeding-competent aggregate     (uM)
    4 MVB      multivesicular body pool        (uM equiv)
    5 exo      aggregate in exosomes           (uM)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

RAB35_TOT = 1.0
RAB10_TOT = 1.0
LRRK2_ACTIVE = 0.1

# ---------------------------------------------------------------------------
# FIXED parameters. Pinned to literature or declared unconstrainable.
# These are NOT calibrated and NOT counted as degrees of freedom.
# ---------------------------------------------------------------------------
P_FIXED = dict(
    Km_CMA    = 0.50,   # uM   Dice (2007); Kaushik & Cuervo (2009). PINNED.
    Km_clear  = 0.50,   # uM   Dice (2007). PINNED.
    k_deg_MVB = 0.020,  # 1/s  Hessvik & Llorente (2018). PINNED to break
                        #      the k_MVB / k_deg_MVB degeneracy.
    k_secrete = 0.100,  # 1/s  Emmanouilidou (2010). NUISANCE: invisible to
                        #      any steady-state flux measurement.
    k_MVB     = 0.020,  # uM/s PINNED EQUAL TO k_deg_MVB. This normalises the
                        #      MVB pool to fractional occupancy in [0,1], since
                        #      MVB_ss = (k_MVB/k_deg_MVB) * g(pRab35) = g(pRab35).
                        #      MVB appears only linearly in the loading flux, so
                        #      k_load x k_MVB is the sole identifiable product
                        #      and their ratio is not determinable. k_load now
                        #      carries the export magnitude.
)

# ---------------------------------------------------------------------------
# FREE parameters. Calibrated. Must be full rank (verified in identifiability.py)
# ---------------------------------------------------------------------------
P_FREE_NOMINAL = dict(
    V_lrrk2      = 0.050,   # uM/s     WT LRRK2 max velocity
    g2019s_kin   = 2.00,    # -        G2019S kinase multiplier
    Km_lrrk2     = 1.00,    # uM       Rab substrate affinity
    k_dephos     = 0.080,   # 1/s      PPM1H dephosphorylation
    k_prod_WT    = 0.0030,  # uM/s     alpha-syn synthesis, wildtype
    g2019s_prod  = 1.40,    # -        G2019S synthesis multiplier      [NEW 21]
    Vmax_CMA     = 0.0111,  # 1/s      CMA capacity per unit LAMP2A
    k_nuc        = 0.461,   # 1/(uM s) primary nucleation
    Vmax_clear   = 0.220,   # 1/s      lysosomal aggregate clearance per LAMP2A
    Km_MVB       = 0.100,   # uM       pRab35 half-saturation      [see item 22]
    k_load       = 0.0550,  # 1/s      aggregate loading into MVB
    Km_load      = 0.827,   # uM       loading half-saturation
)

FREE_NAMES = list(P_FREE_NOMINAL.keys())


def full_params(p_free=None):
    p = dict(P_FIXED)
    p.update(P_FREE_NOMINAL if p_free is None else p_free)
    return p


CONDITIONS = {
    'wildtype': dict(genotype='WT',     rab35_block=1.00, LAMP2A=0.40),
    'control':  dict(genotype='G2019S', rab35_block=1.00, LAMP2A=0.20),
    'rab35i':   dict(genotype='G2019S', rab35_block=0.25, LAMP2A=0.20),
    'lamp2a':   dict(genotype='G2019S', rab35_block=1.00, LAMP2A=0.50),
    'combined': dict(genotype='G2019S', rab35_block=0.25, LAMP2A=0.50),
}


def rhs(t, y, p, cond):
    pRab35, pRab10, aSyn, aggr, MVB, exo = np.maximum(y, 0.0)
    is_pd = (cond['genotype'] == 'G2019S')
    Vmax = p['V_lrrk2'] * (p['g2019s_kin'] if is_pd else 1.0)
    k_prod = p['k_prod_WT'] * (p['g2019s_prod'] if is_pd else 1.0)   # [21]
    L = cond['LAMP2A']

    Rab35 = max(RAB35_TOT - pRab35, 0.0)
    Rab10 = max(RAB10_TOT - pRab10, 0.0)
    d_p35 = Vmax * LRRK2_ACTIVE * Rab35 / (p['Km_lrrk2'] + Rab35) - p['k_dephos'] * pRab35
    d_p10 = Vmax * LRRK2_ACTIVE * Rab10 / (p['Km_lrrk2'] + Rab10) - p['k_dephos'] * pRab10

    # CMA: Michaelis-Menten, no self-inhibition term (removed, see item 23)
    v_cma = p['Vmax_CMA'] * L * aSyn / (p['Km_CMA'] + aSyn)
    v_nuc = p['k_nuc'] * aSyn ** 2
    d_aSyn = k_prod - v_cma - v_nuc

    # aggregate: two competing sinks, both draining the SAME pool.
    # This shared-pool structure is what makes synergy expressible.
    v_clear = p['Vmax_clear'] * L * aggr / (p['Km_clear'] + aggr)
    v_load = p['k_load'] * MVB * aggr / (p['Km_load'] + aggr)
    d_aggr = v_nuc - v_clear - v_load

    eff = pRab35 * cond['rab35_block']
    floor_frac = p.get('floor_frac', 0.0)
    v_mvb = p['k_deg_MVB'] * ((1.0 - floor_frac) * eff / (p['Km_MVB'] + eff) + floor_frac)
    d_MVB = v_mvb - p['k_deg_MVB'] * MVB

    d_exo = v_load - p['k_secrete'] * exo
    return [d_p35, d_p10, d_aSyn, d_aggr, d_MVB, d_exo]


def steady_state(cond, p=None, fast=False):
    p = p or full_params()
    y0 = np.array([0.05, 0.05, 0.10, 0.02, 0.20, 0.005])

    def f(y):
        return np.asarray(rhs(0.0, np.abs(y), p, cond))

    y = None
    for guess in (y0, y0 * 3, y0 * 0.3):
        s = root(f, guess, method='hybr', tol=1e-14)
        if s.success:
            c = np.abs(s.x)
            if np.max(np.abs(f(c))) < 1e-12:
                y = c
                break
    if y is None:
        if fast:
            return None
        sol = solve_ivp(rhs, (0, 50000.0), y0, args=(p, cond), method='LSODA',
                        rtol=1e-9, atol=1e-13, t_eval=[50000.0], first_step=1e-3)
        y = np.maximum(sol.y[:, -1], 0.0)

    pRab35, pRab10, aSyn, aggr, MVB, exo = y
    return dict(pRab35=pRab35, pRab10=pRab10, aSyn=aSyn, aggregate=aggr,
                MVB=MVB, exosomal=exo,
                secretion_flux=p['k_secrete'] * exo,
                total_aSyn=aSyn + aggr,
                aggregate_fraction=aggr / (aSyn + aggr) if aSyn + aggr > 0 else 0.0,
                residual=np.max(np.abs(rhs(0.0, y, p, cond))))


def run_all(p=None, fast=False):
    p = p or full_params()
    out = {}
    for k, c in CONDITIONS.items():
        r = steady_state(c, p, fast=fast)
        if r is None:
            return None
        out[k] = r
    return out


def combination_index(res, key='secretion_flux'):
    c = res['control'][key]
    fr, fl, fc = res['rab35i'][key] / c, res['lamp2a'][key] / c, res['combined'][key] / c
    return dict(f_rab35i=fr, f_lamp2a=fl, f_combined=fc,
                bliss_expected=fr * fl, CI=fc / (fr * fl),
                bliss_excess_pp=((1 - fc) - (1 - fr * fl)) * 100)


# ---------------------------------------------------------------------------
# TIME-COURSE EXTENSION
#
# Everything above this point only reports steady states. This block exposes
# full trajectories, needed for:
#   - comparing against time-resolved data (e.g. Volpicelli-Daley 2016 Fig 2B,
#     which shows no genotype difference at day 7 but a significant one at
#     day 18 -- a pattern steady_state() cannot represent at all)
#   - checking how fast each condition approaches its steady state, which
#     matters for interpreting any fixed-timepoint experimental comparison
#
# Does not touch or redefine anything above -- purely additive.
# ---------------------------------------------------------------------------

def time_course(cond, p=None, t_span=(0.0, 50000.0), n_points=500,
                 y0=None, rtol=1e-9, atol=1e-13):
    """Full trajectory for one condition, returned as a dict of arrays.

    Parameters
    ----------
    cond : dict         -- one of the CONDITIONS values, or a custom dict
                            with the same keys (genotype, rab35_block, LAMP2A)
    p : dict or None    -- parameter set; defaults to full_params()
    t_span : (t0, t1)   -- integration window, in the same time units as the
                            rate constants (seconds, given k_dephos ~ 0.08 /s
                            etc. -- so t_span=(0, 50000) is roughly 14 hours)
    n_points : int       -- number of evenly spaced output points
    y0 : array or None  -- initial state; defaults to the same guess used by
                            steady_state()

    Returns
    -------
    dict with keys 't', 'pRab35', 'pRab10', 'aSyn', 'aggregate', 'MVB',
    'exosomal', 'secretion_flux', 'total_aSyn', 'aggregate_fraction'
    """
    p = p or full_params()
    if y0 is None:
        y0 = np.array([0.05, 0.05, 0.10, 0.02, 0.20, 0.005])

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(rhs, t_span, y0, args=(p, cond), method='LSODA',
                     rtol=rtol, atol=atol, t_eval=t_eval, first_step=1e-3)

    if not sol.success:
        raise RuntimeError(f"time_course integration failed: {sol.message}")

    pRab35, pRab10, aSyn, aggr, MVB, exo = np.maximum(sol.y, 0.0)
    total_aSyn = aSyn + aggr
    with np.errstate(divide='ignore', invalid='ignore'):
        agg_frac = np.where(total_aSyn > 0, aggr / total_aSyn, 0.0)

    return dict(
        t=sol.t,
        pRab35=pRab35, pRab10=pRab10, aSyn=aSyn, aggregate=aggr,
        MVB=MVB, exosomal=exo,
        secretion_flux=p['k_secrete'] * exo,
        total_aSyn=total_aSyn,
        aggregate_fraction=agg_frac,
    )


def run_all_time_courses(p=None, t_span=(0.0, 50000.0), n_points=500):
    """Time-course version of run_all() -- same CONDITIONS dict, full
    trajectories instead of endpoints only."""
    p = p or full_params()
    return {k: time_course(c, p, t_span=t_span, n_points=n_points)
            for k, c in CONDITIONS.items()}


def time_to_fraction_of_steady_state(traj, ss_value, field='secretion_flux',
                                       frac=0.90):
    """Find the first timepoint at which `field` reaches `frac` of its final
    (steady-state) value. Useful for checking whether a fixed experimental
    timepoint (e.g. day 7 vs day 18) falls before or after the model has
    effectively equilibrated -- if the model reaches steady state much
    faster than the experimental timescale, steady-state comparisons remain
    valid; if not, only the time-course comparison is meaningful."""
    target = frac * ss_value
    vals = traj[field]
    idx = np.argmax(vals >= target) if ss_value >= 0 else np.argmax(vals <= target)
    if idx == 0 and vals[0] < target:
        return None  # never reached within t_span
    return traj['t'][idx]


def genotype_gap_over_time(p=None, t_span=(0.0, 50000.0), n_points=500,
                            field='total_aSyn'):
    """Compute the WT-vs-G2019S ('control') fold gap in `field` at every
    timepoint, using the same custom conditions used elsewhere in this
    module (genotype label drives both production and phosphorylation).

    Returns t, gap(t) = control(t) / wildtype(t), the WT trajectory, and the
    G2019S trajectory. This is the direct model-side analogue of
    Volpicelli-Daley et al. 2016 Fig. 2B: no genotype difference at an early
    timepoint, growing difference later.
    """
    p = p or full_params()
    cond_wt = dict(genotype='WT', rab35_block=1.0, LAMP2A=0.40)
    cond_gs = dict(genotype='G2019S', rab35_block=1.0, LAMP2A=0.20)
    tc_wt = time_course(cond_wt, p, t_span=t_span, n_points=n_points)
    tc_gs = time_course(cond_gs, p, t_span=t_span, n_points=n_points)
    with np.errstate(divide='ignore', invalid='ignore'):
        gap = np.where(tc_wt[field] > 0, tc_gs[field] / tc_wt[field], np.nan)
    return tc_wt['t'], gap, tc_wt, tc_gs


    # -----------------------------------------------------------------
    # TIME-COURSE DEMONSTRATION  [item 22 extension]
    # -----------------------------------------------------------------
    print("\n" + "=" * 74)
    print("TIME-COURSE EXTENSION")
    print("=" * 74)

    # How long does it take each condition to reach steady state, at the
    # native (second-scale) rate constants? Establishes whether the
    # existing steady-state-only calibration is even valid on the
    # timescale we care about.
    t_span_native = (0.0, 50000.0)
    tcs = run_all_time_courses(p, t_span=t_span_native, n_points=2000)
    print(f"\nTime (native units implied by rate constants, ~seconds) to reach "
          f"90% of steady-state secretion_flux:")
    for k, tc in tcs.items():
        ss = res[k]['secretion_flux']
        t90 = time_to_fraction_of_steady_state(tc, ss, field='secretion_flux',
                                                 frac=0.90)
        t90_str = f"{t90:8.1f}" if t90 is not None else "  not reached"
        print(f"  {k:<10} t90 = {t90_str}   (steady state = {ss:.5e})")

    # WT vs G2019S genotype gap over time, on total_aSyn -- the model-side
    # analogue of Volpicelli-Daley et al. 2016 Fig. 2B (no genotype
    # difference at an early fibril-exposure timepoint, a significant
    # difference by a later one).
    t, gap, tc_wt, tc_gs = genotype_gap_over_time(p, t_span=t_span_native,
                                                    n_points=2000)
    print(f"\nG2019S/WT fold-gap in total_aSyn over time (native units):")
    checkpoints = [0.0, 0.05, 0.10, 0.25, 0.50, 1.0]
    for frac in checkpoints:
        idx = int(frac * (len(t) - 1))
        print(f"  t={t[idx]:8.1f}  ({frac*100:5.1f}% of window)   "
              f"gap = {gap[idx]:.4f}x")

    early_gap = gap[int(0.05 * (len(t) - 1))]
    late_gap = gap[-1]
    print(f"\n  early-window gap  = {early_gap:.4f}x")
    print(f"  late-window  gap  = {late_gap:.4f}x")
    print(f"  gap growth        = {(late_gap - early_gap):+.4f}x")

    print("\n" + "-" * 74)
    print("PROPOSED CHECKPOINT T7 -- genotype gap should EMERGE over time,")
    print("not be present from t=0 (qualitative match to Volpicelli-Daley 2016")
    print("Fig. 2B: no significant difference at day 7, significant by day 18)")
    print("-" * 74)
    t7_pass = (early_gap < 1.15) and (late_gap > early_gap)
    print(f"  early gap near 1.0x (no early difference)?  "
          f"{'YES' if early_gap < 1.15 else 'NO'}  ({early_gap:.3f}x)")
    print(f"  gap increases over the window?               "
          f"{'YES' if late_gap > early_gap else 'NO'}  "
          f"({early_gap:.3f}x -> {late_gap:.3f}x)")
    print(f"  T7 VERDICT: {'PASS' if t7_pass else 'FAIL'}")
    print("\n  NOTE: t_span above is in the model's native (second-scale) time")
    print("  units, NOT calendar days -- it has not been mapped onto the")
    print("  18-day experimental timescale. A PASS here is a qualitative")
    print("  shape match only (delayed emergence vs immediate divergence),")
    print("  not a quantitative day-7-vs-day-18 comparison. Mapping native")
    print("  time to real days requires either a literature time constant")
    print("  for one of the fast reactions (e.g. k_dephos) or a dedicated")
    print("  fit against Volpicelli-Daley's day-7/day-18 data, TBD.")
    print("\n  To generate steady-state, synergy, therapeutic window, and parameter identifiability figures, run: python3 generate_figures.py")
    print("\n  To generate time-course figures, run: python3 generate_figures.py")


