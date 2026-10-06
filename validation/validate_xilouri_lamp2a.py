"""
Held-out validation against the Xilouri et al. LAMP2A/CMA papers -- the
clearance-side counterpart to validate_bae2018.py (which tested the
RAB35/export side). Together these two scripts test the two halves of
the shared-pool mechanism pd_model_v3.py's v3 rebuild is built on
(d_aggr = v_nuc - v_clear - v_load).

Sources (both independent of calibration -- neither is cited anywhere in
Model_Parameters_v3.xlsx; only Volpicelli-Daley 2016 and Dice/Kaushik-Cuervo
are cited for the CMA-pathway constants Km_CMA/Km_clear, not for a
LAMP2A dose-response target):

  V5  Xilouri et al. 2013, Brain 136(7):2130-2146. doi:10.1093/brain/awt131
      "Boosting chaperone-mediated autophagy in vivo mitigates
      alpha-synuclein-induced neurodegeneration"
      -> LAMP2A OVEREXPRESSION reduces total alpha-synuclein / aberrant
         species and neurodegeneration.
      https://academic.oup.com/brain/article/136/7/2130/276870

  V6  Xilouri et al. 2016, Autophagy 12(11):2230-2247.
      doi:10.1080/15548627.2016.1214777
      "Impairment of chaperone-mediated autophagy induces dopaminergic
      neurodegeneration in rats"
      -> LAMP2A KNOCKDOWN (shRNA) causes accumulation of SNCA-positive
         puncta and dopaminergic neurodegeneration -- the opposite arm.
      https://pmc.ncbi.nlm.nih.gov/articles/PMC5103347/

CAVEAT (state in any write-up): Xilouri et al. 2013 explicitly report that
protection was observed even when the steady-state levels of alpha-synuclein
were unchanged in some of their systems -- i.e. they report the LAMP2A ->
alpha-syn-LEVEL relationship as present but not uniformly strong across
every condition tested. This script checks DIRECTION only (does more LAMP2A
mean less total_aSyn/aggregate, and vice versa for knockdown), consistent
with the same "direction, not magnitude" standard used in validate_bae2018.py.
"""

# Uncomment if running from a subfolder structure where pd_model_v3.py
# is not in the same directory as this script:
# import sys, os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

import numpy as np
import pd_model_v3 as M

# Uncomment if running from a subfolder structure where params_v3.npy is not in the same directory as this script: 
# PARAMS_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'params_v3.npy')

try:
    # Uncomment if running from a subfolder structure where params_v3.npy is not in the same directory as this script:     
    # PF = np.load(PARAMS_PATH, allow_pickle=True).item()
    PF = np.load('params_v3.npy', allow_pickle=True).item() # If using subfolder structure with sys.path directory, comment out
    P = dict(M.P_FIXED)
    P.update(PF)
    PARAM_SOURCE = "CALIBRATED (params_v3.npy)"
except FileNotFoundError:
    print("!" * 74)
    print("WARNING: params_v3.npy not found -- using NOMINAL parameters.")
    print("Results below are illustrative only.")
    print("!" * 74)
    print()
    P = M.full_params()
    PARAM_SOURCE = "NOMINAL (illustrative only)"

print(f"Parameter source: {PARAM_SOURCE}\n")


def steady(LAMP2A, genotype='WT', rab35_block=1.0, p=None):
    p = p or P
    cond = dict(genotype=genotype, rab35_block=rab35_block, LAMP2A=LAMP2A)
    return M.steady_state(cond, p)


# Baseline: use the model's own WT LAMP2A level (0.40, per CONDITIONS) as
# the reference point, consistent with how 'wildtype' is defined elsewhere
# in this codebase.
BASELINE_LAMP2A = 0.40

print("=" * 84)
print("V5  [Xilouri et al. 2013]  LAMP2A OVEREXPRESSION should REDUCE")
print("    total alpha-synuclein / aggregate levels")
print("=" * 84)

overexpr_levels = [0.40, 0.60, 0.80, 1.00, 1.50, 2.00]
print(f"{'LAMP2A':>10}{'total_aSyn':>14}{'aggregate':>14}{'agg_fraction':>14}")
prev_total = None
overexpr_monotonic = True
for L in overexpr_levels:
    r = steady(L)
    print(f"{L:>10.2f}{r['total_aSyn']:>14.5f}{r['aggregate']:>14.5f}"
          f"{r['aggregate_fraction']:>14.4f}")
    if prev_total is not None and r['total_aSyn'] > prev_total:
        overexpr_monotonic = False
    prev_total = r['total_aSyn']

r_base = steady(BASELINE_LAMP2A)
r_high = steady(2.00)
v5_pass = r_high['total_aSyn'] < r_base['total_aSyn']
print(f"\n  total_aSyn at baseline LAMP2A={BASELINE_LAMP2A}:  "
      f"{r_base['total_aSyn']:.5f}")
print(f"  total_aSyn at 5x LAMP2A={2.00}:            "
      f"{r_high['total_aSyn']:.5f}")
print(f"  Monotonically decreasing across the full sweep?  "
      f"{'YES' if overexpr_monotonic else 'NO'}")
print(f"  V5 VERDICT: {'PASS' if v5_pass else 'FAIL'} -- "
      f"{'overexpression reduces total alpha-syn, matching Xilouri 2013' if v5_pass else 'overexpression does NOT reduce total alpha-syn -- mismatch'}")

print()
print("=" * 84)
print("V6  [Xilouri et al. 2016]  LAMP2A KNOCKDOWN should INCREASE")
print("    total alpha-synuclein / aggregate levels")
print("=" * 84)

knockdown_levels = [0.40, 0.30, 0.20, 0.10, 0.05, 0.02]
print(f"{'LAMP2A':>10}{'total_aSyn':>14}{'aggregate':>14}{'agg_fraction':>14}")
prev_total = None
knockdown_monotonic = True
for L in knockdown_levels:
    r = steady(L)
    print(f"{L:>10.2f}{r['total_aSyn']:>14.5f}{r['aggregate']:>14.5f}"
          f"{r['aggregate_fraction']:>14.4f}")
    if prev_total is not None and r['total_aSyn'] < prev_total:
        knockdown_monotonic = False
    prev_total = r['total_aSyn']

r_low = steady(0.02)
v6_pass = r_low['total_aSyn'] > r_base['total_aSyn']
print(f"\n  total_aSyn at baseline LAMP2A={BASELINE_LAMP2A}:  "
      f"{r_base['total_aSyn']:.5f}")
print(f"  total_aSyn at near-total knockdown LAMP2A={0.02}: "
      f"{r_low['total_aSyn']:.5f}")
print(f"  Monotonically increasing as LAMP2A decreases?  "
      f"{'YES' if knockdown_monotonic else 'NO'}")
print(f"  V6 VERDICT: {'PASS' if v6_pass else 'FAIL'} -- "
      f"{'knockdown increases total alpha-syn, matching Xilouri 2016' if v6_pass else 'knockdown does NOT increase total alpha-syn -- mismatch'}")

print()
print("=" * 84)
print("SUMMARY")
print("=" * 84)
for name, ok in [("V5  LAMP2A overexpression reduces total_aSyn (Xilouri 2013)", v5_pass),
                  ("V6  LAMP2A knockdown increases total_aSyn (Xilouri 2016)", v6_pass)]:
    print(f"  {name:<62} {'PASS' if ok else 'FAIL'}")

print()
print("Note: these check DIRECTION (monotonic response to LAMP2A dose) only,")
print("consistent with validate_bae2018.py's standard. Neither source paper")
print("gives a fold-change target in units comparable to this model's uM")
print("output, so magnitude is not compared. Both papers were confirmed")
print("independent of calibration -- not cited in Model_Parameters_v3.xlsx")
print("for any T1-T6 checkpoint or free-parameter bound.")
