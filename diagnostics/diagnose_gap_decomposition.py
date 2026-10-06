"""
Phase 1 diagnostic, follow-up: does the residual gap under intervention
survive even at g2019s_prod=1.0 (full genotype-independence in production)?
If so, production is not the mechanism -- something downstream (most likely
the quadratic nucleation term, v_nuc = k_nuc * aSyn^2) is amplifying even a
tiny numerical residual into a persistent gap.

This script also DECOMPOSES the gap stage-by-stage (aSyn -> aggregate ->
secretion_flux) under both interventions, at both the calibrated
g2019s_prod and at 1.0, to find exactly where the resistant gap enters the
pipeline.
"""

# Uncomment if running from a subfolder structure where pd_model_v3.py 
# is not in the same directory as this script: 
# import sys, os 
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model')) 

import numpy as np
import pd_model_v3 as M

# Uncomment if running from a subfolder structure where params_v3.npy is not in the same directory as this script: 
# PARAMS_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'params_v3.npy')
# PF = np.load(PARAMS_PATH, allow_pickle=True).item()
PF = np.load('params_v3.npy', allow_pickle=True).item() # If using subfolder structure with sys.path directory, comment out
P = dict(M.P_FIXED)
P.update(PF)


def run(genotype, rab35_block=1.0, LAMP2A=0.20, kinase_scale=1.0, p=None):
    p = dict(p)
    p['V_lrrk2'] = p['V_lrrk2'] * kinase_scale
    cond = dict(genotype=genotype, rab35_block=rab35_block, LAMP2A=LAMP2A)
    return M.steady_state(cond, p)


def fold_gap(a, b):
    return abs(np.log(max(a, 1e-300) / max(b, 1e-300)))


def report_stagewise(label, p, kinase_scale=1.0, rab35_block=1.0):
    r_wt = run('WT', rab35_block=rab35_block, kinase_scale=kinase_scale, p=p, LAMP2A=0.40)
    r_gs = run('G2019S', rab35_block=rab35_block, kinase_scale=kinase_scale, p=p, LAMP2A=0.20)
    print(f"  {label}")
    for field in ['aSyn', 'aggregate', 'secretion_flux']:
        ratio = r_gs[field] / max(r_wt[field], 1e-300)
        print(f"    {field:<16} WT={r_wt[field]:.5e}  GS={r_gs[field]:.5e}  "
              f"ratio(GS/WT)={ratio:.4f}")
    return r_wt, r_gs


print("=" * 84)
print("PART 1 -- g2019s_prod = 1.0 exactly: does the residual gap under")
print("intervention finally collapse, or does it survive?")
print("=" * 84)

P_zero = dict(P)
P_zero['g2019s_prod'] = 1.0

print("\n-- Uninhibited / unblocked baseline --")
report_stagewise("baseline (kinase_scale=1.0, rab35_block=1.0)", P_zero)

print("\n-- Under near-total kinase inhibition --")
report_stagewise("kinase_scale=0.01", P_zero, kinase_scale=0.01)

print("\n-- Under near-total RAB35 blockade --")
report_stagewise("rab35_block=0.02", P_zero, rab35_block=0.02)

print()
print("=" * 84)
print("PART 2 -- Stage-wise decomposition AT THE CALIBRATED g2019s_prod")
print("(for comparison -- where does the gap enter the pipeline normally?)")
print("=" * 84)

print("\n-- Uninhibited / unblocked baseline --")
report_stagewise("baseline (calibrated g2019s_prod)", P)

print("\n-- Under near-total kinase inhibition --")
report_stagewise("kinase_scale=0.01 (calibrated g2019s_prod)", P, kinase_scale=0.01)

print("\n-- Under near-total RAB35 blockade --")
report_stagewise("rab35_block=0.02 (calibrated g2019s_prod)", P, rab35_block=0.02)

print()
print("=" * 84)
print("INTERPRETATION GUIDE")
print("=" * 84)
print("  If, at g2019s_prod=1.0, the aSyn ratio under intervention is ~1.0")
print("  (production gap gone) but aggregate/secretion_flux ratios are NOT")
print("  ~1.0 -- that confirms the resistant gap is being generated/amplified")
print("  DOWNSTREAM of production (most likely by the quadratic v_nuc term,")
print("  or by another structural feature), not by g2019s_prod itself.")
print()
print("  If aSyn, aggregate, AND secretion_flux ratios all collapse to ~1.0")
print("  at g2019s_prod=1.0, then production genuinely was the sole driver,")
print("  and the earlier g2019s_prod=1.4 result was likely a sensitivity/")
print("  nonlinearity artifact rather than evidence of a second mechanism.")
