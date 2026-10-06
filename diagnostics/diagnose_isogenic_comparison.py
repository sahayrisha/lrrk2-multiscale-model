"""
Phase 1 diagnostic, part 3: ISOGENIC comparison.

pd_model_v3.py's CONDITIONS dict pairs WT with LAMP2A=0.40 and G2019S with
LAMP2A=0.20 -- these are NOT isogenic conditions. Any WT-vs-G2019S
comparison using those named conditions confounds genotype with LAMP2A
dose. The previous diagnostic (diagnose_gap_decomposition.py) showed the
aSyn ratio stays at ~1.08 even with g2019s_prod=1.0 -- evidence this
confound, not a missing structural mechanism, may be driving at least part
of the "resistant gap under intervention" finding.

This script re-runs the same intervention checks (kinase inhibition,
RAB35 blockade) but holds LAMP2A IDENTICAL between WT and G2019S, at two
LAMP2A levels (0.40 and 0.20), and at two g2019s_prod values (calibrated,
and 1.0), to isolate the true genotype-driven (kinase/production) effect
from the LAMP2A confound.
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


def run(genotype, LAMP2A, rab35_block=1.0, kinase_scale=1.0, p=None):
    p = dict(p)
    p['V_lrrk2'] = p['V_lrrk2'] * kinase_scale
    cond = dict(genotype=genotype, rab35_block=rab35_block, LAMP2A=LAMP2A)
    return M.steady_state(cond, p)


def report(label, p, LAMP2A, kinase_scale=1.0, rab35_block=1.0):
    r_wt = run('WT', LAMP2A, rab35_block=rab35_block, kinase_scale=kinase_scale, p=p)
    r_gs = run('G2019S', LAMP2A, rab35_block=rab35_block, kinase_scale=kinase_scale, p=p)
    print(f"  {label}  (LAMP2A={LAMP2A} for BOTH genotypes)")
    for field in ['aSyn', 'aggregate', 'secretion_flux']:
        ratio = r_gs[field] / max(r_wt[field], 1e-300)
        print(f"    {field:<16} WT={r_wt[field]:.5e}  GS={r_gs[field]:.5e}  "
              f"ratio(GS/WT)={ratio:.4f}")
    return r_wt, r_gs


for g2019s_prod_val, tag in [(P['g2019s_prod'], "CALIBRATED g2019s_prod"),
                              (1.0, "g2019s_prod = 1.0 (production fully genotype-independent)")]:
    p = dict(P)
    p['g2019s_prod'] = g2019s_prod_val

    print("=" * 84)
    print(f"{tag}  (g2019s_prod = {g2019s_prod_val:.4f})")
    print("=" * 84)

    for LAMP2A_level in [0.40, 0.20]:
        print(f"\n-- isogenic LAMP2A = {LAMP2A_level} --")
        print("\n  Baseline (no intervention):")
        report("baseline", p, LAMP2A_level)
        print("\n  Under near-total kinase inhibition:")
        report("kinase_scale=0.01", p, LAMP2A_level, kinase_scale=0.01)
        print("\n  Under near-total RAB35 blockade:")
        report("rab35_block=0.02", p, LAMP2A_level, rab35_block=0.02)
    print()

print("=" * 84)
print("INTERPRETATION GUIDE")
print("=" * 84)
print("  Compare the isogenic ratios above to the NON-isogenic ratios from")
print("  diagnose_gap_decomposition.py (WT@0.40 vs GS@0.20, confounded).")
print()
print("  If, at g2019s_prod=1.0 AND isogenic LAMP2A, the aSyn/aggregate/")
print("  secretion_flux ratios collapse to ~1.0 under BOTH interventions --")
print("  the LAMP2A confound was the (or a major) source of the 'resistant")
print("  gap', not a missing structural mechanism. In that case, V2/V3's")
print("  original failure against Bae et al. may be substantially a")
print("  condition-definition artifact, and the fix is to correct how")
print("  conditions are defined/compared, not to add new model structure.")
print()
print("  If a meaningful gap still survives even under isogenic LAMP2A and")
print("  g2019s_prod=1.0, that final residual IS the genuine structural")
print("  signal (likely g2019s_kin acting through some other channel, or a")
print("  real missing genotype-independent floor) -- worth pursuing the")
print("  MVB-floor fix for THAT residual specifically, now correctly sized.")
