"""
diagnose_validation_failure_structural_cause: structural cause of V2/V3 failure.

Three checks, run in sequence:
  A. Is the system saturated (pRab35 >> Km_MVB) at baseline?
  B. Does the GS/WT pRab35 ratio stay pinned at g2019s_kin regardless
     of kinase_scale?
  C. Does g2019s_prod=1.4 make V2 pass while V3 still fails (confirming
     two separate root causes, not one)?

Does not modify params_v3.npy, pd_model_v3.py, or calibrate_v3.py.
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

print("=" * 74)
print("STEP A -- Km_MVB vs baseline pRab35 (saturation check)")
print("=" * 74)
cond_wt = dict(genotype='WT', rab35_block=1.0, LAMP2A=0.40)
ss = M.steady_state(cond_wt, P)
ratio_a = ss['pRab35'] / P['Km_MVB']
print(f"  baseline pRab35        = {ss['pRab35']:.5f}")
print(f"  Km_MVB                 = {P['Km_MVB']:.5f}")
print(f"  ratio pRab35/Km_MVB    = {ratio_a:.3f}")
print(f"  {'SATURATED (ratio >> 1)' if ratio_a > 3 else 'NOT clearly saturated'}")

print()
print("=" * 74)
print("STEP B -- GS/WT pRab35 ratio across kinase_scale (invariance check)")
print("=" * 74)
for kinase_scale in [1.0, 0.5, 0.1, 0.01, 0.001]:
    p = dict(P)
    p['V_lrrk2'] = P['V_lrrk2'] * kinase_scale
    r_wt = M.steady_state(dict(genotype='WT', rab35_block=1.0, LAMP2A=0.40), p)
    r_gs = M.steady_state(dict(genotype='G2019S', rab35_block=1.0, LAMP2A=0.20), p)
    ratio_b = r_gs['pRab35'] / r_wt['pRab35']
    print(f"  kinase_scale={kinase_scale:<8} pRab35 ratio GS/WT = {ratio_b:.5f}  "
          f"(g2019s_kin = {P['g2019s_kin']:.5f})")

print()
print("=" * 74)
print("STEP C -- g2019s_prod=1.4 diagnostic (does it fix V2, not V3?)")
print("=" * 74)
P_test = dict(P)
P_test['g2019s_prod'] = 1.4

def flux(genotype, rab35_block=1.0, LAMP2A=0.20, kinase_scale=1.0, p=None):
    p = dict(p)
    p['V_lrrk2'] = p['V_lrrk2'] * kinase_scale
    cond = dict(genotype=genotype, rab35_block=rab35_block, LAMP2A=LAMP2A)
    r = M.steady_state(cond, p)
    return r['secretion_flux']

def fold_gap(a, b):
    return abs(np.log(a / b))

f_wt = flux('WT', p=P_test)
f_gs = flux('G2019S', p=P_test)

f_wt_inhib = flux('WT', kinase_scale=0.01, p=P_test)
f_gs_inhib = flux('G2019S', kinase_scale=0.01, p=P_test)
gap_uninhib = fold_gap(f_gs, f_wt)
gap_inhib = fold_gap(f_gs_inhib, f_wt_inhib)
v2_collapse = 1.0 - gap_inhib / max(gap_uninhib, 1e-12)

f_wt_block = flux('WT', rab35_block=0.02, p=P_test)
f_gs_block = flux('G2019S', rab35_block=0.02, p=P_test)
gap_block = fold_gap(f_gs_block, f_wt_block)
v3_collapse = 1.0 - gap_block / max(gap_uninhib, 1e-12)

print(f"  uninhibited gap        = {gap_uninhib:.4f}")
print(f"  V2 (kinase-inhibited)  = {gap_inhib:.4f}   collapse = {v2_collapse*100:.1f}%   "
      f"{'PASS' if v2_collapse > 0.8 else 'FAIL'}")
print(f"  V3 (RAB35-blocked)     = {gap_block:.4f}   collapse = {v3_collapse*100:.1f}%   "
      f"{'PASS' if v3_collapse > 0.8 else 'FAIL'}")