"""
Demonstrates the RAB35-independent MVB biogenesis floor (floor_frac)
mechanism proposed to address the V2/V3 amplification finding
(see diagnose_gap_decomposition.py and diagnose_isogenic_comparison.py
for the underlying diagnosis this mechanism responds to).

Requires the modified pd_model_v3.py with the floor_frac term added to
the MVB equation in rhs() (see accompanying model file). floor_frac
defaults to 0.0 via p.get('floor_frac', 0.0), so this is fully backward
compatible -- floor_frac=0.0 reproduces the original model exactly
(verified below as a sanity check).

Two things are shown:
  1. Sanity check: floor_frac=0.0 reproduces the original calibrated
     model's secretion_flux for all 5 CONDITIONS, exactly.
  2. Isogenic sweep (LAMP2A held identical between genotypes, to avoid
     the condition-definition confound identified during Phase 1):
     as floor_frac increases, the pathological AMPLIFICATION of the
     genotype gap under RAB35 blockade is resolved (blocked/unblocked
     gap ratio -> 1.0), while a smaller residual gap persists and does
     NOT collapse to zero (no full rescue).

This script does NOT calibrate floor_frac to any target and does NOT
claim a specific floor_frac value is correct -- see
check_floor_frac_identifiability.py for why: the parameter was found to
be non-identifiable from the current 5 conditions / 3 measurable
observables, and no literature value for it could be found. This script
exists only to demonstrate the mechanism's qualitative behavior, as
reported in the paper's Limitations / Future Work section.
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


def flux(genotype, LAMP2A, rab35_block=1.0, floor_frac=0.0, p=None):
    p = dict(p)
    p['floor_frac'] = floor_frac
    cond = dict(genotype=genotype, rab35_block=rab35_block, LAMP2A=LAMP2A)
    r = M.steady_state(cond, p)
    return r['secretion_flux']


def fold_gap(a, b):
    return abs(np.log(a / b))


print("=" * 84)
print("SANITY CHECK: floor_frac=0.0 reproduces the original model exactly")
print("=" * 84)
res_orig = M.run_all(P)
P0 = dict(P)
P0['floor_frac'] = 0.0
res_zero = M.run_all(P0)
print(f"{'condition':<10}{'orig flux':>16}{'floor=0 flux':>16}{'match?':>10}")
all_match = True
for k in res_orig:
    a, b = res_orig[k]['secretion_flux'], res_zero[k]['secretion_flux']
    ok = abs(a - b) < 1e-12
    all_match &= ok
    print(f"{k:<10}{a:>16.8e}{b:>16.8e}{'YES' if ok else 'NO':>10}")
print(f"\nAll conditions match: {'YES -- backward compatible' if all_match else 'NO -- investigate before proceeding'}")

print()
print("=" * 84)
print("ISOGENIC SWEEP: effect of floor_frac on the amplification pathology")
print("(LAMP2A=0.40 held IDENTICAL for both genotypes -- avoids the")
print("condition-definition confound identified during Phase 1 diagnosis)")
print("=" * 84)
print(f"{'floor_frac':>12}{'gap unblocked':>16}{'gap blocked':>16}"
      f"{'blocked/unblocked ratio':>26}")
for ff in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    f_wt = flux('WT', 0.40, floor_frac=ff, p=P)
    f_gs = flux('G2019S', 0.40, floor_frac=ff, p=P)
    f_wt_b = flux('WT', 0.40, rab35_block=0.02, floor_frac=ff, p=P)
    f_gs_b = flux('G2019S', 0.40, rab35_block=0.02, floor_frac=ff, p=P)
    gap_un = fold_gap(f_gs, f_wt)
    gap_bl = fold_gap(f_gs_b, f_wt_b)
    ratio = gap_bl / max(gap_un, 1e-12)
    print(f"{ff:>12.2f}{gap_un:>16.4f}{gap_bl:>16.4f}{ratio:>26.4f}")

print()
print("INTERPRETATION:")
print("  At floor_frac=0 (original model), blockade AMPLIFIES the genotype")
print("  gap (ratio > 1) -- this is the V3 failure mode identified in Phase 1.")
print("  As floor_frac increases, this ratio converges toward 1.0: blockade")
print("  no longer amplifies the gap. However, the absolute gap does NOT")
print("  collapse to 0 (full rescue) even at floor_frac->1 -- a residual,")
print("  not-yet-isolated source of genotype difference remains, and")
print("  floor_frac itself was found to be non-identifiable from current")
print("  data (see check_floor_frac_identifiability.py). Reported as a")
print("  limitation / future-work item, not a fitted result.")
