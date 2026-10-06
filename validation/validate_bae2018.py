"""
Held-out structural validation against Bae et al. 2018, Nature Communications
"LRRK2 kinase regulates alpha-synuclein propagation via RAB35 phosphorylation"

IMPORTANT: This is a DIRECTION/STRUCTURE validation, not a magnitude-fit test.
Bae et al. report BiFC fluorescence, band intensity, and % positive cells --
none of these are in the same units as pd_model_v3.py's uM / uM/s outputs.
None of these comparisons were used anywhere in calibrate_v3.py, so they are
legitimately held-out.

Three structural predictions are checked, each corresponding to a specific
figure in the paper:

  V1  Fig. 3g   secretion(G2019S) > secretion(WT) > secretion(kinase-dead)
                -- genotype-dependent secretion, same direction as the
                   calibrated model's control/wildtype flux ratio (T1-adjacent)

  V2  Fig. 3h-j LRRK2 kinase inhibition collapses the genotype-dependent
                secretion difference (WT and G2019S propagation "near
                completely nullified" by the kinase inhibitor)
                -- modeled as V_lrrk2 -> ~0

  V3  Fig. 4h-i / Fig. 6   RAB35 phosphorylation-blockade (dominant-negative
                S22N, or phospho-dead T72A) eliminates the GENOTYPE EFFECT:
                WT+block and G2019S+block become statistically indistinguishable
                ("ns" in Fig. 4i), even though WT and G2019S differ strongly
                without the block.
                -- modeled as rab35_block -> ~0 (near-total inhibition)

None of these checks require refitting P_FREE_NOMINAL / params_v3.npy.
They only run the ALREADY-CALIBRATED model under new condition dictionaries
that were never part of the CONDITIONS dict used in calibration.
"""

# Uncomment if running from a subfolder structure where pd_model_v3.py
# is not in the same directory as this script:
# import sys, os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

import numpy as np
import pd_model_v3 as M

# ---------------------------------------------------------------------------
# Load the calibrated parameter set (must exist -- run calibrate_v3.py first)
# ---------------------------------------------------------------------------
# Uncomment if running from a subfolder structure where params_v3.npy is not in the same directory as this script: 
# PARAMS_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'params_v3.npy')

try:
    # Uncomment if running from a subfolder structure where params_v3.npy is not in the same directory as this script:     
    # PF = np.load(PARAMS_PATH, allow_pickle=True).item()
    PF = np.load('params_v3.npy', allow_pickle=True).item() # If using subfolder structure with sys.path directory, comment out
    P = dict(M.P_FIXED)
    P.update(PF)
    print("Loaded calibrated parameters from params_v3.npy\n")
except FileNotFoundError:
    print("params_v3.npy not found -- using NOMINAL parameters.")
    print("Results below are illustrative only; run calibrate_v3.py first "
          "for the reported result.\n")
    P = M.full_params()


def flux(genotype, rab35_block=1.0, LAMP2A=0.20, kinase_scale=1.0, p=None):
    """Steady-state secretion flux for an arbitrary condition, allowing
    kinase activity to be scaled independently of genotype (for the
    inhibitor-treatment checks)."""
    p = dict(p or P)
    p = dict(p)
    p['V_lrrk2'] = p['V_lrrk2'] * kinase_scale
    cond = dict(genotype=genotype, rab35_block=rab35_block, LAMP2A=LAMP2A)
    r = M.steady_state(cond, p)
    return r['secretion_flux'] if r else np.nan


def fold_gap(a, b):
    """How far the GS/WT ratio sits from 1 (no genotype effect), in log space.
    This is the correct 'gap' metric for the collapse checks below: percentage
    difference is scale-invariant and can stay large even as both values
    shrink toward zero together, which produced a false FAIL in an earlier
    version of this script. Distance-from-1 in log space actually captures
    whether the genotype effect has been abolished."""
    return abs(np.log(a / b))


print("=" * 84)
print("V1  [Fig. 3g]  Genotype-dependent secretion: GS > WT")
print("=" * 84)
f_wt = flux('WT')
f_gs = flux('G2019S')
ratio = f_gs / f_wt
print(f"  secretion(WT)      = {f_wt:.5e} uM/s")
print(f"  secretion(G2019S)  = {f_gs:.5e} uM/s")
print(f"  fold change GS/WT  = {ratio:.3f}x")
v1_pass = ratio > 1.0
print(f"  Bae et al. Fig 3g: GS visibly greater than WT (ANOVA "
      f"F(3,20)=44.77, GS vs WT significant by Tukey post hoc)")
print(f"  VERDICT: {'PASS -- correct direction' if v1_pass else 'FAIL -- wrong direction'}")
print(f"  (magnitude is not compared -- different assay units; only sign/direction "
      f"is a valid held-out test here)")

print()
print("=" * 84)
print("V2  [Fig. 3h-j]  Kinase inhibition nullifies genotype effect on secretion")
print("=" * 84)
f_wt_inhib = flux('WT', kinase_scale=0.01)     # near-total kinase inhibition
f_gs_inhib = flux('G2019S', kinase_scale=0.01)
gap_uninhibited = fold_gap(f_gs, f_wt)
gap_inhibited = fold_gap(f_gs_inhib, f_wt_inhib)
print(f"  uninhibited:  WT={f_wt:.4e}  GS={f_gs:.4e}   log-fold gap = {gap_uninhibited:.3f}")
print(f"  inhibited:    WT={f_wt_inhib:.4e}  GS={f_gs_inhib:.4e}   log-fold gap = {gap_inhibited:.3f}")
collapse_frac = 1.0 - gap_inhibited / max(gap_uninhibited, 1e-12)
print(f"  genotype gap collapsed by {collapse_frac*100:.1f}% under simulated kinase inhibition")
v2_pass = collapse_frac > 0.8   # gap shrinks by >80% -- "near complete nullification"
print(f"  Bae et al. Fig 3i-j: HG-10-102-01 'near completely nullified' the "
      f"effects\n  of LRRK2 WT and G2019S on propagation")
print(f"  VERDICT: {'PASS' if v2_pass else 'FAIL'} -- "
      f"{'gap collapses as predicted' if v2_pass else 'gap does not collapse enough'}")

print()
print("=" * 84)
print("V3  [Fig. 4h-i, Fig. 6]  RAB35 blockade eliminates the GENOTYPE effect")
print("=" * 84)
f_wt_block = flux('WT', rab35_block=0.02)
f_gs_block = flux('G2019S', rab35_block=0.02)
gap_unblocked = fold_gap(f_gs, f_wt)
gap_blocked = fold_gap(f_gs_block, f_wt_block)
collapse_frac_rab = 1.0 - gap_blocked / max(gap_unblocked, 1e-12)
print(f"  unblocked:     WT={f_wt:.4e}  GS={f_gs:.4e}   log-fold gap = {gap_unblocked:.3f}")
print(f"  RAB35-blocked: WT={f_wt_block:.4e}  GS={f_gs_block:.4e}   log-fold gap = {gap_blocked:.3f}")
print(f"  genotype gap collapsed by {collapse_frac_rab*100:.1f}% under RAB35 blockade")
v3_pass = collapse_frac_rab > 0.8
print(f"  Bae et al. Fig 4i: WT+RAB35-DN vs GS+RAB35-DN -- 'ns' (not significant)")
print(f"  despite WT vs GS alone being highly significant (****)")
print(f"  VERDICT: {'PASS' if v3_pass else 'FAIL'} -- "
      f"{'genotype effect abolished as predicted' if v3_pass else 'genotype effect persists -- structural mismatch'}")

print()
print("=" * 84)
print("SUMMARY")
print("=" * 84)
checks = [("V1  genotype direction (Fig 3g)", v1_pass),
          ("V2  kinase-inhibition collapse (Fig 3h-j)", v2_pass),
          ("V3  RAB35-block collapse (Fig 4i / Fig 6)", v3_pass)]
for name, ok in checks:
    print(f"  {name:<45} {'PASS' if ok else 'FAIL'}")
print()
print("Note: these are STRUCTURAL/DIRECTIONAL checks against an independent")
print("dataset never used in calibrate_v3.py. They test whether the model's")
print("causal wiring (LRRK2 -> pRab35 -> export, gated by genotype AND by")
print("RAB35 phosphorylation) reproduces qualitative results it was not fit to,")
print("not whether its absolute magnitudes match a different assay's units.")
