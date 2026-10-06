"""
Identifiability check for the new floor_frac parameter, BEFORE deciding
whether to pin it from literature or fit it via calibrate_v3.py.

Extends the existing sens_matrix/rank_of machinery (calibrate_v3.py) to a
13-parameter free set: the original 12 free parameters + floor_frac.
Checks whether this set is full rank against MEASURABLE observables only
(secretion_flux, pRab35, total_aSyn), the same standard the original 12
were held to.

This does NOT calibrate floor_frac to any target -- it only asks: IF we
tried to fit floor_frac alongside the existing 12, would the data (steady
states across the 5 existing CONDITIONS) contain enough independent
information to pin all 13 down uniquely? If not, no amount of fitting
effort will produce a trustworthy value, and fitting should not be
attempted before that is resolved (pin from literature instead, or find
an additional measurable observable / condition that breaks the
degeneracy).
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

# Use an arbitrary, non-degenerate floor_frac value for the LOCAL sensitivity
# check (identifiability is a local property of the Jacobian; 0.0 exactly
# would be a singular/degenerate point for a multiplicatively-entered
# parameter in some formulations, so evaluate away from the boundary).
FLOOR_FRAC_EVAL_POINT = 0.30

FREE_EXT = list(M.FREE_NAMES) + ['floor_frac']
MEASURABLE = ['secretion_flux', 'pRab35', 'total_aSyn']
ALL_OBS = ['secretion_flux', 'pRab35', 'aSyn', 'aggregate', 'MVB']


def full_params_ext(pf_ext):
    p = dict(M.P_FIXED)
    p.update(pf_ext)
    return p


def sens_matrix_ext(pf_ext, obs, eps=1e-4):
    """Same construction as calibrate_v3.py's sens_matrix, extended to the
    13-parameter set (FREE_EXT) including floor_frac."""
    p0 = full_params_ext(pf_ext)
    r0 = M.run_all(p0)
    base = np.array([np.log(max(r0[c][o], 1e-300))
                      for c in M.CONDITIONS for o in obs])
    S = np.zeros((len(base), len(FREE_EXT)))
    for j, k in enumerate(FREE_EXT):
        pp = dict(pf_ext)
        # floor_frac is bounded in [0,1); perturb multiplicatively same as
        # everything else, consistent with the original sens_matrix
        pp[k] = pf_ext[k] * (1 + eps)
        p = full_params_ext(pp)
        r = M.run_all(p)
        v = np.array([np.log(max(r[c][o], 1e-300))
                       for c in M.CONDITIONS for o in obs])
        S[:, j] = (v - base) / np.log(1 + eps)
    return S


def rank_of(S):
    s = np.linalg.svd(S, compute_uv=False)
    return int(np.sum(s > max(S.shape) * np.finfo(float).eps * s[0])), s


pf_ext = dict(PF)
pf_ext['floor_frac'] = FLOOR_FRAC_EVAL_POINT

print("=" * 84)
print(f"IDENTIFIABILITY CHECK: 13-parameter free set (12 original + floor_frac)")
print(f"Evaluated at floor_frac = {FLOOR_FRAC_EVAL_POINT} (arbitrary, non-degenerate point)")
print("=" * 84)

for tag, obs in [("all species (incl. unmeasurable)", ALL_OBS),
                  ("MEASURABLE observables only", MEASURABLE)]:
    S = sens_matrix_ext(pf_ext, obs)
    rk, sv = rank_of(S)
    verdict = "FULL RANK" if rk == len(FREE_EXT) else f"DEFICIENT by {len(FREE_EXT) - rk}"
    print(f"  {tag:<38} {S.shape[0]:>3} obs x {len(FREE_EXT)} par -> "
          f"rank {rk}/{len(FREE_EXT)}  {verdict}")

print()
S = sens_matrix_ext(pf_ext, MEASURABLE)
rk, sv = rank_of(S)
ev = sv ** 2
print(f"MEASURABLE-only singular values: {np.round(sv, 4)}")
print(f"FIM eigenvalue span: {np.log10(ev[0] / max(ev[-1], 1e-300)):.1f} decades")
print()

if rk == len(FREE_EXT):
    print("VERDICT: floor_frac IS identifiable alongside the existing 12 free")
    print("parameters, against measurable observables alone. Safe to add to")
    print("P_FREE_NOMINAL and fit via calibrate_v3.py -- no literature pin needed.")
else:
    print("VERDICT: the 13-parameter set is RANK-DEFICIENT. floor_frac trades")
    print("off against at least one other free parameter (most likely k_load,")
    print("per the earlier structural argument) and CANNOT be reliably fit from")
    print("the current 5 conditions x 3 measurable-observable data alone.")
    print()
    print("Next options, in order of preference:")
    print("  1. Pin floor_frac from literature (ESCRT/RAB35-independent MVB")
    print("     fraction estimate) and treat it as P_FIXED, not P_FREE.")
    print("  2. Find an additional measurable observable or a new condition")
    print("     (e.g. a partial-blockade level not currently in CONDITIONS)")
    print("     that breaks the specific degeneracy -- identify which pair")
    print("     of parameters share a near-null singular vector first.")
    print("  3. If neither is feasible in the current timeline, do not add")
    print("     floor_frac as a free parameter -- report the amplification")
    print("     finding (Phase 1) as the deliverable, without the fix.")

    # Identify which parameters are involved in the near-null direction(s)
    print()
    print("Diagnostic: contribution of each parameter to the smallest singular")
    print("vector(s) (large entries = involved in the unidentifiable direction):")
    U, s_vals, Vt = np.linalg.svd(S, full_matrices=False)
    n_deficient = len(FREE_EXT) - rk
    for i in range(max(1, n_deficient)):
        vec = Vt[-(i+1)]
        order = np.argsort(-np.abs(vec))
        contribs = ", ".join(f"{FREE_EXT[j]}={vec[j]:+.3f}" for j in order[:4])
        print(f"  null direction {i+1} (sv={s_vals[-(i+1)]:.2e}): {contribs}")
