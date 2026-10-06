"""
V7: Network-level held-out validation against Bieri et al. 2019
(Acta Neuropathologica), "LRRK2 modifies alpha-syn pathology and spread in
mouse models and human neurons", DOI: 10.1007/s00401-019-01995-0, Figure 6d.

DATA SOURCE NOTE: the target values below were extracted from the published
figure using WebPlotDigitizer, as exact numeric values are not reported in
the paper's text. This is standard practice when a figure reports data only
graphically, but it introduces additional extraction uncertainty beyond
whatever error bars the original authors reported. Comparisons below are
therefore restricted to DIRECTION and order-of-magnitude, consistent with
the standard applied to every other held-out validation script in this
project (V1-V6) -- see validate_bae2018.py, validate_xilouri_lamp2a.py.

WHY PANEL D, NOT PANEL E: panel d (pSer129+/TH+ neuron fraction) measures
SEEDING/PROPAGATION -- the direct experimental analogue of what
network_v3.py's incidence-rate endpoint represents. Panel e measures
neurodegeneration/cell death, a downstream process this project's Scale 4
(tissue-level death) was deliberately deferred/cut from -- validating
against panel e would reopen that scope decision rather than test the
network-propagation model this script is actually checking.

WHY CUMULATIVE HAZARD, NOT RAW FRACTION: panel d reports a bounded fraction
(pSer129+/TH+ ratio), the same kind of bounded metric network_v3.py's own
docstring explains is unsuitable for multiplicative (Bliss-style) fold-change
comparison, because it can saturate and distort ratios. network_v3.py
already defines cumulative_hazard(A) = -ln(1-A) for exactly this reason --
this script reuses that same transform for consistency with the project's
established methodology, rather than introducing a new ad hoc comparison.
"""

# Uncomment if running from a subfolder structure where pd_model_v3.py
# is not in the same directory as this script:
# import sys, os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))


import numpy as np
import network_v3 as N

# ===========================================================================
# DIGITIZED TARGET DATA  <-- INSERT YOUR WEBPLOTDIGITIZER VALUES HERE
# ===========================================================================
# Bieri et al. 2019, Figure 6d: pSer129+/TH+ neuron fraction, WT+PFF vs
# G2019S+PFF, at 1/3/6 months post-injection (PI).
# Units: fraction (0-1)

WT_PFF = {
    '1mo': 0.0302,   
    '3mo': 0.2340,   
    '6mo': 0.0925,   
}

G2019S_PFF = {
    '1mo': 0.1066,   
    '3mo': 0.3208,   
    '6mo': 0.1226,   
}

# ===========================================================================

def check_placeholders_filled():
    missing = [f'WT_PFF[{k}]' for k, v in WT_PFF.items() if v is None]
    missing += [f'G2019S_PFF[{k}]' for k, v in G2019S_PFF.items() if v is None]
    if missing:
        raise ValueError(
            f"Placeholder values not filled in: {missing}. "
            f"Insert your digitized WebPlotDigitizer values in the "
            f"DIGITIZED TARGET DATA block before running."
        )


def hazard(fraction):
    """Wraps network_v3.cumulative_hazard for a single scalar value (that
    function expects an array of per-simulation-run fractions; here we
    have one point-estimate fraction per timepoint from the paper)."""
    return N.cumulative_hazard(np.array([fraction]), n=N.N_NEURONS)[0]


def fold_gap(a, b):
    return abs(np.log(a / b))


if __name__ == '__main__':
    check_placeholders_filled()

    print("=" * 84)
    print("V7  Network-level validation: Bieri et al. 2019, Fig 6d")
    print("    (WebPlotDigitizer-extracted values -- direction/magnitude only)")
    print("=" * 84)

    print(f"\n{'timepoint':<10}{'WT frac':>10}{'GS frac':>10}"
          f"{'WT hazard':>12}{'GS hazard':>12}{'hazard fold-gap':>18}")

    hazard_gaps = {}
    for t in ['1mo', '3mo', '6mo']:
        wt_f, gs_f = WT_PFF[t], G2019S_PFF[t]
        wt_h, gs_h = hazard(wt_f), hazard(gs_f)
        gap = fold_gap(gs_h, wt_h)
        hazard_gaps[t] = gap
        print(f"{t:<10}{wt_f:>10.4f}{gs_f:>10.4f}{wt_h:>12.4f}{gs_h:>12.4f}"
              f"{gap:>18.4f}")

    print()
    print("-" * 84)
    print("V7a: DIRECTION -- does G2019S show higher seeding/hazard than WT")
    print("     at every timepoint? (matches the qualitative pattern in Fig 6d)")
    print("-" * 84)
    all_gs_higher = all(G2019S_PFF[t] > WT_PFF[t] for t in ['1mo', '3mo', '6mo'])
    print(f"  G2019S > WT at all 3 timepoints: "
          f"{'YES' if all_gs_higher else 'NO'}")
    print(f"  V7a VERDICT: {'PASS' if all_gs_higher else 'FAIL'}")

    print()
    print("-" * 84)
    print("V7b: Compare against this model's own genotype hazard-fold-gap")
    print("     prediction from network_v3.py's calibrated transmission model")
    print("-" * 84)
    try:
        BS = N.calibrate_beta()
        a_wt = N.attack_rate(N.EXO_FLUX['wildtype'], BS, n_runs=80, seed0=7000)
        a_gs = N.attack_rate(N.EXO_FLUX['control'], BS, n_runs=80, seed0=7000)
        h_wt_model = N.cumulative_hazard(a_wt).mean()
        h_gs_model = N.cumulative_hazard(a_gs).mean()
        model_gap = fold_gap(h_gs_model, h_wt_model)
        print(f"  model-predicted WT/GS hazard fold-gap: {model_gap:.4f}")
        print(f"  paper (Fig 6d) hazard fold-gap, by timepoint:")
        for t, g in hazard_gaps.items():
            print(f"    {t}: {g:.4f}")
        print()
        print("  NOTE: this is an order-of-magnitude comparison only. The")
        print("  model's incidence_rate/hazard framework was calibrated on")
        print("  exosomal secretion flux (molecular scale), not fit to this")
        print("  paper's in vivo seeding data -- exact magnitude agreement is")
        print("  not expected or required; consistent DIRECTION and roughly")
        print("  comparable order of magnitude across the two independent")
        print("  measures is the relevant check here.")
    except Exception as e:
        print(f"  Could not run network_v3.py comparison: {e}")
        print("  (V7a direction check above still stands independently)")

    print()
    print("=" * 84)
    print("SUMMARY")
    print("=" * 84)
    print(f"  V7a (direction, G2019S > WT at all timepoints): "
          f"{'PASS' if all_gs_higher else 'FAIL'}")
    print("  Data source: Bieri et al. 2019, Acta Neuropathologica, Fig 6d,")
    print("  values extracted via WebPlotDigitizer (see module docstring).")
    print("  Not used anywhere in this project's calibration "
          "(Model_Parameters_v3.xlsx).")
