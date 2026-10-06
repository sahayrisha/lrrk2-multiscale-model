"""
Network propogation model.

Analyse on the CUMULATIVE HAZARD scale.
 
    Lambda = -ln(1 - A)      where A is the attack rate
 
This maps [0, 1) onto [0, infinity). It is the standard unbounded transform of
a proportion and is exactly the scale on which survival analysis operates.
 
The justification is mechanistic, not cosmetic. If an intervention multiplies
the per-contact force of infection by a factor f, it multiplies the cumulative
hazard by f. Two independent interventions therefore multiply the hazard by
f1 * f2, which IS Bliss independence correctly applied to a rate. The attack
rate is a saturating transform of the hazard, so multiplicativity is preserved
on the hazard scale and destroyed on the attack-rate scale.

"""

import numpy as np
import networkx as nx
from scipy import stats
from scipy.optimize import brentq
import json
import warnings
warnings.filterwarnings('ignore')

# Scale 1/2 outputs from pd_model_v3 (exosomal secretion flux, uM/s)
EXO_FLUX = {
    'wildtype': 3.98523e-03,
    'control':  1.12882e-02,
    'rab35i':   8.28531e-03,
    'lamp2a':   7.22916e-03,
    'combined': 4.21295e-03,
}
REF = EXO_FLUX['control']

N_NEURONS   = 50
N_SEED      = 5
N_RUNS      = 200
OBS_DAYS    = 90
MEAN_DEGREE = 6
TARGET_CONTROL_ATTACK = 0.70
HILL_K, HILL_N = 1.0, 1.0


def transmission(flux, beta_scale):
    x = (flux / REF) ** HILL_N
    return beta_scale * x / (HILL_K ** HILL_N + x)


def simulate(flux, beta_scale, seed, days=OBS_DAYS):
    """Returns (cumulative curve, n_events, susceptible-days at risk)."""
    rng = np.random.default_rng(seed)
    g = nx.watts_strogatz_graph(n=N_NEURONS, k=MEAN_DEGREE, p=0.3, seed=seed)
    adj = [list(g.neighbors(i)) for i in range(N_NEURONS)]
    seeded = np.zeros(N_NEURONS, dtype=bool)
    seeded[rng.choice(N_NEURONS, N_SEED, replace=False)] = True
    beta = transmission(flux, beta_scale)
    curve = np.empty(days, dtype=int)
    events = 0
    at_risk_days = 0
    for t in range(days):
        at_risk_days += int((~seeded).sum())
        newly = np.zeros(N_NEURONS, dtype=bool)
        for i in np.flatnonzero(seeded):
            for j in adj[i]:
                if not seeded[j] and rng.random() < beta:
                    newly[j] = True
        events += int(newly.sum())
        seeded |= newly
        curve[t] = seeded.sum()
    return curve, events, at_risk_days


def incidence_rate(flux, beta_scale, n_runs=N_RUNS, seed0=0):
    """Events per susceptible-neuron-day. This is the force of infection: the
    quantity an intervention multiplies, and therefore the correct scale for
    a multiplicative (Bliss) combination analysis. Unlike attack rate and
    cumulative hazard it does not saturate, because it conditions on the
    population still at risk."""
    out = []
    for r in range(n_runs):
        _, ev, risk = simulate(flux, beta_scale, seed0 + r)
        out.append(ev / max(risk, 1))
    return np.array(out)


def attack_rate(flux, beta_scale, n_runs=N_RUNS, seed0=0):
    return np.array([simulate(flux, beta_scale, seed0 + r)[0][-1] / N_NEURONS
                     for r in range(n_runs)])


def cumulative_hazard(A, n=N_NEURONS):
    """Lambda = -ln(1 - A), with a continuity correction so that A = 1 maps to
    a finite value rather than infinity."""
    A = np.clip(A, 0.0, 1.0 - 0.5 / n)
    return -np.log(1.0 - A)


def calibrate_beta():
    def gap(bs):
        return attack_rate(REF, bs, n_runs=60, seed0=10_000).mean() - TARGET_CONTROL_ATTACK
    return brentq(gap, 1e-4, 0.95, xtol=1e-5)


def combo_index(vals):
    """vals: dict of condition -> array. Returns CI on whatever scale is given."""
    c = np.mean(vals['control'])
    fr, fl, fc = (np.mean(vals['rab35i']) / c, np.mean(vals['lamp2a']) / c,
                  np.mean(vals['combined']) / c)
    return dict(f_rab=fr, f_lamp=fl, f_comb=fc, expected=fr * fl,
                CI=fc / (fr * fl), excess_pp=((1 - fc) - (1 - fr * fl)) * 100)


def bootstrap_ci(a, b, n_boot=10_000, seed=1):
    rng = np.random.default_rng(seed)
    d = b - a
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    bs = d[idx].mean(axis=1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def holm(p):
    p = np.asarray(p); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(run, 1.0)
    return adj


if __name__ == '__main__':
    print("=" * 80)
    print("NETWORK PROPAGATION v3 -- unbounded endpoint  [item 19]")
    print("=" * 80)
    BS = calibrate_beta()
    print(f"  calibrated beta_scale = {BS:.5f}   (control attack rate "
          f"anchored to {TARGET_CONTROL_ATTACK:.2f})")

    A = {k: attack_rate(f, BS) for k, f in EXO_FLUX.items()}
    H = {k: cumulative_hazard(v) for k, v in A.items()}
    I = {k: incidence_rate(f, BS) for k, f in EXO_FLUX.items()}

    print(f"\n{'condition':<11}{'flux':>12}{'attack rate':>18}{'cumulative hazard':>22}")
    print("-" * 80)
    for k in EXO_FLUX:
        print(f"{k:<11}{EXO_FLUX[k]:>12.4e}{A[k].mean():>11.3f} +/- {A[k].std(ddof=1):.3f}"
              f"{H[k].mean():>15.3f} +/- {H[k].std(ddof=1):.3f}")

    print("\n" + "=" * 80)
    print("SYNERGY ON BOTH SCALES")
    print("=" * 80)
    for tag, vals in [("attack rate       (BOUNDED, v2 endpoint)", A),
                      ("cumulative hazard (unbounded but SATURATES)", H),
                      ("incidence rate    (force of infection, v3 endpoint)", I)]:
        ci = combo_index(vals)
        print(f"  {tag}")
        print(f"    f_rab={ci['f_rab']:.4f}  f_lamp={ci['f_lamp']:.4f}  "
              f"f_comb={ci['f_comb']:.4f}  expected={ci['expected']:.4f}")
        print(f"    CI = {ci['CI']:.4f}   Bliss excess = {ci['excess_pp']:+.2f} pp\n")

    print("=" * 80)
    print("REGIME INVARIANCE TEST -- the decisive comparison")
    print("=" * 80)
    print(f"{'beta_scale':>11}{'ctrl attack':>13}{'CI attack':>12}{'CI hazard':>12}"
          f"{'CI incidence':>14}")
    print("-" * 80)
    ci_att, ci_haz, ci_inc = [], [], []
    for m in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        bs = BS * m
        a = {k: attack_rate(f, bs, n_runs=80, seed0=5000) for k, f in EXO_FLUX.items()}
        h = {k: cumulative_hazard(v) for k, v in a.items()}
        i_ = {k: incidence_rate(f, bs, n_runs=80, seed0=5000) for k, f in EXO_FLUX.items()}
        ca, ch, cinc = combo_index(a)['CI'], combo_index(h)['CI'], combo_index(i_)['CI']
        ci_att.append(ca); ci_haz.append(ch); ci_inc.append(cinc)
        star = "  <-" if m == 1.0 else ""
        print(f"{bs:>11.5f}{a['control'].mean():>13.3f}{ca:>12.4f}{ch:>12.4f}"
              f"{cinc:>14.4f}{star}")
    print("-" * 80)
    for nm, arr in [("attack rate   ", ci_att), ("cumulative hazard", ci_haz),
                    ("INCIDENCE RATE", ci_inc)]:
        print(f"  {nm:<18} CI range {min(arr):.4f} to {max(arr):.4f}   "
              f"spread {max(arr)-min(arr):.4f}")

    print("\n" + "=" * 80)
    print("PAIRED WILCOXON ON INCIDENCE RATE (common random numbers)")
    print("=" * 80)
    arms = ['wildtype', 'rab35i', 'lamp2a', 'combined']
    raw = [stats.wilcoxon(I['control'], I[a], alternative='two-sided').pvalue for a in arms]
    adj = holm(raw)
    print(f"{'comparison':<28}{'incidence difference (95% CI)':>34}{'p Holm':>12}")
    print("-" * 80)
    for a, pa in zip(arms, adj):
        d, lo, hi = bootstrap_ci(I['control'], I[a])
        print(f"  control vs {a:<15}{d:>+11.5f} [{lo:+.5f}, {hi:+.5f}]{pa:>12.2e}")

    json.dump({'beta_scale': BS,
               'attack': {k: v.tolist() for k, v in A.items()},
               'hazard': {k: v.tolist() for k, v in H.items()},
               'incidence': {k: v.tolist() for k, v in I.items()},
               'ci_incidence_sweep': ci_inc,
               'ci_attack_sweep': ci_att, 'ci_hazard_sweep': ci_haz},
              open('network_results_v3.json', 'w'), indent=2)
    print("\n  saved -> network_results_v3.json")
    print("\n To generate network scale figures, run: generate_figures_network.py")
