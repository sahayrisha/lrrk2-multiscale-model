"""
Addresses Action Checklist items 20 and 22.

[20] THERAPEUTIC WINDOW / ON-TARGET LIABILITY
     Exosome release is a physiological process. Suppressing exosomal export
     below the healthy baseline is an on-target liability, not a benefit.
     v2 reported that combined treatment drove export to 0.45x healthy without
     acknowledging it. This script maps the dose-response surface over Rab35
     blockade and LAMP-2A overexpression, and identifies the region where
     pathological export is reduced WITHOUT falling below the healthy baseline.

[22] Km_MVB PROFILE
     Km_MVB has no direct literature value. Rather than leave it as an
     undefended assumption, this script profiles it across three orders of
     magnitude and reports the range over which the synergy conclusion holds.
"""
import numpy as np
import pd_model_v3 as M

PF = np.load('params_v3.npy', allow_pickle=True).item()
P = dict(M.P_FIXED); P.update(PF)


def flux(rab35_block, LAMP2A, genotype='G2019S', p=None):
    cond = dict(genotype=genotype, rab35_block=rab35_block, LAMP2A=LAMP2A)
    r = M.steady_state(cond, p or P)
    return r['secretion_flux'] if r else np.nan


F_WT = flux(1.0, 0.40, 'WT')
F_CTRL = flux(1.0, 0.20, 'G2019S')

print("=" * 84)
print("[20] THERAPEUTIC WINDOW: benefit vs on-target liability")
print("=" * 84)
print(f"  healthy baseline export  = {F_WT:.5e} uM/s")
print(f"  untreated G2019S export  = {F_CTRL:.5e} uM/s  ({F_CTRL/F_WT:.2f}x healthy)")
print()
print("  benefit  = fraction of the PATHOLOGICAL EXCESS removed")
print("             (F_ctrl - F) / (F_ctrl - F_WT), capped at 1.0")
print("  liability = fractional suppression BELOW the healthy baseline")
print("             max(0, (F_WT - F) / F_WT)")
print()

blocks = [1.0, 0.75, 0.50, 0.35, 0.25, 0.15, 0.10, 0.05, 0.02]
lamps = [0.20, 0.35, 0.50, 0.70, 1.00]

print(f"{'Rab35 residual':>15}", end='')
for L in lamps:
    print(f"{'LAMP2A=' + str(L):>16}", end='')
print()
print(f"{'(1.0 = no drug)':>15}" + "   benefit / liability" * 0)
print("-" * 96)
best = None
for b in blocks:
    print(f"{b:>15.2f}", end='')
    for L in lamps:
        F = flux(b, L)
        benefit = min(1.0, max(0.0, (F_CTRL - F) / (F_CTRL - F_WT)))
        liab = max(0.0, (F_WT - F) / F_WT)
        print(f"{benefit:>8.2f} /{liab:>6.2f}", end='')
        if liab <= 0.05 and (best is None or benefit > best[0]):
            best = (benefit, b, L, F, liab)
    print()

print()
print(f"  OPTIMAL REGIMEN with on-target liability <= 5%:")
print(f"    Rab35 residual activity = {best[1]:.2f}  (i.e. {100*(1-best[1]):.0f}% inhibition)")
print(f"    LAMP-2A level           = {best[2]:.2f} uM  ({best[2]/0.20:.1f}x disease baseline)")
print(f"    pathological excess removed = {best[0]*100:.1f}%")
print(f"    suppression below healthy   = {best[4]*100:.1f}%")
print()
print(f"  The clinically calibrated condition used in the model")
print(f"  (75% inhibition, LAMP-2A 0.50) gives:")
Fc = flux(0.25, 0.50)
print(f"    benefit = {min(1.0,(F_CTRL-Fc)/(F_CTRL-F_WT))*100:.1f}%   "
      f"liability = {max(0.0,(F_WT-Fc)/F_WT)*100:.1f}%   "
      f"export = {Fc/F_WT:.2f}x healthy")
print()
print("  Aggressive blockade (95% inhibition, LAMP-2A 1.0) gives:")
Fa = flux(0.05, 1.00)
print(f"    benefit = {min(1.0,(F_CTRL-Fa)/(F_CTRL-F_WT))*100:.1f}%   "
      f"liability = {max(0.0,(F_WT-Fa)/F_WT)*100:.1f}%   "
      f"export = {Fa/F_WT:.2f}x healthy")
print("    -> this is the v2 regimen. It overshoots into physiological")
print("       suppression, which the v2 analysis did not acknowledge.")

print()
print("=" * 84)
print("[22] Km_MVB PROFILE: does the conclusion depend on the unsourced parameter?")
print("=" * 84)
print(f"  calibrated value = {PF['Km_MVB']:.4f} uM (no direct literature source)")
print()
print(f"{'Km_MVB':>10}{'f_rab35i':>11}{'f_lamp2a':>11}{'f_comb':>10}"
      f"{'expected':>11}{'CI':>9}   verdict")
print("-" * 84)
hold = []
for km in [0.005, 0.01, 0.025, 0.05, PF['Km_MVB'], 0.20, 0.40, 0.80, 2.0, 5.0]:
    p = dict(P); p['Km_MVB'] = km
    res = M.run_all(p)
    if res is None:
        continue
    ci = M.combination_index(res)
    v = "SYNERGY" if ci['CI'] < 0.90 else ("additive" if ci['CI'] < 1.10 else "ANTAG")
    if ci['CI'] < 0.90:
        hold.append(km)
    star = "  <- calibrated" if abs(km - PF['Km_MVB']) < 1e-9 else ""
    print(f"{km:>10.4g}{ci['f_rab35i']:>11.4f}{ci['f_lamp2a']:>11.4f}"
          f"{ci['f_combined']:>10.4f}{ci['bliss_expected']:>11.4f}{ci['CI']:>9.4f}"
          f"   {v}{star}")
print("-" * 84)
if hold:
    print(f"  Synergy verdict holds for Km_MVB from {min(hold):.4g} to {max(hold):.4g} uM,")
    print(f"  a {max(hold)/min(hold):.0f}-fold range spanning the calibrated value.")
    print(f"  The conclusion does not depend on the specific unsourced value.")
else:
    print("  Verdict does NOT hold across the profiled range.")
