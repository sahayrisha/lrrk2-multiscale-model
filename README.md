\# Rab35/LAMP-2A Synergy in LRRK2 G2019S Parkinson's Disease: A Mechanistic, Identifiability-Constrained Model



\## Overview



Parkinson’s disease progression is associated with the cell-to-cell propagation of misfolded α-synuclein, with the disease-linked LRRK2 G2019S mutation affecting pathways involved in α-synuclein handling. This project develops a mechanistic multiscale computational model of α-synuclein dynamics in LRRK2-linked Parkinson’s disease, with a focus on evaluating combination therapeutic strategies.



This repository contains a mechanistic computational model testing whether combined Rab35 pathway blockade and LAMP-2A overexpression act synergistically to reduce pathological α-synuclein secretion in LRRK2 G2019S Parkinson's disease neurons, relative to either intervention alone.  A molecular-scale model of α-synuclein handling is coupled to a network-level model of neuronal propagation, allowing changes in intracellular aggregate dynamics to be connected to their potential effects on cell-to-cell transmission.



The model is calibrated against literature-derived steady-state biomarker ratios, explicitly partitioned into identifiable and non-identifiable parameters, validated against multiple independent (held-out) published datasets, and uncertainty-quantified via profile likelihood and ensemble sampling. 



\---



\## Headline Result



\*\*Combined Rab35 blockade + LAMP-2A overexpression is synergistic\*\*, not merely additive, in reducing pathological α-synuclein secretion:



\- Bliss combination index (CI) = \*\*0.81\*\*, 90% uncertainty interval \*\*\[0.78, 0.87]\*\*

\- \*\*99.7%\*\* of parameter sets consistent with the calibration data predict synergy (CI < 0.90), despite substantial uncertainty in individual parameter values

\- A therapeutic window exists in which pathological export is reduced without suppressing exosome secretion below the healthy physiological baseline



\---



\## Model Architecture



\### Modular Framework



This model was designed as a modular multiscale framework, with a separate computational model at each scale coupled through offline sequential coupling. This enables each scale to be constructed, modified, and validated independently before integration into the full system. Sequential validation provides a structured pipeline for debugging and identifying errors within individual model components before coupling. The modular structure also allows the software implementation of each scale to be updated or replaced without requiring changes to the other scales.



\### Molecular Scale



The molecular scale models intracellular processes involved in pathological α-synuclein clearance and export, with a focus on the effects of hyperactive LRRK2 signaling on Rab35-mediated exosomal export and chaperone-mediated intracellular clearance in the G2019S genotype. The model represents the dynamics of soluble and aggregated α-synuclein, intracellular trafficking, and exosomal export.



Key outputs include aggregate fold change across experimental conditions and exosomal α-synuclein secretion flux. Exosomal secretion fluxes are passed to the network scale as inputs governing neuronal propagation.



\### Network Scale



The network scale models the transmission of pathological, seeding-competent α-synuclein aggregates between neurons. Molecular-scale exosomal secretion is used to determine the probability of neuronal transmission, allowing changes in intracellular aggregate handling to be translated into predicted effects on cell-to-cell propagation.



The primary network outcome is cumulative neuronal seeding, represented by the proportion of cells that become seeded over the simulation. This is additionally expressed as a cumulative hazard using:



$$

H = -\\ln(1-A)

$$



where \\(A\\) is the proportion of seeded cells.



\### Model Specification



\* \*\*6-state molecular ODE system:\*\* The molecular model represents the dynamics of phosphorylated Rab35, phosphorylated Rab10, soluble α-synuclein, aggregated α-synuclein, the multivesicular body (MVB) pool, and exosomal α-synuclein.

\* \*\*5 experimental conditions:\*\* wildtype, G2019S control, Rab35 blockade, LAMP-2A overexpression, and combined treatment.

\* \*\*12 free parameters: The final model contains 12 calibrated free parameters, with remaining parameters fixed to literature-derived values or treated as structurally unconstrainable nuisance parameters. Identifiability was assessed using Fisher information and sensitivity-matrix rank analysis against measurable model observables.



\---



\## Calibration



This model was calibrated against 6 steady-state checkpoints (T1–T6) derived from independent literature sources (see `Model\_Parameters\_v3.xlsx`). T1–T4 are quantitative fold-change targets; T5–T6 are plausibility/inequality constraints (not point measurements), implemented as soft penalty terms in the optimization cost function rather than point targets.



\---



\## Held-out Validation



Validated against experimental data \*\*never used in calibration\*\*, testing direction and (where units are comparable) magnitude:



| Check | Source | Result |

|---|---|---|

| V1 | Bae et al. 2018 (\*Nat Commun\*) — genotype direction | \*\*PASS\*\* |

| V2 | Bae et al. 2018 — kinase-inhibition collapse | FAIL (structural, see below) |

| V3 | Bae et al. 2018 — RAB35-blockade collapse | FAIL (structural, see below) |

| V4 | Fan et al. 2018 (\*Biochem J\*) — dephosphorylation timescale | Resolved (see Phase 0) |

| V5 | Xilouri et al. 2013 (\*Brain\*) — LAMP2A overexpression | \*\*PASS\*\* |

| V6 | Xilouri et al. 2016 (\*Autophagy\*) — LAMP2A knockdown | \*\*PASS\*\* |

| V7 | Network-level propagation (genotype comparison) | \*\*PASS\*\* |



V5/V6 together form a clean bidirectional test of the clearance/CMA axis and are the strongest validation result. V1 provides secondary support on the export axis.



\---



\## Uncertainty quantification



Uncertainty quantification is performed via profile likelihood (parameter-level) and ensemble rejection sampling(prediction-level). Of 12 free parameters, only `g2019s\_prod` is identifiable from current data (median 1.66, 90% CI \[0.51, 3.60]); the remaining 11 are flat or one-sided, consistent with having only 4 quantitative calibration checkpoints against 12 parameters. Despite this, the synergy conclusion and the validation limitation sturctural finding(See \*\*Known Limitations\*\*) are both robust across the full uncertainty ensemble.



\---



\## Repository contents



\*\*Core model\*\*

\- `pd\_model\_v3.py` — the ODE model, steady-state solver, time-course extension, graph generation. 

\- `calibrate\_v3.py` — calibration via differential evolution; identifiability rank check via sensitivity-matrix SVD.

\- `params\_v3.npy` — final calibrated parameter set.

\- `network\_v3.py` — network-level propagation model (incidence-rate endpoint).

\- `therapeutic\_window.py` — benefit-vs-liability dose-response analysis, Km\_MVB robustness profile.

\- `SBML\_v3\_control.xml` — the SBML-formatted moelcular-scale ODE model defining LRRK2-Rab35 axis



\*\*Held-out validation scripts\*\*

\- `validate\_bae2018.py` — V1/V2/V3 (see \*\*Held-out Validation\*\*)

\- `validate\_fan2018\_timecourse.py` — V4 (see \*\*Held-out Validation\*\*)

\- `validate\_xilouri\_lamp2a.py` — V5/V6 (see \*\*Held-out Validation\*\*)

\- `validate\_network\_level.py` — V7 (see \*\*Held-out Validation\*\*)



\*\*Structural diagnosis scripts (evidence trail for the validation strucutral findings, see Known Limitations)\*\*

\- `diagnose\_validation\_failure\_structural\_cause.py` — runs 3 checks to test hypotheses for failure of V2/V3 validation checkpoints(see \*\*Held-out Validation\*\*)

\- `diagnose\_gap\_decomposition.py` — stage-wise decomposition isolating where the genotype gap enters the pipeline

\- `diagnose\_isogenic\_comparison.py` — controls for a condition-definition (LAMP2A) confound identified during diagnosis

\- `demonstrate\_floor\_frac\_mechanism.py` — reproducible demonstration of the candidate structural fix's partial effect

\- `check\_floor\_frac\_identifiability.py` — rank check showing `floor\_frac` is not identifiable from current data



\*\*Uncertainty Quantification\*\*

\- `profile\_and\_uq.py` — uncertainty quantification (profile likelihood + ensemble).

\- `uq\_output` — results and figures produced by `profile\_and\_uq.py`.



\*\*Figures\*\*

\- `generate\_figures\_molecular.py` — generates `Fig1\_v3\_steady\_states.png`, `Fig2\_v3\_synergy.png`, `Fig3\_v3\_therapeutic\_window.png`, and `Fig4\_v3\_identifiability.png`

\- `generate\_figures\_network.py` — generates `Fig5\_v3\_network\_synergy.png` and `Fig6\_v3\_network\_wilcoxon.png

\- `time\_course\_figures.py` —

\- `Fig1\_v3\_steady\_states.png` — moecular steady states from `pd\_model\_v3.py`

\- `Fig2\_v3\_synergy.png` — comparison of Bliss CI for reduction in exosomal flux(%) across independent interventions, expected CI, and observed CI with combined interventions

\- `Fig3\_v3\_therapeutic\_window.png` — therapeutic benefit and on-target liability across varying Rab35 inhibition(%) and LAMP-2A overexpression

\- `Fig4\_v3\_identifiability.png` — full parameter rank identifiability and Km\_MVB(unsourced parameter) profile, synergy conclusion holds over 40-fold range

\- `Fig5\_v3\_network\_synergy.png` — comparison of Bliss CI for reduction in incidence rate(%) across independent interventions, expected CI, and observed CI with combined interventions

\- `Fig6\_v3\_network\_wilcoxon.png` — forest plot of Paired Wilcoxon test results on incidence rate (common random numbers, Holm-corrected p-value) of control vs each condition





\*\*Documentation\*\*

\- Model\_Parameters\_v3.xlsx — full parameter table with literature sources, calibration checkpoint definitions, and bounds



\---



\## How to reproduce



```bash

\# 1. Calibrate (or use the provided params\_v3.npy)

python3 calibrate\_v3.py



\# 2. Run the core model, generate time-course graphs

python3 pd\_model\_v3.py



\# 3. Run held-out validation

python3 validate\_bae2018.py

python3 validate\_fan2018\_timecourse.py

python3 validate\_xilouri\_lamp2a.py



\# 4. Run structural diagnosis (evidence for the V2/V3 finding, see \*\*Known Limitations\*\*)

python3 diagnose\_gap\_decomposition.py

python3 diagnose\_isogenic\_comparison.py

python3 demonstrate\_floor\_frac\_mechanism.py

python3 check\_floor\_frac\_identifiability.py



\# 5. Run uncertainty quantification

python3 profile\_and\_uq.py            # full run, \~10-20 min

python3 profile\_and\_uq.py --quick    # coarse run, \~1-2 min

```



\---



\## Known Limitations 



\- Calibration constraints: Only 4 of 6 calibration checkpoints (T1–T4) have point estimates with estimable uncertainty; T5–T6 are implemented as inequality constraints. Uncertainty quantification indicates that 11 of the 12 free parameters are individually non-identifiable from the current data. However, the primary synergy conclusion is robust across the accepted parameter uncertainty ensemble, with 99.7% of accepted parameter sets predicting synergy.



\- Structural validation limitation: Held-out validation revealed that the final model cannot reproduce the collapse of the genotype-dependent secretion difference under LRRK2 kinase inhibition or RAB35 blockade reported by Bae et al. (2018)(Checkpoints V2 and V3, see \*\*Held-out Validation\*\*). The discrepancy persists across the full accepted parameter uncertainty ensemble, indicating that the finding is robust to parameter uncertainty rather than being a parameter-fitting artifact. Analytically, the current intervention formulation provides no genotype-independent route to secretion that can close this genotype-dependent gap. A candidate structural extension, floor\_frac, was implemented to represent a RAB35-independent, ESCRT-dependent contribution to MVB biogenesis and exosomal secretion. This extension reduced the intervention-induced gap amplification but did not fully resolve the validation discrepancy. floor\_frac was not adopted in the final model because it was found to be non-identifiable from the current experimental conditions and measurable observables, and no independent literature value was available to constrain it. The discrepancy is therefore retained as an unresolved structural limitation and a target for future model development.



\- Deferred tissue-level modeling: Scale 4, representing tissue-level neurodegeneration using PhysiCell, was deferred because no sufficiently supported, citable dose-dependent stress/death mechanism was identified to parameterize the model.



\- Held-out validation: Validation against Bae et al., Fan et al., and Xilouri et al. is interpreted primarily at the level of directionality and order of magnitude, as the experimental datasets use different assay types and measurement units that do not support direct numerical comparison.

