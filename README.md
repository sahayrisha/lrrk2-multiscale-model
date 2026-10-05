\# Mechanistic Multiscale Modeling of Rab35 Inhibition and LAMP-2A Enhancement for Combination Therapy in LRRK2-Linked Parkinson's Disease



\## Overview



Parkinson’s disease progression is associated with the cell-to-cell propagation of misfolded α-synuclein, with the disease-linked LRRK2 G2019S mutation affecting pathways involved in α-synuclein handling. This project develops a mechanistic multiscale computational model of α-synuclein dynamics in LRRK2-linked Parkinson’s disease, with a focus on evaluating combination therapeutic strategies.



The model examines two complementary intervention targets: Rab35-mediated exosomal export of α-synuclein aggregates and LAMP-2A-mediated chaperone-mediated autophagy (CMA). A molecular-scale model of α-synuclein handling is coupled to a network-level model of neuronal propagation, allowing changes in intracellular aggregate dynamics to be connected to their potential effects on cell-to-cell transmission.





\## Research Objective



The primary objective of this project is to investigate whether simultaneous targeting of α-synuclein export and intracellular clearance can produce a synergistic reduction in the pool of seeding-competent α-synuclein aggregates in LRRK2 G2019S neurons.



Specifically, the model is designed to evaluate:



* the effects of Rab35 inhibition and LAMP-2A enhancement individually and in combination;
* whether the combined intervention produces greater-than-additive effects(synergy defined as Bliss combination index < 0.9);
* how changes in the molecular aggregate pool influence predicted neuronal propagation; and
* whether therapeutic predictions remain robust across uncertainty in model parameters.





\## Model Architecture



\### Modular Framework



This model was designed with a modular framework with a separate model at each scale coupled via offline sequential coupling. This enables construction, modification, and validation of each scale independently without affecting the structure or results of the other scales. Sequential validation before coupling creates a cleaner pipeline for debugging and identifying errors before running the entire system. Because of modular framework, the software for each scale can be updated or replaced without changing other scales. 



\### Molecular Scale



The molecular scale models interactions between intracellular components involved in pathological alpha-synuclein clearance and export in cell-to-cell propogation, specifically focusing on the axis of hyperactive LRRK2-mediated effects on Rab35-mediated exosomal export and chaperone-mediated intracellular clearance in G2019S genotypes. Key outputs include aggregate fold change across conditions and exosomal secretion fluxes. Exosomal secretion fluxes are passed up to Network Scale as input. 



\### Network Scale



The network scale models transmission of pathological alpha-synuclein seeding-competent aggregates between neurons. Primary metric is incidence rate, using cumulative hazard function -ln(1-A) where A is percentage of seeded cells. 

