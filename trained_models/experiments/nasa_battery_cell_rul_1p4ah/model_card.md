# Laboratory battery-cell RUL experiment

This experimental pipeline estimates remaining discharge cycles until the first positive measured capacity at or below 1.4 Ah in selected NASA laboratory Li-ion cell-aging sequences. It is not a vehicle battery-pack model, fleet maintenance predictor, safety monitor, or production-ready predictor.

Evaluation uses whole-cell groups held out from training. The dataset contains only three independent cells, with different discharge cutoff voltages. Other NASA cohorts with unresolved anomalous capacities were excluded. Results do not establish generalization to vehicle packs, other chemistries, operating environments, or real fleet usage. Do not integrate into the Django service or use for maintenance decisions.
