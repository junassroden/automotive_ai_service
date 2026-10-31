# Model card: service_need_v3_3a435be143

**Status:** evaluated_rejected  
**Task:** 1 = the service item was performed at the recorded workshop visit. Scheduled-service need, not an observed or predicted failure.

## Data

- Source: `Vehicle Maintenance- Service Records.csv` (sha256 `7b78521108206474…`)
- Rows read / used: 1139 / 1139; rejected: none
- Train groups: city|chennai, city|mumbai, fortuner|mumbai, jazz|chennai, jazz|mumbai
- Held-out test groups: amaze|chennai, amaze|mumbai

## Held-out results (served targets, macro)

Served estimator: milestone_frequency (training-group service frequency at the nearest 10,000 km milestone, smoothed towards prevalence).

| Metric | Served estimator | Best ML candidate | Prevalence baseline |
|---|---:|---:|---:|
| Balanced accuracy | 0.7429 | 0.7251 | 0.5 |
| F1 | 0.5364 | 0.4927 | 0.0 |
| PR-AUC | 0.6312 | 0.6871 | 0.2448 |
| Brier | 0.1481 | 0.1392 | 0.1788 |
| ECE | 0.1557 | | |

Best ML candidate: logistic_regression {'C': 0.1, 'feature_set': 'full'} (calibrated).

## Per target (held-out groups)

| Target | Served | Test positives | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier | Best ML F1 | Inner-CV ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| washer_plug_drain | no | 291 | 0.9065 | 1.0 | 0.951 | 0.9965 | 0.9811 | 0.0631 | 0.951 | 0.1798 |
| dust_and_pollen_filter | yes | 0 | 0.0 | 0.0 | 0.0 | None | None | 0.3016 | 0.0 | 0.8694 |
| whell_alignment_and_balancing | no | 88 | 0.2897 | 0.7045 | 0.4106 | 0.4797 | 0.6497 | 0.1876 | 0.4324 | 0.4415 |
| air_clean_filter | yes | 142 | 0.9324 | 0.9718 | 0.9517 | 0.9784 | 0.9817 | 0.075 | 0.7874 | 0.8235 |
| fuel_filter | yes | 105 | 0.529 | 0.781 | 0.6308 | 0.5378 | 0.7021 | 0.1872 | 0.6961 | 0.8088 |
| spark_plug | no | 35 | 0.0 | 0.0 | 0.0 | 0.5 | 0.9337 | 0.0792 | 0.0 | 0.0794 |
| brake_fluid | yes | 52 | 1.0 | 0.75 | 0.8571 | 0.8761 | 0.9536 | 0.0447 | 0.8696 | 0.8962 |
| brake_and_clutch_oil | yes | 91 | 0.4286 | 0.2308 | 0.3 | 0.4146 | 0.7113 | 0.2488 | 0.0 | 0.8502 |
| transmission_fluid | yes | 53 | 1.0 | 0.7358 | 0.8478 | 0.8528 | 0.9453 | 0.0416 | 0.809 | 0.9622 |
| brake_pads | yes | 61 | 0.2857 | 0.1967 | 0.233 | 0.3382 | 0.7438 | 0.1695 | 0.2913 | 0.7535 |
| clutch | yes | 46 | 0.5128 | 0.4348 | 0.4706 | 0.4205 | 0.85 | 0.1161 | 0.4878 | 0.8607 |
| coolant | no | 8 | 0.08 | 0.75 | 0.1446 | 0.2354 | 0.8323 | 0.0288 | 0.0521 | 0.5153 |

## Design history

v1 (one calibrated ML estimator for all targets) and v2 (per-target ML vs milestone-frequency choice plus feature-set ablation on inner folds) were both rejected: on the held-out Honda Amaze groups the ML models did not beat the milestone-frequency baseline (v1 macro F1 0.730 vs 0.752; v2 0.724 vs 0.752; see data/reports/service_need_v1_rejected_evaluation.json and service_need_v2_rejected_evaluation.json). Both also chose served targets from held-out metrics, which biases their aggregates upward. v3 serves the milestone-frequency estimator, declared as the baseline before any held-out result was seen, and chooses served targets from inner training-group folds only. The decision to deploy it instead of an ML model was made after seeing v1 and v2, so a fresh validation on new fleet service records is still required.


## Limitations

- Four car models from two Indian cities, 2016-2018; no motorcycles, pickups, vans or minivans.
- Targets are workshop service decisions, which largely follow manufacturer schedules; they are not wear measurements or failures.
- No visit dates or vehicle IDs, so no service-window or failure-date estimate can be validated.
- Vehicles outside the training domain (body type, fuel type, odometer range) are refused rather than extrapolated.
