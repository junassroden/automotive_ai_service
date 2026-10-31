# Model card: service_need_v2_e1961052d4

**Status:** evaluated_rejected  
**Task:** 1 = the service item was performed at the recorded workshop visit. Scheduled-service need, not an observed or predicted failure.

## Data

- Source: `Vehicle Maintenance- Service Records.csv` (sha256 `7b78521108206474…`)
- Rows read / used: 1139 / 1139; rejected: none
- Train groups: city|chennai, city|mumbai, fortuner|mumbai, jazz|chennai, jazz|mumbai
- Held-out test groups: amaze|chennai, amaze|mumbai

## Held-out results (served targets, macro)

| Metric | Model | Milestone-lookup baseline |
|---|---:|---:|
| Balanced accuracy | 0.7979 | 0.821 |
| F1 | 0.724 | 0.7516 |
| PR-AUC | 0.7651 | 0.7331 |
| Brier | 0.0985 | prevalence: 0.1749 |
| ECE | 0.0942 | |

## Per target

| Target | Served | Test positives | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier | Baseline F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| washer_plug_drain | no | 291 | 0.9065 | 1.0 | 0.951 | 0.7592 | 0.0105 | 0.0996 | 0.951 |
| dust_and_pollen_filter | no | 0 | 0.0 | 0.0 | 0.0 | None | None | 0.2371 | 0.0 |
| whell_alignment_and_balancing | no | 88 | 0.2759 | 1.0 | 0.4324 | 0.1889 | 0.2826 | 0.2173 | 0.4106 |
| air_clean_filter | yes | 142 | 0.8929 | 0.7042 | 0.7874 | 0.9276 | 0.9514 | 0.1236 | 0.9517 |
| fuel_filter | yes | 105 | 0.7172 | 0.6762 | 0.6961 | 0.7375 | 0.8011 | 0.1616 | 0.6308 |
| spark_plug | no | 35 | 0.0 | 0.0 | 0.0 | 0.0583 | 0.007 | 0.109 | 0.0 |
| brake_fluid | yes | 52 | 1.0 | 0.75 | 0.8571 | 0.8761 | 0.9536 | 0.0633 | 0.8571 |
| brake_and_clutch_oil | no | 91 | 0.0 | 0.0 | 0.0 | 0.4992 | 0.8322 | 0.2582 | 0.3 |
| transmission_fluid | yes | 53 | 1.0 | 0.6792 | 0.809 | 0.8636 | 0.9225 | 0.0448 | 0.8478 |
| brake_pads | no | 61 | 0.3571 | 0.2459 | 0.2913 | 0.3519 | 0.7906 | 0.1445 | 0.233 |
| clutch | yes | 46 | 0.5128 | 0.4348 | 0.4706 | 0.4205 | 0.85 | 0.0992 | 0.4706 |
| coolant | no | 8 | 0.0513 | 0.75 | 0.096 | 0.2354 | 0.8323 | 0.0253 | 0.1446 |

## Limitations

- Four car models from two Indian cities, 2016-2018; no motorcycles, pickups, vans or minivans.
- Targets are workshop service decisions, which largely follow manufacturer schedules; they are not wear measurements or failures.
- No visit dates or vehicle IDs, so no service-window or failure-date estimate can be validated.
- Vehicles outside the training domain (body type, fuel type, odometer range) are refused rather than extrapolated.
