# Dataset manifest

Generated 2026-10-31T09:01:02.629872+00:00 by `python -m data_pipeline.run_all`.

- Files discovered under `data/raw`: **468**
- File status: audited_excluded 2, duplicate 27, excluded 144, redundant 2, rejected 1, supporting 107, used 185
- Unclassified files: 0
- Rejected files: 1
  - `navarra_telemetry-raw/navarra_telemetry/raw/LOG00010.MBL`: Empty file (zero bytes).

Status meanings: **used** = read by a pipeline; **audited_excluded** = trained/evaluated, then excluded on evidence; **excluded** = not usable (reason given); **redundant** / **duplicate** = same content as another file; **supporting** = documentation/code/metadata (never executed); **rejected** = empty or unreadable.

## Datasets

| Dataset | Nature | Files | Rows (non-duplicate) | Pipelines | File status | Note |
|---|---|---:|---:|---|---|---|
| AI4I 2020 Predictive Maintenance (UCI) | synthetic | 1 | 10,000 | — | excluded 1 | Industrial machine tool, not a road vehicle; its components are not interchangeable with vehicle components. |
| Auto repair orders, parts and invoices (relational, v1) | synthetic | 7 | 976 | — | excluded 7 | Synthetic sample of 238 orders; its prices and repair patterns are not real workshop evidence, so they are neither indexed as facts nor used for cost estimates. |
| Automotive fault diagnosis flowcharts (AKTC/oBike et al.) | reference | 5 | 99 | knowledge_index | excluded 4, used 1 | Compiled diagnostic flowcharts; source attribution in the file name only. |
| carOBD Toyota Etios OBD-II telemetry | observed | 153 | 304,299 | carobd_sensor_anomaly_v1 | supporting 24, used 129 | github.com/eron93br/carOBD |
| Chinese Vehicle Fleet Fault Association Dataset (CVFFAD) | synthetic | 9 | 60,363 | cvffad_fault_system_experiment | duplicate 4, excluded 3, supporting 1, used 1 | Its own records carry source='generated'; the README DOI is a placeholder and the referenced licence file is absent. Root copies are byte-identical to the folder copies. |
| Diesel Vehicle OBD-II PID Dataset (ANCEL ELITE 610) | observed | 21 | 152,315 | diesel_dtc_presence_experiment | duplicate 10, excluded 1, redundant 1, supporting 7, used 2 | README documents 8 named vehicles; DTC columns are constant per vehicle, so they identify the vehicle rather than a time-varying fault state. |
| DTC diagnostic chat examples (train/validation parquet) | synthetic | 2 | 100 | chatbot_intent_training | excluded 1, used 1 | No source documentation. Assistant answers are templated and contain truncated, unrelated 'technical context' text, so they are not used as reference answers; only user questions are used. |
| Engine condition sensor tables (raw_data/train/test) | unverified | 3 | 19,535 | engine_condition_experiment | redundant 1, used 2 | No source documentation. raw_data.csv is the exact union of train.csv and test.csv. |
| EngineFaultDB | observed | 3 | 55,999 | — | excluded 1, supporting 2 | No acquisition-run identifier, so any split mixes rows from the same run (leakage); the numeric class meanings are not documented locally. |
| EV Predictive Maintenance Dataset (15-minute) | synthetic | 1 | 175,393 | synthetic_signal_audit | audited_excluded 1 | No source documentation. Columns are statistically independent of each other and of the labels (|r| < 0.005, lag-1 autocorrelation 0), which indicates independently generated random values. |
| Experimental Energy Characterization (NEDC/WLTP) | unverified | 8 | 31,418 | — | excluded 8 | No documented vehicle, units, labels or licence; nothing maintenance-related to predict. |
| Honda (AHM) motorcycle master parts list, Sep 2025 update 19 | reference | 1 | 61,970 | knowledge_index | used 1 | Dealer price-list style export; no licence file. |
| Hyundai component readings (cars_hyundai.csv) | synthetic | 1 | 1,100 | synthetic_signal_audit | audited_excluded 1 | No source documentation. Values are uniformly distributed and uncorrelated with the anomaly label. |
| KIT Automotive OBD-II Dataset (Seat Leon) | observed | 102 | 2,693,824 | — | duplicate 3, excluded 82, supporting 17 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| LEVIN open data (documentation only) | documentation | 13 | 0 | — | supporting 13 | The measurement export itself is not present locally. |
| Standardized engine/EGR parquet (main_df_standardized) | unverified | 1 | 1,572,243 | — | excluded 1 | fault_type values and the standardization reference are undocumented; there is no group identifier for a leakage-free split. |
| NASA Li-ion Battery Aging Datasets | observed | 48 | 0 | nasa_battery_cell_rul_1p4ah | duplicate 5, supporting 9, used 34 | NASA Ames Prognostics Center of Excellence. |
| Kawasaki Ninja 400 track telemetry (Circuito de Navarra) | observed | 42 | 1,924,111 | — | duplicate 5, excluded 34, rejected 1, supporting 2 | Track racing telemetry from one motorcycle with no fault, wear or maintenance label; it cannot be used to infer road-fleet maintenance. |
| OBDex generic OBD-II DTC and PID reference | reference | 45 | 9,669 | knowledge_index | supporting 32, used 13 | github.com/foerbsnavi/obdex |
| Generated vehicle diagnostic samples (VehicleDiagnosticSample.txt) | synthetic | 1 | 0 | — | excluded 1 | Generated text with random measurements; using it as knowledge would present fabricated diagnostics as reference facts. |
| Vehicle Maintenance - Service Records (Honda/Toyota, India) | observed | 1 | 1,139 | service_need_model, knowledge_index | used 1 | Public Kaggle-style export; no local source or licence file. Records are internally consistent with manufacturer service schedules (e.g. 60,000 km major service). |

## Files used, audited or excluded (data files only)

| File | Status | Rows | Reason |
|---|---|---:|---|
| `10.35097-1130/data/dataset/OBD-II-Dataset.zip` | excluded |  | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0005.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0006.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0007.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0018.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/2. BatteryAgingARC_25_26_27_28_P1/B0025.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/2. BatteryAgingARC_25_26_27_28_P1/B0026.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/2. BatteryAgingARC_25_26_27_28_P1/B0027.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/2. BatteryAgingARC_25_26_27_28_P1/B0028.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/3. BatteryAgingARC_25-44/B0029.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/3. BatteryAgingARC_25-44/B0030.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/3. BatteryAgingARC_25-44/B0031.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/3. BatteryAgingARC_25-44/B0032.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/3. BatteryAgingARC_25-44/B0033.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/3. BatteryAgingARC_25-44/B0034.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/3. BatteryAgingARC_25-44/B0036.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/3. BatteryAgingARC_25-44/B0038.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/3. BatteryAgingARC_25-44/B0039.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/3. BatteryAgingARC_25-44/B0040.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/3. BatteryAgingARC_25-44/B0041.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/3. BatteryAgingARC_25-44/B0042.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/3. BatteryAgingARC_25-44/B0043.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/3. BatteryAgingARC_25-44/B0044.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/4. BatteryAgingARC_45_46_47_48/B0045.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/4. BatteryAgingARC_45_46_47_48/B0046.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/4. BatteryAgingARC_45_46_47_48/B0047.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/4. BatteryAgingARC_45_46_47_48/B0048.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/5. BatteryAgingARC_49_50_51_52/B0049.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/5. BatteryAgingARC_49_50_51_52/B0050.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/5. BatteryAgingARC_49_50_51_52/B0051.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/5. BatteryAgingARC_49_50_51_52/B0052.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/6. BatteryAgingARC_53_54_55_56/B0053.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/6. BatteryAgingARC_53_54_55_56/B0054.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/6. BatteryAgingARC_53_54_55_56/B0055.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `5. Battery Data Set/6. BatteryAgingARC_53_54_55_56/B0056.mat` | used |  | Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah). |
| `ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv` | excluded | 10000 | Industrial machine tool, not a road vehicle; its components are not interchangeable with vehicle components. |
| `auto-repair-orders-parts-and-invoices-dataset-v1-csv-customers.csv` | excluded | 60 | Synthetic sample of 238 orders; its prices and repair patterns are not real workshop evidence, so they are neither indexed as facts nor used for cost estimates. |
| `auto-repair-orders-parts-and-invoices-dataset-v1-csv-invoices.csv` | excluded | 238 | Synthetic sample of 238 orders; its prices and repair patterns are not real workshop evidence, so they are neither indexed as facts nor used for cost estimates. |
| `auto-repair-orders-parts-and-invoices-dataset-v1-csv-mechanics.csv` | excluded | 6 | Synthetic sample of 238 orders; its prices and repair patterns are not real workshop evidence, so they are neither indexed as facts nor used for cost estimates. |
| `auto-repair-orders-parts-and-invoices-dataset-v1-csv-order_parts.csv` | excluded | 323 | Synthetic sample of 238 orders; its prices and repair patterns are not real workshop evidence, so they are neither indexed as facts nor used for cost estimates. |
| `auto-repair-orders-parts-and-invoices-dataset-v1-csv-parts.csv` | excluded | 20 | Synthetic sample of 238 orders; its prices and repair patterns are not real workshop evidence, so they are neither indexed as facts nor used for cost estimates. |
| `auto-repair-orders-parts-and-invoices-dataset-v1-csv-repair_orders.csv` | excluded | 238 | Synthetic sample of 238 orders; its prices and repair patterns are not real workshop evidence, so they are neither indexed as facts nor used for cost estimates. |
| `auto-repair-orders-parts-and-invoices-dataset-v1-csv-vehicles.csv` | excluded | 91 | Synthetic sample of 238 orders; its prices and repair patterns are not real workshop evidence, so they are neither indexed as facts nor used for cost estimates. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-05_Seat_Leon_RT_S_Stau.csv` | excluded | 46349 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-05_Seat_Leon_S_KA_Normal.csv` | excluded | 30817 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-06_Seat_Leon_KA_KA_Normal.csv` | excluded | 18111 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-06_Seat_Leon_KA_RT_Normal.csv` | excluded | 57249 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-07_Seat_Leon_RT_S_Normal.csv` | excluded | 38080 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-07_Seat_Leon_S_RT_Normal.csv` | excluded | 27306 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-10_Seat_Leon_KA_KA_Stau.csv` | excluded | 14570 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-11_Seat_Leon_KA_KA_Stau.csv` | excluded | 17934 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-11_Seat_Leon_KA_S_Normal.csv` | excluded | 41050 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-11_Seat_Leon_S_RT_Frei.csv` | excluded | 28114 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-12_Seat_Leon_RT_S_Normal.csv` | excluded | 43252 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-12_Seat_Leon_S_RT_Normal.csv` | excluded | 32373 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-13_Seat_Leon_KA_KA_Normal.csv` | excluded | 11216 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-13_Seat_Leon_RT_KA_Stau_Messfehler.csv` | excluded | 59634 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-14_Seat_Leon_KA_KA_Frei.csv` | excluded | 6829 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-14_Seat_Leon_KA_RT_Normal.csv` | excluded | 54037 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-24_Seat_Leon_RT_KA_Normal.csv` | excluded | 54866 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-26_Seat_Leon_RT_S_Stau.csv` | excluded | 47350 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-26_Seat_Leon_S_KA_Normal.csv` | excluded | 21618 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-27_Seat_Leon_KA_KA_2_Normal.csv` | excluded | 16961 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-27_Seat_Leon_KA_KA_Normal.csv` | excluded | 13583 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-28_Seat_Leon_KA_KA_Normal.csv` | excluded | 13917 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-28_Seat_Leon_KA_RT_Stau.csv` | excluded | 73200 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-31_Seat_Leon_KA_KA_Frei.csv` | excluded | 10637 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-07-31_Seat_Leon_RT_KA_Normal.csv` | excluded | 57331 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-01_Seat_Leon_KA_KA_Frei.csv` | excluded | 14510 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-01_Seat_Leon_KA_RT_Normal.csv` | excluded | 47797 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-02_Seat_Leon_RT_S_Normal.csv` | excluded | 38524 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-03_Seat_Leon_S_RT_Normal.csv` | excluded | 25540 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-04_Seat_Leon_S_RT_Normal.csv` | excluded | 32903 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-05_Seat_Leon_S_CW_Normal.csv` | excluded | 35469 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-07_Seat_Leon_KA_RT_Frei.csv` | excluded | 27371 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-07_Seat_Leon_RT_KA_Frei_Vollbremsung.csv` | excluded | 39110 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-08_Seat_Leon_RT_S_Normal.csv` | excluded | 24535 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-08_Seat_Leon_S_RT_Normal.csv` | excluded | 24442 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-09_Seat_Leon_RT_S_Normal.csv` | excluded | 38662 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-09_Seat_Leon_S_RT_Normal.csv` | excluded | 28057 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-10_Seat_Leon_RT_S_Normal.csv` | excluded | 29475 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2017-08-10_Seat_Leon_S_RT_Frei.csv` | excluded | 19645 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-14_Seat_Leon_RT_S_Normal.csv` | excluded | 30973 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-14_Seat_Leon_S_RT_Normal.csv` | excluded | 26102 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-16_Seat_Leon_S_RT_Normal.csv` | excluded | 28040 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-17_Seat_Leon_BB_RT_Normal_Glatteis.csv` | excluded | 41611 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-17_Seat_Leon_RT_BB_Frei.csv` | excluded | 29341 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-18_Seat_Leon_RT_KA_Stau.csv` | excluded | 53843 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-20_Seat_Leon_KA_KA_Frei.csv` | excluded | 8903 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-21_Seat_Leon_BB_RT_Normal.csv` | excluded | 24129 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-21_Seat_Leon_KA_BB_Normal.csv` | excluded | 43590 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-21_Seat_Leon_KA_KA_Normal.csv` | excluded | 15372 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-22_Seat_Leon_RT_S_Normal.csv` | excluded | 36251 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-23_Seat_Leon_RT_RT_Frei_Beschleunigung.csv` | excluded | 10176 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-23_Seat_Leon_RT_S_Normal.csv` | excluded | 31958 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-23_Seat_Leon_S_RT_Normal.csv` | excluded | 20413 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-23_Seat_Leon_S_S_Stau.csv` | excluded | 22521 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-26_Seat_Leon_KA_KA_Frei.csv` | excluded | 10980 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-26_Seat_Leon_RT_KA_Normal.csv` | excluded | 56676 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-26_Seat_Leon_RT_RT_Frei.csv` | excluded | 11534 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-27_Seat_Leon_KA_KA_Normal.csv` | excluded | 17405 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-27_Seat_Leon_KA_RT_Normal.csv` | excluded | 47345 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-28_Seat_Leon_RT_S_Normal.csv` | excluded | 37385 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-02-28_Seat_Leon_S_RT_Normal.csv` | excluded | 34380 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-01_Seat_Leon_RT_S_Normal.csv` | excluded | 32097 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-01_Seat_Leon_S_RT_Normal.csv` | excluded | 54784 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-02_Seat_Leon_RT_S_Normal.csv` | excluded | 34648 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-06_Seat_Leon_KA_RT_Normal.csv` | excluded | 46002 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-07_Seat_Leon_RT_S_Normal.csv` | excluded | 34082 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-07_Seat_Leon_S_RT_Normal.csv` | excluded | 34710 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-08_Seat_Leon_RT_S_Normal.csv` | excluded | 37067 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-19_Seat_Leon_KA_KA_Frei.csv` | excluded | 11307 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-19_Seat_Leon_RT_KA_Normal.csv` | excluded | 70522 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-20_Seat_Leon_KA_KA_Normal.csv` | excluded | 19070 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-21_Seat_Leon_KA_RT_Normal.csv` | excluded | 51566 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-22_Seat_Leon_RT_S_Normal.csv` | excluded | 37082 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-22_Seat_Leon_S_RT_Normal.csv` | excluded | 27579 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-23_Seat_Leon_RT_S_Normal.csv` | excluded | 37556 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-23_Seat_Leon_S_RT_Stau.csv` | excluded | 38509 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-26_Seat_Leon_RT_S_Normal.csv` | excluded | 33563 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-26_Seat_Leon_S_RT_Stau.csv` | excluded | 19510 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-03-29_Seat_Leon_KA_RT_Stau.csv` | excluded | 86654 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-04-23_Seat_Leon_KA_KA_Frei.csv` | excluded | 9701 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/2018-04-23_Seat_Leon_RT_KA_Normal.csv` | excluded | 76463 | No fault, repair or maintenance labels; one vehicle. The sensor-anomaly role is already covered by the carOBD pipeline, which has idle/drive baselines. |
| `automotive_faults_aktc_obike_et_al.json` | used | 99 | Diagnostic flowcharts indexed for the chatbot (marked unverified source). |
| `carOBD-master/obdiidata/drive1.csv` | used | 2709 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/drive10.csv` | used | 5454 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/drive11.csv` | used | 3271 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/drive12.csv` | used | 3836 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/drive13.csv` | used | 4092 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/drive2.csv` | used | 2324 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/drive3.csv` | used | 2926 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/drive4.csv` | used | 4389 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/drive5.csv` | used | 627 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/drive6.csv` | used | 3468 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/drive7.csv` | used | 557 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/drive8.csv` | used | 2798 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/drive9.csv` | used | 3328 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle1.csv` | used | 1213 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle10.csv` | used | 1465 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle11.csv` | used | 975 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle12.csv` | used | 1275 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle13.csv` | used | 1266 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle14.csv` | used | 1196 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle15.csv` | used | 1207 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle16.csv` | used | 1095 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle17.csv` | used | 1903 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle18.csv` | used | 2115 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle19.csv` | used | 1396 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle2.csv` | used | 1225 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle20.csv` | used | 1479 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle21.csv` | used | 775 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle22.csv` | used | 1864 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle23.csv` | used | 1593 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle24.csv` | used | 1772 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle25.csv` | used | 1297 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle26.csv` | used | 1447 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle27.csv` | used | 2112 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle28.csv` | used | 2319 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle29.csv` | used | 2812 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle3.csv` | used | 1360 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle30.csv` | used | 1295 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle31.csv` | used | 1360 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle32.csv` | used | 1255 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle33.csv` | used | 1126 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle34.csv` | used | 1800 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle35.csv` | used | 1359 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle36.csv` | used | 1877 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle37.csv` | used | 1136 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle38.csv` | used | 1289 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle39.csv` | used | 1359 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle4.csv` | used | 1347 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle40.csv` | used | 1941 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle41.csv` | used | 2196 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle42.csv` | used | 1410 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle43.csv` | used | 2204 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle44.csv` | used | 2437 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle45.csv` | used | 1385 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle46.csv` | used | 2474 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle47.csv` | used | 1882 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle5.csv` | used | 1405 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle6.csv` | used | 1070 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle7.csv` | used | 1930 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle8.csv` | used | 795 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/idle9.csv` | used | 733 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live1.csv` | used | 3853 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live10.csv` | used | 1176 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live11.csv` | used | 1727 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live12.csv` | used | 1151 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live13.csv` | used | 2418 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live14.csv` | used | 1291 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live15.csv` | used | 2309 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live16.csv` | used | 3082 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live17.csv` | used | 3791 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live18.csv` | used | 3084 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live19.csv` | used | 2671 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live2.csv` | used | 2823 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live20.csv` | used | 2407 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live21.csv` | used | 3057 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live22.csv` | used | 2685 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live23.csv` | used | 3082 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live24.csv` | used | 2627 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live25.csv` | used | 3804 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live26.csv` | used | 1840 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live27.csv` | used | 2381 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live28.csv` | used | 1855 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live29.csv` | used | 2283 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live3.csv` | used | 3022 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live30.csv` | used | 2233 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live31.csv` | used | 3769 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live32.csv` | used | 3148 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live33.csv` | used | 2914 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live34.csv` | used | 2188 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live35.csv` | used | 1989 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live36.csv` | used | 2266 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live37.csv` | used | 1815 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live38.csv` | used | 1912 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live39.csv` | used | 1745 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live4.csv` | used | 3178 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live5.csv` | used | 2384 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live6.csv` | used | 2287 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live7.csv` | used | 2453 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live8.csv` | used | 2413 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/live9.csv` | used | 2168 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/long1.csv` | used | 3528 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/long10.csv` | used | 5035 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/long11.csv` | used | 5528 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/long12.csv` | used | 5616 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/long2.csv` | used | 7516 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/long3.csv` | used | 4041 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/long4.csv` | used | 5863 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/long5.csv` | used | 4653 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/long6.csv` | used | 4342 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/long7.csv` | used | 6066 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/long8.csv` | used | 4152 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/long9.csv` | used | 4607 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/ufpe1.csv` | used | 1651 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/ufpe10.csv` | used | 2691 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/ufpe11.csv` | used | 1394 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/ufpe12.csv` | used | 3075 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/ufpe13.csv` | used | 2573 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/ufpe14.csv` | used | 2662 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/ufpe15.csv` | used | 2007 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/ufpe16.csv` | used | 1865 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/ufpe17.csv` | used | 949 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/ufpe18.csv` | used | 2271 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/ufpe2.csv` | used | 1380 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/ufpe3.csv` | used | 808 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/ufpe4.csv` | used | 1349 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/ufpe5.csv` | used | 1354 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/ufpe6.csv` | used | 1105 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/ufpe7.csv` | used | 2366 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/ufpe8.csv` | used | 2671 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `carOBD-master/obdiidata/ufpe9.csv` | used | 1895 | Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1). |
| `cars_hyundai.csv` | audited_excluded | 1100 | Trained against baselines in the synthetic-signal audit; no learnable signal, excluded from served models. |
| `EngineFaultDB-main/EngineFaultDB-main/EngineFaultDB_Final.csv` | excluded | 55999 | No acquisition-run identifier, so any split mixes rows from the same run (leakage); the numeric class meanings are not documented locally. |
| `EV_Predictive_Maintenance_Dataset_15min.csv` | audited_excluded | 175393 | Trained against baselines in the synthetic-signal audit; no learnable signal, excluded from served models. |
| `Experimental Energy Characterization/NEDC EXTRAURBAN 1.csv` | excluded | 2106 | No documented vehicle, units, labels or licence; nothing maintenance-related to predict. |
| `Experimental Energy Characterization/NEDC EXTRAURBAN 2.csv` | excluded | 2164 | No documented vehicle, units, labels or licence; nothing maintenance-related to predict. |
| `Experimental Energy Characterization/NEDC URBAN 1.csv` | excluded | 3028 | No documented vehicle, units, labels or licence; nothing maintenance-related to predict. |
| `Experimental Energy Characterization/NEDC URBAN 2.csv` | excluded | 3229 | No documented vehicle, units, labels or licence; nothing maintenance-related to predict. |
| `Experimental Energy Characterization/WLTP EXTRAURBAN 1.csv` | excluded | 4584 | No documented vehicle, units, labels or licence; nothing maintenance-related to predict. |
| `Experimental Energy Characterization/WLTP EXTRAURBAN 2.csv` | excluded | 4530 | No documented vehicle, units, labels or licence; nothing maintenance-related to predict. |
| `Experimental Energy Characterization/WLTP URBAN 1.csv` | excluded | 5544 | No documented vehicle, units, labels or licence; nothing maintenance-related to predict. |
| `Experimental Energy Characterization/WLTP URBAN 2.csv` | excluded | 6233 | No documented vehicle, units, labels or licence; nothing maintenance-related to predict. |
| `fault_events.csv` | used | 29431 | CVFFAD fault-system experiment (evaluation only; generated data, not served). |
| `knowledge_graph_triples.csv` | excluded | 565 | Generated CVFFAD triples (Chinese); not indexed because generated content would be presented as fact. |
| `main_df_standardized.parquet` | excluded | 1572243 | fault_type values and the standardization reference are undocumented; there is no group identifier for a leakage-free split. |
| `masterparts.xlsx` | used | 61970 | Exact part-number lookup table for the chatbot. |
| `metadata.json` | excluded | 1 | Track racing telemetry from one motorcycle with no fault, wear or maintenance label; it cannot be used to infer road-fleet maintenance. |
| `navarra-050926-tanda1/metadata.json` | excluded | 1 | Track racing telemetry from one motorcycle with no fault, wear or maintenance label; it cannot be used to infer road-fleet maintenance. |
| `navarra-050926-tanda1/raw_manifest.json` | excluded | 1 | Track racing telemetry from one motorcycle with no fault, wear or maintenance label; it cannot be used to infer road-fleet maintenance. |
| `navarra-050926-tanda1/samples.csv/samples.csv` | excluded | 19565 | Track racing telemetry from one motorcycle with no fault, wear or maintenance label; it cannot be used to infer road-fleet maintenance. |
| `navarra-050926-tanda1/updates.csv/updates.csv` | excluded | 314402 | Track racing telemetry from one motorcycle with no fault, wear or maintenance label; it cannot be used to infer road-fleet maintenance. |
| `navarra-050926-tanda1-raw/navarra-050926-tanda1/raw/LOG00004.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra-050926-tanda1-raw/navarra-050926-tanda1/raw/LOG00005.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra-050926-tanda1-raw/navarra-050926-tanda1/raw/LOG00006.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra-050926-tanda1-raw/navarra-050926-tanda1/raw/LOG00010.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra-050926-tandas2-4/metadata.json` | excluded | 1 | Track racing telemetry from one motorcycle with no fault, wear or maintenance label; it cannot be used to infer road-fleet maintenance. |
| `navarra-050926-tandas2-4/raw_manifest.json` | excluded | 1 | Track racing telemetry from one motorcycle with no fault, wear or maintenance label; it cannot be used to infer road-fleet maintenance. |
| `navarra-050926-tandas2-4/samples.csv/samples.csv` | excluded | 31709 | Track racing telemetry from one motorcycle with no fault, wear or maintenance label; it cannot be used to infer road-fleet maintenance. |
| `navarra-050926-tandas2-4/updates.csv/updates.csv` | excluded | 601953 | Track racing telemetry from one motorcycle with no fault, wear or maintenance label; it cannot be used to infer road-fleet maintenance. |
| `navarra-050926-tandas2-4-raw/navarra-050926-tandas2-4/raw/LOG00000.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra-050926-tandas2-4-raw/navarra-050926-tandas2-4/raw/LOG00001.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra-050926-tandas2-4-raw/navarra-050926-tandas2-4/raw/LOG00002.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra-050926-tandas2-4-raw/navarra-050926-tandas2-4/raw/LOG00003.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra-050926-tandas2-4-raw/navarra-050926-tandas2-4/raw/LOG00004.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra-050926-tandas2-4-raw/navarra-050926-tandas2-4/raw/LOG00005.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra_telemetry-raw/navarra_telemetry/raw/LOG00000.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra_telemetry-raw/navarra_telemetry/raw/LOG00001.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra_telemetry-raw/navarra_telemetry/raw/LOG00002.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra_telemetry-raw/navarra_telemetry/raw/LOG00003.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra_telemetry-raw/navarra_telemetry/raw/LOG00004.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra_telemetry-raw/navarra_telemetry/raw/LOG00005.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra_telemetry-raw/navarra_telemetry/raw/LOG00006.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra_telemetry-raw/navarra_telemetry/raw/LOG00007.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra_telemetry-raw/navarra_telemetry/raw/LOG00008.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra_telemetry-raw/navarra_telemetry/raw/LOG00009.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra_telemetry-raw/navarra_telemetry/raw/LOG00010.MBL` | rejected |  | Empty file (zero bytes). |
| `navarra_telemetry-raw/navarra_telemetry/raw/LOG00011.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `navarra_telemetry-raw/navarra_telemetry/raw/LOG00012.MBL` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `ner_annotations.csv` | excluded | 29431 | Generated CVFFAD NER annotations (Chinese); no NER task in this system. |
| `OBD-Dataset-main/FILE.cbf` | excluded |  | Proprietary binary log; its decoded copy is profiled separately. |
| `OBD-Dataset-main/MASTER_TEST_X.xlsx` | used | 33838 | Input to: diesel_dtc_presence_experiment. |
| `OBD-Dataset-main/MASTER_TRAIN_X.xlsx` | used | 118477 | Input to: diesel_dtc_presence_experiment. |
| `OBD-Dataset-main.zip` | redundant |  | Archive of the extracted OBD-Dataset-main folder; the extracted workbooks are used. |
| `OBDex-main/data/generic/B0xxx_enriched.yaml` | used | 323 | Parsed into the chatbot knowledge index (DTC and PID definitions). |
| `OBDex-main/data/generic/C0xxx_enriched.yaml` | used | 626 | Parsed into the chatbot knowledge index (DTC and PID definitions). |
| `OBDex-main/data/generic/P0xxx_enriched.yaml` | used | 3705 | Parsed into the chatbot knowledge index (DTC and PID definitions). |
| `OBDex-main/data/generic/P2xxx_enriched.yaml` | used | 3495 | Parsed into the chatbot knowledge index (DTC and PID definitions). |
| `OBDex-main/data/generic/P3xxx_enriched.yaml` | used | 155 | Parsed into the chatbot knowledge index (DTC and PID definitions). |
| `OBDex-main/data/generic/U0xxx_enriched.yaml` | used | 1055 | Parsed into the chatbot knowledge index (DTC and PID definitions). |
| `OBDex-main/data/generic/U3xxx_enriched.yaml` | used | 174 | Parsed into the chatbot knowledge index (DTC and PID definitions). |
| `OBDex-main/data/pids/mode01.yaml` | used | 119 | Parsed into the chatbot knowledge index (DTC and PID definitions). |
| `OBDex-main/data/pids/mode09.yaml` | used | 13 | Parsed into the chatbot knowledge index (DTC and PID definitions). |
| `OBDex-main/package-lock.json` | used | 1 | Input to: knowledge_index. |
| `OBDex-main/package.json` | used | 1 | Input to: knowledge_index. |
| `OBDex-main/schema/code.schema.json` | used | 1 | Input to: knowledge_index. |
| `OBDex-main/schema/pid.schema.json` | used | 1 | Input to: knowledge_index. |
| `raw_data.csv` | redundant | 19535 | Exact union of train.csv and test.csv; verified, not used again. |
| `raw_manifest.json` | excluded | 1 | Track racing telemetry from one motorcycle with no fault, wear or maintenance label; it cannot be used to infer road-fleet maintenance. |
| `samples.csv` | excluded | 134984 | Track racing telemetry from one motorcycle with no fault, wear or maintenance label; it cannot be used to infer road-fleet maintenance. |
| `test.csv` | used | 3907 | Engine-condition experiment test split (evaluation only, not served). |
| `train-00000-of-00001.parquet` | excluded | 85 | Synthetic DTC chat examples; answers are templated and contain unrelated truncated text, so they are not used for training. |
| `train.csv` | used | 15628 | Engine-condition experiment training split (evaluation only, not served). |
| `updates.csv` | excluded | 821492 | Track racing telemetry from one motorcycle with no fault, wear or maintenance label; it cannot be used to infer road-fleet maintenance. |
| `validation-00000-of-00001.parquet` | used | 15 | User questions only, as DTC-routing cases in the chatbot evaluation set (answers not used). |
| `Vehicle Maintenance- Service Records.csv` | used | 1139 | Training/evaluation rows for the service-need model; service statistics indexed for the chatbot. |
| `vehicle_fleet.csv` | excluded | 936 | Generated CVFFAD fleet register; not needed by the experiment and not real fleet data. |

Duplicates and supporting files (134) are listed per file in `dataset_manifest.csv`.
