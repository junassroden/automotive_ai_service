# Predictive-Maintenance Dataset Assessment

## Verdict: `NO_SUITABLE_DATASET_FOUND`

The recursively inventoried local collection contains vehicle telemetry, engine
fault-classification data, generated fault-event records, component service rows,
and laboratory battery-aging records. None currently provides a provenance-verified,
representative set of real fleet vehicles with time-ordered sensor/vehicle history,
confirmed maintenance or failure outcomes, and identifiers that support an
independent vehicle/run holdout. Therefore none supports a defensible first model
for predicting future maintenance needs across cars, motorcycles, pickups, vans,
and minivans.

No separate training script was created: the prerequisite dataset is not suitable.
No training, evaluation, API/configuration change, or artifact modification was
performed for this assessment. The existing Django 503 safeguard remains required.

## Evidence basis

- The existing inventory covers **237 files across 14 top-level directories**;
  the inventory reported no content-inspection errors. See
  [`dataset_inventory.md`](dataset_inventory.md) and
  [`dataset_inventory.json`](dataset_inventory.json).
- The prior training audit records the maintenance training attempt as blocked
  before fitting: `acquisition_run_id` is missing, with no training/test split,
  fitted model, or evaluation. Its `training_records: 55998` is the available
  prepared-row count, not the number used to fit a model.
- The legacy `trained_models/vehicle_maintenance_model.joblib` is unverified and
  cannot be attributed to any of these datasets. It is not accepted by the
  current inference registry.
- The chatbot's independent intent-classification training does not provide
  evidence for vehicle fault diagnosis or maintenance prediction.

## Candidate dataset assessment

Paths below are relative to `data/raw/` unless identified as prepared data. Counts
and field descriptions are from the local inventory and accompanying local
documentation; unsupported source claims are identified rather than treated as
verified evidence.

| Candidate; path | Vehicle/component; size/features | Documented target and meaning | IDs and leakage-safe split | Task fit and disposition |
|---|---|---|---|---|
| **EngineFaultDB** — `EngineFaultDB-main/EngineFaultDB-main/EngineFaultDB_Final.csv`; prepared copy `../processed/enginefaultdb/enginefaultdb_normalized.csv` | One C14NE spark-ignition engine under controlled laboratory conditions. 55,999 source rows; 55,998 after one exact duplicate was removed. 14 numeric signals plus `Fault`; no missing values. Class counts after preparation: 0: 16,000; 1: 10,998; 2: 15,000; 3: 14,000. | Four numeric classes described as engine operating/fault states. The README does not give a verified mapping from 0–3 to named fault mechanisms. No future failure, service event, or RUL target. | No vehicle, trip, run, or acquisition-run column in the CSV or prepared metadata. README describes one engine and laboratory acquisition, but does not map rows to experimental runs. A row-random split is unsafe; a valid group holdout cannot be constructed locally. | Could be a laboratory engine-state/fault-classification research dataset only if labels and independent runs are verified. Not fleet maintenance prediction. **Not trainable for the requested task.** |
| **Diesel OBD workbooks** — `OBD-Dataset-main/OBD-Dataset-main/MASTER_TRAIN_X.xlsx`, `MASTER_TEST_X.xlsx` | README claims eight diesel vehicles and provides 11 OBD PID definitions. Workbook inventory reports 118,477 training and 33,838 test rows, 26 columns, zero missing cells, and 90,657 / 26,280 exact duplicate rows respectively. | README claims manually documented DTCs and describes healthy/faulty and multi-label tasks. The local workbooks are named feature matrices (`*_X`); row-level target/label linkage and acquisition provenance are not established by the available evidence. | No vehicle/trip/run ID in the workbook evidence; supplied split's independence by vehicle is unverified. Cannot establish a leakage-safe holdout or even a reliably linked target. | Potential fault diagnosis only if source, labels, row-to-vehicle linkage, split semantics, and license are verified. **Not usable now for prediction or evaluation.** |
| **Vehicle Maintenance service records** — `Vehicle Maintenance- Service Records.csv` | 1,139 rows × 24 columns; all rows say `car`, two makes and four models. 14 service-related binary fields and `cost`; `oil_filter` and `engine_oil` are constant 1. No missing or exact duplicate rows. | Service-item indicators and cost, but no documentation establishes whether values are past work, scheduled recommendations, or future outcomes. No service timestamps or confirmed failure labels. | `slno` is unique but is not documented as a vehicle identifier. No vehicle identity or event chronology, so no group/time holdout. | Could describe service-row associations only after provenance and temporal meaning are supplied. It cannot support future maintenance prediction. |
| **KIT Automotive OBD-II** — `10.35097-1130/data/dataset/OBD-II-Dataset.zip` and mirrored `Automotive_Diagnostics-main/Automotive_Diagnostics-main/OBD-II-Dataset/` CSVs | 2,693,824 time-stamped samples across 81 CSV files for a Seat Leon vehicle model, with ten operating signals and road-condition filename tags. The extracted files are byte-identical to archive members. | Road/traffic operating conditions, not documented component faults, repairs, or maintenance outcomes. | File/session names permit session-level separation for descriptive signal studies, but no stable fleet vehicle identity or outcome labels are established. | Telemetry only; neither supervised maintenance prediction nor fault classification is supported. |
| **Navarra motorcycle telemetry** — session folders `navarra_telemetry-raw/`, `navarra-050926-tanda1/`, and `navarra-050926-tandas2-4/` | Kawasaki Ninja 400 track motorcycle; 134,984 + 19,565 + 31,709 decoded sample rows (186,258 total) and 95 channels per session. One rider and one machine. | No confirmed failure, repair, maintenance, loss-of-grip, or RUL outcomes. ECU `cut_function` codes describe control functions, not fault labels. | Logs and metadata supply per-file `boot_id`/continuity segments that can prevent some temporal overlap, but there is only one machine/rider and no labeled outcome. | Useful for single-motorcycle sensor analysis, not maintenance/failure prediction or vehicle-type generalization. |
| **Root engine-condition table** — `raw_data.csv`, `train.csv`, `test.csv` | 19,535 total rows, six numeric measurements; supplied partition is 15,628 train and 3,907 test rows and exactly partitions `raw_data.csv`. | Binary `Engine Condition` / `engine_condition` label; its source definition and relationship to diagnosed faults or maintenance are undocumented. | No vehicle/run IDs, source, units, or license. A fixed row split exists but cannot be verified as group-disjoint. | At best an unproven engine-state classification target, not future maintenance prediction. |
| **Unattributed Parquet** — `main_df_standardized.parquet` | 1,572,243 rows × 21 columns: 18 sensor-like fields plus `drive_cycle`, `fault_type`, and `flag`; 136 exact duplicate rows. `flag` is missing in 1,572,210 rows. | Ten values in `fault_type`, but no documentation defines them or says these are verified faults. | No source, vehicle/run IDs, units, or license. No defensible group split. | Labels are semantically unverified; neither fault diagnosis nor predictive maintenance can be claimed. |
| **CVFFAD generated event records** — `Chinese Vehicle Fleet Fault Association Dataset/Chinese Vehicle Fleet Fault Association Dataset/fault_events.csv`; linked metadata `.../vehicle_fleet.csv` | 29,431 fault-event rows; 33 columns. Includes `vehicle_id`, `event_id`, dates, mileage, fault attributes and some derived sensor/trend fields. Companion fleet table has 936 rows / 20 columns. Many event fields are generated; `related_faults` is missing in 28,098 rows. | `fault_code`, `fault_category`, and `fault_name` are presented as fault-event labels; README explicitly says generated. These are not verified real maintenance outcomes or prospective labels. DOI is placeholder `10.5281/zenodo.XXXXXXX`; referenced license/provenance files are absent locally. | A `vehicle_id` column exists and could technically group the generated records if its semantics held, but it does not establish independently collected real vehicles or valid prospective labels. | Synthetic/generated event classification only; unsuitable as evidence for commercial fleet maintenance or real-world predictive performance. |
| **AI4I 2020** — `ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv` | 10,000 synthetic industrial-machine records, 14 fields, 339 `Machine failure` positives, no missing values or exact duplicates. `Product ID` exists. | Machine failure and component failure-mode flags (TWF/HDF/PWF/OSF/RNF), for industrial equipment. | Product ID may allow grouping within the synthetic dataset, but no road vehicles are represented. | Industrial machine predictive-maintenance benchmark only; not automotive fleet training. |
| **NASA Li-ion battery aging** — `5. Battery Data Set/` (38 MAT files including byte-identical copies; 34 distinct cell files; 7,885 cycle records in the inventory counting duplicates) | Laboratory lithium-ion cells with charge/discharge/impedance cycle measurements; cycle capacity and electrical/temperature signals. | Cycle capacity and documented end-of-life at 30% capacity fade support lab-cell aging/RUL targets. | Cell IDs are available and a cell-disjoint split is conceptually possible, subject to a carefully constructed cycle-level dataset and adequate independent cells. Duplicated cell files must not cross splits. | A separate laboratory battery-cell degradation/RUL project is plausible. This is not a validated installed automotive battery or whole-vehicle maintenance dataset. |
| **Experimental Energy Characterization** — `Experimental Energy Characterization/` | Eight CSV files, 31,418 rows total, 14–41 columns; telemetry-like temperature, RPM, voltage/current, acceleration and time fields, with substantial missingness. | No target labels or documented maintenance/failure outcomes. | No vehicle/run IDs, source URL, or acquisition protocol. | Unlabeled telemetry with unverified provenance; neither supervised task. |
| **LEVIN telematics** — `levin-openData-master/` | Local folder has documentation and notebooks, not raw measurement exports. README describes a multi-vehicle telematics project but no local records are measurable. | No local outcomes available. | No usable local event/group records to split. | Cannot train from the files present. |

## EngineFaultDB acquisition-run identifier finding

The missing `acquisition_run_id` cannot be resolved from the local data or its
metadata. The source CSV has only `Fault` and 14 signal columns. The prepared
metadata records the source path, row counts, transformations, and class counts,
but no row-to-run mapping. The README documents a single C14NE engine and controlled
laboratory capture, but not run identifiers, row ordering boundaries, or a mapping
from each row to an acquisition session. The raw inventory/manifests contain file
identity, sizes, and hashes; they do not carry row-level acquisition metadata.

Consequently, do not synthesize groups from row order, numeric labels, adjacent
values, or file timestamps. Resolve this only by obtaining authoritative original
acquisition logs or a source-side record mapping each CSV row to its true run/session,
then verify the mapping against the unchanged source file. Even with valid groups,
the fault-class meanings must be confirmed and the dataset would remain a single
laboratory-engine classification study, not fleet maintenance prediction.

## Recommendation and data required

There is **no recommended dataset for the first fleet maintenance model** in the
current local collection. EngineFaultDB is the closest automotive fault-labeled
candidate but is not an appropriate substitute: it is single-engine laboratory
classification with undocumented class names and missing acquisition groups. Do
not train or evaluate a fleet maintenance model from it.

Collect a properly licensed, provenance-documented fleet dataset with:

1. Stable pseudonymous vehicle IDs, component IDs where applicable, vehicle class,
   make/model/year and configuration.
2. Timestamped and unit-defined telemetry/odometer readings, trip or acquisition
   session IDs, and preserved temporal order.
3. Confirmed faults, inspections, repairs/replacements and service dates, linked to
   the vehicle/component and mileage; include healthy/no-event observation windows.
4. An explicit prediction horizon and target, such as a confirmed component failure
   within a defined future interval or a scheduled service outcome. For RUL, provide
   degradation/run-to-failure traces and censoring semantics.
5. Enough independent vehicles and failure events for vehicle-disjoint, time-aware
   holdout evaluation; no same-vehicle windows across train and test.
6. Documented source, collection protocol, label definitions, missingness, units,
   license, and coverage for the intended car, motorcycle, pickup, van, and minivan
   population.

Once such data exists, prepare the independent workflow before fitting: freeze a
vehicle/time-disjoint holdout; compare a majority/most-frequent baseline and a
simple domain baseline; report per-class precision, recall and F1, balanced
accuracy, confusion matrix, and appropriate PR-AUC/ROC-AUC; inspect leakage,
calibration and class prevalence; bind the report to dataset and artifact hashes.
Fault classification should be named as such and must not be described as
prediction of future failure unless outcomes are explicitly time-forward.

## Direct answers

- **Can the downloaded data support a reliable fleet predictive-maintenance model
  now?** No; verdict is `NO_SUITABLE_DATASET_FOUND`.
- **Is there a candidate for a separate, limited research task?** NASA supports
  lab-cell aging/RUL; EngineFaultDB may support lab engine-state classification
  only after authoritative run IDs and label meanings are supplied. Neither
  supports the requested cross-vehicle fleet maintenance task.
- **Was a suitable training script prepared?** No. Under the requested condition,
  a separate script is only warranted if a suitable dataset exists; creating one
  now would encode unsupported targets/splits.
- **What is the next action?** Obtain the fleet records and label/group metadata
  listed above, then reassess target validity and split feasibility before
  implementing or running training.
