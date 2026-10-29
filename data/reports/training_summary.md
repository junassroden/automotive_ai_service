# Predictive-maintenance pipeline summary

Audit completed 2026-10-29 in the existing Django service. Raw files were preserved.

## Inventory and preparation

- Inspected **237 files** in **14 top-level directories** and the loose root-file
  collection. The 12 evidence groups used for this report are nine named collection
  bundles plus three unprovenanced root table sources; duplicate archives, mirrored
  files, splits and multi-session folders are not counted as new datasets.
- File types included 105 CSV, 38 MATLAB MAT, 17 notebooks, 23 MBL, two XLSX, one
  CBF, one Parquet, one ZIP, and supporting metadata, documentation and code.
- The scanner reported **zero content-inspection errors**. It identified 23 opaque
  MBL telemetry files (one zero-byte file), one opaque CBF file, and one WebP image
  not decoded. No archive was extracted; the ZIP was checked for corruption,
  encryption and unsafe paths.
- Verified byte-identical duplicates: 81 KIT CSVs match the 81 CSV members inside the
  KIT ZIP; battery B0025–B0028 files are repeated byte-for-byte in two folders.
- Source verification and license findings: see
  [`dataset_research.md`](dataset_research.md). Complete file-level statistics:
  [`dataset_inventory.md`](dataset_inventory.md) and
  [`dataset_inventory.json`](dataset_inventory.json).
- Prepared `EngineFaultDB` as a separate normalized CSV: 55,999 source rows became
  55,998 rows after removing one exact duplicate. No missing or non-finite numeric
  values were found. Column/unit normalization and label counts are recorded in
  `data/processed/enginefaultdb/preparation_metadata.json`. No labels were invented.

## Training decision and results

**No project predictive model was trained by this pipeline. No evaluation metrics are
available.** The data does not justify a defensible held-out fleet performance claim:

- EngineFaultDB has numeric labels 0–3 but no verified mapping to fault names and no
  vehicle, run, trip or acquisition-group IDs. A row-random split would risk leakage
  between related laboratory samples, so the trainer refuses it.
- The root maintenance-service table lacks source/license, service-date semantics and
  vehicle identity. Two target fields are constant.
- The engine-condition CSV split has no vehicle/run IDs, units, source or license.
- The diesel OBD workbooks have no vehicle IDs and contain 116,937 exact duplicate
  rows across the supplied train/test sheets; source and split-by-vehicle semantics
  remain unverified.
- The large Parquet file has fault-like labels but no source/units/license; its
  `flag` field is missing in 1,572,210 of 1,572,243 rows.
- AI4I is explicitly synthetic industrial-machine data. Battery-aging files concern
  laboratory cells, and the KIT and motorcycle telemetry have no failure/repair
  labels.

The existing `trained_models/vehicle_maintenance_model.joblib` is a legacy artifact
already present in the project. Its training source and evaluation are not reproduced
by the supplied code/data and it was not loaded, retrained or evaluated for this task.
The existing prediction and fleet endpoints are disabled because the legacy
artifact's training source and evaluation are unverified. They return HTTP 503
with an unavailable status; chat prediction and fleet requests similarly report
unavailable without returning model scores. No environment-variable bypass is
provided. Inference requires an accepted registry entry, artifact/report hashes,
and evaluation metrics meeting the minimum group, baseline, and probability
calibration criteria. The
`GET /api/capabilities/` endpoint reports no validated predictive model.

The grouped trainer uses a vehicle/trip/run-disjoint holdout, an in-pipeline
preprocessor, a most-frequent baseline, class-wise metrics and a saved evaluation
record. Generic grouped candidates remain separate from production inference.
For the current EngineFaultDB copy, maintenance training is intentionally blocked.
Chatbot intent training is separate; its report does not validate maintenance.

## Integration, tests and commands

- Django endpoint added: `GET /api/capabilities/`. Existing React and LavaLust
  contracts were not changed.
- Existing local automotive knowledge retrieval remains general reference guidance;
  it is not presented as a trained prediction or confirmed diagnosis.
- Test run: `python manage.py test` — **41 tests passed**, Django system check clean.
- Dataset preparation run: `python -m data_pipeline.cli prepare-enginefaultdb` —
  **55,998 processed rows**, one exact duplicate removed.
- Inventory run: `python -m data_pipeline.cli inventory` — **237 files**, zero
  content-inspection errors.
- Chatbot intent classifier trained from `data/chatbot_intents.json`: **2,780**
  authored examples across **23 intents**; stratified holdout **2,085 train / 695
  test**; accuracy **0.9381**, macro F1 **0.8618**, weighted F1 **0.9400**.
  The most-frequent baseline macro F1 was **0.0355**.
  These scores measure a holdout of related authored examples, not independent
  real-user generalization and not maintenance-prediction performance. Model and
  report: `trained_models/chatbot_nlp.joblib` and
  `data/reports/chatbot_training.json`.
- Maintenance training attempt: **blocked** before fitting; no evaluation split,
  classifier, or metrics were produced because `acquisition_run_id` is absent.
- Install readers after activating the service virtual environment:
  `python -m pip install -r requirements.txt`.
- Refresh inventory: `python -m data_pipeline.cli inventory`.
- Recreate normalized EngineFaultDB: `python -m data_pipeline.cli prepare-enginefaultdb`.
- Safe extraction is available for archives under `data/raw/`, with destinations
  restricted to `data/processed/archives/`. It validates paths, symlinks, encryption,
  duplicate output paths and archive integrity before writing. No extraction was
  needed for this audit.
- Train only when source data includes a documented group identifier:

  ```powershell
  python -m data_pipeline.cli train `
    --input data/processed/enginefaultdb/enginefaultdb_normalized.csv `
    --target fault_class `
    --group-column acquisition_run_id `
    --output models/enginefaultdb
  ```

  This command intentionally fails for the current CSV because
  `acquisition_run_id` is not present. Do not add a synthetic grouping column.

## Coverage and next data needed

The current data supports only limited, single-vehicle sensor/engine analysis and
general local knowledge retrieval. It does **not** support validated passenger-car,
motorcycle, pickup, van or minivan component-failure risk, service scheduling or
vehicle RUL. The full component matrix is in
[`vehicle_component_coverage.md`](vehicle_component_coverage.md).

To expand capability, collect legally reusable, provenance-documented fleet data with:

1. Stable pseudonymous vehicle IDs, vehicle type/make/model/year and relevant
   configuration; keep trips and time ordering.
2. Timestamped sensor measurements with units, calibration, operating context and
   missingness semantics.
3. Confirmed DTCs/faults, inspection outcomes, repair actions, component replacements,
   service dates and odometer readings, including negative/healthy observations.
4. For risk/RUL, failure timestamps or run-to-failure/degradation targets and
   enough independent vehicles/runs for group- and time-disjoint testing.
5. Verified maintenance schedules/manuals for exact vehicle variants, with citations
   and redistribution/commercial-use terms.
6. Representativeness across cars, motorcycles, pickups, vans and minivans; do not
   infer a category from synthetic or undocumented rows.
