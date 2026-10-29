# AI Training Verification Audit

**Audit time:** 2026-10-29 09:28 Asia/Manila  
**Scope:** Read-only verification of the existing model artifacts, training records,
dataset evidence, and Django inference paths. No model was retrained, no project
settings or production behavior were changed, and no existing model artifact was
deserialized except the chatbot model through its trusted Django loader.

## Executive result

| Model/task | Result | Inference status |
|---|---|---|
| Legacy vehicle-maintenance artifact | `ARTIFACT_EXISTS_UNVERIFIED` | Blocked; not loaded or usable |
| Current EngineFaultDB engine-classification attempt | `TRAINING_BLOCKED` | No classifier created |
| Remaining-useful-life model | `NOT_FOUND` | Unsupported; no model found |
| Chatbot intent classifier | `VERIFIED_TRAINED_AND_EVALUATED` for authored-intent routing only | Loads and serves chat; not a maintenance model |

The maintenance `.joblib` file is not proof of a trained or evaluated
maintenance model. The current training status explicitly records a blocked
attempt because no acquisition-run identifier exists. The chatbot is different:
its training code, source-data hash, evaluation, artifact hash, and trusted
inference loading all match.

## Artifact inventory and provenance

The recursive model-format scan found **two project model artifacts** outside
the Python virtual environment. Eleven additional `.pkl` files are package test
fixtures under `venv/Lib/site-packages`; they are not project models and were not
loaded. No project `.pt`, `.pth`, `.onnx`, `.h5`, `.keras`, `.safetensors`,
`.ckpt`, or other model-format files were found.

| Model | Artifact; size; modified (UTC) | SHA-256 | Provenance and safe-load result |
|---|---|---|---|
| Maintenance legacy artifact | [`trained_models/vehicle_maintenance_model.joblib`](../../trained_models/vehicle_maintenance_model.joblib); 21,678,344 bytes; 2026-10-28 15:22:35Z | `876ae394173a696ac02fa98d7ee17489a310f6d2a4860b4a3019a1391b621af7` | Exists, but no producing script, training record, matching evaluation, or accepted registry entry was found. The placeholder file in the same folder instructs the owner to copy an existing artifact. It is not referenced by the current maintenance inference loader. **Not deserialized**: its provenance is unknown and the service correctly blocks before load. |
| Chatbot intent classifier | [`trained_models/chatbot_nlp.joblib`](../../trained_models/chatbot_nlp.joblib); 811,706 bytes; 2026-10-29 01:26:17Z | `72fe528173c1fec703b75f5ed179a0169667b255ab89226c375733429b66a01e` | Produced by the current `train_chatbot.py` workflow, as supported by its training timestamp, status record, source-data hash, and matching artifact hash in both evaluation reports. Loaded successfully through `ai_api.chatbot_service.load_chatbot`; it exposes 23 intent classes and 23 response entries. |

The maintenance artifact predates the 2026-10-29 training-status record and
current chatbot artifact timestamp. Its file timestamp alone does not establish
when, how, or on what data it was trained. The current project has no
`AI_MODEL_PATH` setting. The previous `AI_ENABLE_UNVERIFIED_MAINTENANCE_PREDICTIONS`
switch is also absent; maintenance eligibility is enforced by
`models/model_registry.json` instead.

## Model-by-model findings

### 1. Legacy vehicle-maintenance model

- **Status:** `ARTIFACT_EXISTS_UNVERIFIED`. Separately, the current maintenance
  training attempt is `TRAINING_BLOCKED`.
- **Expected task:** Vehicle maintenance risk/recommendations for fleet vehicles.
  Fleet prioritization invokes the same per-vehicle predictor; it is not a
  separately trained model.
- **Artifact provenance:** Unknown. The maintenance artifact path is not a current
  CLI output path; the current generic grouped trainer saves
  `grouped_classifier.joblib` under a candidate directory and explicitly does
  not promote candidates to production. No reproducible maintenance-model
  training source or artifact-specific historical run report was located.
- **Configured path / registry:** There is no `AI_MODEL_PATH` setting. The
  current loader reads `models/model_registry.json`; that registry has a
  `maintenance_model` entry with `status: blocked`, `artifact_path: null`, no
  evaluation, and `eligible_for_inference: false`. Therefore the legacy file is
  not the registered model and the service does not deserialize it.
- **Expected schema:** Project documentation describes a legacy interface with
  feature names `vehicle_type`, `brand`, `model`, `engine_type`, `make_year`,
  `region`, `mileage_range`, and `mileage`, plus maintenance target classifiers.
  Those are documentation claims only: the serialized artifact was not opened,
  so its actual features, targets, preprocessing, class labels, probabilities,
  and fitted estimator state remain unverified. Earlier notes about 12
  classifiers and two constant targets are also unverified.
- **Datasets / records:** No dataset can be attributed to this artifact.
  The current pipeline did **not** train it on downloaded data. The downloaded
  EngineFaultDB dataset is not a vehicle-maintenance-label dataset.
- **Evaluation:** No metrics, baseline, confusion matrix, group/time split,
  calibration record, or matching model evaluation were found. There are no
  verified training/test record counts or independent groups for this artifact.
- **Inference checks:** The current registry gate returned blocked. The
  maintenance API returned HTTP 503 with `status: unavailable` and
  `code: model_not_eligible`, without returning scores. A controlled test
  fixture verified that a correctly registered accepted-format model can pass
  through the loader and produce a prediction; this tests plumbing only and is
  not evidence about the existing maintenance artifact.
- **Suitability:** Not suitable for vehicle-specific or safety-related
  maintenance predictions until provenance, target semantics, compatible
  features, group-disjoint evaluation, probability validity/calibration, and
  acceptance evidence are established.
- **Required evidence/action:** Obtain the original training code and source
  data, or train a versioned candidate from eligible records with vehicle/run
  identifiers and confirmed maintenance outcomes. Create an evaluation report
  for that candidate and register/promote it only after it passes the
  configured acceptance checks. Do not copy or register the current artifact
  as accepted based solely on its existence.

### 2. EngineFaultDB laboratory engine-state classification

- **Status:** `TRAINING_BLOCKED`. No classifier artifact was created.
- **Task and target:** Four-class laboratory engine-state classification using
  numeric `fault_class` values 0–3. The dataset documentation does not provide
  a verified mapping from these numeric values to named fault mechanisms.
- **Dataset:** `data/raw/EngineFaultDB-main/EngineFaultDB-main/EngineFaultDB_Final.csv`,
  prepared separately at
  `data/processed/enginefaultdb/enginefaultdb_normalized.csv`.
  Preparation reported 55,999 source rows and 55,998 processed rows after
  dropping one exact duplicate; zero missing values and no non-finite numeric
  values. Class counts are 0: 16,000; 1: 10,998; 2: 15,000; 3: 14,000.
- **Features:** 14 numerical laboratory signals: MAP, throttle position, force,
  power, RPM, fuel-consumption measures, speed, CO, HC, CO2, O2, lambda, and
  air-fuel ratio, normalized with unit-bearing names. The target is not included
  as a feature by the grouped trainer.
- **Training/evaluation:** The actual `train-maintenance` run failed its
  prerequisite check because `acquisition_run_id` is absent. It therefore
  created no training split, fitted classifier, test rows, metrics, or model
  artifact. The status field's `training_records: 55998` reflects prepared
  candidate rows, **not fitted training records**; actual fitted training and
  test records are both zero.
- **Leakage and scope:** No vehicle/run/trip grouping is supplied, so related
  observations cannot be separated reliably. The code refuses row-random
  training. The data describes a controlled single C14NE engine, not fleet
  vehicle-maintenance outcomes; it cannot establish future failure risk for
  cars, motorcycles, pickups, vans, or minivans.
- **Suitability / next step:** Not an accepted maintenance model. To conduct
  even laboratory engine-state evaluation, obtain a documented run or
  acquisition-group identifier and verified class mapping. To train fleet
  maintenance, collect separate, provenance-documented fleet data with
  confirmed outcomes and independent vehicle/run IDs.

### 3. Remaining-useful-life model

- **Status:** `NOT_FOUND`.
- **Artifact, training source, dataset, evaluation, feature schema, and target
  definition:** None were found for a vehicle-fleet remaining-useful-life
  model. Project capability metadata explicitly says remaining useful life is
  unsupported for fleet vehicles.
- **Suitability / next step:** Not available. A supported RUL task requires
  time-ordered component degradation or run-to-failure records, timestamps,
  vehicle/component identifiers, and independent-group/time-separated
  evaluation. Laboratory battery-cell aging alone does not establish vehicle
  battery-pack or fleet-component RUL.

### 4. Chatbot intent classifier

- **Status:** `VERIFIED_TRAINED_AND_EVALUATED`, limited to the defined
  authored-example intent-classification task.
- **Training source:** `train_chatbot.py`; source
  `data/chatbot_intents.json`, SHA-256
  `5c9f4709e01036b2285ff7354c10631a4ff686b40d30d361cb3d11bb7f7469b5`.
  This is the project's authored chatbot dataset, **not a downloaded file from
  `data/raw/`**.
- **Data/targets:** 2,780 text examples across 23 intent tags. Minimum class
  size is 23, maximum 1,920; `troubleshooting` is about 69.1% of all examples,
  so the classes are materially imbalanced. The target is the intent tag;
  mapped response text is stored in the artifact. No vehicle measurement or
  maintenance outcome is a target.
- **Features/model:** Text is vectorized with TF-IDF (lowercase, Unicode
  accent stripping, unigrams and bigrams, sublinear term frequency), followed
  by balanced `LogisticRegression` (`max_iter=3000`, random state 42). This is
  a classifier, not a generative language model.
- **Training/evaluation split:** A fixed, stratified 75/25 split (`random_state=42`)
  evaluates 2,085 train and 695 test examples. The final serialized classifier
  is then fitted on all 2,780 examples. The reported holdout metrics therefore
  assess the fixed model-building procedure, not an untouched evaluation of
  the final all-data estimator. The training report binds the final artifact
  hash to the run.
- **Metrics:** Accuracy 0.9381; macro F1 0.8618; weighted F1 0.9400.
  Most-frequent baseline accuracy 0.6906 and macro F1 0.0355. The configured
  internal macro-F1 threshold is 0.75 and the candidate exceeds both that
  threshold and the baseline. Per-class precision/recall/F1 and the full
  23-by-23 confusion matrix are in
  [`chatbot_training.json`](chatbot_training.json) and
  [`trained_models/chatbot_evaluation.json`](../../trained_models/chatbot_evaluation.json).
  ROC-AUC/PR-AUC and balanced accuracy are not reported.
- **Artifact/report consistency:** The SHA-256 in the report and training
  status matches the current `chatbot_nlp.joblib`; the data hash matches the
  current JSON source. Both evaluation files contain the same run metrics.
- **Load/inference test:** Trusted Django loader successfully loaded 23 classes
  and 23 response entries. A live Django test-client smoke request returned
  HTTP 200 and classified “hello” as `greeting`. This verifies loading and
  intent routing, not correctness of generated mechanical advice or
  maintenance predictions.
- **Limitations / suitability:** Suitable only as the small local intent router
  it was trained to be. Authored examples may share wording/templates across
  the random split, and the report itself warns that this can inflate scores.
  It needs independent real user questions and class-wise review before
  treating its scores as real-world generalization. It does not diagnose a
  vehicle, estimate failure probabilities, or validate the maintenance model.

## Dataset and training-history verification

The existing recursive inventory records **237 files across 14 top-level
directories** under `data/raw/`, with CSV, MAT, MBL, XLSX, Parquet, JSON, XML,
ZIP, notebooks, code, and supporting documentation. The raw inventory and
dataset reports were used as the dataset evidence; original files were not
changed. No downloaded dataset is recorded as the source for the chatbot.

The only dataset explicitly attempted by the maintenance training command was
the prepared EngineFaultDB copy described above. It was blocked before fitting.
The downloaded datasets with telemetry but no repair/failure labels, unknown
provenance, synthetic industrial-machine data, or laboratory cells do not
constitute verified fleet-maintenance outcomes. No valid maintenance evaluation
set is available.

Training code existing in the repository is not training history. The
`training_status.json` entry, `model_registry.json`, and current CLI run all
show maintenance status blocked. Generic `train_grouped_classifier()` has not
been run on project data: no grouped candidate artifact or evaluation output
is present. Its ability to train when given valid inputs does not prove that a
maintenance model was trained.

## Django inference verification

- Current maintenance configuration uses
  `AI_MAINTENANCE_MODEL_REGISTRY_PATH` (`models/model_registry.json`), not
  `AI_MODEL_PATH`. The old `AI_ENABLE_UNVERIFIED_MAINTENANCE_PREDICTIONS`
  setting is not present or used.
- The service checks registry status, artifact/report hashes, feature schema,
  group-disjoint metrics, baseline criteria, and calibrated-probability
  evidence before deserializing a maintenance artifact. Current status stops
  at `blocked`.
- A registered test fixture with controlled evaluation metadata and a
  compatible feature schema generated a prediction. This fixture is synthetic
  test plumbing and does not change model status or supply evidence for any
  real model.
- The unregistered legacy artifact was not deserialized. `POST /api/predict/`
  returned HTTP 503 with a clear unavailable response and no risk score.
- The chatbot has an independent loader and model path. It loaded successfully
  and produced an intent result. That does not make maintenance inference
  available.

## Verification commands and results

Run from `automotive_ai_service`:

```powershell
.\venv\Scripts\python.exe manage.py test
.\venv\Scripts\python.exe manage.py test ai_api.tests.HealthEndpointTests ai_api.tests.PredictionEndpointTests
```

Results during this audit: full Django suite **41 passed**, system check clean;
focused maintenance endpoint tests **6 passed**, system check clean. The actual
chatbot endpoint smoke request returned 200; the maintenance prediction request
returned 503 unavailable. No training or inventory-refresh command was run in
this verification-only audit.

## Intended-task completion percentage

For this calculation, the denominator is the **four distinct learned-model
tasks explicitly named in the current service's capabilities/training status**:

1. Fleet vehicle-maintenance risk prediction (fleet prioritization uses this
   same predictor, so it is not counted twice).
2. EngineFaultDB laboratory engine-fault/state classification.
3. Fleet remaining-useful-life estimation.
4. Chatbot intent classification.

General local automotive-knowledge retrieval is excluded because it is not a
trained model. The deployed capability/readiness fields enumerate the above
model tasks; this percentage is limited to that inspectable project task
inventory, not any unrecorded future AI scope.

**Fully verified trained and evaluated:** 1 task (chatbot intent classification).  
**Identified task denominator:** 4.  
**Verified completion:** **1 / 4 = 25%**.

## Direct answers

- **Was the maintenance AI actually trained?** Not verifiably. A maintenance
  artifact exists, but its provenance is unknown; the current reproducible
  maintenance training run was blocked before fitting.
- **Was it trained using the downloaded datasets?** No evidence supports that.
  The legacy artifact has no traceable dataset; the current EngineFaultDB attempt
  used prepared downloaded data only to discover the missing group identifier
  and did not fit a model.
- **Did it pass a valid evaluation?** No. No maintenance evaluation metrics or
  matching report exist.
- **Can Django use it safely?** No. The registry marks the maintenance task
  blocked; the API fails closed with HTTP 503 and no fabricated score.
- **Was the chatbot AI trained successfully?** Yes, as a 23-class authored
  intent classifier. Its source/report/artifact hashes match, its holdout
  metrics pass the configured internal threshold, and the trusted loader and
  endpoint smoke test succeeded. This does not verify general mechanical
  advice or maintenance prediction.
- **What percentage of intended AI tasks are verified complete?** **25%**
  (1 of the 4 explicitly enumerated model tasks above).

## Files created by this audit

- `data/reports/ai_training_audit.md`
- `data/reports/ai_training_audit.json`

