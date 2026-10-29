# carOBD anomaly detection experiment

## Scope

Research-only unsupervised anomaly scoring. It identifies telemetry rows that are statistically unusual relative to training data; it does not diagnose a fault or predict future component failure.

## Data and preparation

- Raw sessions: 129
- Source rows: 304,299
- Unique prepared rows: 274,971
- Exact duplicate rows removed within files: 29,328
- Blank rows skipped: 5
- Rows with short field counts padded to header: 2,290
- Extra trailing empty fields removed: 288,537
- Missing selected-feature values: 0
- Selected features: ENGINE_RPM (), VEHICLE_SPEED (), THROTTLE (), ENGINE_LOAD (), COOLANT_TEMPERATURE (), LONG_TERM_FUEL_TRIM_BANK_1 (), SHORT_TERM_FUEL_TRIM_BANK_1 (), INTAKE_MANIFOLD_PRESSURE (), FUEL_TANK (), ABSOLUTE_THROTTLE_B ()
- CSV headers contain empty unit parentheses; physical units are not documented locally.
- Filenames identify idle/drive/live/long/ufpe sessions. `idle` labels follow README; stationary engine-running is separately described as speed=0 and RPM>0.
- The elapsed engine-run-time field was non-decreasing in source row order within every session, but many values tie and no calendar timestamps are present.

Exact full-row duplicates are removed only inside each file, retaining the first; repeated sensor values at different run times and all session boundaries are kept. Non-numeric/non-finite values become missing and the training-fitted median imputer handles missing features. No undocumented physical range filters were applied.

## Split and model

- Training: 192,293 earlier rows (approximately first 70% of each of 129 sessions).
- Evaluation: 82,678 later rows after the run-time boundary; rows tied at the boundary remain in training, so fractions may differ.
- Model: 300-tree Isolation Forest, random seed 42, in a scikit-learn Pipeline with median imputation fit on training rows only.
- Score is `-score_samples`; larger values are more unusual relative to the fitted training distribution.

## Evaluation results

| Threshold from training anomaly scores | Score threshold | Flagged | Evaluation rows | Flagged percent |
|---|---:|---:|---:|---:|
| training_99.0th_percentile | 0.622836 | 793 | 82,678 | 0.959% |
| training_95.0th_percentile | 0.544428 | 4,386 | 82,678 | 5.305% |
| training_90.0th_percentile | 0.506806 | 7,385 | 82,678 | 8.932% |

Default reporting threshold: **95th percentile of training anomaly scores**, flagging 4,386 of 82,678 evaluation records (5.305%). This is a thresholded outlier rate, not model accuracy or a mechanical failure rate.

### Score and temporal stability

- Evaluation score median: 0.421173; p95: 0.547967; p99: 0.620253.
- early evaluation segment: 27,606 rows; 95th-training-percentile flag rate 6.480%.
- middle evaluation segment: 27,558 rows; 95th-training-percentile flag rate 5.356%.
- late evaluation segment: 27,514 rows; 95th-training-percentile flag rate 4.074%.

### Baseline

A simple maximum absolute median/MAD-scaled feature deviation is fitted on training data. Its flag-rate sensitivity at matching training percentiles is recorded in `evaluation.json`; there are no labels to compare either method for correctness.

### Operating conditions

- `drive`: 11,063 evaluation rows; 95th-training-percentile flag rate 3.742%
- `idle`: 18,179 evaluation rows; 95th-training-percentile flag rate 0.000%
- `live`: 26,917 evaluation rows; 95th-training-percentile flag rate 2.879%
- `long`: 16,961 evaluation rows; 95th-training-percentile flag rate 17.098%
- `ufpe`: 9,558 evaluation rows; 95th-training-percentile flag rate 3.107%
- `stationary_engine_running`: 24,981 evaluation rows; 95th-training-percentile flag rate 0.068%
- `moving`: 57,693 evaluation rows; 95th-training-percentile flag rate 7.566%

No accuracy, precision, recall, F1, false-positive rate, or false-negative rate is reported: no independent normal/fault/anomaly labels exist. Flag rates by mode may reflect legitimate operating-condition differences.

## Artifacts

- Pipeline: `trained_models\experiments\carobd_sensor_anomaly_v1\pipeline.joblib`
- Dataset: `trained_models\experiments\carobd_sensor_anomaly_v1\prepared_carobd.csv`
- Evaluation: `trained_models\experiments\carobd_sensor_anomaly_v1\evaluation.json`
- Provenance, split, configuration, schema, and model card are in this same directory.

## Limitations and next work

The data represents one car and is not evidence of performance on any other vehicle. No fault labels exist; apparent outliers may be benign mode changes or data-acquisition effects. Sensor units and absolute dates are not documented. Validate next against mechanic-confirmed annotations and a separate vehicle cohort, after confirming data reuse terms. Keep these scores out of production decisions.
