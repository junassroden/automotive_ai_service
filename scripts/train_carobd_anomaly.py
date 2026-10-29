import json
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

from scripts.carobd_anomaly import (
    EXPERIMENT_ROOT,
    EXPERIMENTS_ROOT,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TRAIN_FRACTION,
    build_pipeline,
    chronological_group_split,
    resolve_within,
    sha256_file,
    train_thresholds,
    validate_feature_frame,
    write_json,
)


def train(experiment_dir=EXPERIMENT_ROOT):
    experiment_dir = resolve_within(experiment_dir, EXPERIMENTS_ROOT)
    if not experiment_dir.is_dir():
        raise FileNotFoundError(
            f"Prepare the experiment first using scripts/prepare_carobd_anomaly.py: {experiment_dir}"
        )
    prepared_path = experiment_dir / "prepared_carobd.csv"
    provenance_path = experiment_dir / "dataset_provenance.json"
    for path in (prepared_path, provenance_path):
        if not path.is_file():
            raise FileNotFoundError(f"Prepared experiment input is missing: {path}")
    model_path = experiment_dir / "pipeline.joblib"
    config_path = experiment_dir / "training_config.json"
    if model_path.exists() or config_path.exists():
        raise FileExistsError("Refusing to overwrite an existing trained experiment.")

    import pandas as pd

    frame = pd.read_csv(prepared_path, low_memory=False)
    training, evaluation, session_splits = chronological_group_split(
        frame,
        TRAIN_FRACTION,
    )
    train_features = validate_feature_frame(training)
    evaluation_features = validate_feature_frame(evaluation)
    if train_features.isna().all(axis=0).any():
        raise ValueError("A selected feature is entirely missing in training data.")

    pipeline = build_pipeline(RANDOM_STATE)
    pipeline.fit(train_features)
    training_scores = -pipeline.score_samples(train_features)
    thresholds = train_thresholds(training_scores)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(pipeline, model_path)

    split = {
        "split_strategy": (
            "within-session chronological forward holdout with tied elapsed run times "
            "kept wholly on one side of the boundary"
        ),
        "train_fraction": TRAIN_FRACTION,
        "train_rows": int(len(training)),
        "evaluation_rows": int(len(evaluation)),
        "source_sessions": int(frame["source_session"].nunique()),
        "session_splits": session_splits,
        "cross_session_overlap": "source session is intentionally shared; later rows only in evaluation",
        "absolute_calendar_time_available": False,
        "elapsed_time_order_check": (
            "source order, verified non-decreasing engine run time within each CSV; "
            "training and evaluation run-time values do not overlap"
        ),
    }
    write_json(experiment_dir / "split_manifest.json", split)
    schema = {
        "features": [
            {
                "name": name,
                "unit": "not specified in local CSV header (empty `()` unit field)",
                "order": index,
            }
            for index, name in enumerate(FEATURE_COLUMNS)
        ],
        "excluded_metadata": [
            "ENGINE_RUN_TINE () elapsed-time field",
            "source_session",
            "operating_mode",
            "source_row_index",
            "within_session_order",
            "stationary_engine_running",
            "moving",
        ],
        "missing_value_handling": (
            "median SimpleImputer fit on training rows only; nonnumeric/nonfinite values "
            "are represented as missing during preparation."
        ),
    }
    write_json(experiment_dir / "feature_schema.json", schema)
    config = {
        "experiment_name": experiment_dir.name,
        "task": "unsupervised OBD-II sensor-pattern anomaly scoring",
        "not_a_failure_prediction_model": True,
        "features": FEATURE_COLUMNS,
        "random_state": RANDOM_STATE,
        "estimator": {
            "class": "sklearn.ensemble.IsolationForest",
            "n_estimators": 300,
            "contamination": "auto",
            "max_samples": "auto",
        },
        "score_definition": (
            "anomaly_score = -Pipeline.score_samples(X); larger scores are more unusual "
            "relative to the fitted training distribution."
        ),
        "thresholds": {
            f"training_score_percentile_{key}": value
            for key, value in thresholds.items()
        },
        "training_rows": int(len(training)),
        "evaluation_rows": int(len(evaluation)),
        "train_fraction_per_session": TRAIN_FRACTION,
        "prepared_dataset_sha256": sha256_file(prepared_path),
        "provenance_sha256": sha256_file(provenance_path),
        "pipeline_sha256": sha256_file(model_path),
        "training_session_modes": training["operating_mode"].value_counts().to_dict(),
        "evaluation_session_modes": evaluation["operating_mode"].value_counts().to_dict(),
    }
    write_json(config_path, config)
    card = (
        "# carOBD sensor anomaly experiment\n\n"
        "This research-only pipeline scores unusual patterns in ten numeric OBD-II "
        "telemetry fields from the locally documented 2014 Toyota Etios data. It does "
        "not identify confirmed mechanical faults, component failures, maintenance needs, "
        "or future failures. No verified anomaly/fault labels are available.\n\n"
        "The split targets 70% training per source CSV in non-decreasing engine-run-time/"
        "source order; rows sharing the boundary run-time value stay together, so the exact "
        "fraction may differ. The later tail is held out with no run-time value overlap. "
        "Median imputation "
        "is fitted on training records only. File mode and session IDs are descriptive "
        "metadata, not model inputs. Units are not supplied in local headers. The data "
        "represents one vehicle and cannot establish generalization to any other vehicle.\n\n"
        "Anomaly flags are threshold-relative statistical outliers. They may reflect "
        "operating-mode changes, sensor update behavior, missingness, or unusual-but-normal "
        "conditions. They must not be interpreted as diagnostic findings or used for "
        "maintenance decisions. Threshold sensitivity is reported rather than presenting "
        "an arbitrary flag rate as accuracy.\n"
    )
    (experiment_dir / "model_card.md").write_text(card, encoding="utf-8")
    print(
        f"Trained Isolation Forest on {len(training)} rows; held out "
        f"{len(evaluation)} later rows across {frame['source_session'].nunique()} sessions."
    )
    print(json.dumps(config["thresholds"], indent=2))
    return config


if __name__ == "__main__":
    train()
