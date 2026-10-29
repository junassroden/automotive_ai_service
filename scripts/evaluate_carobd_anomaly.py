import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

from scripts.carobd_anomaly import (
    EXPERIMENT_ROOT,
    EXPERIMENTS_ROOT,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    chronological_group_split,
    load_pipeline,
    resolve_within,
    robust_distance,
    sha256_file,
    train_thresholds,
    validate_feature_frame,
    write_json,
)


def period_summary(frame, flagged, scores, thresholds):
    evaluation = frame.reset_index(drop=True).copy()
    evaluation["anomaly_score"] = scores
    evaluation["flagged"] = flagged
    evaluation["temporal_period"] = ""
    for _, indices in evaluation.groupby("source_session", sort=True).groups.items():
        ordered = evaluation.loc[list(indices)].sort_values(
            "within_session_order",
            kind="stable",
        ).index
        labels = np.array_split(np.asarray(ordered), 3)
        for label, positions in zip(("early", "middle", "late"), labels):
            evaluation.loc[positions, "temporal_period"] = label
    results = {}
    for period, group in evaluation.groupby("temporal_period", sort=False):
        if group.empty:
            continue
        results[period] = {
            "rows": int(len(group)),
            "median_anomaly_score": float(group["anomaly_score"].median()),
            "p95_anomaly_score": float(group["anomaly_score"].quantile(0.95)),
            "flag_rates_by_training_percentile": {
                key: float((group["anomaly_score"] >= threshold).mean())
                for key, threshold in thresholds.items()
            },
        }
    return results


def mode_summary(frame, scores, thresholds):
    evaluation = frame.reset_index(drop=True).copy()
    evaluation["anomaly_score"] = scores
    results = {}
    for mode, group in evaluation.groupby("operating_mode", sort=True):
        results[mode] = {
            "rows": int(len(group)),
            "median_anomaly_score": float(group["anomaly_score"].median()),
            "p95_anomaly_score": float(group["anomaly_score"].quantile(0.95)),
            "flag_rates_by_training_percentile": {
                key: float((group["anomaly_score"] >= threshold).mean())
                for key, threshold in thresholds.items()
            },
        }
    for name, mask in (
        (
            "stationary_engine_running",
            frame["stationary_engine_running"].astype(bool).to_numpy(),
        ),
        ("moving", frame["moving"].astype(bool).to_numpy()),
    ):
        subset = evaluation.loc[mask]
        results[name] = {
            "rows": int(len(subset)),
            "median_anomaly_score": (
                float(subset["anomaly_score"].median()) if len(subset) else None
            ),
            "flag_rates_by_training_percentile": {
                key: (
                    float((subset["anomaly_score"] >= threshold).mean())
                    if len(subset)
                    else None
                )
                for key, threshold in thresholds.items()
            },
        }
    return results


def evaluate(experiment_dir=EXPERIMENT_ROOT):
    experiment_dir = resolve_within(experiment_dir, EXPERIMENTS_ROOT)
    prepared_path = experiment_dir / "prepared_carobd.csv"
    provenance_path = experiment_dir / "dataset_provenance.json"
    model_path = experiment_dir / "pipeline.joblib"
    config_path = experiment_dir / "training_config.json"
    split_path = experiment_dir / "split_manifest.json"
    for path in (prepared_path, provenance_path, model_path, config_path, split_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required anomaly experiment input missing: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if sha256_file(prepared_path) != config["prepared_dataset_sha256"]:
        raise ValueError("Prepared dataset hash differs from the training config.")
    if sha256_file(provenance_path) != config["provenance_sha256"]:
        raise ValueError("Dataset provenance hash differs from the training config.")
    if sha256_file(model_path) != config["pipeline_sha256"]:
        raise ValueError("Pipeline hash differs from the training config.")
    if config["features"] != FEATURE_COLUMNS:
        raise ValueError("Feature schema/order differs from the saved training config.")

    frame = pd.read_csv(prepared_path, low_memory=False)
    training, evaluation, session_splits = chronological_group_split(
        frame,
        split["train_fraction"],
    )
    if session_splits != split["session_splits"]:
        raise ValueError("Chronological session split no longer matches its saved manifest.")
    model = load_pipeline(model_path)
    training_features = validate_feature_frame(training)
    evaluation_features = validate_feature_frame(evaluation)
    if list(training_features.columns) != config["features"]:
        raise ValueError("Training feature order differs from the saved feature schema.")
    train_scores = -model.score_samples(training_features)
    scores = -model.score_samples(evaluation_features)
    if not np.isfinite(scores).all() or len(scores) != len(evaluation):
        raise ValueError("Loaded pipeline produced non-finite or misaligned scores.")
    thresholds = train_thresholds(train_scores)

    sensitivity = {}
    for percentile_text, threshold in thresholds.items():
        flags = scores >= threshold
        sensitivity[f"training_{percentile_text}th_percentile"] = {
            "score_threshold": threshold,
            "flagged_records": int(flags.sum()),
            "evaluation_records": int(len(flags)),
            "flagged_percentage": float(flags.mean() * 100),
        }

    baseline_train, baseline_eval = robust_distance(training, evaluation)
    baseline_thresholds = train_thresholds(baseline_train)
    baseline_sensitivity = {}
    for percentile_text, threshold in baseline_thresholds.items():
        flags = baseline_eval >= threshold
        baseline_sensitivity[f"training_{percentile_text}th_percentile"] = {
            "score_threshold": threshold,
            "flagged_records": int(flags.sum()),
            "flagged_percentage": float(flags.mean() * 100),
        }

    default_key = "95.0"
    default_threshold = thresholds[default_key]
    default_flags = scores >= default_threshold
    results = {
        "status": "evaluated_unsupervised_experiment",
        "task": "unlabeled OBD-II sensor-pattern anomaly scoring",
        "not_a_fault_classifier_or_future_failure_predictor": True,
        "dataset": "carOBD Toyota Etios telemetry",
        "source_vehicle_claim": "one 2014 Toyota Etios, according to local README",
        "training_rows": int(len(training)),
        "evaluation_rows": int(len(evaluation)),
        "features": config["features"],
        "feature_units": "not specified in local CSV headers",
        "training_period": {
            "definition": (
                "approximately first 70% of each CSV session in source order / "
                "non-decreasing engine run time; tied boundary times stay on one side"
            ),
            "session_count": int(training["source_session"].nunique()),
        },
        "evaluation_period": {
            "definition": (
                "later session tail after the run-time boundary; boundary ties stay in "
                "training, with no elapsed run-time value overlap"
            ),
            "session_count": int(evaluation["source_session"].nunique()),
            "absolute_calendar_timestamps_available": False,
        },
        "score_definition": config["score_definition"],
        "score_distribution": {
            "minimum": float(np.min(scores)),
            "p25": float(np.quantile(scores, 0.25)),
            "median": float(np.median(scores)),
            "p75": float(np.quantile(scores, 0.75)),
            "p90": float(np.quantile(scores, 0.90)),
            "p95": float(np.quantile(scores, 0.95)),
            "p99": float(np.quantile(scores, 0.99)),
            "maximum": float(np.max(scores)),
        },
        "threshold_sensitivity": sensitivity,
        "default_threshold": {
            "selection": "95th percentile of training anomaly scores",
            "score_threshold": default_threshold,
            "flagged_records": int(default_flags.sum()),
            "evaluation_records": int(len(default_flags)),
            "flagged_percentage": float(default_flags.mean() * 100),
            "interpretation": "flag rate, not accuracy or error rate",
        },
        "simple_baseline": {
            "method": "maximum absolute median/MAD-scaled feature deviation",
            "feature statistics_fit_on_training_only": True,
            "threshold_sensitivity": baseline_sensitivity,
        },
        "temporal_stability": period_summary(
            evaluation,
            default_flags,
            scores,
            thresholds,
        ),
        "operating_mode_analysis": mode_summary(evaluation, scores, thresholds),
        "supervised_metrics": {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "reason": "No independently verified normal/fault/anomaly labels are present.",
        },
        "duplicate_and_missing_handling": json.loads(
            provenance_path.read_text(encoding="utf-8")
        )["statistics"],
        "model_loading_smoke_test": {
            "passed": True,
            "loaded_pipeline_class": type(model.named_steps["isolation_forest"]).__name__,
            "finite_evaluation_scores": True,
            "scored_records": int(len(scores)),
        },
        "random_state": RANDOM_STATE,
        "limitations": [
            "One vehicle only; no cross-vehicle generalization has been demonstrated.",
            "No ground-truth fault/anomaly labels; flag rates are not accuracy or fault rates.",
            "Unusual scores may reflect mode/drive changes, sensor polling/update behavior, or benign operating variation.",
            "Sensor units are absent from the local CSV headers.",
            "Engine run time is a within-session relative clock, not a calendar timestamp; time ties are common.",
            "Threshold is selected from training score quantiles and is a sensitivity setting, not validated operational calibration.",
            "Exact duplicate records are removed within sessions, but nearby correlated observations remain.",
            "The local dataset has no verified formal data license; confirm reuse terms before redistribution.",
        ],
        "artifact_sha256": sha256_file(model_path),
        "prepared_dataset_sha256": sha256_file(prepared_path),
        "cross_session_overlap": split["cross_session_overlap"],
    }
    write_json(experiment_dir / "evaluation.json", results)
    write_report(experiment_dir, results, provenance_path)
    print(json.dumps(results, indent=2))
    return results


def write_report(experiment_dir, results, provenance_path):
    stats = results["duplicate_and_missing_handling"]
    lines = [
        "# carOBD anomaly detection experiment",
        "",
        "## Scope",
        "",
        "Research-only unsupervised anomaly scoring. It identifies telemetry rows that "
        "are statistically unusual relative to training data; it does not diagnose a fault "
        "or predict future component failure.",
        "",
        "## Data and preparation",
        "",
        f"- Raw sessions: {json.loads(provenance_path.read_text(encoding='utf-8'))['file_count']}",
        f"- Source rows: {stats['source_rows']:,}",
        f"- Unique prepared rows: {stats['usable_rows']:,}",
        f"- Exact duplicate rows removed within files: {stats['duplicate_rows_removed']:,}",
        f"- Blank rows skipped: {stats['blank_rows']:,}",
        f"- Rows with short field counts padded to header: {stats['short_rows_padded']:,}",
        f"- Extra trailing empty fields removed: {stats['trailing_empty_fields_removed']:,}",
        f"- Missing selected-feature values: {stats['selected_feature_missing_cells']:,}",
        f"- Selected features: {', '.join(results['features'])}",
        "- CSV headers contain empty unit parentheses; physical units are not documented locally.",
        "- Filenames identify idle/drive/live/long/ufpe sessions. `idle` labels follow README; "
        "stationary engine-running is separately described as speed=0 and RPM>0.",
        "- The elapsed engine-run-time field was non-decreasing in source row order within every "
        "session, but many values tie and no calendar timestamps are present.",
        "",
        "Exact full-row duplicates are removed only inside each file, retaining the first; "
        "repeated sensor values at different run times and all session boundaries are kept. "
        "Non-numeric/non-finite values become missing and the training-fitted median imputer "
        "handles missing features. No undocumented physical range filters were applied.",
        "",
        "## Split and model",
        "",
        f"- Training: {results['training_rows']:,} earlier rows (approximately first 70% of each of "
        f"{results['training_period']['session_count']} sessions).",
        f"- Evaluation: {results['evaluation_rows']:,} later rows after the run-time "
        "boundary; rows tied at the boundary remain in training, so fractions may differ.",
        "- Model: 300-tree Isolation Forest, random seed 42, in a scikit-learn Pipeline with "
        "median imputation fit on training rows only.",
        "- Score is `-score_samples`; larger values are more unusual relative to the fitted "
        "training distribution.",
        "",
        "## Evaluation results",
        "",
        "| Threshold from training anomaly scores | Score threshold | Flagged | Evaluation rows | Flagged percent |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, value in results["threshold_sensitivity"].items():
        lines.append(
            f"| {key} | {value['score_threshold']:.6f} | {value['flagged_records']:,} | "
            f"{value['evaluation_records']:,} | {value['flagged_percentage']:.3f}% |"
        )
    lines += [
        "",
        f"Default reporting threshold: **{results['default_threshold']['selection']}**, "
        f"flagging {results['default_threshold']['flagged_records']:,} of "
        f"{results['evaluation_rows']:,} evaluation records "
        f"({results['default_threshold']['flagged_percentage']:.3f}%). This is a thresholded "
        "outlier rate, not model accuracy or a mechanical failure rate.",
        "",
        "### Score and temporal stability",
        "",
        f"- Evaluation score median: {results['score_distribution']['median']:.6f}; "
        f"p95: {results['score_distribution']['p95']:.6f}; "
        f"p99: {results['score_distribution']['p99']:.6f}.",
    ]
    for period, value in results["temporal_stability"].items():
        rate = value["flag_rates_by_training_percentile"]["95.0"]
        lines.append(
            f"- {period} evaluation segment: {value['rows']:,} rows; "
            f"95th-training-percentile flag rate {rate * 100:.3f}%."
        )
    lines += [
        "",
        "### Baseline",
        "",
        "A simple maximum absolute median/MAD-scaled feature deviation is fitted on training "
        "data. Its flag-rate sensitivity at matching training percentiles is recorded in "
        "`evaluation.json`; there are no labels to compare either method for correctness.",
        "",
        "### Operating conditions",
        "",
    ]
    for mode, value in results["operating_mode_analysis"].items():
        rate = value["flag_rates_by_training_percentile"]["95.0"]
        lines.append(
            f"- `{mode}`: {value['rows']:,} evaluation rows; "
            f"95th-training-percentile flag rate "
            f"{rate * 100:.3f}%" if rate is not None else f"- `{mode}`: no rows."
        )
    lines += [
        "",
        "No accuracy, precision, recall, F1, false-positive rate, or false-negative rate is "
        "reported: no independent normal/fault/anomaly labels exist. Flag rates by mode may "
        "reflect legitimate operating-condition differences.",
        "",
        "## Artifacts",
        "",
        f"- Pipeline: `{(experiment_dir / 'pipeline.joblib').relative_to(SERVICE_ROOT)}`",
        f"- Dataset: `{(experiment_dir / 'prepared_carobd.csv').relative_to(SERVICE_ROOT)}`",
        f"- Evaluation: `{(experiment_dir / 'evaluation.json').relative_to(SERVICE_ROOT)}`",
        "- Provenance, split, configuration, schema, and model card are in this same directory.",
        "",
        "## Limitations and next work",
        "",
        "The data represents one car and is not evidence of performance on any other vehicle. "
        "No fault labels exist; apparent outliers may be benign mode changes or data-acquisition "
        "effects. Sensor units and absolute dates are not documented. Validate next against "
        "mechanic-confirmed annotations and a separate vehicle cohort, after confirming data "
        "reuse terms. Keep these scores out of production decisions.",
        "",
    ]
    (experiment_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    evaluate()
