import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

from scripts.maintenance_experiment import (
    EXPERIMENT_ROOT,
    EXPERIMENTS_ROOT,
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    PROCESSED_ROOT,
    SERVICE_ROOT,
    TARGET_COLUMN,
    read_prepared_data,
    resolve_within,
    sha256_file,
    write_json,
)


def regression_metrics(y_true, prediction):
    return {
        "mae_cycles": float(mean_absolute_error(y_true, prediction)),
        "rmse_cycles": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "r2": float(r2_score(y_true, prediction)) if len(y_true) > 1 else None,
    }


def evaluate(experiment_dir=EXPERIMENT_ROOT):
    experiment_dir = resolve_within(experiment_dir, EXPERIMENTS_ROOT)
    config_path = experiment_dir / "training_config.json"
    split_path = experiment_dir / "split_manifest.json"
    artifact_path = experiment_dir / "pipeline.joblib"
    for path in (config_path, split_path, artifact_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required training/evaluation file missing: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_path = resolve_within(
        SERVICE_ROOT / config["prepared_dataset_path"],
        PROCESSED_ROOT,
    )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Prepared dataset missing: {dataset_path}")
    experiment_manifest_path = experiment_dir / "dataset_manifest.json"
    if not experiment_manifest_path.is_file():
        raise FileNotFoundError(f"Dataset provenance manifest missing from {experiment_dir}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if sha256_file(artifact_path) != config["artifact_sha256"]:
        raise ValueError("Model artifact hash does not match the saved training config.")
    if sha256_file(dataset_path) != config["prepared_dataset_sha256"]:
        raise ValueError("Prepared dataset hash differs from the training dataset.")
    if sha256_file(experiment_manifest_path) != config["dataset_manifest_sha256"]:
        raise ValueError("Copied dataset manifest differs from the training manifest.")
    if config["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Saved model feature schema/order does not match evaluator schema.")
    if split["group_overlap"] != 0 or set(split["train_cells"]) & set(split["test_cells"]):
        raise ValueError("Saved split contains cell-group leakage.")

    frame = read_prepared_data(dataset_path)
    train_frame = frame[frame[GROUP_COLUMN].isin(split["train_cells"])]
    test_frame = frame[frame[GROUP_COLUMN].isin(split["test_cells"])]
    if len(train_frame) != split["train_rows"] or len(test_frame) != split["test_rows"]:
        raise ValueError("Prepared rows no longer match the saved split manifest.")
    if set(train_frame[GROUP_COLUMN]) & set(test_frame[GROUP_COLUMN]):
        raise ValueError("Cell-group overlap detected during evaluation.")

    model = joblib.load(artifact_path)
    y_test = test_frame[TARGET_COLUMN]
    predictions = model.predict(test_frame[FEATURE_COLUMNS])
    if len(predictions) != len(test_frame) or not np.isfinite(predictions).all():
        raise ValueError("Loaded artifact returned invalid prediction values.")
    baseline = DummyRegressor(strategy="median")
    baseline.fit(train_frame[FEATURE_COLUMNS], train_frame[TARGET_COLUMN])
    baseline_predictions = baseline.predict(test_frame[FEATURE_COLUMNS])

    per_cell = {}
    for cell_id, group in test_frame.assign(prediction=predictions).groupby(GROUP_COLUMN):
        per_cell[cell_id] = regression_metrics(
            group[TARGET_COLUMN],
            group["prediction"],
        )
    results = {
        "task": config["task"],
        "status": "evaluated_experiment",
        "target": TARGET_COLUMN,
        "test_rows": len(test_frame),
        "test_cells": sorted(test_frame[GROUP_COLUMN].unique().tolist()),
        "train_cells": split["train_cells"],
        "group_overlap": 0,
        "split_strategy": split["splitter"],
        "random_state": split["random_state"],
        "features": FEATURE_COLUMNS,
        "metrics": {
            "random_forest": regression_metrics(y_test, predictions),
            "median_baseline": regression_metrics(y_test, baseline_predictions),
        },
        "per_test_cell": per_cell,
        "artifact_path": str(artifact_path.relative_to(SERVICE_ROOT)),
        "artifact_sha256": sha256_file(artifact_path),
        "prepared_dataset_sha256": sha256_file(dataset_path),
        "inference_smoke_test": {
            "passed": True,
            "rows_predicted": len(predictions),
            "finite_predictions": True,
        },
        "limitations": [
            "Evaluation has one held-out laboratory cell and two training cells; the estimate is highly uncertain.",
            "The selected cells use different discharge cutoff voltages; this can confound cross-cell generalization.",
            "There is no fleet maintenance or vehicle failure evaluation.",
            "This experiment is not registered, promoted, or used by Django inference.",
        ],
    }
    write_json(experiment_dir / "evaluation.json", results)
    write_training_report(results, config, split)
    print(json.dumps(results, indent=2))
    return results


def write_training_report(results, config, split):
    model_metrics = results["metrics"]["random_forest"]
    baseline = results["metrics"]["median_baseline"]
    lines = [
        "# Training and evaluation report",
        "",
        "## Selected task",
        "",
        "A separately isolated NASA laboratory Li-ion battery-cell RUL experiment. The "
        "target is discharge cycles remaining until first positive measured capacity "
        "at or below 1.4 Ah. This does not predict future failure in a car, motorcycle, "
        "pickup, van, minivan, or fleet.",
        "",
        "## Dataset and split",
        "",
        f"- Prepared rows: {config['training_rows'] + config['test_rows']}",
        f"- Training rows / cells: {config['training_rows']} / {len(split['train_cells'])} "
        f"({', '.join(split['train_cells'])})",
        f"- Test rows / cells: {config['test_rows']} / {len(split['test_cells'])} "
        f"({', '.join(split['test_cells'])})",
        f"- Split: {split['splitter']} with random seed {split['random_state']}; cell overlap: 0.",
        f"- Features: {', '.join(config['feature_columns'])}",
        "- Preprocessing: median imputation fitted only on training groups.",
        "",
        "## Holdout results",
        "",
        "| Model | MAE (cycles) | RMSE (cycles) | R-squared |",
        "|---|---:|---:|---:|",
        f"| Random forest | {model_metrics['mae_cycles']:.4f} | "
        f"{model_metrics['rmse_cycles']:.4f} | {model_metrics['r2'] if model_metrics['r2'] is not None else 'n/a'} |",
        f"| Training-median baseline | {baseline['mae_cycles']:.4f} | "
        f"{baseline['rmse_cycles']:.4f} | {baseline['r2'] if baseline['r2'] is not None else 'n/a'} |",
        "",
        "MAE/RMSE are measured on held-out cell rows. Per-cell results and hashes are "
        "stored in `evaluation.json`. Only one cell is held out, so these metrics are "
        "highly uncertain and do not establish population generalization; they are an "
        "experimental result, not production acceptance.",
        "",
        "## Artifacts",
        "",
        f"- Pipeline: `{results['artifact_path']}`",
        f"- Artifact SHA-256: `{results['artifact_sha256']}`",
        f"- Dataset SHA-256: `{results['prepared_dataset_sha256']}`",
        "- Split manifest, feature schema, training configuration, dataset provenance, "
        "metrics, and model card are stored beside the pipeline under "
        f"`{Path(results['artifact_path']).parent.as_posix()}`.",
        "",
        "## Exclusions, safety, and limits",
        "",
        "Only the selected cohort with a documented 1.4 Ah end-of-life threshold and cells with a valid measured "
        "crossing are used. Nonpositive capacities are treated as invalid measurements, "
        "not end-of-life labels. Other battery cohorts are excluded when their cutoff is "
        "different/unknown, the cell starts below the threshold, or a valid crossing is "
        "absent. There is no vehicle-specific battery pack data, no fleet labels, and no "
        "validated future component-failure target. The experiment is not connected to "
        "the Django service; production safeguards and the legacy model remain unchanged.",
        "",
        "## Reproduction",
        "",
        "From `automotive_ai_service/` with the project virtual environment active:",
        "",
        "```powershell",
        "python scripts\\audit_datasets.py",
        "python scripts\\prepare_maintenance_data.py",
        "python scripts\\train_maintenance_model.py",
        "python scripts\\evaluate_maintenance_model.py",
        "```",
        "",
        "For a separate repeat run, use paths that do not already exist, for example:",
        "",
        "```powershell",
        "python scripts\\prepare_maintenance_data.py --output data\\processed\\nasa_battery_rul_repeat",
        "python scripts\\train_maintenance_model.py --input data\\processed\\nasa_battery_rul_repeat\\discharge_cycles.csv --output trained_models\\experiments\\nasa_battery_cell_rul_repeat",
        "python scripts\\evaluate_maintenance_model.py --experiment-dir trained_models\\experiments\\nasa_battery_cell_rul_repeat",
        "```",
        "",
        "The scripts refuse to overwrite existing prepared data or experiment artifacts.",
        "```",
        "",
    ]
    (SERVICE_ROOT / "data" / "reports" / "training_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_ROOT)
    evaluate(parser.parse_args().experiment_dir)
