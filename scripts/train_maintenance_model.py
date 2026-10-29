import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

from scripts.maintenance_experiment import (
    EXPERIMENT_ROOT,
    EXPERIMENTS_ROOT,
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    PREPARED_ROOT,
    RANDOM_STATE,
    TARGET_COLUMN,
    PROCESSED_ROOT,
    read_prepared_data,
    resolve_within,
    sha256_file,
    write_json,
)


def train(input_path=PREPARED_ROOT / "discharge_cycles.csv", output=EXPERIMENT_ROOT):
    input_path = resolve_within(input_path, PROCESSED_ROOT)
    output = resolve_within(output, EXPERIMENTS_ROOT)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing experiment: {output}"
        )
    dataset_path = input_path
    dataset_manifest_path = dataset_path.parent / "dataset_manifest.json"
    if not dataset_path.is_file() or not dataset_manifest_path.is_file():
        raise FileNotFoundError(
            "Run scripts/prepare_maintenance_data.py before training."
        )

    frame = read_prepared_data(dataset_path)
    if frame[GROUP_COLUMN].nunique() < 3:
        raise ValueError("At least three independent cells are required.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_indices, test_indices = next(
        splitter.split(frame[FEATURE_COLUMNS], frame[TARGET_COLUMN], frame[GROUP_COLUMN])
    )
    train_groups = sorted(frame.iloc[train_indices][GROUP_COLUMN].unique().tolist())
    test_groups = sorted(frame.iloc[test_indices][GROUP_COLUMN].unique().tolist())
    if set(train_groups).intersection(test_groups):
        raise RuntimeError("Cell leakage detected across the train/test split.")
    if not train_groups or not test_groups:
        raise ValueError("Group-aware split produced an empty partition.")

    output.mkdir(parents=True, exist_ok=False)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    train_frame = frame.iloc[train_indices]
    model.fit(train_frame[FEATURE_COLUMNS], train_frame[TARGET_COLUMN])
    artifact_path = output / "pipeline.joblib"
    joblib.dump(model, artifact_path)

    split = {
        "random_state": RANDOM_STATE,
        "splitter": "GroupShuffleSplit",
        "test_size": 0.25,
        "group_column": GROUP_COLUMN,
        "train_cells": train_groups,
        "test_cells": test_groups,
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "group_overlap": 0,
    }
    write_json(output / "split_manifest.json", split)
    shutil.copyfile(dataset_manifest_path, output / "dataset_manifest.json")
    config = {
        "experiment_name": output.name,
        "task": "laboratory battery-cell RUL to 1.4 Ah",
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "group_column": GROUP_COLUMN,
        "model": {
            "estimator": "RandomForestRegressor",
            "n_estimators": 300,
            "min_samples_leaf": 3,
            "random_state": RANDOM_STATE,
            "preprocessing": "median imputer fitted on training cells only",
        },
        "baseline": "DummyRegressor(strategy='median') fitted on training rows",
        "training_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "training_cells": train_groups,
        "test_cells": test_groups,
        "prepared_dataset_sha256": sha256_file(dataset_path),
        "prepared_dataset_path": dataset_path.relative_to(SERVICE_ROOT).as_posix(),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "artifact_sha256": sha256_file(artifact_path),
    }
    write_json(output / "training_config.json", config)
    (output / "feature_schema.json").write_text(
        json.dumps(
            {
                "features": [
                    {"name": "discharge_cycle_index", "unit": "cycle"},
                    {"name": "capacity_ah", "unit": "Ah"},
                    {"name": "ambient_temperature_c", "unit": "degrees C"},
                ],
                "target": {
                    "name": TARGET_COLUMN,
                    "unit": "discharge cycles",
                    "meaning": "Remaining cycles to first observed positive capacity <= 1.4 Ah.",
                },
                "group_only": {"name": GROUP_COLUMN, "used_as_feature": False},
                "feature_order": FEATURE_COLUMNS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "model_card.md").write_text(
        "# Laboratory battery-cell RUL experiment\n\n"
        "This experimental pipeline estimates remaining discharge cycles until the "
        "first positive measured capacity at or below 1.4 Ah in selected NASA laboratory "
        "Li-ion cell-aging sequences. It is not a vehicle battery-pack model, fleet "
        "maintenance predictor, safety monitor, or production-ready predictor.\n\n"
        "Evaluation uses whole-cell groups held out from training. The dataset contains "
        "only three independent cells, with different discharge cutoff voltages. Other "
        "NASA cohorts with unresolved anomalous capacities were excluded. Results do not establish generalization to vehicle "
        "packs, other chemistries, operating environments, or real fleet usage. Do not "
        "integrate into the Django service or use for maintenance decisions.\n",
        encoding="utf-8",
    )
    print(
        f"Trained experiment with {len(train_indices)} rows / {len(train_groups)} cells; "
        f"held out {len(test_indices)} rows / {len(test_groups)} cells."
    )
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PREPARED_ROOT / "discharge_cycles.csv")
    parser.add_argument("--output", type=Path, default=EXPERIMENT_ROOT)
    arguments = parser.parse_args()
    train(arguments.input, arguments.output)
