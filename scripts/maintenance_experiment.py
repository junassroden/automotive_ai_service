import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SERVICE_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = SERVICE_ROOT / "data" / "raw"
PREPARED_ROOT = SERVICE_ROOT / "data" / "processed" / "nasa_battery_rul"
PROCESSED_ROOT = SERVICE_ROOT / "data" / "processed"
EXPERIMENTS_ROOT = SERVICE_ROOT / "trained_models" / "experiments"
EXPERIMENT_ROOT = (
    SERVICE_ROOT
    / "trained_models"
    / "experiments"
    / "nasa_battery_cell_rul_1p4ah"
)
EOL_CAPACITY_AH = 1.4
RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "discharge_cycle_index",
    "capacity_ah",
    "ambient_temperature_c",
]
TARGET_COLUMN = "remaining_discharge_cycles"
GROUP_COLUMN = "cell_id"
SUPPORTED_CELLS = {"B0005", "B0006", "B0018"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def resolve_within(path, parent):
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(Path(parent).resolve())
    except ValueError as error:
        raise ValueError(f"Path must be inside {Path(parent).resolve()}: {resolved}") from error
    return resolved


def prepare_cell_records(cell_id, discharge_records, eol_capacity_ah=EOL_CAPACITY_AH):
    valid = []
    invalid_capacity_count = 0
    for cycle_index, record in enumerate(discharge_records):
        capacity = record.get("capacity_ah")
        try:
            capacity = float(capacity)
            temperature = float(record["ambient_temperature_c"])
        except (TypeError, ValueError, KeyError):
            invalid_capacity_count += 1
            continue
        if not np.isfinite(capacity) or capacity <= 0 or not np.isfinite(temperature):
            invalid_capacity_count += 1
            continue
        valid.append(
            {
                "cell_id": cell_id,
                "discharge_cycle_index": cycle_index,
                "capacity_ah": capacity,
                "ambient_temperature_c": temperature,
            }
        )

    if not valid:
        return [], invalid_capacity_count, "no valid positive discharge capacities"
    if valid[0]["capacity_ah"] <= eol_capacity_ah:
        return [], invalid_capacity_count, "initial valid capacity is at or below EOL threshold"

    crossing = next(
        (
            index
            for index, row in enumerate(valid)
            if row["capacity_ah"] <= eol_capacity_ah
        ),
        None,
    )
    if crossing is None:
        return [], invalid_capacity_count, "no observed crossing of documented EOL threshold"

    trajectory = valid[: crossing + 1]
    last_cycle_index = trajectory[-1]["discharge_cycle_index"]
    for row in trajectory:
        row[TARGET_COLUMN] = last_cycle_index - row["discharge_cycle_index"]
    return trajectory, invalid_capacity_count, None


def extract_cell_cycles(path):
    from scipy.io import loadmat

    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    variable_names = [name for name in data if not name.startswith("__")]
    if len(variable_names) != 1:
        raise ValueError(f"Expected one battery variable in {path}, got {variable_names}")
    battery = data[variable_names[0]]
    if not hasattr(battery, "cycle"):
        raise ValueError(f"Missing cycle structure in {path}")

    discharge_records = []
    for cycle in np.atleast_1d(battery.cycle):
        operation = str(cycle.type).strip().lower()
        if operation != "discharge":
            continue
        capacity = getattr(cycle.data, "Capacity", None)
        capacity_values = np.asarray(capacity, dtype=float).reshape(-1) if capacity is not None else []
        capacity_value = float(capacity_values[0]) if len(capacity_values) else None
        ambient = float(np.asarray(cycle.ambient_temperature).reshape(-1)[0])
        discharge_records.append(
            {
                "capacity_ah": capacity_value,
                "ambient_temperature_c": ambient,
            }
        )
    return discharge_records


def read_prepared_data(path=None):
    path = Path(path or PREPARED_ROOT / "discharge_cycles.csv")
    frame = pd.read_csv(path)
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN, GROUP_COLUMN])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prepared dataset is missing required columns: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Prepared dataset is empty or contains missing required values.")
    if (frame[TARGET_COLUMN] < 0).any():
        raise ValueError("RUL target cannot contain negative values.")
    return frame
