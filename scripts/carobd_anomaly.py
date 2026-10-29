import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


SERVICE_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = SERVICE_ROOT / "data" / "raw"
SOURCE_ROOT = RAW_ROOT / "carOBD-master" / "obdiidata"
EXPERIMENTS_ROOT = SERVICE_ROOT / "trained_models" / "experiments"
EXPERIMENT_ROOT = EXPERIMENTS_ROOT / "carobd_sensor_anomaly_v1"
FEATURE_COLUMNS = [
    "ENGINE_RPM ()",
    "VEHICLE_SPEED ()",
    "THROTTLE ()",
    "ENGINE_LOAD ()",
    "COOLANT_TEMPERATURE ()",
    "LONG_TERM_FUEL_TRIM_BANK_1 ()",
    "SHORT_TERM_FUEL_TRIM_BANK_1 ()",
    "INTAKE_MANIFOLD_PRESSURE ()",
    "FUEL_TANK ()",
    "ABSOLUTE_THROTTLE_B ()",
]
TIME_COLUMN = "ENGINE_RUN_TINE ()"
RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
THRESHOLD_PERCENTILES = (99.0, 95.0, 90.0)
MODE_PATTERN = re.compile(r"^(drive|idle|live|long|ufpe)(\d+)$", re.IGNORECASE)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def resolve_within(path, parent):
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(Path(parent).resolve())
    except ValueError as error:
        raise ValueError(f"Path must be inside {Path(parent).resolve()}: {resolved}") from error
    return resolved


def _read_source_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header:
            raise ValueError(f"CSV has no header: {path}")
        header = [name.strip() for name in header]
        if len(set(header)) != len(header):
            raise ValueError(f"CSV has duplicate column names: {path}")

        rows = []
        blank_rows = 0
        trailing_empty_fields = 0
        short_rows = 0
        for line_number, row in enumerate(reader, start=2):
            if not row or all(not value.strip() for value in row):
                blank_rows += 1
                continue
            row = [value.strip() for value in row]
            if len(row) == len(header) + 1 and not row[-1]:
                row = row[:-1]
                trailing_empty_fields += 1
            if len(row) > len(header):
                raise ValueError(
                    f"Unexpected non-empty extra CSV field at {path}:{line_number}"
                )
            if len(row) < len(header):
                short_rows += 1
                row.extend([""] * (len(header) - len(row)))
            rows.append(row)

    frame = pd.DataFrame(rows, columns=header)
    if TIME_COLUMN not in frame:
        raise ValueError(f"Required elapsed-time column missing from {path}: {TIME_COLUMN}")
    missing_features = sorted(set(FEATURE_COLUMNS).difference(frame.columns))
    if missing_features:
        raise ValueError(f"Required sensor columns missing from {path}: {missing_features}")

    non_numeric_values = {}
    non_finite_values = {}
    for column in header:
        original = frame[column]
        blank = original.isna() | original.astype(str).str.strip().eq("")
        numeric = pd.to_numeric(original, errors="coerce")
        nonnumeric_count = int((numeric.isna() & ~blank).sum())
        nonfinite_count = int(numeric.isin([np.inf, -np.inf]).sum())
        if nonnumeric_count:
            non_numeric_values[column] = nonnumeric_count
        if nonfinite_count:
            non_finite_values[column] = nonfinite_count
        frame[column] = numeric.replace([np.inf, -np.inf], np.nan)

    stats = {
        "source_rows": len(frame),
        "blank_rows": blank_rows,
        "trailing_empty_fields_removed": trailing_empty_fields,
        "short_rows_padded": short_rows,
        "duplicate_rows_removed": int(frame.duplicated(keep="first").sum()),
        "missing_values_by_column": {
            column: int(count)
            for column, count in frame.isna().sum().items()
            if count
        },
        "non_numeric_values_by_column": non_numeric_values,
        "non_finite_values_by_column": non_finite_values,
    }
    return frame, stats, header


def mode_from_filename(path):
    match = MODE_PATTERN.match(Path(path).stem)
    if not match:
        raise ValueError(f"Unknown carOBD session filename: {Path(path).name}")
    return match.group(1).lower()


def prepare_dataset(source_root=SOURCE_ROOT, output_path=None):
    source_root = Path(source_root).resolve()
    files = sorted(source_root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No carOBD CSV files found in {source_root}")

    records = []
    file_manifest = []
    total_stats = {
        "source_rows": 0,
        "blank_rows": 0,
        "trailing_empty_fields_removed": 0,
        "short_rows_padded": 0,
        "duplicate_rows_removed": 0,
        "usable_rows": 0,
        "all_selected_features_missing_rows_removed": 0,
        "missing_values_by_column": {},
        "non_numeric_values_by_column": {},
        "non_finite_values_by_column": {},
    }
    expected_header = None

    for source_path in files:
        frame, stats, header = _read_source_csv(source_path)
        if expected_header is None:
            expected_header = header
        elif header != expected_header:
            raise ValueError(f"CSV schema differs from the first file: {source_path}")

        original_count = len(frame)
        deduplicated = frame.drop_duplicates(keep="first").copy()
        if len(deduplicated) != original_count - stats["duplicate_rows_removed"]:
            raise RuntimeError(f"Duplicate accounting mismatch in {source_path}")
        no_features = deduplicated[FEATURE_COLUMNS].isna().all(axis=1)
        total_stats["all_selected_features_missing_rows_removed"] += int(no_features.sum())
        deduplicated = deduplicated.loc[~no_features].copy()
        deduplicated["source_session"] = source_path.stem
        deduplicated["operating_mode"] = mode_from_filename(source_path)
        deduplicated["source_row_index"] = np.arange(original_count, dtype=np.int64)[
            frame.index.isin(deduplicated.index)
        ]

        elapsed = deduplicated[TIME_COLUMN]
        if elapsed.notna().sum() and (elapsed.dropna().diff().dropna() < 0).any():
            raise ValueError(
                f"Elapsed run-time moves backwards in source order: {source_path.name}"
            )
        deduplicated["stationary_engine_running"] = (
            deduplicated["VEHICLE_SPEED ()"].eq(0)
            & deduplicated["ENGINE_RPM ()"].gt(0)
        )
        deduplicated["moving"] = deduplicated["VEHICLE_SPEED ()"].gt(0)
        deduplicated["within_session_order"] = np.arange(len(deduplicated), dtype=np.int64)
        records.append(deduplicated)

        stats["usable_rows"] = len(deduplicated)
        stats["session"] = source_path.stem
        stats["operating_mode"] = mode_from_filename(source_path)
        stats["path"] = source_path.relative_to(RAW_ROOT).as_posix()
        stats["sha256"] = sha256_file(source_path)
        stats["bytes"] = source_path.stat().st_size
        stats["elapsed_time_min"] = (
            float(elapsed.min()) if elapsed.notna().any() else None
        )
        stats["elapsed_time_max"] = (
            float(elapsed.max()) if elapsed.notna().any() else None
        )
        stats["elapsed_time_missing"] = int(elapsed.isna().sum())
        stats["stationary_engine_running_rows"] = int(
            deduplicated["stationary_engine_running"].sum()
        )
        stats["moving_rows"] = int(deduplicated["moving"].sum())
        file_manifest.append(stats)

        for key in (
            "source_rows",
            "blank_rows",
            "trailing_empty_fields_removed",
            "short_rows_padded",
            "duplicate_rows_removed",
            "usable_rows",
        ):
            total_stats[key] += stats[key]
        for column, count in stats["missing_values_by_column"].items():
            total_stats["missing_values_by_column"][column] = (
                total_stats["missing_values_by_column"].get(column, 0) + count
            )
        for field in ("non_numeric_values_by_column", "non_finite_values_by_column"):
            for column, count in stats[field].items():
                total_stats[field][column] = total_stats[field].get(column, 0) + count

    combined = pd.concat(records, ignore_index=True)
    selected_missing = combined[FEATURE_COLUMNS].isna().sum()
    total_stats["selected_feature_missing_values"] = {
        column: int(count) for column, count in selected_missing.items()
    }
    total_stats["selected_feature_missing_cells"] = int(selected_missing.sum())
    total_stats["elapsed_time_backward_steps"] = 0
    total_stats["stationary_engine_running_rows"] = int(
        combined["stationary_engine_running"].sum()
    )
    total_stats["moving_rows"] = int(combined["moving"].sum())
    total_stats["idle_named_session_rows"] = int(
        combined["operating_mode"].eq("idle").sum()
    )

    provenance = {
        "dataset": "carOBD Toyota Etios OBD-II telemetry",
        "source_description": (
            "Local README describes a 2014 Toyota Etios captured at nominal 1 Hz with "
            "27 OBD-II PID columns across idle, drive, live, long, and ufpe session files."
        ),
        "source_root": str(source_root),
        "source_readme": str(RAW_ROOT / "carOBD-master" / "README.md"),
        "source_readme_sha256": sha256_file(RAW_ROOT / "carOBD-master" / "README.md"),
        "files": file_manifest,
        "file_count": len(file_manifest),
        "raw_record_count": total_stats["source_rows"],
        "preprocessed_record_count": len(combined),
        "statistics": total_stats,
        "features": [
            {
                "name": name,
                "unit": "not specified in local CSV header (empty `()` unit field)",
            }
            for name in FEATURE_COLUMNS
        ],
        "elapsed_time": {
            "field": TIME_COLUMN,
            "meaning": "Local README identifies this as engine run time since start.",
            "units": "not specified in local CSV header",
            "monotonic_in_source_order": True,
            "ties_retained": True,
            "absolute_calendar_timestamp_available": False,
        },
        "duplicate_policy": (
            "Remove exact full-row duplicates within each source file, keeping the first "
            "occurrence. Repeated sensor values at different run times are retained. "
            "Never deduplicate across session boundaries."
        ),
        "invalid_value_policy": (
            "Parse values numerically; non-numeric and non-finite values become missing "
            "and are counted. Pad short rows at the end to match the declared header. "
            "Do not apply undocumented physical ranges or label unusual values as faults."
        ),
        "operating_conditions": {
            "file_mode_source": "filename prefixes documented in local README",
            "stationary_engine_running": (
                "VEHICLE_SPEED == 0 and ENGINE_RPM > 0; descriptive condition only."
            ),
            "moving": "VEHICLE_SPEED > 0; descriptive condition only.",
            "training_features_exclude_condition_and_session_columns": True,
        },
    }
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
    return combined, provenance


def validate_feature_frame(frame, feature_columns=FEATURE_COLUMNS):
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Required anomaly features are missing: {missing}")
    matrix = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    if matrix.isna().all(axis=1).any():
        raise ValueError("Rows with all anomaly features missing cannot be scored.")
    if matrix.isna().all(axis=0).any():
        empty_columns = matrix.columns[matrix.isna().all(axis=0)].tolist()
        raise ValueError(f"Anomaly features are entirely missing: {empty_columns}")
    return matrix


def chronological_group_split(frame, train_fraction=0.70):
    if not 0.5 <= train_fraction < 1:
        raise ValueError("train_fraction must be at least 0.5 and below 1.")
    if not {"source_session", "within_session_order"}.issubset(frame.columns):
        raise ValueError("Chronological split requires source-session order metadata.")
    train_parts = []
    evaluation_parts = []
    session_splits = {}
    for session, group in frame.groupby("source_session", sort=True):
        group = group.sort_values("within_session_order", kind="stable")
        if len(group) < 2:
            raise ValueError(f"Session has too few rows for a chronological split: {session}")
        elapsed_times = pd.to_numeric(group[TIME_COLUMN], errors="coerce")
        if elapsed_times.isna().any():
            raise ValueError(
                f"Elapsed run time is missing in session; chronological split is unsafe: {session}"
            )
        if elapsed_times.diff().dropna().lt(0).any():
            raise ValueError(f"Elapsed run time moves backwards in session: {session}")
        cutoff = int(np.floor(len(group) * train_fraction))
        cutoff = min(max(cutoff, 1), len(group) - 1)
        evaluation_start_time = float(elapsed_times.iloc[cutoff])
        cutoff = int(
            np.searchsorted(
                elapsed_times.to_numpy(),
                evaluation_start_time,
                side="left",
            )
        )
        if cutoff < 1:
            raise ValueError(
                f"Could not separate training/evaluation run-time windows in {session}"
            )
        train_parts.append(group.iloc[:cutoff])
        evaluation_parts.append(group.iloc[cutoff:])
        session_splits[session] = {
            "train_rows": int(cutoff),
            "evaluation_rows": int(len(group) - cutoff),
            "cutoff_run_time": float(elapsed_times.iloc[cutoff]),
            "train_last_run_time": (
                float(group.iloc[cutoff - 1][TIME_COLUMN])
                if pd.notna(group.iloc[cutoff - 1][TIME_COLUMN])
                else None
            ),
            "evaluation_first_run_time": (
                float(group.iloc[cutoff][TIME_COLUMN])
                if pd.notna(group.iloc[cutoff][TIME_COLUMN])
                else None
            ),
        }
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(evaluation_parts, ignore_index=True),
        session_splits,
    )


def train_thresholds(training_scores):
    scores = np.asarray(training_scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Training anomaly scores must be a non-empty finite vector.")
    return {
        str(percentile): float(np.percentile(scores, percentile))
        for percentile in THRESHOLD_PERCENTILES
    }


def build_pipeline(random_state=RANDOM_STATE):
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "isolation_forest",
                IsolationForest(
                    n_estimators=300,
                    contamination="auto",
                    max_samples="auto",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def load_pipeline(path):
    import joblib
    from sklearn.pipeline import Pipeline

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Anomaly experiment pipeline not found: {path}")
    model = joblib.load(path)
    if not isinstance(model, Pipeline) or list(model.named_steps) != [
        "imputer",
        "isolation_forest",
    ]:
        raise ValueError(f"Unexpected anomaly pipeline structure: {path}")
    return model


def robust_distance(training_frame, evaluation_frame):
    train = validate_feature_frame(training_frame)
    evaluation = validate_feature_frame(evaluation_frame)
    medians = train.median()
    deviations = (train - medians).abs().median()
    scales = deviations.replace(0, np.nan)
    scales = scales.fillna(train.std().replace(0, np.nan)).fillna(1.0)
    train_filled = train.fillna(medians)
    evaluation_filled = evaluation.fillna(medians)
    train_scores = ((train_filled - medians).abs() / scales).max(axis=1).to_numpy()
    evaluation_scores = (
        (evaluation_filled - medians).abs() / scales
    ).max(axis=1).to_numpy()
    return train_scores, evaluation_scores


def condition_label(frame):
    if "operating_mode" not in frame:
        raise ValueError("Condition summaries require operating_mode metadata.")
    return frame["operating_mode"].astype(str)
