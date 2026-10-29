import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import re
import stat
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from xml.etree import ElementTree

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.io import loadmat, whosmat
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from data_pipeline.status import record_training_status, read_training_status


SERVICE_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = SERVICE_ROOT / "data" / "raw"
REPORT_ROOT = SERVICE_ROOT / "data" / "reports"

ENGINEFAULTDB_COLUMNS = {
    "Fault": "fault_class",
    "MAP": "map_kpa",
    "TPS": "throttle_position_pct",
    "Force": "force_n",
    "Power": "power_kw",
    "RPM": "rpm",
    "Consumption L/H": "fuel_consumption_lph",
    "Consumption L/100KM": "fuel_consumption_l_per_100km",
    "Speed": "speed_kmh",
    "CO": "co_pct",
    "HC": "hc_ppm",
    "CO2": "co2_pct",
    "O2": "o2_pct",
    "Lambda": "lambda",
    "AFR": "air_fuel_ratio",
}
LEAKAGE_COLUMN_PATTERN = re.compile(
    r"(^|[_\s])(target|label|outcome|post[_\s]?(fault|failure|repair)|"
    r"after[_\s]?(fault|failure|repair)|failure[_\s]?(time|date|result)|"
    r"repair[_\s]?(result|date)|diagnos(ed|is))($|[_\s])",
    re.IGNORECASE,
)


def _safe_member_path(name):
    posix_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    parts = [part for part in posix_path.parts if part not in {"", "."}]
    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
    return not (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
        or ".." in windows_path.parts
        or any(
            not part
            or part.endswith((".", " "))
            or any(character in part for character in '<>:"|?*')
            or part.split(".", 1)[0].upper() in reserved_names
            for part in parts
        )
    )


def safe_extract_archive(archive_path, destination):
    archive_path = Path(archive_path).resolve()
    destination = Path(destination).resolve()
    extracted = []
    if destination.exists():
        if not destination.is_dir():
            raise ValueError("Extraction destination exists and is not a directory.")
        if any(destination.iterdir()):
            raise ValueError("Extraction destination must be empty.")

    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > 20000:
            raise ValueError("Archive contains too many members to extract safely.")
        if sum(info.file_size for info in members) > 2 * 1024 * 1024 * 1024:
            raise ValueError("Archive exceeds the 2 GiB uncompressed extraction limit.")
        if any(info.flag_bits & 1 for info in members):
            raise ValueError("Encrypted archives are not supported.")
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError(f"Corrupt archive member: {corrupt_member}")
        planned = []
        targets = set()
        for info in members:
            if not _safe_member_path(info.filename):
                raise ValueError(f"Unsafe archive member path: {info.filename}")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Archive symlinks are not supported: {info.filename}")

            target = (destination / Path(*PurePosixPath(info.filename).parts)).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Archive member escapes destination: {info.filename}")
            target_key = str(target).casefold()
            if target_key in targets:
                raise ValueError(f"Archive contains duplicate output paths: {info.filename}")
            targets.add(target_key)
            if target == archive_path:
                raise ValueError("Archive extraction would overwrite its source file.")
            planned.append((info, target))

        for info, target in planned:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            extracted.append(str(target))

    return extracted


def _inspect_file(path, relative_path):
    extension = path.suffix.lower()
    result = {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "extension": extension or "[no extension]",
        "inspection_status": "metadata_only",
    }

    if result["bytes"] == 0:
        result["inspection_status"] = "empty"
        result["issue"] = "File is zero bytes."
        return result

    if extension == ".csv":
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
            sample = stream.read(8192)
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
        frame = pd.read_csv(
            path,
            sep=delimiter,
            low_memory=False,
            encoding="utf-8-sig",
            encoding_errors="replace",
        )
        target_columns = {
            "fault",
            "fault_class",
            "fault_code",
            "fault_type",
            "fault_category",
            "engine condition",
            "engine_condition",
            "machine failure",
            "label",
            "mode",
            "vehicle_type",
        }
        target_distributions = {}
        for column in frame.columns:
            if str(column).strip().lower() not in target_columns:
                continue
            counts = frame[column].value_counts(dropna=False)
            if len(counts) <= 50:
                target_distributions[str(column)] = {
                    str(label): int(count)
                    for label, count in counts.items()
                }
        result.update(
            inspection_status="inspected",
            delimiter=delimiter,
            rows=int(len(frame)),
            columns=[
                {"name": str(name), "dtype": str(dtype)}
                for name, dtype in frame.dtypes.items()
            ],
            missing_values={
                str(name): int(count)
                for name, count in frame.isna().sum().items()
            },
            duplicate_rows=int(frame.duplicated().sum()),
            observed_target_value_counts=target_distributions,
        )
    elif extension in {".xlsx", ".xlsm"}:
        sheets = pd.read_excel(path, sheet_name=None)
        result.update(
            inspection_status="inspected",
            sheets={
                name: {
                    "rows": int(len(frame)),
                    "columns": [
                        {"name": str(column), "dtype": str(dtype)}
                        for column, dtype in frame.dtypes.items()
                    ],
                    "missing_values": {
                        str(column): int(count)
                        for column, count in frame.isna().sum().items()
                    },
                    "duplicate_rows": int(frame.duplicated().sum()),
                    "observed_target_value_counts": {
                        str(column): {
                            str(label): int(count)
                            for label, count in frame[column].value_counts(dropna=False).items()
                        }
                        for column in frame.columns
                        if str(column).strip().lower()
                        in {
                            "fault", "fault_class", "fault_code", "fault_type",
                            "fault_category", "label", "mode",
                        }
                        and frame[column].nunique(dropna=False) <= 50
                    },
                }
                for name, frame in sheets.items()
            },
        )
    elif extension == ".parquet":
        import pyarrow.parquet as parquet

        metadata = parquet.read_metadata(path)
        schema = parquet.read_schema(path)
        frame = pd.read_parquet(path)
        result.update(
            inspection_status="inspected",
            rows=int(metadata.num_rows),
            row_groups=int(metadata.num_row_groups),
            columns=[
                {"name": field.name, "dtype": str(field.type)}
                for field in schema
            ],
            missing_values={
                str(name): int(count)
                for name, count in frame.isna().sum().items()
            },
            duplicate_rows=int(frame.duplicated().sum()),
            observed_target_value_counts={
                str(column): {
                    str(label): int(count)
                    for label, count in frame[column].value_counts(dropna=False).items()
                }
                for column in frame.columns
                if str(column).strip().lower()
                in {"fault", "fault_class", "fault_type", "fault_category", "label", "flag"}
                and frame[column].nunique(dropna=False) <= 50
            },
        )
    elif extension == ".mat":
        variables = whosmat(path)
        structure_summaries = []
        if len(variables) == 1 and variables[0][2] == "struct":
            contents = loadmat(path, squeeze_me=True, struct_as_record=False)
            structure = contents.get(variables[0][0])
            if isinstance(structure, np.ndarray) and structure.size == 1:
                structure = structure.item()
            cycles = getattr(structure, "cycle", None)
            if cycles is not None:
                cycle_list = np.atleast_1d(cycles).ravel()
                operation_counts = Counter(
                    str(getattr(cycle, "type", "unknown"))
                    for cycle in cycle_list
                )
                structure_summaries.append(
                    {
                        "variable": variables[0][0],
                        "cycle_count": int(len(cycle_list)),
                        "operation_counts": dict(operation_counts),
                    }
                )
        result.update(
            inspection_status="inspected",
            variables=[
                {"name": name, "shape": list(shape), "dtype": dtype}
                for name, shape, dtype in variables
            ],
            structure_summaries=structure_summaries,
        )
    elif extension == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            unsafe = [member.filename for member in members if not _safe_member_path(member.filename)]
            corrupt_member = archive.testzip()
            result.update(
                inspection_status="inspected",
                member_count=len(members),
                encrypted_members=sum(bool(member.flag_bits & 1) for member in members),
                unsafe_members=unsafe,
                corrupt_member=corrupt_member,
                member_extensions=dict(
                    Counter(
                        PurePosixPath(member.filename).suffix.lower() or "[no extension]"
                        for member in members
                        if not member.is_dir()
                    )
                ),
            )
    elif extension == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        result.update(
            inspection_status="inspected",
            json_type=type(value).__name__,
            top_level_keys=list(value.keys()) if isinstance(value, dict) else None,
            item_count=len(value) if isinstance(value, (dict, list)) else None,
        )
    elif extension in {".xml"}:
        root = ElementTree.parse(path).getroot()
        result.update(
            inspection_status="inspected",
            root_element=root.tag,
            child_elements=len(root),
        )
    elif extension in {".md", ".txt", ".rst", ".license", ".gitignore"} or not extension:
        content = path.read_text(encoding="utf-8", errors="replace")
        result.update(
            inspection_status="inspected",
            lines=len(content.splitlines()),
            characters=len(content),
        )
    elif extension == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = notebook.get("cells", [])
        result.update(
            inspection_status="inspected",
            cell_count=len(cells),
            cell_types=dict(Counter(cell.get("cell_type", "unknown") for cell in cells)),
            executed=False,
        )
    elif extension in {".py"}:
        result.update(
            inspection_status="not_executed",
            lines=len(path.read_text(encoding="utf-8", errors="replace").splitlines()),
        )
    elif extension in {".mbl", ".cbf"}:
        result["inspection_status"] = "unsupported"
        result["issue"] = (
            "Binary telemetry/diagnostic format was not decoded; inspect the "
            "accompanying dataset documentation and decoder requirements."
        )
    else:
        result["inspection_status"] = "unsupported"
        result["issue"] = f"No content reader is configured for {extension or 'this file type'}."

    return result


def build_inventory(raw_root=RAW_ROOT):
    raw_root = Path(raw_root).resolve()
    files = []
    errors = []
    for path in sorted(item for item in raw_root.rglob("*") if item.is_file()):
        relative = path.relative_to(raw_root).as_posix()
        try:
            entry = _inspect_file(path, relative)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            entry["sha256"] = digest.hexdigest()
            files.append(entry)
        except Exception as error:
            files.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "extension": path.suffix.lower() or "[no extension]",
                    "inspection_status": "error",
                    "issue": f"{type(error).__name__}: {error}",
                    "sha256": None,
                }
            )
            errors.append(relative)

    groups = defaultdict(list)
    digest_groups = defaultdict(list)
    for item in files:
        group = item["path"].split("/", 1)[0] if "/" in item["path"] else "[root files]"
        groups[group].append(item)
        if item.get("sha256"):
            digest_groups[item["sha256"]].append(item["path"])

    return {
        "raw_root": str(raw_root),
        "files_inspected": len(files),
        "top_level_directories": sum(path.is_dir() for path in raw_root.iterdir()),
        "file_count_by_extension": dict(Counter(item["extension"] for item in files)),
        "identical_file_groups": [
            {"sha256": digest, "paths": paths}
            for digest, paths in sorted(digest_groups.items())
            if len(paths) > 1
        ],
        "error_paths": errors,
        "datasets": [
            {
                "name": name,
                "file_count": len(group_files),
                "total_bytes": sum(item["bytes"] for item in group_files),
                "files": group_files,
            }
            for name, group_files in sorted(groups.items())
        ],
        "files": files,
    }


def _render_inventory_markdown(inventory):
    lines = [
        "# Raw dataset inventory",
        "",
        f"- Files inspected: **{inventory['files_inspected']}**",
        f"- Top-level directories: **{inventory['top_level_directories']}**",
        f"- Content inspection errors: **{len(inventory['error_paths'])}**",
        "",
        "Raw files were only read. Archives were checked for unsafe paths and corruption; "
        "nothing was extracted or modified. Code and notebooks were not executed. "
        "SHA-256 hashes were calculated for byte-level duplicate checks. The JSON "
        "companion contains the complete per-file manifest and schemas.",
        "",
        "## Collection overview",
        "",
        "| Top-level collection | Files | Total size (MiB) |",
        "|---|---:|---:|",
    ]
    for dataset in inventory["datasets"]:
        lines.append(
            f"| `{dataset['name']}` | {dataset['file_count']} | "
            f"{dataset['total_bytes'] / (1024 * 1024):.2f} |"
        )

    lines += ["", "## Byte-identical files", ""]
    if inventory["identical_file_groups"]:
        for duplicate in inventory["identical_file_groups"]:
            lines.append(
                f"- SHA-256 `{duplicate['sha256']}`: "
                + ", ".join(f"`{path}`" for path in duplicate["paths"])
            )
    else:
        lines.append("None found.")

    lines += ["", "## Tabular data and structured files", ""]
    for item in inventory["files"]:
        if item["extension"] not in {
            ".csv", ".xlsx", ".xlsm", ".parquet", ".mat", ".json", ".zip"
        }:
            continue
        lines.append(f"### `{item['path']}`")
        lines.append("")
        lines.append(
            f"- Size: {item['bytes']:,} bytes; inspection: `{item['inspection_status']}`."
        )
        if "rows" in item:
            lines.append(
                f"- Rows: {item['rows']:,}; columns: {len(item.get('columns', []))}; "
                f"duplicate rows: {item.get('duplicate_rows', 'not checked')}."
            )
            columns = item.get("columns", [])
            if columns:
                lines.append(
                    "- Schema: "
                    + ", ".join(f"`{col['name']}` ({col['dtype']})" for col in columns)
                    + "."
                )
            missing = item.get("missing_values", {})
            nonzero_missing = {key: value for key, value in missing.items() if value}
            lines.append(f"- Missing values: `{json.dumps(nonzero_missing, ensure_ascii=True)}`.")
            if item.get("observed_target_value_counts"):
                lines.append(
                    "- Observed target-like values: "
                    f"`{json.dumps(item['observed_target_value_counts'], ensure_ascii=True)}`."
                )
        elif "sheets" in item:
            lines.append(
                "- Sheets: "
                + ", ".join(
                    f"`{name}` ({details['rows']:,} rows, "
                    f"{len(details['columns'])} columns, "
                    f"{sum(details['missing_values'].values()):,} missing cells, "
                    f"{details['duplicate_rows']:,} duplicate rows)"
                    for name, details in item["sheets"].items()
                )
                + "."
            )
            for name, details in item["sheets"].items():
                if details["observed_target_value_counts"]:
                    lines.append(
                        f"- `{name}` target-like counts: "
                        f"`{json.dumps(details['observed_target_value_counts'], ensure_ascii=True)}`."
                    )
        elif "member_count" in item:
            lines.append(
                f"- Archive members: {item['member_count']}; encrypted: "
                f"{item['encrypted_members']}; unsafe paths: {len(item['unsafe_members'])}; "
                f"corrupt member: `{item['corrupt_member']}`."
            )
        elif "variables" in item:
            lines.append(f"- MATLAB variables: `{json.dumps(item['variables'])}`.")
            if item.get("structure_summaries"):
                lines.append(
                    "- Structure summary: "
                    f"`{json.dumps(item['structure_summaries'], ensure_ascii=True)}`."
                )
        elif "issue" in item:
            lines.append(f"- Inspection issue: {item['issue']}")
        lines.append("")

    lines += ["## Files requiring format-specific follow-up", ""]
    unsupported = [
        item
        for item in inventory["files"]
        if item["inspection_status"] in {"unsupported", "error", "empty"}
    ]
    if unsupported:
        for item in unsupported:
            lines.append(
                f"- `{item['path']}` ({item['bytes']:,} bytes): "
                f"{item.get('issue', 'zero-byte file; no content to inspect')}"
            )
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def write_inventory(raw_root=RAW_ROOT, report_root=REPORT_ROOT):
    inventory = build_inventory(raw_root)
    report_root = Path(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "dataset_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (report_root / "dataset_inventory.md").write_text(
        _render_inventory_markdown(inventory),
        encoding="utf-8",
    )
    return inventory


def prepare_enginefaultdb(raw_root=RAW_ROOT, processed_root=None):
    raw_root = Path(raw_root)
    source = (
        raw_root
        / "EngineFaultDB-main"
        / "EngineFaultDB-main"
        / "EngineFaultDB_Final.csv"
    )
    if not source.is_file():
        raise FileNotFoundError(f"EngineFaultDB source CSV not found: {source}")

    frame = pd.read_csv(source, low_memory=False)
    if set(frame.columns) != set(ENGINEFAULTDB_COLUMNS):
        raise ValueError("EngineFaultDB columns do not match the documented schema.")
    frame = frame.rename(columns=ENGINEFAULTDB_COLUMNS)
    frame = frame.loc[:, list(ENGINEFAULTDB_COLUMNS.values())]
    if frame.isna().any().any():
        raise ValueError("EngineFaultDB contains missing values; refusing silent imputation.")
    if not all(pd.api.types.is_numeric_dtype(frame[column]) for column in frame.columns):
        raise ValueError("EngineFaultDB contains non-numeric measurements or labels.")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("EngineFaultDB contains non-finite values.")

    rows_before = len(frame)
    duplicate_count = int(frame.duplicated().sum())
    frame = frame.drop_duplicates(ignore_index=True)
    if processed_root is None:
        processed_root = Path(__file__).resolve().parents[1] / "data" / "processed"
    output_dir = Path(processed_root) / "enginefaultdb"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "enginefaultdb_normalized.csv"
    frame.to_csv(output, index=False)
    target_counts = {
        str(key): int(value)
        for key, value in frame["fault_class"].value_counts().sort_index().items()
    }
    metadata = {
        "dataset": "EngineFaultDB",
        "source_files": [source.relative_to(raw_root).as_posix()],
        "output_file": output.name,
        "rows_before": rows_before,
        "rows_after": len(frame),
        "exact_duplicates_removed": duplicate_count,
        "columns": list(frame.columns),
        "missing_values": {
            column: int(count) for column, count in frame.isna().sum().items()
        },
        "label": {
            "column": "fault_class",
            "observed_counts": target_counts,
            "documented_semantics": (
                "Four numeric fault classes are reported by the source README; "
                "a verified class-to-fault-name mapping is not supplied."
            ),
        },
        "transformations": [
            "Renamed source columns to normalized snake_case names with units in names.",
            "Verified numeric values and absence of missing/non-finite values.",
            "Removed exact duplicate rows only.",
        ],
        "limitations": [
            "Controlled laboratory data for a single C14NE engine, not a fleet dataset.",
            "No vehicle, run, trip, or acquisition-group identifier is present.",
            "Adjacent/related sample independence cannot be established.",
            "The four class labels are not mapped to documented fault mechanisms.",
            "No vehicle-health or future-failure claim is supported.",
        ],
        "training_status": (
            "Blocked: a group-disjoint holdout cannot be constructed without "
            "a documented run or vehicle identifier."
        ),
    }
    (output_dir / "preparation_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if raw_root.resolve() == RAW_ROOT.resolve():
        record_training_status(
            "maintenance_model",
            {
                "status": "blocked",
                "model_name": "maintenance_model",
                "task": "vehicle maintenance prediction",
                "dataset": "EngineFaultDB",
                "target_definitions": {
                    "fault_class": "Numeric laboratory engine-state class; class mapping unverified."
                },
                "source_dataset": source.relative_to(raw_root).as_posix(),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "training_records": len(frame),
                "test_records": 0,
                "training_groups": 0,
                "test_groups": 0,
                "metrics": None,
                "artifact_path": None,
                "eligible_for_inference": False,
                "reason": (
                    "The source has no vehicle/acquisition-run identifiers for "
                    "group-disjoint evaluation. Numeric class meanings are also "
                    "unverified, and this lab dataset does not represent fleet "
                    "maintenance outcomes."
                ),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    return metadata


def train_grouped_classifier(
    frame,
    target_column,
    group_column,
    output_directory,
    random_state=42,
):
    if target_column not in frame.columns:
        raise ValueError(f"Target column not found: {target_column}")
    if group_column not in frame.columns:
        raise ValueError(
            f"Group column not found: {group_column}. "
            "A vehicle, trip, or acquisition-run identifier is required; "
            "row-random splitting is intentionally disabled."
        )
    if frame[target_column].isna().any() or frame[group_column].isna().any():
        raise ValueError("Target and group columns must not contain missing values.")
    if frame[group_column].nunique() < 2:
        raise ValueError("At least two independent groups are required.")

    groups = frame[group_column]
    features = frame.drop(columns=[target_column, group_column])
    suspected_leakage = [
        column for column in features.columns
        if LEAKAGE_COLUMN_PATTERN.search(str(column))
    ]
    if suspected_leakage:
        raise ValueError(
            "Potential target-leakage feature columns must be reviewed and removed "
            "before training: " + ", ".join(map(str, suspected_leakage))
        )
    target = frame[target_column]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_indices, test_indices = next(splitter.split(features, target, groups))
    train_groups = set(groups.iloc[train_indices])
    test_groups = set(groups.iloc[test_indices])
    if train_groups & test_groups:
        raise RuntimeError("Group leakage detected between train and test splits.")
    if target.iloc[train_indices].nunique() < 2:
        raise ValueError("The training split contains fewer than two target classes.")

    numeric_columns = [
        column
        for column in features.columns
        if pd.api.types.is_numeric_dtype(features[column])
    ]
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    train_features, test_features = features.iloc[train_indices], features.iloc[test_indices]
    train_target, test_target = target.iloc[train_indices], target.iloc[test_indices]
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(train_features, train_target)
    model.fit(train_features, train_target)
    predictions = model.predict(test_features)
    baseline_predictions = baseline.predict(test_features)
    labels = sorted(target.unique().tolist())

    metrics = {
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "train_group_count": len(train_groups),
        "test_group_count": len(test_groups),
        "group_overlap": 0,
        "model": {
            "balanced_accuracy": float(balanced_accuracy_score(test_target, predictions)),
            "macro_precision": float(
                precision_score(test_target, predictions, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(test_target, predictions, average="macro", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(test_target, predictions, average="macro", zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(
                test_target, predictions, labels=labels
            ).tolist(),
            "class_report": classification_report(
                test_target, predictions, labels=labels, zero_division=0,
                output_dict=True,
            ),
        },
        "most_frequent_baseline": {
            "balanced_accuracy": float(
                balanced_accuracy_score(test_target, baseline_predictions)
            ),
            "macro_f1": float(
                f1_score(
                    test_target,
                    baseline_predictions,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
        },
        "class_distribution": {
            str(label): int(count)
            for label, count in target.value_counts().sort_index().items()
        },
    }
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_directory / "grouped_classifier.joblib")
    metadata = {
        "status": "evaluated_rejected",
        "eligible_for_inference": False,
        "target_column": target_column,
        "group_column": group_column,
        "feature_columns": list(features.columns),
        "random_state": random_state,
        "split": "GroupShuffleSplit(test_size=0.2)",
        "preprocessing_version": "numeric-median-standard-scale+categorical-mode-onehot-v1",
        "feature_schema": [
            {"name": str(column), "dtype": str(features[column].dtype)}
            for column in features.columns
        ],
        "model_parameters": {
            "estimator": "RandomForestClassifier",
            "n_estimators": 300,
            "class_weight": "balanced",
            "random_state": random_state,
        },
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
        "metrics": metrics,
        "limitations": [
            "Holdout metrics only apply to the observed groups and source population.",
            "No risk probability is calibrated by this evaluation.",
            "Performance does not establish prospective vehicle-failure prediction.",
            "This generic grouped classifier is a research candidate, not an accepted production maintenance model.",
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata["artifact_path"] = str(output_directory / "grouped_classifier.joblib")
    metadata["reason"] = (
        "Evaluated grouped classifier is retained as a research candidate only; "
        "no production maintenance acceptance criteria are defined for this task."
    )
    (output_directory / "evaluation.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _record_blocked_maintenance(reason, dataset="EngineFaultDB"):
    processed_source = (
        SERVICE_ROOT / "data" / "processed" / "enginefaultdb"
        / "enginefaultdb_normalized.csv"
    )
    source_hash = None
    records = 0
    if processed_source.is_file():
        source_hash = hashlib.sha256(processed_source.read_bytes()).hexdigest()
        with processed_source.open("r", encoding="utf-8") as source_stream:
            records = sum(1 for _ in source_stream) - 1
    record_training_status(
        "maintenance_model",
        {
            "status": "blocked",
            "model_name": "maintenance_model",
            "task": "vehicle maintenance prediction",
            "dataset": dataset,
            "source_dataset": str(processed_source) if source_hash else None,
            "source_sha256": source_hash,
            "target_definitions": {},
            "training_records": max(records, 0),
            "test_records": 0,
            "training_groups": 0,
            "test_groups": 0,
            "metrics": None,
            "artifact_path": None,
            "eligible_for_inference": False,
            "reason": reason,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _run_chatbot_training():
    import subprocess
    import sys

    record_training_status(
        "chatbot_intent_classifier",
        {
            "status": "running",
            "task": "custom automotive chatbot intent classification",
            "eligible_for_inference": False,
            "reason": "Chatbot classifier training and holdout evaluation are in progress.",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    script = SERVICE_ROOT / "train_chatbot.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=SERVICE_ROOT,
        check=False,
        text=True,
    )
    if completed.returncode:
        record_training_status(
            "chatbot_intent_classifier",
            {
                "status": "failed",
                "task": "automotive chatbot intent classification",
                "eligible_for_inference": False,
                "reason": f"train_chatbot.py exited with code {completed.returncode}.",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise RuntimeError(
            f"Chatbot training failed with exit code {completed.returncode}."
        )


def _main():
    parser = argparse.ArgumentParser(description="Automotive dataset evidence pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inventory")
    commands.add_parser("prepare-enginefaultdb")
    commands.add_parser("train-maintenance")
    commands.add_parser("train-chatbot")
    commands.add_parser("status")
    extract = commands.add_parser("safe-extract")
    extract.add_argument("archive", type=Path)
    extract.add_argument("--destination", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--input", type=Path, required=True)
    train.add_argument("--target", required=True)
    train.add_argument("--group-column", required=True)
    train.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.command == "inventory":
        inventory = write_inventory()
        print(
            f"Inspected {inventory['files_inspected']} files; "
            f"{len(inventory['error_paths'])} content inspection errors."
        )
    elif arguments.command == "prepare-enginefaultdb":
        result = prepare_enginefaultdb()
        print(
            f"Prepared {result['rows_after']} rows; removed "
            f"{result['exact_duplicates_removed']} exact duplicate."
        )
    elif arguments.command == "status":
        print(json.dumps(read_training_status(), indent=2))
    elif arguments.command == "train-chatbot":
        _run_chatbot_training()
        print(json.dumps(read_training_status(), indent=2))
    elif arguments.command == "train-maintenance":
        record_training_status(
            "maintenance_model",
            {
                "status": "running",
                "model_name": "maintenance_model",
                "task": "vehicle maintenance prediction",
                "eligible_for_inference": False,
                "reason": "Dataset preparation and group-aware evaluation are in progress.",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        try:
            prepare_enginefaultdb()
            source = (
                SERVICE_ROOT / "data" / "processed" / "enginefaultdb"
                / "enginefaultdb_normalized.csv"
            )
            frame = pd.read_csv(source, low_memory=False)
            train_grouped_classifier(
                frame,
                "fault_class",
                "acquisition_run_id",
                SERVICE_ROOT / "data" / "training" / "candidates"
                / "enginefaultdb",
            )
        except (FileNotFoundError, ValueError) as error:
            _record_blocked_maintenance(str(error))
            print(f"Maintenance training blocked: {error}")
            raise SystemExit(2) from error
        except Exception as error:
            record_training_status(
                "maintenance_model",
                {
                    "status": "failed",
                    "model_name": "maintenance_model",
                    "task": "vehicle maintenance prediction",
                    "eligible_for_inference": False,
                    "reason": str(error),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            raise
        _record_blocked_maintenance(
            "The grouped laboratory classifier is not a validated vehicle "
            "maintenance model and cannot be promoted for inference."
        )
        print("Candidate evaluation completed; maintenance inference remains blocked.")
    elif arguments.command == "safe-extract":
        archive = arguments.archive.resolve()
        destination = arguments.destination.resolve()
        archive_root = RAW_ROOT.resolve()
        destination_root = (
            SERVICE_ROOT / "data" / "processed" / "archives"
        ).resolve()
        try:
            archive.relative_to(archive_root)
            destination.relative_to(destination_root)
        except ValueError as error:
            raise ValueError(
                "CLI extraction is restricted to archives in data/raw and "
                "destinations in data/processed/archives."
            ) from error
        extracted = safe_extract_archive(arguments.archive, arguments.destination)
        print(f"Extracted {len(extracted)} files to {arguments.destination}.")
    else:
        record_training_status(
            "grouped_classifier_candidate",
            {
                "status": "running",
                "task": "grouped classification research candidate",
                "eligible_for_inference": False,
                "reason": "Grouped holdout training and evaluation are in progress.",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        try:
            frame = pd.read_csv(arguments.input, low_memory=False)
            result = train_grouped_classifier(
                frame,
                arguments.target,
                arguments.group_column,
                arguments.output,
            )
        except (FileNotFoundError, ValueError) as error:
            record_training_status(
                "grouped_classifier_candidate",
                {
                    "status": "blocked",
                    "task": "grouped classification research candidate",
                    "eligible_for_inference": False,
                    "reason": str(error),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(f"Training blocked: {error}")
            raise SystemExit(2) from error
        except Exception as error:
            record_training_status(
                "grouped_classifier_candidate",
                {
                    "status": "failed",
                    "task": "grouped classification research candidate",
                    "eligible_for_inference": False,
                    "reason": str(error),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            raise
        source_hash = hashlib.sha256(arguments.input.read_bytes()).hexdigest()
        evaluation_path = arguments.output / "evaluation.json"
        result["source_dataset"] = str(arguments.input)
        result["source_sha256"] = source_hash
        evaluation_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        record_training_status(
            "grouped_classifier_candidate",
            {
                "status": "evaluated_rejected",
                "task": "grouped classification research candidate",
                "source_dataset": str(arguments.input),
                "source_sha256": source_hash,
                "target": arguments.target,
                "group_column": arguments.group_column,
                "artifact_path": result["artifact_path"],
                "metrics": result["metrics"],
                "eligible_for_inference": False,
                "reason": result["reason"],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    _main()
