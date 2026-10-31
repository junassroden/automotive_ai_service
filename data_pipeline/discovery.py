"""Recursive discovery and profiling of every file under data/raw.

Raw files are only read: nothing is extracted, executed or modified. Large
delimited files are profiled in chunks so memory use stays bounded. The result
is one record per file plus a per-dataset roll-up, which ``run_all`` merges
with each pipeline's usage report into the dataset manifest.
"""
import csv
import fnmatch
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.dataset_rules import DATASET_RULES, SUPPORT_EXTENSIONS


CSV_CHUNK_ROWS = 200_000
TABULAR_EXTENSIONS = {".csv", ".tsv"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def match_rule(relative_path):
    lowered = relative_path.casefold()
    for rule in DATASET_RULES:
        if any(fnmatch.fnmatchcase(lowered, pattern) for pattern in rule["patterns"]):
            return rule
    return None


def _frame_profile(frame):
    nulls = frame.isna().sum()
    return {
        "columns": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": {str(column): int(count) for column, count in nulls.items() if count},
    }


def _profile_delimited(path):
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        sample = stream.read(16384)
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    rows = 0
    duplicate_rows = 0
    seen = set()
    nulls = None
    first = None
    for chunk in pd.read_csv(
        path, sep=delimiter, chunksize=CSV_CHUNK_ROWS, low_memory=False,
        encoding="utf-8-sig", encoding_errors="replace", on_bad_lines="warn",
    ):
        if first is None:
            first = chunk.head(0)
            nulls = chunk.isna().sum()
        else:
            nulls = nulls.add(chunk.isna().sum(), fill_value=0)
        rows += len(chunk)
        hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy()
        for value in hashes:
            if value in seen:
                duplicate_rows += 1
            else:
                seen.add(value)
    profile = _frame_profile(first if first is not None else pd.DataFrame())
    profile["missing_values"] = {
        str(column): int(count) for column, count in (nulls if nulls is not None else {}).items() if count
    }
    profile.update(rows=rows, duplicate_rows=duplicate_rows, delimiter=delimiter)
    return profile


def _profile_frame(frame):
    profile = _frame_profile(frame)
    hashable = frame.copy()
    for column in hashable.columns:
        if hashable[column].dtype == object:
            hashable[column] = hashable[column].map(
                lambda value: json.dumps(value.tolist() if isinstance(value, np.ndarray) else value, default=str)
            )
    profile["rows"] = int(len(frame))
    profile["duplicate_rows"] = int(hashable.duplicated().sum())
    return profile


def _profile_excel(path):
    workbook = pd.ExcelFile(path)
    sheets = {}
    for sheet in workbook.sheet_names:
        frame = workbook.parse(sheet)
        sheets[sheet] = _profile_frame(frame)
    first = next(iter(sheets.values()), {"rows": 0, "columns": []})
    return {"sheets": sheets, "rows": sum(sheet["rows"] for sheet in sheets.values()),
            "columns": first["columns"], "duplicate_rows": sum(s["duplicate_rows"] for s in sheets.values())}


def _profile_json(path):
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        keys = sorted({key for item in payload[:500] if isinstance(item, dict) for key in item})
        return {"rows": len(payload), "columns": keys, "json_type": "array"}
    if isinstance(payload, dict):
        return {"rows": 1, "columns": sorted(payload)[:60], "json_type": "object"}
    return {"rows": 1, "json_type": type(payload).__name__}


def _profile_jsonl(path):
    rows = 0
    invalid = 0
    keys = set()
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            if not line.strip():
                continue
            rows += 1
            try:
                item = json.loads(line)
                if isinstance(item, dict) and len(keys) < 200:
                    keys.update(item)
            except json.JSONDecodeError:
                invalid += 1
    return {"rows": rows, "columns": sorted(keys), "invalid_lines": invalid}


def _profile_yaml(path):
    # PyYAML is not a dependency; count top-level list entries ("- key:").
    entries = 0
    first_keys = defaultdict(int)
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("- ") and ":" in line:
                entries += 1
                first_keys[line[2:].split(":", 1)[0].strip()] += 1
    return {"rows": entries, "entry_keys": dict(first_keys)}


def _profile_mat(path):
    from scipy.io import whosmat
    return {"mat_variables": [[name, list(shape), kind] for name, shape, kind in whosmat(str(path))]}


def _profile_zip(path):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        corrupt = archive.testzip()
        return {
            "archive_members": len(members),
            "uncompressed_bytes": sum(member.file_size for member in members),
            "corrupt_member": corrupt,
            "member_sample": [member.filename for member in members[:15]],
        }


def _looks_tabular_text(path):
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        lines = [stream.readline() for _ in range(5)]
    lines = [line for line in lines if line.strip()]
    if len(lines) < 3:
        return False
    counts = {line.count(",") for line in lines}
    return len(counts) == 1 and counts.pop() >= 2


def profile_file(path):
    extension = path.suffix.lower() if path.suffix else path.name.lower()
    if path.stat().st_size == 0:
        return {"format": "empty", "profile_status": "empty", "issue": "File is zero bytes."}
    try:
        if extension in TABULAR_EXTENSIONS or (extension == ".txt" and _looks_tabular_text(path)):
            return {"format": "delimited", "profile_status": "profiled", **_profile_delimited(path)}
        if extension == ".parquet":
            return {"format": "parquet", "profile_status": "profiled", **_profile_frame(pd.read_parquet(path))}
        if extension in {".xlsx", ".xlsm", ".xls"}:
            return {"format": "excel", "profile_status": "profiled", **_profile_excel(path)}
        if extension == ".json":
            return {"format": "json", "profile_status": "profiled", **_profile_json(path)}
        if extension in {".jsonl", ".ndjson"}:
            return {"format": "jsonl", "profile_status": "profiled", **_profile_jsonl(path)}
        if extension == ".yaml":
            return {"format": "yaml", "profile_status": "profiled", **_profile_yaml(path)}
        if extension == ".mat":
            return {"format": "matlab", "profile_status": "profiled", **_profile_mat(path)}
        if extension == ".zip":
            return {"format": "zip", "profile_status": "profiled", **_profile_zip(path)}
        if extension in {".mbl", ".cbf"}:
            return {"format": "binary_log", "profile_status": "opaque",
                    "issue": "Proprietary binary log; decoded copies (if any) are profiled separately."}
        kind = SUPPORT_EXTENSIONS.get(extension)
        if kind is None and path.name.upper().startswith("LICENSE"):
            kind = "documentation"
        if kind is None and path.name == "DataDescription":
            kind = "documentation"
        if kind:
            return {"format": kind, "profile_status": "not_tabular"}
        return {"format": "unknown", "profile_status": "unsupported",
                "issue": f"No reader for extension '{extension}'."}
    except Exception as error:  # a corrupt file must be reported, not skip the scan
        return {"format": extension, "profile_status": "error", "issue": f"{type(error).__name__}: {error}"}


def discover(raw_root):
    raw_root = Path(raw_root)
    records = []
    for path in sorted(item for item in raw_root.rglob("*") if item.is_file()):
        relative = path.relative_to(raw_root).as_posix()
        rule = match_rule(relative)
        record = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "dataset_id": rule["dataset_id"] if rule else "unclassified",
            "data_nature": rule["data_nature"] if rule else "unreviewed",
        }
        record.update(profile_file(path))
        records.append(record)

    by_hash = defaultdict(list)
    for record in records:
        if record["bytes"]:
            by_hash[record["sha256"]].append(record)
    for group in by_hash.values():
        if len(group) > 1:
            canonical = min(group, key=lambda item: (item["path"].count("/"), item["path"]))
            for record in group:
                if record is not canonical:
                    record["duplicate_of"] = canonical["path"]
    return records
