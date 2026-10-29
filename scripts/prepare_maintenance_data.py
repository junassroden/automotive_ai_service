import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

from scripts.maintenance_experiment import (
    EOL_CAPACITY_AH,
    PREPARED_ROOT,
    PROCESSED_ROOT,
    RAW_ROOT,
    SUPPORTED_CELLS,
    extract_cell_cycles,
    prepare_cell_records,
    resolve_within,
    sha256_file,
    write_json,
)


def prepare(output=PREPARED_ROOT):
    output = resolve_within(output, PROCESSED_ROOT)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing prepared data: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    raw_files = {
        path.stem.upper(): path
        for path in RAW_ROOT.glob("5. Battery Data Set/**/*.mat")
        if path.stem.upper() in SUPPORTED_CELLS
    }
    missing = SUPPORTED_CELLS.difference(raw_files)
    if missing:
        raise FileNotFoundError(f"Required documented NASA cell files missing: {sorted(missing)}")

    rows = []
    included = []
    excluded = []
    for cell_id in sorted(SUPPORTED_CELLS):
        source = raw_files[cell_id]
        discharge_cycles = extract_cell_cycles(source)
        cell_rows, invalid_count, reason = prepare_cell_records(
            cell_id,
            discharge_cycles,
            EOL_CAPACITY_AH,
        )
        if reason:
            excluded.append(
                {
                    "cell_id": cell_id,
                    "path": source.relative_to(RAW_ROOT).as_posix(),
                    "reason": reason,
                    "invalid_capacity_records": invalid_count,
                }
            )
            continue
        rows.extend(cell_rows)
        included.append(
            {
                "cell_id": cell_id,
                "path": source.relative_to(RAW_ROOT).as_posix(),
                "sha256": sha256_file(source),
                "file_bytes": source.stat().st_size,
                "discharge_records": len(discharge_cycles),
                "prepared_records": len(cell_rows),
                "invalid_capacity_records": invalid_count,
                "first_capacity_ah": cell_rows[0]["capacity_ah"],
                "eol_capacity_ah": cell_rows[-1]["capacity_ah"],
                "eol_cycle_index": cell_rows[-1]["discharge_cycle_index"],
            }
        )

    if len(included) < 3:
        raise ValueError(
            f"At least three independently identified cells are required; found {len(included)}."
        )
    frame = pd.DataFrame(rows)
    if frame.empty or frame["cell_id"].nunique() != len(included):
        raise RuntimeError("Prepared rows do not match the included cell groups.")
    csv_path = output / "discharge_cycles.csv"
    frame.to_csv(csv_path, index=False)
    manifest = {
        "task": "laboratory battery-cell remaining useful life",
        "target": "remaining_discharge_cycles",
        "target_definition": (
            "Number of discharge cycles remaining until the first observed positive "
            f"capacity at or below {EOL_CAPACITY_AH} Ah."
        ),
        "eol_capacity_ah": EOL_CAPACITY_AH,
        "feature_columns": [
            "discharge_cycle_index",
            "capacity_ah",
            "ambient_temperature_c",
        ],
        "group_column": "cell_id",
        "source": "NASA Li-ion Battery Aging Datasets",
        "source_url": "https://data.nasa.gov/dataset/li-ion-battery-aging-datasets",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "prepared_csv": csv_path.name,
        "prepared_csv_sha256": sha256_file(csv_path),
        "row_count": len(frame),
        "cell_count": int(frame["cell_id"].nunique()),
        "raw_files": included,
        "excluded_files": excluded,
        "transformations": [
            "Read MATLAB files with scipy.io.loadmat; raw files were not changed.",
            "Kept only documented discharge operations and their measured capacity (Ah).",
            "Excluded non-finite, zero, and negative capacity values as invalid measurements.",
            "Required the first valid capacity to be above 1.4 Ah and an observed crossing at or below 1.4 Ah.",
            "Kept each cell only through its first valid EOL-threshold crossing.",
            "Defined RUL as the difference between source discharge-cycle index and the first crossing index.",
            "Retained cell_id only as an evaluation grouping key, not as a model feature.",
            "No scaler, imputer, selector, or other learned transform was fit during preparation.",
        ],
        "limits": [
            "This is laboratory cell-aging prognosis, not installed vehicle battery-pack RUL.",
            "Only three independent cell groups are included; one cell is held out, so the test result is highly uncertain.",
            "The selected cohort is described as room temperature with 2 A discharge; cutoff voltage differs by cell.",
            "Local data bundle does not include a separate license statement; do not redistribute without confirming reuse terms.",
        ],
        "cohort_exclusions": [
            "B0007: observed capacity does not cross 1.4 Ah.",
            "B0042-B0044: unusually low capacity records are explicitly unresolved in the local README; observed first threshold values are approximately 0.06-0.08 Ah.",
            "B0045: initial capacity is already below 1.4 Ah.",
            "B0046-B0048: local README warns of unexplained low-capacity runs and the cohort uses different protocols/cutoffs.",
            "B0049-B0052: local README reports the experiment ended after control software crashed.",
            "B0053-B0056: low capacities are flagged in the local README and initial capacities are below the selected threshold.",
            "B0025-B0032: local documentation does not establish the selected common EOL threshold.",
            "B0033-B0040: local documentation specifies a different 1.6 Ah/20% threshold.",
        ],
    }
    write_json(output / "dataset_manifest.json", manifest)
    print(
        f"Prepared {len(frame)} discharge-cycle rows from {len(included)} cells; "
        f"excluded {len(excluded)} cells."
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PREPARED_ROOT)
    prepare(parser.parse_args().output)
