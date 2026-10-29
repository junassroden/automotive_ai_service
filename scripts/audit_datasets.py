import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

from data_pipeline.cli import build_inventory


RAW_ROOT = SERVICE_ROOT / "data" / "raw"
REPORT_ROOT = SERVICE_ROOT / "data" / "reports"

DATASET_INFO = {
    "nasa_battery": {
        "name": "NASA Li-ion Battery Aging Datasets",
        "vehicle_component": "Laboratory lithium-ion battery cells; not installed vehicle packs.",
        "target": "For selected cells, remaining discharge cycles to the documented 1.4 Ah capacity threshold.",
        "identifier_evidence": "Battery IDs are present in MAT filenames and used as cell groups; no fleet vehicle IDs.",
        "decision": "Suitable for a narrow laboratory battery-cell RUL experiment only.",
        "reason": "NASA battery documentation describes cycle-level discharge capacity and an explicit end-of-life threshold for selected cohorts. Cells without a reliable threshold crossing are excluded; cell ID enables group-disjoint evaluation.",
        "source": "https://data.nasa.gov/dataset/li-ion-battery-aging-datasets",
        "license_evidence": "No separate license statement was found in the local NASA bundle; verify reuse terms before redistribution.",
        "limitations": "Small number of independent cells; mixed temperatures and protocols; several cohorts lack a confirmed common stop threshold or contain anomalous capacities. Not evidence for vehicle/fleet maintenance.",
    },
    "kit_obd": {
        "name": "KIT Automotive OBD-II Dataset",
        "vehicle_component": "One Seat Leon vehicle model; OBD-II operating telemetry.",
        "target": "No confirmed component-failure, repair, or maintenance target.",
        "identifier_evidence": "Session/file names exist; no stable fleet vehicle identifier was established.",
        "decision": "Supporting/reference telemetry only.",
        "reason": "Signals and road-condition sessions are present, but no verified fault or maintenance outcomes are provided.",
        "source": "https://doi.org/10.35097/1130",
        "license_evidence": "Local descriptive metadata states CC BY 4.0; verify applicable item terms before redistribution.",
        "limitations": "Not suitable for supervised fault/failure prediction.",
    },
    "obdex": {
        "name": "OBDex generic OBD-II code and PID reference",
        "vehicle_component": "Generic DTC definitions and OBD-II PID reference data; no measured vehicles.",
        "target": "No observation-level model target; code-to-description reference entries only.",
        "identifier_evidence": "DTC and PID codes are reference identifiers, not vehicle/acquisition IDs.",
        "decision": "Supporting/reference data only; not a predictive-maintenance training dataset.",
        "reason": "The local README describes generic DTC/PID lookup content. It is not a set of confirmed vehicle events or repair outcomes.",
        "source": "https://github.com/foerbsnavi/obdex",
        "license_evidence": "Local LICENSE-DATA identifies CC0 1.0 for data; LICENSE-CODE separately identifies MIT for code.",
        "limitations": "The README reports 9,533 generic codes and 132 PIDs; local YAML/JSON files were inventoried but not used to infer fault prevalence or predictive labels.",
    },
    "carobd": {
        "name": "carOBD Toyota Etios telemetry",
        "vehicle_component": "One 2014 Toyota Etios; 27 OBD-II PIDs at 1 Hz across named driving/idle logs.",
        "target": "No confirmed fault, repair, maintenance, or EOL label; coolant-sensor anomalies are discussed as a research use.",
        "identifier_evidence": "Log filenames distinguish acquisition sessions/modes; the data describes one vehicle, not independent vehicle groups.",
        "decision": "Supporting telemetry only; suitable for exploratory single-vehicle sensor anomaly scoring, not validated fault or future-failure prediction.",
        "reason": "129 CSV files contain 304,299 rows and 27 columns. The local README describes one Toyota Etios and 1 Hz collection; no ground-truth anomaly labels or confirmed outcomes are present.",
        "source": "https://github.com/eron93br/carOBD (local README cites a thesis/article on coolant-sensor anomaly detection).",
        "license_evidence": "No formal dataset license file was located; README says free to everyone and requests citation, which is not a precise license grant.",
        "limitations": "There are 42,537 within-file duplicate rows and substantial missingness, especially in warm-ups-since-clear. The repository has no clearly documented dataset license; verify reuse terms.",
    },
    "enginefaultdb": {
        "name": "EngineFaultDB",
        "vehicle_component": "One controlled C14NE laboratory engine.",
        "target": "Numeric engine-state/fault classes 0-3; named class meanings are not verified locally.",
        "identifier_evidence": "No vehicle, run, trip, or acquisition-run identifier in the local table.",
        "decision": "Unsuitable until class definitions and acquisition-run identifiers are resolved.",
        "reason": "No vehicle/run/trip identifiers; cannot create group-disjoint evaluation. Numeric label mapping to fault mechanisms is undocumented in the local README.",
        "source": "https://github.com/Leo-Thomas/EngineFaultDB",
        "license_evidence": "A GPL-3.0 license file is present in the local repository copy.",
        "limitations": "A row split would risk leakage; not future failure or fleet maintenance data.",
    },
    "diesel_obd": {
        "name": "Diesel OBD workbooks",
        "vehicle_component": "Claimed diesel OBD telemetry; vehicle identities not present in workbook columns.",
        "target": "Target linkage and meanings not established from local workbook evidence.",
        "identifier_evidence": "No vehicle/trip/run identifiers in the workbook columns.",
        "decision": "Unsuitable until provenance, labels, license, and vehicle-disjoint split are verified.",
        "reason": "No vehicle/run IDs; source and test split independence are not established; exact duplicates are extensive.",
        "source": "Local OBD-Dataset-main documentation (source link not verified).",
        "license_evidence": "README claims CC BY 4.0, but no separate license file or verified source record is present locally.",
        "limitations": "The local README's source and label claims are not independently validated.",
    },
    "cvffad": {
        "name": "Chinese Vehicle Fleet Fault Association Dataset (CVFFAD)",
        "vehicle_component": "Generated fleet fault-event records across claimed vehicle types.",
        "target": "Generated fault-event/category fields; not verified real maintenance outcomes.",
        "identifier_evidence": "Generated vehicle_id/event_id values are present but do not establish real vehicle identities.",
        "decision": "Unsuitable until provenance and redistribution terms are resolved.",
        "reason": "Local DOI is a placeholder; referenced license and data-availability files are absent.",
        "source": "Local README only; DOI is a placeholder.",
        "license_evidence": "README claims CC BY-NC-SA 4.0; the referenced license file is absent and the DOI is a placeholder.",
        "limitations": "Synthetic records do not establish real fleet coverage or failure prevalence.",
    },
    "ai4i": {
        "name": "AI4I 2020 Predictive Maintenance",
        "vehicle_component": "Synthetic industrial machinery.",
        "target": "Machine failure and failure modes.",
        "identifier_evidence": "Synthetic rows have no independent road-vehicle identity/group for automotive evaluation.",
        "decision": "Supporting/reference data only; excluded from automotive training.",
        "reason": "The dataset is explicitly synthetic industrial-machine data, not road-vehicle data.",
        "source": "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset",
        "license_evidence": "UCI identifies the dataset as CC BY 4.0.",
        "limitations": "No automotive transfer or fleet generalization claim.",
    },
    "navarra": {
        "name": "Navarra motorcycle telemetry",
        "vehicle_component": "One motorcycle and rider across recorded sessions.",
        "target": "No confirmed failure, repair, maintenance, or RUL label.",
        "identifier_evidence": "Session/boot continuity identifiers exist for one motorcycle; no independent vehicle groups.",
        "decision": "Supporting sensor telemetry only.",
        "reason": "Session telemetry has grouping context but lacks confirmed target outcomes and independent vehicles.",
        "source": "Local files and notes; origin/license claims not independently verified.",
        "license_evidence": "No locally verified dataset license; do not assume the cited unrelated campaign terms apply.",
        "limitations": "Cannot validate maintenance or failure predictions.",
    },
    "experimental_energy": {
        "name": "Experimental Energy Characterization",
        "vehicle_component": "Vehicle/bench context not documented.",
        "target": "No documented target.",
        "identifier_evidence": "No vehicle, trip, or acquisition identifiers documented.",
        "decision": "Unsuitable until source, units, identifiers, labels, and license are documented.",
        "reason": "Local CSVs have sensor-like values but no accompanying provenance or outcome labels.",
        "source": "No source documented locally.",
        "license_evidence": "No source or license documented locally.",
        "limitations": "Cannot establish vehicle scope or a valid evaluation split.",
    },
    "levin": {
        "name": "LEVIN vehicle telematics",
        "vehicle_component": "Four-wheel vehicle telematics described by documentation.",
        "target": "No local measurement export or maintenance outcome.",
        "identifier_evidence": "Documentation describes a multi-vehicle source, but no raw local data/IDs are available.",
        "decision": "Supporting documentation only.",
        "reason": "The local folder contains documentation/notebooks but no raw measurement export; notebooks were not executed.",
        "source": "https://github.com/YunSolutions/levin-openData",
        "license_evidence": "Local README states CC BY-NC-SA for the data; repository code MIT license is separate, and no raw data export is present.",
        "limitations": "No local data to prepare or evaluate.",
    },
    "root_parquet": {
        "name": "Unattributed standardized Parquet",
        "vehicle_component": "Unknown.",
        "target": "fault_type values are undocumented; flag is mostly missing.",
        "identifier_evidence": "No verified vehicle, trip, or acquisition-group IDs.",
        "decision": "Unsuitable until provenance, labels, units, and identifiers are verified.",
        "reason": "No accompanying source/documentation or group IDs; target semantics cannot be inferred safely.",
        "source": "No source documented locally.",
        "license_evidence": "No source or license documented locally.",
        "limitations": "Cannot treat fault-like names as confirmed faults.",
    },
    "root_engine": {
        "name": "Unattributed engine-condition tables",
        "vehicle_component": "Unknown engine sensor records.",
        "target": "Binary engine-condition label with undocumented meaning.",
        "identifier_evidence": "No vehicle/run identifiers; supplied partition is not proven group-disjoint.",
        "decision": "Unsuitable until provenance, label definition, units, and group IDs are verified.",
        "reason": "A supplied row split does not establish vehicle/run independence.",
        "source": "No source documented locally.",
        "license_evidence": "No source or license documented locally.",
        "limitations": "No fleet or future-maintenance claim supported.",
    },
    "root_service": {
        "name": "Unattributed vehicle service records",
        "vehicle_component": "Rows say car; specific provenance is unavailable.",
        "target": "Service flags/cost with no documented future-outcome meaning.",
        "identifier_evidence": "slno is unique but is not documented as a vehicle identifier; no service timestamps.",
        "decision": "Unsuitable until vehicle IDs, service dates, source, and target semantics are supplied.",
        "reason": "No chronology or documented vehicle key exists.",
        "source": "No source documented locally.",
        "license_evidence": "No source or license documented locally.",
        "limitations": "Cannot form a temporal or vehicle-disjoint prediction target.",
    },
    "supporting": {
        "name": "Supporting files and metadata",
        "vehicle_component": "Not applicable.",
        "target": "No model target.",
        "identifier_evidence": "Not applicable for documentation/code; opaque telemetry identifiers are not decoded.",
        "decision": "Supporting/reference material only.",
        "reason": "Documentation, source code, notebook, archive metadata, or opaque files are not direct labeled observations.",
        "source": "See exact local path in the inventory CSV.",
        "license_evidence": "Varies by file; inspect the exact local source/license documents before reuse.",
        "limitations": "Opaque MBL/CBF files were not decoded; downloaded notebooks/scripts were not executed.",
    },
}


def dataset_key(relative_path):
    path = relative_path.replace("\\", "/")
    lower = path.lower()
    if path.startswith("5. Battery Data Set/"):
        return "nasa_battery"
    if path.startswith("EngineFaultDB-main/"):
        return "enginefaultdb"
    if path.startswith("OBD-Dataset-main/"):
        return "diesel_obd"
    if path.startswith("Chinese Vehicle Fleet Fault Association Dataset/"):
        return "cvffad"
    if path.startswith("ai4i+2020+predictive+maintenance+dataset/"):
        return "ai4i"
    if path.startswith("levin-openData-master/"):
        return "levin"
    if path.startswith("Experimental Energy Characterization/"):
        return "experimental_energy"
    if path.startswith("10.35097-1130/") or path.startswith("Automotive_Diagnostics-main/"):
        return "kit_obd"
    if path.startswith("OBDex-main/"):
        return "obdex"
    if path.startswith("carOBD-master/"):
        return "carobd"
    if path.startswith("navarra") or "/navarra" in lower:
        return "navarra"
    if path == "main_df_standardized.parquet":
        return "root_parquet"
    if path.lower().endswith("vehicle maintenance- service records.csv"):
        return "root_service"
    if Path(path).name.lower() in {"raw_data.csv", "train.csv", "test.csv"}:
        return "root_engine"
    if "/" not in path:
        return "supporting"
    return "supporting"


def write_audit():
    inventory = build_inventory(RAW_ROOT)
    grouped = defaultdict(list)
    for item in inventory["files"]:
        key = dataset_key(item["path"])
        grouped[key].append(item)

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_ROOT / "dataset_inventory.csv"
    columns = [
        "dataset_key", "dataset_name", "path", "extension", "bytes",
        "inspection_status",         "rows", "columns", "column_names", "missing_values",
        "duplicate_rows", "sha256", "classification", "candidate_task",
        "license_evidence", "audit_notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in inventory["files"]:
            key = dataset_key(item["path"])
            info = DATASET_INFO[key]
            writer.writerow(
                {
                    "dataset_key": key,
                    "dataset_name": info["name"],
                    "path": item["path"],
                    "extension": item.get("extension"),
                    "bytes": item.get("bytes"),
                    "inspection_status": item.get("inspection_status"),
                    "rows": item.get("rows", ""),
                    "columns": len(item.get("columns", [])) if item.get("columns") is not None else "",
                    "column_names": json.dumps(
                        [column.get("name") for column in item.get("columns", [])],
                        ensure_ascii=True,
                    ),
                    "missing_values": json.dumps(item.get("missing_values", {}), ensure_ascii=True),
                    "duplicate_rows": item.get("duplicate_rows", ""),
                    "sha256": item.get("sha256", ""),
                    "classification": info["decision"],
                    "candidate_task": (
                        "laboratory battery-cell RUL to documented 1.4 Ah threshold"
                        if key == "nasa_battery"
                        else ""
                    ),
                    "license_evidence": info.get(
                        "license_evidence",
                        "No source-specific license evidence recorded; inspect local files.",
                    ),
                    "audit_notes": info["reason"],
                }
            )

    suitability = {
        "raw_root": str(RAW_ROOT),
        "files_inspected": inventory["files_inspected"],
        "top_level_directories": inventory["top_level_directories"],
        "inspection_error_paths": inventory["error_paths"],
        "duplicate_hash_groups": inventory["identical_file_groups"],
        "datasets": [
            {
                "key": key,
                **DATASET_INFO[key],
                "license_evidence": DATASET_INFO[key].get(
                    "license_evidence",
                    "No source-specific license evidence recorded; inspect local files.",
                ),
                "file_count": len(files),
                "total_bytes": sum(item["bytes"] for item in files),
                "files": [
                    {
                        "path": item["path"],
                        "sha256": item.get("sha256"),
                        "bytes": item.get("bytes"),
                        "rows": item.get("rows"),
                        "columns": item.get("columns"),
                        "missing_values": item.get("missing_values"),
                        "duplicate_rows": item.get("duplicate_rows"),
                        "observed_target_value_counts": item.get("observed_target_value_counts"),
                        "structure_summaries": item.get("structure_summaries"),
                        "inspection_status": item.get("inspection_status"),
                    }
                    for item in files
                ],
            }
            for key, files in sorted(grouped.items())
        ],
        "task_decisions": {
            "sensor_anomaly_detection": (
                "No anomaly ground truth is available to validate anomalies as component faults. "
                "Unlabeled telemetry may support exploratory scoring only."
            ),
            "existing_fault_classification": (
                "No local vehicle dataset has both verified fault semantics and an independent "
                "vehicle/run holdout."
            ),
            "dtc_classification": "No verified DTC-linked observations and group IDs were established.",
            "future_component_failure_prediction": (
                "Unsupported: no verified time-linked fleet failure outcomes and vehicle-disjoint "
                "holdout were identified."
            ),
            "remaining_useful_life": (
                "Supported only as a limited laboratory battery-cell experiment using documented "
                "1.4 Ah EOL cohorts and cells with observed valid threshold crossings. This is not "
                "vehicle battery-pack or fleet maintenance RUL."
            ),
        },
        "selected_rul_cells": ["B0005", "B0006", "B0018"],
        "battery_cohort_decisions": [
            {
                "cells": ["B0005", "B0006", "B0018"],
                "decision": "selected for laboratory RUL experiment",
                "reason": "Their local cohort README documents a 1.4 Ah/30% fade EOL and each positive capacity trajectory begins above and crosses 1.4 Ah.",
            },
            {
                "cells": ["B0007"],
                "decision": "excluded",
                "reason": "The observed positive capacity sequence does not cross 1.4 Ah (minimum is about 1.4005 Ah); do not invent a crossing.",
            },
            {
                "cells": ["B0042", "B0043", "B0044"],
                "decision": "excluded",
                "reason": "The local README says very low capacities occurred and their causes are not analyzed. The first observed <=1.4 Ah capacities are abrupt values around 0.06-0.08 Ah, so they are not accepted as reliable EOL targets.",
            },
            {
                "cells": ["B0045"],
                "decision": "excluded",
                "reason": "The first valid positive capacity is already below the 1.4 Ah threshold.",
            },
            {
                "cells": ["B0046", "B0047", "B0048"],
                "decision": "excluded",
                "reason": "The local README flags unexplained very low capacity readings; protocols/cutoff voltages differ, and a conservative first experiment uses the cleaner room-temperature cohort.",
            },
            {
                "cells": ["B0049", "B0050", "B0051", "B0052"],
                "decision": "excluded",
                "reason": "The local README says the experiment ended when control software crashed; the endpoint is not a documented EOL event.",
            },
            {
                "cells": ["B0053", "B0054", "B0055", "B0056"],
                "decision": "excluded",
                "reason": "The local README flags unexplained low capacities; observed initial capacities are below 1.4 Ah, so valid RUL-to-threshold trajectories cannot be formed.",
            },
            {
                "cells": ["B0025", "B0026", "B0027", "B0028", "B0029", "B0030", "B0031", "B0032"],
                "decision": "excluded",
                "reason": "The local cohort documentation does not establish a common 1.4 Ah stopping criterion; the observed capacity series do not provide the selected target.",
            },
            {
                "cells": ["B0033", "B0034", "B0036", "B0038", "B0039", "B0040"],
                "decision": "excluded",
                "reason": "Their local README specifies a different 1.6 Ah/20% fade criterion; they are not relabeled to the selected 1.4 Ah task.",
            },
        ],
        "excluded_battery_reason": (
            "Cohort-level reasons and observed capacity checks are enumerated in "
            "battery_cohort_decisions. No undocumented or unobserved EOL label is inferred."
        ),
    }
    (REPORT_ROOT / "dataset_suitability.json").write_text(
        json.dumps(suitability, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Raw dataset audit",
        "",
        f"- Raw root: `{RAW_ROOT}`",
        f"- Files recursively inspected: **{inventory['files_inspected']}**",
        f"- Top-level directories: **{inventory['top_level_directories']}**",
        f"- Content inspection errors: **{len(inventory['error_paths'])}**",
        f"- Byte-identical duplicate groups: **{len(inventory['identical_file_groups'])}**",
        "",
        "The inventory script reused the existing safe scanner in `data_pipeline/cli.py`. "
        "It reads raw files, computes SHA-256 hashes, inventories tabular schemas and missing/"
        "duplicate values, checks archive metadata, and does not extract archives or execute "
        "downloaded scripts/notebooks. Unsupported MBL/CBF payloads remain opaque.",
        "",
        "## Dataset decisions",
        "",
        "| Dataset | Local files | Classification | Target/task and evidence | Identifier evidence | License evidence | Limitation |",
        "|---|---:|---|---|---|---|---|",
    ]
    for key, files in sorted(grouped.items()):
        info = DATASET_INFO[key]
        lines.append(
            f"| {info['name']} | {len(files)} | {info['decision']} | "
            f"{info['target']} {info['reason']} | {info['identifier_evidence']} | "
            f"{info.get('license_evidence', 'No source-specific license evidence recorded; inspect local files.')} | "
            f"{info['limitations']} |"
        )
    lines += [
        "",
        "## Task selection",
        "",
        "No dataset supports a verified road-vehicle future-failure predictor, DTC classifier, "
        "or broad fleet maintenance model. The only selected training task is a narrow "
        "laboratory-cell prognostics experiment: estimate remaining discharge cycles to 1.4 Ah "
        "for NASA cells whose local README documents that threshold and whose measured positive "
        "capacity sequence crosses it. EOL is not inferred for other cohorts. Evaluation holds "
        "out whole cells, never random rows.",
        "",
        "The selected cells are B0005, B0006, and B0018. The 1.4 Ah criterion is documented "
        "for their cohort; cells with no positive initial capacity above threshold, no valid "
        "crossing, or unresolved very-low capacity anomalies are excluded. Capacity records "
        "at or below zero are treated as invalid measurements, not EOL observations. Cohort "
        "inclusion/exclusion decisions are listed in `dataset_suitability.json`.",
        "",
        "## Dataset inventory CSV",
        "",
        "`dataset_inventory.csv` has one row per recursively inspected local file, with "
        "path, file type, bytes, SHA-256, row/column counts where measurable, column names, "
        "missing-value counts, duplicate-row counts, and inspection status. The JSON report "
        "contains the grouped per-file schema, source/license evidence, and task decisions. Row totals across duplicate "
        "copies are not treated as independent observations.",
        "",
        "The scanner does not impose universal valid-value ranges where dataset documentation "
        "does not define them. Field-specific checks for the selected battery task reject "
        "non-finite and nonpositive capacity values; exact exclusions and counts are recorded "
        "in the prepared dataset manifest.",
        "",
        "## Filesystem limitations",
        "",
        "- KIT extracted CSVs duplicate ZIP members; B0025-B0028 battery copies are byte-identical.",
        "- EngineFaultDB lacks run IDs and named mappings for numeric class labels 0-3.",
        "- Diesel workbook labels, source, license and supplied split grouping are unverified.",
        "- CVFFAD is marked generated locally and has a placeholder DOI/missing license materials.",
        "- Root Parquet/engine/service tables lack verified provenance or targets.",
        "- Navarra sessions have telemetry but no confirmed maintenance outcomes; one bike is not a fleet.",
        "- LEVIN has local documentation but no raw measurements. AI4I is synthetic industrial data.",
        "- No raw data was changed or extracted.",
        "",
    ]
    (REPORT_ROOT / "dataset_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"Audited {inventory['files_inspected']} files; "
        f"{len(inventory['error_paths'])} inspection errors."
    )


if __name__ == "__main__":
    write_audit()
