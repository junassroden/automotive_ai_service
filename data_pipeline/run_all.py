"""One command to rediscover data/raw, retrain, re-evaluate and rebuild the index.

    python -m data_pipeline.run_all                 # everything
    python -m data_pipeline.run_all --only inventory knowledge
    python -m data_pipeline.run_all --skip-experiments

Steps: discovery -> service-need model -> evaluation-only experiments ->
knowledge index -> dataset manifest and reports. New files under data/raw are
always discovered and listed; a file no rule recognises is reported as
``unclassified`` until a rule is added in data_pipeline/dataset_rules.py.
The model registry is only changed through status.record_training_status,
which never replaces an accepted model with a rejected one.
"""
import argparse
import csv
import fnmatch
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from data_pipeline import dataset_rules
from data_pipeline.discovery import discover
from data_pipeline.status import MODEL_REGISTRY_PATH, _read_json, _write_json_atomic, record_training_status


SERVICE_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = SERVICE_ROOT / "data" / "raw"
REPORT_ROOT = SERVICE_ROOT / "data" / "reports"
MODEL_ROOT = SERVICE_ROOT / "trained_models" / "service_need"
STEPS = ("inventory", "service_need", "experiments", "knowledge")

# Finer-grained use of individual files inside a dataset (fnmatch on lower-cased path).
FILE_ROLES = [
    ("vehicle maintenance- service records.csv", "used", "Training/evaluation rows for the service-need model; service statistics indexed for the chatbot."),
    ("train.csv", "used", "Engine-condition experiment training split (evaluation only, not served)."),
    ("test.csv", "used", "Engine-condition experiment test split (evaluation only, not served)."),
    ("raw_data.csv", "redundant", "Exact union of train.csv and test.csv; verified, not used again."),
    ("ev_predictive_maintenance_dataset_15min.csv", "audited_excluded", "Trained against baselines in the synthetic-signal audit; no learnable signal, excluded from served models."),
    ("cars_hyundai.csv", "audited_excluded", "Trained against baselines in the synthetic-signal audit; no learnable signal, excluded from served models."),
    ("*fault_events.csv", "used", "CVFFAD fault-system experiment (evaluation only; generated data, not served)."),
    ("*vehicle_fleet.csv", "excluded", "Generated CVFFAD fleet register; not needed by the experiment and not real fleet data."),
    ("*knowledge_graph_triples.csv", "excluded", "Generated CVFFAD triples (Chinese); not indexed because generated content would be presented as fact."),
    ("*ner_annotations.csv", "excluded", "Generated CVFFAD NER annotations (Chinese); no NER task in this system."),
    ("obd-dataset-main/obd-dataset-main/master_*_x.xlsx", "used", "Diesel DTC-presence experiment (evaluation only, not served)."),
    ("obd-dataset-main.zip", "redundant", "Archive of the extracted OBD-Dataset-main folder; the extracted workbooks are used."),
    ("obdex-main/data/*.yaml", "used", "Parsed into the chatbot knowledge index (DTC and PID definitions)."),
    ("automotive_faults_aktc_obike_et_al.json", "used", "Diagnostic flowcharts indexed for the chatbot (marked unverified source)."),
    ("*_flowchart.png", "excluded", "Images of the same flowcharts; the JSON version is indexed instead (no OCR)."),
    ("masterparts.xlsx", "used", "Exact part-number lookup table for the chatbot."),
    ("vehiclediagnosticsample.txt", "excluded", "Generated diagnostic text with random measurements; not indexed or trained on."),
    ("train-00000-of-00001.parquet", "excluded", "Synthetic DTC chat examples; answers are templated and contain unrelated truncated text, so they are not used for training."),
    ("validation-00000-of-00001.parquet", "used", "User questions only, as DTC-routing cases in the chatbot evaluation set (answers not used)."),
    ("carobd-master/obdiidata/*.csv", "used", "Existing carOBD sensor-anomaly experiment (trained_models/experiments/carobd_sensor_anomaly_v1)."),
    ("5. battery data set/*.mat", "used", "Existing NASA battery-cell RUL experiment (trained_models/experiments/nasa_battery_cell_rul_1p4ah)."),
]
SUPPORT_FORMATS = {"documentation", "code", "image", "metadata"}


def file_role(record, rule):
    path = record["path"].casefold()
    if record.get("duplicate_of"):
        return "duplicate", f"Byte-identical to {record['duplicate_of']}; processed once under that path."
    if record["profile_status"] == "empty":
        return "rejected", "Empty file (zero bytes)."
    if record["profile_status"] == "error":
        return "rejected", f"Unreadable: {record.get('issue')}"
    if record["profile_status"] == "unsupported":
        return "rejected", record.get("issue", "Unsupported format.")
    for pattern, status, reason in FILE_ROLES:
        if fnmatch.fnmatchcase(path, pattern):
            return status, reason
    if rule is None:
        return "unclassified", "No dataset rule matches this file; review its provenance and add a rule in data_pipeline/dataset_rules.py."
    if record["format"] in SUPPORT_FORMATS:
        return "supporting", f"{record['format'].capitalize()} file; read for provenance only, never executed or used as training rows."
    if record["profile_status"] == "opaque":
        return "excluded", "Proprietary binary log; its decoded copy is profiled separately."
    if rule.get("exclusion_reason"):
        return "excluded", rule["exclusion_reason"]
    if rule["pipelines"]:
        return "used", f"Input to: {', '.join(rule['pipelines'])}."
    return "excluded", "Dataset has no pipeline that can use it."


def build_manifest(records, pipeline_usage):
    rules = {rule["dataset_id"]: rule for rule in dataset_rules.DATASET_RULES}
    datasets = defaultdict(lambda: {"files": 0, "bytes": 0, "rows": 0, "status_counts": Counter()})
    for record in records:
        rule = rules.get(record["dataset_id"])
        record["usage_status"], record["usage_reason"] = file_role(record, rule)
        summary = datasets[record["dataset_id"]]
        summary["files"] += 1
        summary["bytes"] += record["bytes"]
        if record.get("rows") and record["usage_status"] not in {"duplicate", "redundant"}:
            summary["rows"] += record["rows"]
        summary["status_counts"][record["usage_status"]] += 1
    dataset_rows = []
    for dataset_id, summary in sorted(datasets.items()):
        rule = rules.get(dataset_id, {})
        dataset_rows.append({
            "dataset_id": dataset_id, "name": rule.get("name", "Unclassified files"),
            "data_nature": rule.get("data_nature", "unreviewed"), "purpose": rule.get("purpose"),
            "vehicle_types": rule.get("vehicle_types", []), "provenance": rule.get("provenance"), "license": rule.get("license"),
            "pipelines": rule.get("pipelines", []), "exclusion_reason": rule.get("exclusion_reason"),
            "files": summary["files"], "bytes": summary["bytes"], "rows_non_duplicate": summary["rows"],
            "file_status_counts": dict(summary["status_counts"]),
        })
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(RAW_ROOT),
        "files_discovered": len(records),
        "file_status_counts": dict(Counter(record["usage_status"] for record in records)),
        "unclassified_files": [record["path"] for record in records if record["usage_status"] == "unclassified"],
        "rejected_files": [{"path": r["path"], "reason": r["usage_reason"]} for r in records if r["usage_status"] == "rejected"],
        "datasets": dataset_rows,
        "pipeline_usage": pipeline_usage,
        "files": records,
    }


def write_manifest(manifest):
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    columns = ["path", "dataset_id", "data_nature", "format", "bytes", "rows", "duplicate_rows", "profile_status",
               "usage_status", "usage_reason", "duplicate_of", "sha256", "columns", "missing_values"]
    with (REPORT_ROOT / "dataset_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in manifest["files"]:
            row = dict(record)
            row["columns"] = "|".join(record.get("columns", []))[:2000]
            row["missing_values"] = json.dumps(record.get("missing_values", {}))[:2000]
            writer.writerow(row)
    lines = [
        "# Dataset manifest", "",
        f"Generated {manifest['generated_at']} by `python -m data_pipeline.run_all`.", "",
        f"- Files discovered under `data/raw`: **{manifest['files_discovered']}**",
        "- File status: " + ", ".join(f"{k} {v}" for k, v in sorted(manifest["file_status_counts"].items())),
        f"- Unclassified files: {len(manifest['unclassified_files'])}" + (" — " + ", ".join(manifest["unclassified_files"][:20]) if manifest["unclassified_files"] else ""),
        f"- Rejected files: {len(manifest['rejected_files'])}" + "".join(f"\n  - `{r['path']}`: {r['reason']}" for r in manifest["rejected_files"]), "",
        "Status meanings: **used** = read by a pipeline; **audited_excluded** = trained/evaluated, then excluded on evidence; "
        "**excluded** = not usable (reason given); **redundant** / **duplicate** = same content as another file; "
        "**supporting** = documentation/code/metadata (never executed); **rejected** = empty or unreadable.", "",
        "## Datasets", "",
        "| Dataset | Nature | Files | Rows (non-duplicate) | Pipelines | File status | Note |", "|---|---|---:|---:|---|---|---|",
    ]
    for item in manifest["datasets"]:
        note = item["exclusion_reason"] or item["provenance"] or ""
        lines.append(
            f"| {item['name']} | {item['data_nature']} | {item['files']} | {item['rows_non_duplicate']:,} | "
            f"{', '.join(item['pipelines']) or '—'} | {', '.join(f'{k} {v}' for k, v in sorted(item['file_status_counts'].items()))} | {note} |"
        )
    lines += ["", "## Files used, audited or excluded (data files only)", "", "| File | Status | Rows | Reason |", "|---|---|---:|---|"]
    for record in manifest["files"]:
        if record["usage_status"] in {"supporting", "duplicate"}:
            continue
        if record["format"] in SUPPORT_FORMATS:
            continue
        lines.append(f"| `{record['path']}` | {record['usage_status']} | {record.get('rows', '') if record.get('rows') is not None else ''} | {record['usage_reason']} |")
    lines += ["", f"Duplicates and supporting files ({sum(1 for r in manifest['files'] if r['usage_status'] in {'supporting', 'duplicate'})}) are listed per file in `dataset_manifest.csv`.", ""]
    (REPORT_ROOT / "dataset_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def _existing_experiment(name):
    path = SERVICE_ROOT / "trained_models" / "experiments" / name / "evaluation.json"
    try:
        evaluation = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "missing", "reason": f"{path} not found; rerun its scripts/ pipeline."}
    return {"status": evaluation.get("status"), "task": evaluation.get("task"), "evaluation": str(path.relative_to(SERVICE_ROOT)),
            "metrics": evaluation.get("metrics"), "eligible_for_inference": False,
            "reason": "Evaluated experiment kept from its own pipeline; not connected to fleet inference."}


def register_experiments(experiments):
    registry = _read_json(MODEL_REGISTRY_PATH, {"schema_version": 1})
    registry.setdefault("experiments", {})
    for name, result in experiments.items():
        registry["experiments"][name] = {
            "status": result.get("status"), "task": result.get("target") or result.get("experiment"),
            "dataset": ", ".join(result.get("files", [])) or None, "eligible_for_inference": False,
            "reason": result.get("reason_not_served") or result.get("conclusion") or result.get("reason"),
            "metrics": result.get("results") or result.get("metrics") or {k: result.get(k) for k in ("macro_f1", "majority_baseline_macro_f1") if k in result} or None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    _write_json_atomic(MODEL_REGISTRY_PATH, registry)


def run(steps, skip_experiments=False):
    started = time.time()
    usage = {}
    log = []

    def note(message):
        stamp = f"[{time.time() - started:7.1f}s] {message}"
        log.append(stamp)
        print(stamp, flush=True)

    if "service_need" in steps:
        from data_pipeline.service_need import train_service_need
        note("training service-need model")
        evaluation, entry, service_usage = train_service_need(RAW_ROOT, MODEL_ROOT)
        record_training_status("maintenance_model", entry)
        usage["service_need_model"] = {**service_usage, "status": evaluation["status"], "metrics": evaluation["metrics"]}
        (REPORT_ROOT / "service_need_evaluation.json").write_text(json.dumps(evaluation, indent=2, default=str) + "\n", encoding="utf-8")
        note(f"service-need model {entry['model_version']}: {evaluation['status']}")

    if "experiments" in steps and not skip_experiments:
        from data_pipeline.experiments import run_experiments
        note("running evaluation-only experiments")
        experiments = run_experiments(RAW_ROOT)
        experiments["carobd_sensor_anomaly_v1"] = _existing_experiment("carobd_sensor_anomaly_v1")
        experiments["nasa_battery_cell_rul_1p4ah"] = _existing_experiment("nasa_battery_cell_rul_1p4ah")
        register_experiments(experiments)
        (REPORT_ROOT / "experiments_evaluation.json").write_text(json.dumps(experiments, indent=2, default=str) + "\n", encoding="utf-8")
        usage["experiments"] = {name: result.get("status") for name, result in experiments.items()}
        note("experiments done")

    if "knowledge" in steps:
        from data_pipeline.knowledge_index import build_index
        note("building knowledge index")
        usage["knowledge_index"] = build_index(RAW_ROOT, SERVICE_ROOT / "data" / "automotive_knowledge.json",
                                               SERVICE_ROOT / "data" / "knowledge" / "system_guide.json")
        note(f"knowledge index: {usage['knowledge_index']['chunk_count']} chunks")

    if "inventory" in steps:
        note("discovering data/raw")
        previous = _read_json(REPORT_ROOT / "dataset_manifest.json", {})
        records = discover(RAW_ROOT)
        manifest = build_manifest(records, {**previous.get("pipeline_usage", {}), **usage})
        write_manifest(manifest)
        note(f"manifest: {manifest['files_discovered']} files, {len(manifest['unclassified_files'])} unclassified")
    from data_pipeline.training_report import write_report
    note(f"report: {write_report().relative_to(SERVICE_ROOT)}")
    (REPORT_ROOT / "pipeline_run_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=STEPS, help="Run only these steps.")
    parser.add_argument("--skip-experiments", action="store_true")
    arguments = parser.parse_args()
    run(arguments.only or STEPS, skip_experiments=arguments.skip_experiments)


if __name__ == "__main__":
    main()
