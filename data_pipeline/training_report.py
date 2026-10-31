"""Renders data/reports/training_report_v2.md from the JSON results the pipeline writes.

Every number in the report is read from a results file; nothing is typed in by hand.
    python -m data_pipeline.training_report
"""
import json
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPORTS = SERVICE_ROOT / "data" / "reports"


def _load(name):
    try:
        return json.loads((REPORTS / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _fmt(value):
    return "—" if value is None else value


def _service_need_section(current, history):
    lines = ["## Service-need model (maintenance model candidate)", ""]
    if not current:
        return lines + ["Not trained yet: run `python -m data_pipeline.run_all --only service_need`.", ""]
    split, metrics = current["group_split"], current["metrics"]
    lines += [
        f"- Version **{current['model_version']}**, status **{current['status']}**.",
        f"- Target: {current['target_meaning']}",
        f"- Data: `{current['dataset']['file']}`, {current['dataset']['rows_used']} of {current['dataset']['rows_read']} rows used; rejected: {current['dataset']['rejected'] or 'none'}.",
        f"- Split: {split['method']}. Train groups {', '.join(split['train_groups'])} ({split['train_records']} rows); "
        f"held-out groups {', '.join(split['test_groups'])} ({split['test_records']} rows); overlap {split['group_overlap']}. {split['time_aware_split']}",
        f"- Served estimator: {metrics.get('served_estimator', current['training_config']['algorithm'])}.",
        f"- Calibration: {current['probability_validation']['method']}; status {current['probability_validation']['status']}.", "",
        "### Held-out metrics, served targets (macro)", "",
        "| Metric | Served estimator | Best ML candidate | Prevalence baseline |", "|---|---:|---:|---:|",
    ]
    for label, key in (("Balanced accuracy", "balanced_accuracy"), ("F1", "macro_f1"), ("PR-AUC", "macro_pr_auc"), ("Brier", "macro_brier")):
        ml_key = "best_ml_candidate_" + (key if key == "balanced_accuracy" else key)
        baseline_key = {"balanced_accuracy": "baseline_balanced_accuracy", "macro_f1": "baseline_macro_f1",
                        "macro_pr_auc": "baseline_macro_pr_auc", "macro_brier": "prevalence_baseline_macro_brier"}[key]
        lines.append(f"| {label} | {_fmt(metrics.get(key))} | {_fmt(metrics.get(ml_key))} | {_fmt(metrics.get(baseline_key))} |")
    lines += [f"| ROC-AUC | {_fmt(metrics.get('macro_roc_auc'))} | | |", f"| ECE | {_fmt(metrics.get('macro_ece'))} | | |", "",
              f"Served targets: {', '.join(metrics['served_targets']) or 'none'}. Best ML candidate: {metrics.get('best_ml_candidate', '—')}.", "",
              "### Per target", "",
              "| Target | Served | Test positives | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier | ECE | Best-ML F1 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for target, item in current["per_target"].items():
        ml = current["baselines"][target].get("best_ml_candidate", {})
        lines.append(f"| {target} | {'yes' if item['served'] else 'no'} | {item['positives']} | {item['precision']} | {item['recall']} | "
                     f"{item['f1']} | {_fmt(item['pr_auc'])} | {_fmt(item['roc_auc'])} | {item['brier']} | {item['ece']} | {_fmt(ml.get('f1'))} |")
    lines += ["", "### Class distribution (positive rate)", "", "| Target | Train | Held-out |", "|---|---:|---:|"]
    lines += [f"| {t} | {d['train_positive_rate']} | {d['test_positive_rate']} |" for t, d in current["class_distribution"].items()]
    lines += ["", "### Feature importance (mean PR-AUC drop when an input is shuffled, held-out groups)", ""]
    importance = current.get("feature_importance") or {}
    lines += [f"- {target}: " + ", ".join(f"{k} {v}" for k, v in drops.items()) for target, drops in importance.items()] or ["- not computed"]
    lines += ["", "### Leave-one-group-out (all 7 groups, served estimator)", "", "| Target | Folds | Mean PR-AUC | Mean ROC-AUC |", "|---|---:|---:|---:|"]
    lines += [f"| {t} | {v['folds']} | {v['mean_pr_auc']} | {v['mean_roc_auc']} |" for t, v in current["leave_one_group_out"].items()]
    lines += ["", "### Version history", "", current.get("design_history", ""), "",
              "| Version | Status | Served-target macro F1 | Baseline used | Baseline macro F1 |", "|---|---|---:|---|---:|"]
    for item in history + [current]:
        lines.append(f"| {item['model_version']} | {item['status']} | {item['metrics']['macro_f1']} | {item['metrics']['baseline'].split(' (')[0]} | {item['metrics']['baseline_macro_f1']} |")
    cost = current.get("cost_experiment") or {}
    lines += ["", f"Visit-cost regression (not served): MAE {cost.get('mae_gradient_boosting')} vs milestone-median {cost.get('mae_milestone_median_baseline')} "
              f"vs global median {cost.get('mae_global_median_baseline')} ({cost.get('target')}). {cost.get('reason', '')}", "",
              "### Limitations", ""] + [f"- {item}" for item in current["limitations"]] + [""]
    return lines


def _experiments_section(experiments):
    lines = ["## Evaluation-only experiments (not served)", ""]
    if not experiments:
        return lines + ["Not run yet: `python -m data_pipeline.run_all --only experiments`.", ""]
    engine = experiments.get("engine_condition_experiment")
    if engine:
        lines += ["### Engine-condition tables", "", f"Split: {engine['split']}. {engine['records']}", "",
                  "| Model | ROC-AUC | PR-AUC | F1 | Balanced accuracy |", "|---|---:|---:|---:|---:|"]
        lines += [f"| {name} | {m['roc_auc']} | {m['pr_auc']} | {m['f1']} | {m['balanced_accuracy']} |" for name, m in engine["results"].items()]
        lines += ["", f"Not served: {engine['reason_not_served']}", ""]
    audit = experiments.get("synthetic_signal_audit")
    if audit:
        ev = audit["ev_predictive_maintenance"]
        lines += ["### Synthetic-signal audit (EV 15-minute dataset, Hyundai readings)", "",
                  f"EV split: {ev['split']}. Leakage columns excluded: {', '.join(ev['excluded_leakage_columns'])}.", "",
                  "| EV failure label model | ROC-AUC | PR-AUC | F1 |", "|---|---:|---:|---:|"]
        lines += [f"| {name} | {m['roc_auc']} | {m['pr_auc']} | {m['f1']} |" for name, m in ev["failure_label_results"].items()]
        hyundai = audit["hyundai_component_readings"]["anomaly_results"]
        lines += ["", f"EV RUL regression R²: {ev['rul_regression_test_r2']}; maintenance-type macro F1 {ev['maintenance_type_macro_f1']} vs majority {ev['maintenance_type_majority_macro_f1']}; "
                  f"max |feature–label correlation| {ev['max_abs_feature_label_correlation']}.",
                  f"Hyundai anomaly (5-fold OOF random forest): ROC-AUC {hyundai['roc_auc']}, PR-AUC {hyundai['pr_auc']}, prevalence {hyundai['prevalence']}.", "",
                  audit["conclusion"], ""]
    cvffad = experiments.get("cvffad_fault_system_experiment")
    if cvffad:
        lines += ["### CVFFAD fault-system classifier", "", f"{cvffad['split']}; vehicle overlap {cvffad['records']['vehicle_overlap']}.",
                  f"Macro F1 {cvffad['macro_f1']} vs majority {cvffad['majority_baseline_macro_f1']}; balanced accuracy {cvffad['balanced_accuracy']} vs {cvffad['majority_baseline_balanced_accuracy']}.",
                  f"Not served: {cvffad['reason_not_served']}", ""]
    diesel = experiments.get("diesel_dtc_presence_experiment")
    if diesel:
        supplied = diesel["supplied_split_results"]
        lines += ["### Diesel OBD DTC presence", "", f"Supplied split (leaky): ROC-AUC {supplied['roc_auc']}, F1 {supplied['f1']}. {diesel['supplied_split_warning']}",
                  f"Leave-one-signature-out vehicle-level correct: {diesel['vehicle_level_correct']}.", f"Not served: {diesel['reason_not_served']}", ""]
    for name in ("carobd_sensor_anomaly_v1", "nasa_battery_cell_rul_1p4ah"):
        item = experiments.get(name)
        if item:
            lines += [f"### {name}", "", f"Status {item.get('status')}; evaluation `{item.get('evaluation')}`. {item.get('reason')}", ""]
    return lines


def _chatbot_section(chatbot):
    lines = ["## Chatbot", ""]
    if not chatbot:
        return lines + ["Not evaluated yet: `python scripts/evaluate_chatbot.py`.", ""]
    new, old = chatbot["router"]["summary"], chatbot["baseline_core"]["summary"]
    lines += [f"Evaluation set `{chatbot['evaluation_set']}`: {chatbot['hand_written_cases']} author-written cases + {chatbot['external_dtc_cases']} external DTC questions "
              f"(review status: {chatbot['review_status']}). Full results: `chatbot_evaluation_v2.md`.", "",
              "| Metric | Router | Previous core |", "|---|---:|---:|",
              f"| Pass rate | {new['pass_rate']} | {old['pass_rate']} |",
              f"| Critical cases | {new['critical_pass_rate']} | {old['critical_pass_rate']} |",
              f"| Permission leaks | {new['permission_leaks']}/{new['permission_cases']} | {old['permission_leaks']}/{old['permission_cases']} |",
              f"| Latency p50/p95 ms | {new['latency_ms']['p50']}/{new['latency_ms']['p95']} | {old['latency_ms']['p50']}/{old['latency_ms']['p95']} |", "",
              f"Retrieval on labelled cases: recall@3 {chatbot['retrieval']['recall_at_3']}, precision@3 {chatbot['retrieval']['precision_at_3']}, MRR {chatbot['retrieval']['mrr']}.", ""]
    lines += ["| Role | Router pass | Previous core |", "|---|---:|---:|"]
    lines += [f"| {role} | {item['pass_rate']} | {old['by_role'].get(role, {}).get('pass_rate')} |" for role, item in new["by_role"].items()]
    return lines + [""]


def write_report():
    manifest = _load("dataset_manifest.json")
    current = _load("service_need_evaluation.json")
    history = [item for item in (_load("service_need_v1_rejected_evaluation.json"), _load("service_need_v2_rejected_evaluation.json")) if item]
    lines = ["# AI training and evaluation report (v2 pipeline)", "",
             "Generated by `python -m data_pipeline.training_report` from the JSON results in `data/reports/`. "
             "Reproduce everything with `python -m data_pipeline.run_all` then `python scripts/evaluate_chatbot.py`.", ""]
    if manifest:
        lines += ["## Data", "", f"{manifest['files_discovered']} files discovered; status counts: "
                  + ", ".join(f"{k} {v}" for k, v in sorted(manifest["file_status_counts"].items()))
                  + f". Unclassified: {len(manifest['unclassified_files'])}. Full manifest: `dataset_manifest.md` / `.csv` / `.json`.", ""]
    lines += _service_need_section(current, history)
    lines += _experiments_section(_load("experiments_evaluation.json"))
    knowledge = (manifest or {}).get("pipeline_usage", {}).get("knowledge_index")
    if knowledge:
        lines += ["## Knowledge index", "", f"{knowledge['chunk_count']} chunks: "
                  + ", ".join(f"{k} {v}" for k, v in knowledge["chunks_by_source"].items())
                  + f". Trust levels: {knowledge['chunks_by_trust']}. Parts lookup entries: {knowledge['parts_lookup']['entries']}. "
                  + "Not indexed: " + "; ".join(f"{k} ({v})" for k, v in knowledge["not_indexed"].items()) + ".", ""]
    lines += _chatbot_section(_load("chatbot_evaluation_v2.json"))
    (REPORTS / "training_report_v2.md").write_text("\n".join(lines), encoding="utf-8")
    return REPORTS / "training_report_v2.md"


if __name__ == "__main__":
    print(write_report())
