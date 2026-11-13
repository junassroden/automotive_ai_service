"""Fleet-specific predictive models, one per assessment category.

Flow: LavaLust owns the fleet database and exports verified records. This module
re-checks them, trains, evaluates on later records, and writes to the model registry.
It never reads the fleet database.

Task: given one assessment, does a verified fault follow within the policy's
horizon (fault) or does a verified check find none (healthy)? Inputs come only
from the assessment payload, through the same ``assessment_features`` function at
training and prediction time, so outcome fields cannot reach the model.

Lifecycle: every run ends as ``evaluated_rejected`` or ``evaluated_pending``. A
pending model serves nothing until an administrator promotes it to
``evaluated_accepted``, which re-checks its evaluation and artifact. Only promotion
changes the active model, so a rejected run can never replace an accepted one.

The score is uncalibrated, so it is stored for traceability and reported as a risk
level only. ``confidence`` stays null.
"""
import hashlib
import json
import logging
import math
import threading
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from django.conf import settings
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_pipeline.status import _read_json, _write_json_atomic

from .assessment_service import AssessmentValidationError, _is_number, load_catalog, screen_component, validate_payload


logger = logging.getLogger(__name__)

FEATURE_SCHEMA_VERSION = "1"
RANDOM_STATE = 42
VALIDATION_FRACTION = 0.25
RANDOM_FOREST_MIN_ROWS = 100
WILSON_Z = 1.96
USAGE_ORDER = {"occasional": 0, "weekly": 1, "several_per_week": 2, "daily": 3}
STATUS_PENDING = "evaluated_pending"
STATUS_ACCEPTED = "evaluated_accepted"
# Simulated demonstration vehicles carry plates with this prefix (the SAMPLE- fleet in the database).
SIMULATED_PLATE_PREFIX = "SAMPLE-"
DEMONSTRATION_NOTICE = (
    "Demonstration model: trained on simulated SAMPLE- vehicles, not on real fleet outcomes. Do not base maintenance "
    "decisions on it; use the condition-based screening and a mechanic's inspection."
)
UNKNOWN_DATA_NOTICE = (
    "The training data of this model was not recorded, so it may include simulated vehicles. Treat it as a demonstration "
    "and rely on the condition-based screening."
)


def training_data_source(rows):
    """How much of a training set comes from simulated SAMPLE- vehicles: real, mixed or simulated."""
    simulated = sum(1 for row in rows if str(row.get("plate_number") or "").upper().startswith(SIMULATED_PLATE_PREFIX))
    source = "real" if simulated == 0 else "simulated" if simulated == len(rows) else "mixed"
    return {"source": source, "rows": len(rows), "simulated_rows": simulated}


def demonstration_notice(entry):
    """The warning a prediction from this registry entry must carry, or None for a model trained only on real outcomes."""
    data = entry.get("training_data") or {}
    if data.get("source") == "real":
        return None
    return DEMONSTRATION_NOTICE if data.get("source") in ("simulated", "mixed") else UNKNOWN_DATA_NOTICE
STATUS_REJECTED = "evaluated_rejected"
STATUS_SUPERSEDED = "superseded"
RISK_ELEVATED = "elevated"
RISK_NOT_ELEVATED = "not_elevated"
FEATURE_LABELS = {
    "rating_score": "Current rating", "previous_rating_score": "Previous rating", "rating_change": "Rating change",
    "critical_findings": "Critical findings", "warning_findings": "Warning findings", "advisory_findings": "Advisory findings",
    "symptom_count": "Symptoms recorded", "answered_fraction": "Questions answered", "fault_code_count": "Fault codes recorded",
    "days_since_maintenance": "Days since maintenance", "vehicle_age_years": "Vehicle age", "days_owned": "Days in the fleet",
    "usage_level": "Usage frequency", "odometer_km": "Odometer", "km_since_previous": "Distance since the previous reading",
    "days_since_previous": "Days since the previous assessment", "km_per_day": "Distance per day",
    "prior_faults": "Earlier verified faults", "prior_outcomes": "Earlier verified outcomes", "service_records": "Earlier service records",
}

_registry_lock = threading.Lock()


class InsufficientTrainingData(ValueError):
    def __init__(self, summary):
        super().__init__("; ".join(summary["blockers"]))
        self.summary = summary


class RegistryError(ValueError):
    pass


def training_policy():
    return load_catalog()["training_policy"]


def safety_critical(category_key):
    return bool(load_catalog()["_categories"][category_key]["safety_critical"])


# ---------------------------------------------------------------------------
# Record eligibility (mirrors ComponentMaintenanceRules::evaluate_training_records)
# ---------------------------------------------------------------------------

def record_label(record):
    if record.get("result") == "no_fault_found":
        return "healthy"
    if record.get("result") == "confirmed_failure":
        return "fault"
    # A completed repair alone does not establish a fault; a mechanic's confirmation does.
    if record.get("result") == "successful_repair" and record.get("mechanic_confirmed") == "confirmed":
        return "fault"
    return None


def _exclusion_reason(record, policy, ratings):
    if record.get("verification_status") != "verified":
        return "not_verified"
    if record_label(record) is None:
        return "no_usable_label"
    for prefix in policy["excluded_plate_prefixes"]:
        if str(record.get("plate_number") or "").upper().startswith(prefix.upper()):
            return "demo_vehicle"
    if not record.get("vehicle_id") or not record.get("assessed_on"):
        return "missing_vehicle_history"
    if record.get("odometer_km") is None:
        return "missing_odometer"
    if record.get("odometer_flag") == "decrease":
        return "odometer_decrease"
    if not record.get("outcome_date"):
        return "missing_outcome_date"
    if (ratings.get(record.get("rating")) or {}).get("score") is None:
        return "category_not_rated"
    days = (date.fromisoformat(record["outcome_date"]) - date.fromisoformat(record["assessed_on"])).days
    if days < 0:
        return "outcome_before_assessment"
    if days == 0:
        return "no_lead_time"
    if days > policy["horizon_days"]:
        return "outside_horizon"
    mileage = record.get("outcome_mileage_km")
    if mileage is not None and mileage < record["odometer_km"]:
        return "invalid_mileage"
    return None


def evaluate_records(records):
    """Splits one category's records into eligible and excluded, and judges sufficiency."""
    catalog = load_catalog()
    policy = catalog["training_policy"]
    ordered = sorted(records, key=lambda item: item["outcome_id"])
    excluded = []
    valid = []
    for record in ordered:
        reason = _exclusion_reason(record, policy, catalog["_ratings"])
        if reason:
            excluded.append({"outcome_id": record["outcome_id"], "reason": reason})
        else:
            valid.append(record)

    # One assessment gives one label. Conflicting verified results cannot be trusted.
    by_assessment = {}
    for record in valid:
        by_assessment.setdefault(record["assessment_id"], []).append(record)
    single = []
    for group in by_assessment.values():
        if len({record_label(record) for record in group}) > 1:
            excluded += [{"outcome_id": record["outcome_id"], "reason": "conflicting_outcomes"} for record in group]
            continue
        single.append(group[0])
        excluded += [{"outcome_id": record["outcome_id"], "reason": "duplicate_outcome"} for record in group[1:]]

    # One real event recorded against several assessments counts once, from the nearest assessment.
    by_event = {}
    for record in single:
        by_event.setdefault((record["vehicle_id"], record_label(record), record["outcome_date"]), []).append(record)
    eligible = []
    for group in by_event.values():
        group.sort(key=lambda item: (item["assessed_on"], item["assessment_id"]))
        eligible.append(group[-1])
        excluded += [{"outcome_id": record["outcome_id"], "reason": "duplicate_event"} for record in group[:-1]]
    eligible.sort(key=lambda item: item["outcome_id"])
    excluded.sort(key=lambda item: item["outcome_id"])

    return eligible, excluded, _summarize(records, eligible, excluded, policy)


def _summarize(records, eligible, excluded, policy):
    per_vehicle = {}
    for record in eligible:
        per_vehicle[record["vehicle_id"]] = per_vehicle.get(record["vehicle_id"], 0) + 1
    faults = sum(1 for record in eligible if record_label(record) == "fault")
    healthy = len(eligible) - faults
    share = round(max(per_vehicle.values()) / len(eligible), 4) if eligible else 0

    blockers = []
    if len(eligible) < policy["min_outcomes_per_category"]:
        blockers.append(f"{policy['min_outcomes_per_category'] - len(eligible)} more eligible outcomes needed (minimum {policy['min_outcomes_per_category']}).")
    if len(per_vehicle) < policy["min_vehicles"]:
        blockers.append(f"Outcomes come from {len(per_vehicle)} vehicles; at least {policy['min_vehicles']} are needed.")
    if eligible and share > policy["max_vehicle_share"]:
        blockers.append(f"One vehicle supplies more than {round(policy['max_vehicle_share'] * 100)}% of the outcomes.")
    if faults < policy["min_failures"]:
        blockers.append(f"{policy['min_failures'] - faults} more confirmed faults needed (minimum {policy['min_failures']}).")
    if healthy < policy["min_healthy"]:
        blockers.append(f"{policy['min_healthy'] - healthy} more healthy outcomes needed (minimum {policy['min_healthy']}).")

    reasons = {}
    for item in excluded:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    return {
        "status": "insufficient" if blockers else "sufficient",
        "total_outcomes": len(records),
        "verified_outcomes": sum(1 for record in records if record.get("verification_status") == "verified"),
        "eligible": len(eligible),
        "remaining": max(0, policy["min_outcomes_per_category"] - len(eligible)),
        "minimum": policy["min_outcomes_per_category"],
        "vehicles": len(per_vehicle),
        "faults": faults,
        "healthy": healthy,
        "largest_vehicle_share": share,
        "excluded_by_reason": reasons,
        "blockers": blockers,
    }


# ---------------------------------------------------------------------------
# Features: only what was known on the assessment date
# ---------------------------------------------------------------------------

def feature_names(category_key):
    category = load_catalog()["_categories"][category_key]
    return list(FEATURE_LABELS) + [f"measure_{item['key']}" for item in category["measurements"]]


def feature_label(category_key, name):
    if name.startswith("measure_"):
        key = name[len("measure_"):]
        for item in load_catalog()["_categories"][category_key]["measurements"]:
            if item["key"] == key:
                return item["label"]
    return FEATURE_LABELS.get(name, name)


def _days_between(earlier, later):
    if not earlier or not later:
        return None
    days = (date.fromisoformat(later) - date.fromisoformat(earlier)).days
    # A date after the assessment was not known then, so it is treated as missing.
    return days if days >= 0 else None


def assessment_features(category_key, payload):
    """Feature values for one category of one assessment. Unknown values stay NaN, never 0."""
    catalog = load_catalog()
    category = catalog["_categories"][category_key]
    assessed_on = payload["assessed_on"]
    component = (payload.get("components") or {}).get(category_key) or {}
    previous = payload.get("previous_assessment") or {}
    previous_component = (previous.get("components") or {}).get(category_key) or {}
    vehicle = payload.get("vehicle") or {}
    history = (payload.get("category_history") or {}).get(category_key) or {}

    score = catalog["_ratings"].get(component.get("rating") or "not_assessed", {}).get("score")
    previous_score = catalog["_ratings"].get(previous_component.get("rating"), {}).get("score")
    findings = screen_component(category_key, component, previous_component or None, assessed_on)[0]
    answers = {name: value for name, value in (component.get("answers") or {}).items() if value and value != "unknown"}
    service = _days_between(component.get("last_maintenance_date") or vehicle.get("previous_maintenance_date"), assessed_on)
    odometer = payload.get("odometer_km")
    prior_odometer = payload.get("previous_odometer_km")
    km_since = odometer - prior_odometer if odometer is not None and prior_odometer is not None and odometer >= prior_odometer else None
    days_since = _days_between(previous.get("assessed_on"), assessed_on)

    values = {
        "rating_score": score,
        "previous_rating_score": previous_score,
        "rating_change": score - previous_score if score is not None and previous_score is not None else None,
        "critical_findings": sum(1 for item in findings if item["severity"] == "critical"),
        "warning_findings": sum(1 for item in findings if item["severity"] == "warning"),
        "advisory_findings": sum(1 for item in findings if item["severity"] == "advisory"),
        "symptom_count": len(component.get("symptoms") or []),
        "answered_fraction": len(answers) / max(1, len(category["questions"])),
        "fault_code_count": len(component.get("fault_codes") or []),
        "days_since_maintenance": service,
        "vehicle_age_years": vehicle.get("age_years"),
        "days_owned": _days_between(vehicle.get("acquisition_date"), assessed_on),
        "usage_level": USAGE_ORDER.get(vehicle.get("usage_frequency")),
        "odometer_km": odometer,
        "km_since_previous": km_since,
        "days_since_previous": days_since,
        "km_per_day": km_since / days_since if km_since is not None and days_since else None,
        "prior_faults": history.get("prior_faults"),
        "prior_outcomes": history.get("prior_outcomes"),
        "service_records": len(payload.get("maintenance_history") or []),
    }
    measurements = component.get("measurements") or {}
    for item in category["measurements"]:
        reading = measurements.get(item["key"])
        values[f"measure_{item['key']}"] = reading if _is_number(reading) else None
    return values


def _vector(category_key, payload):
    values = assessment_features(category_key, payload)
    return [float(values[name]) if values[name] is not None else math.nan for name in feature_names(category_key)]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def chronological_split(rows, fraction):
    """Later assessments are held out. Training rows must also have ended before the
    holdout starts, so no outcome known only later can reach training."""
    ordered = sorted(rows, key=lambda row: (row["assessed_on"], row["outcome_id"]))
    position = max(1, min(len(ordered) - 1, math.floor(len(ordered) * (1 - fraction))))
    start = ordered[position]["assessed_on"]
    before = [row for row in ordered if row["assessed_on"] < start]
    held_out = [row for row in ordered if row["assessed_on"] >= start]
    kept = [row for row in before if row["outcome_date"] < start]
    return kept, held_out, start, len(before) - len(kept)


def _matrix(rows, category_key):
    return np.array([_vector(category_key, row["assessment"]) for row in rows], dtype=float), np.array([1 if record_label(row) == "fault" else 0 for row in rows])


def _candidates(row_count):
    models = {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
        ),
    }
    # A forest needs more data than this fleet is likely to have soon, so it only competes above this size.
    if row_count >= RANDOM_FOREST_MIN_ROWS:
        models["random_forest"] = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
        )
    return models


def _f2(y_true, predicted):
    tp = int(((predicted == 1) & (y_true == 1)).sum())
    fp = int(((predicted == 1) & (y_true == 0)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())
    return 5 * tp / (5 * tp + 4 * fn + fp) if tp else 0.0


def _best_threshold(y_true, scores):
    """Favours recall, because a missed fault costs more than a false alarm. Ties take the higher threshold."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5, 0.0
    best = (0.5, -1.0)
    for threshold in sorted(set(np.round(scores, 6))):
        value = _f2(y_true, (scores >= threshold).astype(int))
        if value >= best[1]:
            best = (float(threshold), value)
    return best


def wilson_lower_bound(successes, total):
    """95% lower bound on a proportion. A small test set cannot prove a high recall."""
    if total == 0:
        return 0.0
    phat = successes / total
    z2 = WILSON_Z ** 2
    centre = phat + z2 / (2 * total)
    margin = WILSON_Z * math.sqrt(phat * (1 - phat) / total + z2 / (4 * total ** 2))
    return (centre - margin) / (1 + z2 / total)


def _metrics(y_true, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    both = 0 < y_true.sum() < len(y_true)
    return {
        "records": int(len(y_true)),
        "faults": int(y_true.sum()),
        "healthy": int(len(y_true) - y_true.sum()),
        "precision": round(float(precision_score(y_true, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predicted, zero_division=0)), 4),
        "f2": round(_f2(y_true, predicted), 4),
        "accuracy": round(float((predicted == y_true).mean()), 4) if len(y_true) else 0.0,
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4) if both else None,
        "recall_lower_bound": round(wilson_lower_bound(tp, tp + fn), 4),
        "false_negative_rate": round(fn / (tp + fn), 4) if tp + fn else None,
        "confusion_matrix": {"true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp},
    }


def acceptance_criteria(category_key, metrics, unseen):
    """Thresholds come from the policy, fixed before training. Safety categories demand more recall."""
    rules = training_policy()["evaluation"]
    safety = safety_critical(category_key)
    min_recall = rules["min_recall_safety"] if safety else rules["min_recall"]
    min_bound = rules["min_recall_lower_bound_safety"] if safety else rules["min_recall_lower_bound"]
    prevalence = metrics["faults"] / metrics["records"] if metrics["records"] else 0
    needed_auc = round(min(1.0, prevalence * rules["min_pr_auc_lift"]), 4)

    def row(name, required, actual, passed):
        return {"name": name, "required": required, "actual": actual, "passed": bool(passed)}

    criteria = [
        row("Faults in the test period", f"at least {rules['min_test_failures']}", metrics["faults"], metrics["faults"] >= rules["min_test_failures"]),
        row("Healthy outcomes in the test period", f"at least {rules['min_test_healthy']}", metrics["healthy"], metrics["healthy"] >= rules["min_test_healthy"]),
        row("Recall", f"at least {min_recall}", metrics["recall"], metrics["recall"] >= min_recall),
        row("Recall, 95% lower bound", f"at least {min_bound}", metrics["recall_lower_bound"], metrics["recall_lower_bound"] >= min_bound),
        row("Precision", f"at least {rules['min_precision']}", metrics["precision"], metrics["precision"] >= rules["min_precision"]),
        row("F1-score", f"at least {rules['min_f1']}", metrics["f1"], metrics["f1"] >= rules["min_f1"]),
        row("PR-AUC against the fault rate", f"at least {needed_auc}", metrics["pr_auc"], metrics["pr_auc"] is not None and metrics["pr_auc"] >= needed_auc),
    ]
    # Vehicles new to the model are only judged when the test period has enough of their faults.
    if unseen and unseen["faults"] >= rules["min_unseen_vehicle_failures"]:
        criteria.append(row("Recall on vehicles not seen in training", f"at least {min_recall}", unseen["recall"], unseen["recall"] >= min_recall))
    return criteria


def _dataset_sha(rows):
    canonical = json.dumps(sorted(rows, key=lambda row: row["outcome_id"]), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_category(category_key, rows, requested_by=None):
    """One training attempt. Raises InsufficientTrainingData before any fitting if the data does not qualify."""
    catalog = load_catalog()
    if category_key not in catalog["_categories"]:
        raise AssessmentValidationError([f"Unknown category '{category_key}'."])
    if not isinstance(rows, list):
        raise AssessmentValidationError(["rows must be a list."])
    for row in rows:
        if not isinstance(row, dict) or row.get("category_key") != category_key or not isinstance(row.get("assessment"), dict):
            raise AssessmentValidationError(["Every row must belong to this category and carry its assessment."])
        if row["assessment"].get("assessment_id") != row.get("assessment_id") or row["assessment"].get("assessed_on") != row.get("assessed_on"):
            raise AssessmentValidationError(["A row's assessment does not match its record."])
        validate_payload(row["assessment"])

    # LavaLust has already filtered these. They are checked again here, independently.
    eligible, excluded, summary = evaluate_records(rows)
    if summary["status"] != "sufficient":
        raise InsufficientTrainingData(summary)

    rules = catalog["training_policy"]["evaluation"]
    development, test, test_start, purged = chronological_split(eligible, rules["test_fraction"])
    report = {
        "category": category_key,
        "category_name": catalog["_categories"][category_key]["name"],
        "safety_critical": safety_critical(category_key),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "catalog_version": catalog["version"],
        "policy_version": catalog["training_policy"]["version"],
        "horizon_days": catalog["training_policy"]["horizon_days"],
        "dataset": {
            "rows": len(eligible),
            "vehicles": summary["vehicles"],
            "faults": summary["faults"],
            "healthy": summary["healthy"],
            "excluded_by_reason": summary["excluded_by_reason"],
            "sha256": _dataset_sha(eligible),
            # Simulated SAMPLE- vehicles make a model a demonstration only (see demonstration_notice).
            "training_data": training_data_source(eligible),
        },
        "methodology": (
            "Chronological holdout: the latest assessments are the test set and are scored once. Training uses only "
            "earlier assessments whose outcomes were also known before the test period began. The threshold and "
            "algorithm are chosen on an earlier validation period, never on the test set."
        ),
        "periods": {
            "training": [development[0]["assessed_on"], development[-1]["assessed_on"]] if development else None,
            "test": [test[0]["assessed_on"], test[-1]["assessed_on"]] if test else None,
            "training_rows": len(development),
            "test_rows": len(test),
            "rows_purged_to_prevent_leakage": purged,
        },
    }

    fit_rows, validation, validation_start, _ = chronological_split(development, VALIDATION_FRACTION) if len(development) >= 4 else ([], [], None, 0)
    report["periods"]["validation_start"] = validation_start
    X_fit, y_fit = _matrix(fit_rows, category_key) if fit_rows else (np.empty((0, 0)), np.array([], dtype=int))
    X_validation, y_validation = _matrix(validation, category_key) if validation else (np.empty((0, 0)), np.array([], dtype=int))

    if len(set(y_fit.tolist())) < 2 or len(set(_matrix(development, category_key)[1].tolist())) < 2:
        return _finish(category_key, report, None, None, None, requested_by, [
            "The training period does not contain both faults and healthy outcomes, so no model could be fitted.",
        ], [])

    best = None
    tried = {}
    for name, model in _candidates(len(development)).items():
        model.fit(X_fit, y_fit)
        scores = model.predict_proba(X_validation)[:, 1] if len(validation) else np.array([])
        threshold, value = _best_threshold(y_validation, scores) if len(validation) else (0.5, 0.0)
        tried[name] = {"validation_f2": round(value, 4), "threshold": round(threshold, 6)}
        if best is None or value > best[2]:
            best = (name, threshold, value)
    selected, threshold, _ = best

    # The final model learns from the whole development period. The test set stays unseen.
    X_development, y_development = _matrix(development, category_key)
    final = _candidates(len(development))[selected]
    final.fit(X_development, y_development)
    X_test, y_test = _matrix(test, category_key)
    scores = final.predict_proba(X_test)[:, 1]
    metrics = _metrics(y_test, scores, threshold)

    # What the same test set scores if every vehicle is simply flagged.
    baseline = _metrics(y_test, np.ones(len(y_test)), 0.5)
    train_vehicles = {row["vehicle_id"] for row in development}
    unseen_mask = np.array([row["vehicle_id"] not in train_vehicles for row in test])
    unseen = _metrics(y_test[unseen_mask], scores[unseen_mask], threshold) if unseen_mask.any() else None
    missed = [
        {"outcome_id": row["outcome_id"], "vehicle_id": row["vehicle_id"], "assessed_on": row["assessed_on"], "outcome_date": row["outcome_date"], "rating": row.get("rating")}
        for row, truth, score in zip(test, y_test, scores) if truth == 1 and score < threshold
    ]
    criteria = acceptance_criteria(category_key, metrics, unseen)
    report.update({
        "algorithm": selected,
        "candidates": tried,
        "threshold": round(threshold, 6),
        "metrics": metrics,
        "baseline_flag_everything": baseline,
        "unseen_vehicle_metrics": unseen,
        "false_negative_analysis": {
            "count": metrics["confusion_matrix"]["false_negative"],
            "rate": metrics["false_negative_rate"],
            "missed_faults": missed,
            "note": "Each of these is a verified fault this model would not have flagged.",
        },
    })
    reasons = [f"{item['name']}: required {item['required']}, got {item['actual']}." for item in criteria if not item["passed"]]
    return _finish(category_key, report, final, threshold, feature_names(category_key), requested_by, reasons, criteria)


def _finish(category_key, report, model, threshold, names, requested_by, reasons, criteria):
    passed = model is not None and bool(criteria) and all(item["passed"] for item in criteria)
    catalog_name = f"category_{category_key}"
    with _registry_lock:
        registry = _read_json(_registry_path(), {"schema_version": 1})
        section = registry.setdefault("category_models", {}).setdefault(category_key, {"active_version": None, "versions": []})
        version = f"{catalog_name}_v{len(section['versions']) + 1}_{report['dataset']['sha256'][:8]}"
        trained_at = datetime.now(timezone.utc).isoformat()
        report.update({
            "model_name": catalog_name,
            "model_version": version,
            "trained_at": trained_at,
            "acceptance_criteria": criteria,
            "acceptance_criteria_passed": passed,
            "status": STATUS_PENDING if passed else STATUS_REJECTED,
            "decision_reasons": (["Passed every criterion. An administrator must review it before it can serve predictions."] if passed else reasons),
        })
        folder = _model_root() / category_key / version
        folder.mkdir(parents=True, exist_ok=True)
        artifact_sha = None
        if model is not None:
            joblib.dump({"model": model, "features": names, "threshold": threshold, "category": category_key, "version": version,
                         "feature_schema_version": FEATURE_SCHEMA_VERSION, "catalog_version": report["catalog_version"]},
                        folder / "pipeline.joblib", compress=3)
            artifact_sha = _sha_file(folder / "pipeline.joblib")
        _write_json_atomic(folder / "evaluation.json", report)
        entry = {
            "model_name": catalog_name,
            "model_version": version,
            "category": category_key,
            "task": f"Verified fault within {report['horizon_days']} days of an assessment",
            "status": report["status"],
            "eligible_for_inference": False,
            "trained_at": trained_at,
            "requested_by": requested_by,
            "algorithm": report.get("algorithm"),
            "artifact_path": (Path(category_key) / version / "pipeline.joblib").as_posix() if artifact_sha else None,
            "artifact_sha256": artifact_sha,
            "evaluation_report_path": (Path(category_key) / version / "evaluation.json").as_posix(),
            "evaluation_report_sha256": _sha_file(folder / "evaluation.json"),
            "evaluation": {"acceptance_criteria_passed": passed},
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "catalog_version": report["catalog_version"],
            "training_data": report["dataset"].get("training_data"),
            "report": report,
        }
        section["versions"].append(entry)
        _write_json_atomic(_registry_path(), registry)
    return entry


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _registry_path():
    return Path(settings.AI_MAINTENANCE_MODEL_REGISTRY_PATH)


def _model_root():
    return Path(settings.CATEGORY_MODEL_ROOT)


def category_registry():
    registry = _read_json(_registry_path(), {})
    return registry.get("category_models") or {} if isinstance(registry, dict) else {}


def models_overview():
    """The registered per-system fault models for explaining them in chat: one entry per category with its status
    (in_use, rejected, not_trained), training data and held-out metrics. Read from the registry only."""
    overview = []
    for category, section in sorted(category_registry().items()):
        versions = section.get("versions") or []
        active = section.get("active_version")
        entry = _find_version(section, active) if active else (versions[-1] if versions else None)
        if entry is None:
            overview.append({"category": category, "status": "not_trained"})
            continue
        report = entry.get("report") or {}
        metrics = report.get("metrics") or {}
        status = "in_use" if active and entry.get("status") == STATUS_ACCEPTED else (
            "rejected" if entry.get("status") == STATUS_REJECTED else "pending" if entry.get("status") == STATUS_PENDING else entry.get("status"))
        overview.append({
            "category": category, "name": report.get("category_name") or category.replace("_", " ").title(),
            "status": status, "horizon_days": report.get("horizon_days"),
            "training_source": (entry.get("training_data") or {}).get("source"), "rows": (report.get("dataset") or {}).get("rows"),
            "vehicles": (report.get("dataset") or {}).get("vehicles"), "recall": metrics.get("recall"), "precision": metrics.get("precision"),
            "reasons": [reason for reason in report.get("decision_reasons") or [] if status == "rejected"],
        })
    return overview


def _find_version(section, version):
    for entry in section.get("versions", []):
        if entry["model_version"] == version:
            return entry
    return None


def _model_file(relative):
    """A registry path (stored relative to the model root, '/'-separated; older entries
    written on Windows use '\\') resolved against the configured model root."""
    return _model_root().joinpath(*relative.replace("\\", "/").split("/"))


def _integrity_check(entry):
    """(status, reason) for why this entry's files cannot be trusted or used now, or None."""
    if not entry.get("artifact_path") or not entry.get("artifact_sha256"):
        return "no_artifact", "It has no model artifact."
    if not entry.get("evaluation_report_path") or not entry.get("evaluation_report_sha256"):
        return "invalid_registry_entry", "Its registry entry has no evaluation report reference."
    artifact = _model_file(entry["artifact_path"])
    report = _model_file(entry["evaluation_report_path"])
    if not artifact.is_file():
        return "artifact_missing", "Its model file is missing."
    if not report.is_file():
        return "report_missing", "Its evaluation report is missing."
    try:
        artifact_sha, report_sha = _sha_file(artifact), _sha_file(report)
    except OSError:
        return "artifact_unreadable", "Its files cannot be read."
    if artifact_sha != entry["artifact_sha256"]:
        return "artifact_changed", "Its model file has changed since evaluation."
    if report_sha != entry["evaluation_report_sha256"]:
        return "report_changed", "Its evaluation report has changed since evaluation."
    if entry.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        return "incompatible_features", "It was built for a different feature schema."
    if entry.get("catalog_version") != load_catalog()["version"]:
        return "incompatible_questionnaire", "It was built for a different questionnaire version."
    return None


def _integrity_problem(entry):
    """Why this entry's files cannot be trusted or used now, or None."""
    problem = _integrity_check(entry)
    return problem[1] if problem else None


def _load_check(category_key, entry):
    """(status, reason) when the verified artifact cannot be loaded or does not fit the
    inputs this service builds for the category, or None when it is ready to predict."""
    try:
        artifact = _load_artifact(entry)
    except Exception as error:
        logger.warning("Category model %s %s failed to load: %s", category_key, entry.get("model_version"), type(error).__name__)
        return "load_failed", "Its model file could not be loaded."
    if not isinstance(artifact, dict) or not hasattr(artifact.get("model"), "predict_proba"):
        return "invalid_artifact", "Its model file does not contain a usable model."
    try:
        float(artifact["threshold"])
    except (KeyError, TypeError, ValueError):
        return "invalid_artifact", "Its model file has no decision threshold."
    if list(artifact.get("features") or []) != list(feature_names(category_key)):
        return "incompatible_features", "Its model expects different inputs than this service provides."
    if not isinstance(entry.get("report"), dict) or "horizon_days" not in entry["report"]:
        return "invalid_registry_entry", "Its registry entry has no evaluation summary."
    return None


def promote_model(category_key, version, promoted_by=None):
    """Review step. Only a pending model that passed evaluation can become active."""
    with _registry_lock:
        registry = _read_json(_registry_path(), {"schema_version": 1})
        section = (registry.get("category_models") or {}).get(category_key) or {}
        entry = _find_version(section, version)
        if entry is None:
            raise RegistryError("That model version does not exist.")
        if entry["status"] != STATUS_PENDING or entry["evaluation"].get("acceptance_criteria_passed") is not True:
            raise RegistryError("Only a model that passed evaluation and awaits review can be accepted.")
        problem = _integrity_problem(entry)
        if problem:
            raise RegistryError(problem)
        previous = _find_version(section, section.get("active_version"))
        if previous is not None:
            previous.update(status=STATUS_SUPERSEDED, eligible_for_inference=False)
        entry.update(status=STATUS_ACCEPTED, eligible_for_inference=True, accepted_by=promoted_by, accepted_at=datetime.now(timezone.utc).isoformat())
        section["active_version"] = version
        _write_json_atomic(_registry_path(), registry)
    return entry


def revoke_model(category_key, reason, revoked_by=None):
    """Withdraws the active model. The category returns to prediction unavailable."""
    with _registry_lock:
        registry = _read_json(_registry_path(), {"schema_version": 1})
        section = (registry.get("category_models") or {}).get(category_key) or {}
        entry = _find_version(section, section.get("active_version"))
        if entry is None:
            raise RegistryError("This category has no active model.")
        entry.update(status=STATUS_REJECTED, eligible_for_inference=False, revoked_by=revoked_by, revoked_reason=reason, revoked_at=datetime.now(timezone.utc).isoformat())
        section["active_version"] = None
        _write_json_atomic(_registry_path(), registry)
    return entry


def category_availability():
    """Per-category serving state, for all eight categories. One category never affects another."""
    registry = category_registry()
    result = {}
    for category in load_catalog()["categories"]:
        key = category["key"]
        section = registry.get(key) or {}
        versions = section.get("versions") or []
        active = section.get("active_version")
        entry = _find_version(section, active)
        if entry is None:
            if active:
                status, reason = "invalid_registry_entry", "The active model version is not in the registry."
            elif not versions:
                status, reason = "not_trained", "No model has been trained for this category."
            elif any(item.get("status") == STATUS_REJECTED for item in versions):
                status, reason = "rejected", "No accepted model exists for this category."
            else:
                status, reason = "no_accepted_model", "No accepted model exists for this category."
            result[key] = {"available": False, "status": status, "reason": reason, "active_version": None}
            continue
        if entry.get("status") != STATUS_ACCEPTED or entry.get("eligible_for_inference") is not True:
            problem = ("not_accepted", "The active model is not accepted.")
        else:
            problem = _integrity_check(entry) or _load_check(key, entry)
        if problem:
            result[key] = {"available": False, "status": problem[0], "reason": problem[1], "active_version": None}
        else:
            notice = demonstration_notice(entry)
            result[key] = {"available": True, "status": "ready", "reason": None, "active_version": entry["model_version"],
                           "demonstration": notice is not None, "notice": notice,
                           "training_data_source": (entry.get("training_data") or {}).get("source", "not_recorded")}
    return result


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

_loaded = {}


def _load_artifact(entry):
    key = entry["artifact_sha256"]
    if key not in _loaded:
        _loaded[key] = joblib.load(_model_file(entry["artifact_path"]))
    return _loaded[key]


def _unavailable(reason):
    return {"available": False, "reason": reason, "risk_level": None}


def predict_category(category_key, payload, availability, section):
    """A prediction from this category's accepted model, or an honest unavailable. Never raises."""
    state = availability.get(category_key) or {"available": False, "reason": "No accepted model exists for this category."}
    if not state["available"]:
        return _unavailable(state["reason"])
    component = (payload.get("components") or {}).get(category_key) or {}
    if load_catalog()["_ratings"].get(component.get("rating") or "not_assessed", {}).get("score") is None or payload.get("odometer_km") is None:
        return _unavailable("A prediction needs a rating for this category and an odometer reading.")
    try:
        entry = _find_version(section, state["active_version"])
        artifact = _load_artifact(entry)
        vector = np.array([_vector(category_key, payload)], dtype=float)
        model = artifact["model"]
        score = float(model.predict_proba(vector)[0, 1])
        threshold = float(artifact["threshold"])
        factors = _top_factors(category_key, model, vector, score, artifact["features"])
    except Exception:
        return _unavailable("The accepted model could not be run for this assessment.")

    elevated = score >= threshold
    horizon = entry["report"]["horizon_days"]
    label = entry["report"]["category_name"]
    notice = demonstration_notice(entry)
    return {
        "available": True,
        "reason": None,
        "risk_level": RISK_ELEVATED if elevated else RISK_NOT_ELEVATED,
        "model_score": round(score, 5),
        "threshold": round(threshold, 6),
        "horizon_days": horizon,
        "model": {"name": entry["model_name"], "version": entry["model_version"], "demonstration": notice is not None},
        "demonstration": notice is not None,
        "notice": notice,
        "training_data": entry.get("training_data"),
        "factors": factors,
        "explanation": (
            ("Demonstration only: the" if notice else "The")
            + f" {label} model {'flags' if elevated else 'does not flag'} a verified fault within {horizon} days. "
            + (f"Most influenced by: {', '.join(item['label'].lower() for item in factors)}." if factors else "")
        ),
        "evaluation": {key: entry["report"]["metrics"][key] for key in ("recall", "precision", "records", "faults")},
    }


def _top_factors(category_key, model, vector, score, names, limit=3):
    """How the score moves when each known input is removed. Local to this assessment."""
    effects = []
    for position, name in enumerate(names):
        if math.isnan(vector[0, position]):
            continue
        changed = vector.copy()
        changed[0, position] = math.nan
        effect = score - float(model.predict_proba(changed)[0, 1])
        if abs(effect) > 1e-6:
            effects.append((abs(effect), name, effect))
    effects.sort(reverse=True)
    return [{"feature": name, "label": feature_label(category_key, name), "direction": "raises" if effect > 0 else "lowers"} for _, name, effect in effects[:limit]]


def predictions_for_assessment(payload):
    """Predictions keyed by category, each decided independently."""
    availability = category_availability()
    registry = category_registry()
    return {
        category["key"]: predict_category(category["key"], payload, availability, registry.get(category["key"]) or {})
        for category in load_catalog()["categories"]
    }
