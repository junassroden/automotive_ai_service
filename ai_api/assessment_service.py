"""Stage 1 analysis of a submitted vehicle assessment: POST /api/assessment/analyze/.

What this does
  - Validates the payload. The service never writes to the fleet database.
  - Runs the explainable rules in assessment_catalog.json for every category.
    Back-End/app/libraries/ComponentMaintenanceRules.php runs the same rules at
    submission, so safety warnings do not depend on this service being reachable.
  - Looks up fault codes and symptom guidance in the knowledge index.

  - Adds a prediction only for categories with an accepted model (category_models.py).

What this deliberately does not do
  - Report risk percentages, remaining useful life or failure dates. Without an
    accepted model a category's ``predictive_status`` is "Insufficient Data", and
    ``ai_model`` and ``prediction_horizon`` are null. Rules are never presented as a
    model. ``confidence`` is always null, because model scores are uncalibrated.
"""
import json
import math
import re
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path

from .prediction_service import maintenance_model_status
from .rag import KnowledgeIndexUnavailable, load_index, retrieve


CATALOG_PATH = Path(__file__).resolve().parent / "assessment_catalog.json"
SEVERITY_ORDER = {"critical": 0, "warning": 1, "advisory": 2}
SCREENING_BY_SEVERITY = {"critical": "critical", "warning": "attention", "advisory": "monitor"}
RECENT_SERVICE_DAYS = 30
MAX_ODOMETER_KM = 2_000_000
MAX_NOTE_LENGTH = 4000
MAX_FAULT_CODES = 40
MAX_HISTORY_ITEMS = 100
DTC_PATTERN = re.compile(r"^[PBCU][0-9][0-9A-F]{3}$")
INSUFFICIENT_DATA = "Insufficient Data"
PREDICTION_AVAILABLE = "Prediction available"
# Below this, retrieved flowcharts were only loosely related to the symptom in a review of all catalog symptoms.
MIN_GUIDANCE_SUPPORT = 0.4
PRIOR_STATUS_LABELS = {"critical": "having a critical finding", "attention": "needing attention"}

# Reference categories in the flowchart source, mapped to assessment categories.
FLOWCHART_CATEGORIES = {
    "ABS System": "brakes", "Air Conditioning System": "air_conditioning", "Cooling System": "cooling",
    "Drivetrain": "transmission", "Transmission": "transmission", "Electrical System": "battery_electrical",
    "Engine Components": "engine", "Engine Compartment": "engine", "Emissions System": "engine",
    "Fuel System": "engine", "Steering": "suspension_steering", "Wheels & Tires": "tires_wheels",
}
# First match wins. Only a display hint for where a code belongs, not a diagnosis.
DTC_CATEGORY_KEYWORDS = [
    ("air_conditioning", r"\b(a/?c|air condition\w*|refrigerant|evaporator|hvac|blower)\b"),
    ("cooling", r"\b(coolant|radiator|thermostat|cooling fan|overheat\w*)\b"),
    ("brakes", r"\b(brake\w*|abs|anti-?lock)\b"),
    ("tires_wheels", r"\b(tire|tyre|tpms)\b"),
    ("suspension_steering", r"\b(steering|suspension|ride height|damper|tie rod)\b"),
    ("transmission", r"\b(transmission|gear|torque converter|clutch|shift|transfer case|tcm)\b"),
    ("battery_electrical", r"\b(battery|charging|alternator|generator|starter|lamp|light)\b"),
]


class AssessmentValidationError(ValueError):
    def __init__(self, errors):
        super().__init__("; ".join(errors))
        self.errors = errors


@lru_cache(maxsize=1)
def load_catalog():
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    catalog["_categories"] = {category["key"]: category for category in catalog["categories"]}
    catalog["_ratings"] = {rating["key"]: rating for rating in catalog["ratings"]}
    catalog["_rules"] = {}
    for rule in catalog["rules"]:
        catalog["_rules"].setdefault(rule["category"], []).append(rule)
    return catalog


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _parse_date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError
    return date.fromisoformat(value)


def _validate_date(value, label, errors, required=False):
    if value in (None, ""):
        if required:
            errors.append(f"{label} is required.")
        return
    try:
        _parse_date(value)
    except ValueError:
        errors.append(f"{label} must be a valid YYYY-MM-DD date.")


def _validate_odometer(value, label, errors):
    # 0 is a real reading; null means it was not recorded.
    if value is not None and not (_is_int(value) and 0 <= value <= MAX_ODOMETER_KM):
        errors.append(f"{label} must be a whole number from 0 to {MAX_ODOMETER_KM}, or null.")


def _validate_component(key, component, category, errors):
    prefix = f"components.{key}"
    if not isinstance(component, dict):
        errors.append(f"{prefix} must be an object.")
        return
    catalog = load_catalog()
    if not isinstance(component.get("rating", "not_assessed"), str) or component.get("rating", "not_assessed") not in catalog["_ratings"]:
        errors.append(f"{prefix}.rating is not a recognised rating.")

    answers = component.get("answers") or {}
    if not isinstance(answers, dict):
        errors.append(f"{prefix}.answers must be an object.")
        answers = {}
    questions = {question["key"] for question in category["questions"]}
    for question, answer in answers.items():
        if question not in questions:
            errors.append(f"{prefix}.answers has an unknown question '{question}'.")
        elif answer is not None and answer not in catalog["answers"]:
            errors.append(f"{prefix}.answers.{question} has an invalid answer.")

    symptoms = component.get("symptoms") or []
    if not isinstance(symptoms, list):
        errors.append(f"{prefix}.symptoms must be a list.")
        symptoms = []
    known_symptoms = {symptom["key"] for symptom in category["symptoms"]}
    for symptom in symptoms:
        if not isinstance(symptom, str) or symptom not in known_symptoms:
            errors.append(f"{prefix}.symptoms has an unknown symptom '{symptom}'.")

    measurements = component.get("measurements") or {}
    if not isinstance(measurements, dict):
        errors.append(f"{prefix}.measurements must be an object.")
        measurements = {}
    ranges = {measurement["key"]: measurement for measurement in category["measurements"]}
    for name, value in measurements.items():
        spec = ranges.get(name)
        if spec is None:
            errors.append(f"{prefix}.measurements has an unknown measurement '{name}'.")
        elif value in (None, "unknown", "not_applicable"):
            continue
        elif not _is_number(value) or not spec["min"] <= value <= spec["max"]:
            errors.append(f"{prefix}.measurements.{name} must be between {spec['min']} and {spec['max']} {spec['unit']}.")

    codes = component.get("fault_codes") or []
    if not isinstance(codes, list) or len(codes) > MAX_FAULT_CODES:
        errors.append(f"{prefix}.fault_codes must be a list of at most {MAX_FAULT_CODES} codes.")
    else:
        if codes and not category["supports_fault_codes"]:
            errors.append(f"{prefix} does not accept fault codes.")
        for code in codes:
            if not isinstance(code, str) or not DTC_PATTERN.fullmatch(code.strip().upper()):
                errors.append(f"{prefix}.fault_codes has an invalid code '{code}'.")

    note = component.get("notes")
    if note is not None and (not isinstance(note, str) or len(note) > MAX_NOTE_LENGTH):
        errors.append(f"{prefix}.notes must be text of at most {MAX_NOTE_LENGTH} characters.")
    _validate_date(component.get("last_maintenance_date"), f"{prefix}.last_maintenance_date", errors)


def validate_payload(payload):
    errors = []
    catalog = load_catalog()
    if not isinstance(payload, dict):
        raise AssessmentValidationError(["Request body must be a JSON object."])

    if not (_is_int(payload.get("assessment_id")) and payload["assessment_id"] > 0):
        errors.append("assessment_id must be a positive whole number.")
    vehicle = payload.get("vehicle")
    if not isinstance(vehicle, dict) or not (_is_int(vehicle.get("id")) and vehicle["id"] > 0):
        errors.append("vehicle.id must be a positive whole number.")
    elif vehicle.get("age_years") is not None and not (_is_int(vehicle["age_years"]) and vehicle["age_years"] >= 0):
        errors.append("vehicle.age_years must be a non-negative whole number or null.")

    _validate_date(payload.get("assessed_on"), "assessed_on", errors, required=True)
    _validate_odometer(payload.get("odometer_km"), "odometer_km", errors)
    _validate_odometer(payload.get("previous_odometer_km"), "previous_odometer_km", errors)

    components = payload.get("components")
    if not isinstance(components, dict):
        errors.append("components must be an object keyed by category.")
    else:
        for key, component in components.items():
            category = catalog["_categories"].get(key)
            if category is None:
                errors.append(f"components has an unknown category '{key}'.")
            else:
                _validate_component(key, component, category, errors)

    previous = payload.get("previous_assessment")
    if previous is not None:
        if not isinstance(previous, dict):
            errors.append("previous_assessment must be an object or null.")
        else:
            _validate_date(previous.get("assessed_on"), "previous_assessment.assessed_on", errors)
            if not isinstance(previous.get("components") or {}, dict):
                errors.append("previous_assessment.components must be an object.")

    history = payload.get("maintenance_history")
    if history is not None and (not isinstance(history, list) or len(history) > MAX_HISTORY_ITEMS):
        errors.append(f"maintenance_history must be a list of at most {MAX_HISTORY_ITEMS} items.")

    if errors:
        raise AssessmentValidationError(errors)
    return payload


# ---------------------------------------------------------------------------
# Rule-based screening (mirrors ComponentMaintenanceRules::screen_component)
# ---------------------------------------------------------------------------

def _format_value(value):
    return str(int(value)) if float(value).is_integer() else str(value)


def _rule_triggered(rule, component):
    when = rule["when"]
    if when["kind"] == "symptom":
        return when["key"] in (component.get("symptoms") or []), None
    if when["kind"] == "answer":
        return (component.get("answers") or {}).get(when["question"]) == when["equals"], None
    if when["kind"] == "measurement":
        value = (component.get("measurements") or {}).get(when["key"])
        if not _is_number(value):
            return False, None
        checks = [
            "lt" not in when or value < when["lt"], "lte" not in when or value <= when["lte"],
            "gt" not in when or value > when["gt"], "gte" not in when or value >= when["gte"],
        ]
        return all(checks), value
    return False, None


def _finding(rule_id, source, severity, safety, category, observation, reason, action):
    return {
        "rule_id": rule_id, "source": source, "severity": severity, "safety": bool(safety),
        "component_key": category["key"], "component_name": category["name"],
        "observation": observation, "reason": reason, "recommended_action": action,
    }


def screen_component(key, component, previous=None, assessed_on=None):
    """Every finding names the observation that triggered it. Returns (findings, status, sufficiency)."""
    catalog = load_catalog()
    category = catalog["_categories"][key]
    component = component or {}
    rating = component.get("rating") or "not_assessed"
    findings = []

    rating_rule = catalog["rating_rules"].get(rating)
    if rating_rule:
        severity = rating_rule["severity"]
        findings.append(_finding(
            f"rating.{rating}", "rating", severity, severity == "critical" and category["safety_critical"], category,
            f"Assessor rated {category['name']} as {catalog['_ratings'][rating]['label']}.",
            rating_rule["reason"], rating_rule["action"],
        ))

    for rule in catalog["_rules"].get(key, []):
        triggered, value = _rule_triggered(rule, component)
        if triggered:
            observation = rule["observation"].replace("{value}", _format_value(value)) if value is not None else rule["observation"]
            findings.append(_finding(rule["id"], rule["when"]["kind"], rule["severity"], rule["safety"], category, observation, rule["reason"], rule["action"]))

    previous = previous or {}
    current_score = catalog["_ratings"].get(rating, {}).get("score")
    previous_rating = previous.get("rating")
    previous_score = catalog["_ratings"].get(previous_rating, {}).get("score")
    if current_score is not None and previous_score is not None and current_score < previous_score:
        drop = previous_score - current_score
        findings.append(_finding(
            "trend.rating_drop", "previous_assessment", "warning" if drop >= 2 else "advisory", False, category,
            f"Rating fell from {catalog['_ratings'][previous_rating]['label']} to {catalog['_ratings'][rating]['label']} since the previous assessment.",
            "Condition has worsened between assessments.",
            "Compare both assessments and find what changed.",
        ))
    previous_status = previous.get("screening_status")
    rated_low = previous_score is not None and previous_score <= 2
    if current_score is None and rating != "not_applicable" and (rated_low or previous_status in ("critical", "attention")):
        earlier = catalog["_ratings"][previous_rating]["label"] if rated_low else PRIOR_STATUS_LABELS[previous_status]
        findings.append(_finding(
            "trend.prior_problem_not_rechecked", "previous_assessment", "warning", False, category,
            f"Previously recorded as {earlier}, but not rated on this assessment.",
            "A recorded problem cannot be treated as resolved without a new rating.",
            "Inspect this component and record a rating.",
        ))

    service_date = component.get("last_maintenance_date")
    if service_date and assessed_on and current_score is not None and current_score <= 2:
        days = (_parse_date(assessed_on) - _parse_date(service_date)).days
        if 0 <= days <= RECENT_SERVICE_DAYS:
            findings.append(_finding(
                "history.problem_after_recent_service", "maintenance_history", "advisory", False, category,
                f"Rated {catalog['_ratings'][rating]['label']} {days} days after maintenance on {service_date}.",
                "A problem soon after service can mean the original fault was not resolved.",
                "Review the earlier repair with the mechanic.",
            ))

    findings.sort(key=lambda item: SEVERITY_ORDER[item["severity"]])

    answers = {name: value for name, value in (component.get("answers") or {}).items() if value}
    measurements = {name: value for name, value in (component.get("measurements") or {}).items() if _is_number(value)}
    has_observations = bool(answers or measurements or component.get("symptoms") or component.get("fault_codes") or (component.get("notes") or "").strip())
    known_answers = [value for value in answers.values() if value != "unknown"]
    missing = [question["text"] for question in category["questions"] if answers.get(question["key"]) in (None, "unknown")]

    if findings:
        status = SCREENING_BY_SEVERITY[findings[0]["severity"]]
    elif rating == "not_applicable":
        status = "not_applicable"
    elif rating == "not_assessed" and not has_observations:
        status = "not_assessed"
    elif current_score is None:
        status = "insufficient_data"
    else:
        status = "no_concerns_recorded"

    if rating == "not_applicable":
        sufficiency = "not_applicable"
    elif rating == "not_assessed" and not has_observations:
        sufficiency = "not_assessed"
    elif current_score is None:
        sufficiency = "partial" if has_observations else "insufficient"
    else:
        sufficiency = "sufficient" if len(known_answers) * 2 >= len(category["questions"]) else "partial"

    return findings, status, {"status": sufficiency, "missing_questions": missing if sufficiency in ("partial", "insufficient") else []}


def odometer_findings(odometer, previous_odometer):
    if odometer is None or previous_odometer is None or odometer >= previous_odometer:
        return []
    return [{
        "rule_id": "odometer.decrease", "source": "odometer", "severity": "warning", "safety": False,
        "component_key": None, "component_name": "Vehicle record",
        "observation": f"Odometer reading {odometer} km is lower than the previous reading of {previous_odometer} km.",
        "reason": "An odometer should not decrease, so one of the readings may be wrong.",
        "recommended_action": "Verify the odometer and correct the source of the wrong reading.",
    }]


# ---------------------------------------------------------------------------
# Diagnostic knowledge
# ---------------------------------------------------------------------------

def dtc_category(title):
    text = (title or "").casefold()
    for key, pattern in DTC_CATEGORY_KEYWORDS:
        if re.search(pattern, text):
            return key
    return None


def lookup_fault_codes(codes, category_key):
    """Exact lookup only. An unknown code is reported as unknown and never approximated."""
    chunks, _, by_dtc = load_index()
    results = []
    for raw in codes:
        code = raw.strip().upper()
        position = by_dtc.get(code)
        if position is None:
            results.append({"code": code, "found": False, "title": None, "explanation": "This code is not in the generic reference; check the manufacturer's definition.", "source": None, "trust": None})
            continue
        chunk = chunks[position]
        belongs_to = dtc_category(chunk["title"])
        results.append({
            "code": code, "found": True, "title": chunk["title"], "explanation": chunk["text"][:600],
            "affected_components": chunk["metadata"].get("affected_components", []),
            "reference_category": belongs_to,
            "recorded_under_other_category": bool(belongs_to and belongs_to != category_key),
            "source": chunk["citation"].get("name"), "trust": chunk["trust"],
        })
    return results


def symptom_guidance(category_key, symptom_labels):
    if not symptom_labels:
        return []
    results = retrieve(" ".join(symptom_labels), top_k=50)
    guidance = []
    for item in results:
        if FLOWCHART_CATEGORIES.get(item["metadata"].get("category")) == category_key and item["support"] >= MIN_GUIDANCE_SUPPORT:
            guidance.append({"title": item["title"], "text": item["text"], "trust": item["trust"], "source": item["citation"].get("name"), "note": item["citation"].get("note")})
        if len(guidance) == 2:
            break
    return guidance


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _explanation(category, status, findings, sufficiency, faults):
    name = category["name"]
    if status == "not_applicable":
        return f"{name} was marked not applicable to this vehicle."
    if status == "not_assessed":
        return f"{name} was not assessed, so no condition can be stated."
    parts = []
    if findings:
        parts.append(f"{len(findings)} screening rule{'s' if len(findings) != 1 else ''} triggered, led by: {findings[0]['observation']}")
    elif status == "insufficient_data":
        parts.append(f"{name} has no usable rating, so no screening conclusion is possible.")
    else:
        parts.append("No screening rule was triggered by the recorded observations. This is not a statement that the component is healthy.")
    if faults:
        parts.append(f"{len(faults)} fault code{'s' if len(faults) != 1 else ''} recorded.")
    if sufficiency["status"] == "partial":
        parts.append("Some questions were left unanswered or unknown.")
    return " ".join(parts)


def _recommended_action(status, findings):
    if findings:
        return findings[0]["recommended_action"]
    if status == "not_applicable":
        return "No action."
    if status == "not_assessed":
        return "Inspect this component and record a rating."
    if status == "insufficient_data":
        return "Gather the missing information and record a rating."
    return "No action triggered. Reassess at the next scheduled assessment."


def analyze_assessment(payload):
    """Result for one validated payload. A knowledge-index failure only removes knowledge lookups."""
    validate_payload(payload)
    catalog = load_catalog()
    previous_components = (payload.get("previous_assessment") or {}).get("components") or {}
    submitted = payload["components"]
    knowledge_error = None
    try:
        load_index()
    except KnowledgeIndexUnavailable as error:
        knowledge_error = str(error)

    # Imported here because category_models builds on this module's screening.
    from .category_models import predictions_for_assessment
    predictions = predictions_for_assessment(payload)

    results = []
    warnings = odometer_findings(payload.get("odometer_km"), payload.get("previous_odometer_km"))
    for category in catalog["categories"]:
        key = category["key"]
        component = submitted.get(key) or {}
        findings, status, sufficiency = screen_component(key, component, previous_components.get(key), payload["assessed_on"])
        warnings.extend(finding for finding in findings if finding["safety"])

        symptoms = [symptom["label"] for symptom in category["symptoms"] if symptom["key"] in (component.get("symptoms") or [])]
        faults, guidance = [], []
        if not knowledge_error:
            faults = lookup_fault_codes(component.get("fault_codes") or [], key)
            guidance = symptom_guidance(key, symptoms)

        prediction = predictions[key]
        rating = component.get("rating") or "not_assessed"
        evidence = [f"Rating: {catalog['_ratings'][rating]['label']}"]
        evidence += [f"Symptom: {label}" for label in symptoms]
        specs = {measurement["key"]: measurement for measurement in category["measurements"]}
        evidence += [f"{specs[name]['label']}: {_format_value(value)} {specs[name]['unit']}" for name, value in (component.get("measurements") or {}).items() if _is_number(value)]
        evidence += [f"Fault code: {code.strip().upper()}" for code in component.get("fault_codes") or []]
        if component.get("last_maintenance_date"):
            evidence.append(f"Last maintenance: {component['last_maintenance_date']}")
        if (previous_components.get(key) or {}).get("rating") in catalog["_ratings"]:
            evidence.append(f"Previous rating: {catalog['_ratings'][previous_components[key]['rating']]['label']}")

        results.append({
            "component_key": key,
            "component_name": category["name"],
            "observed_condition": {"rating": rating, "score": catalog["_ratings"][rating]["score"], "label": catalog["_ratings"][rating]["label"]},
            "screening_status": status,
            "findings": findings,
            "identified_symptoms": symptoms,
            "identified_faults": faults,
            "diagnostic_guidance": guidance,
            "evidence_used": evidence,
            "explanation": _explanation(category, status, findings, sufficiency, faults),
            "recommended_action": _recommended_action(status, findings),
            "data_sufficiency": sufficiency,
            "ai_model": prediction.get("model"),
            "knowledge_used": bool(faults or guidance),
            "predictive_status": PREDICTION_AVAILABLE if prediction["available"] else INSUFFICIENT_DATA,
            "confidence": None,
            "prediction_horizon": f"{prediction['horizon_days']} days" if prediction["available"] else None,
            "prediction": prediction,
        })

    model_status = maintenance_model_status()
    enabled = [item["component_key"] for item in results if item["prediction"]["available"]]
    demonstration = [item["component_key"] for item in results if item["prediction"].get("demonstration")]
    reason = ("Only categories with an accepted model have a prediction. Condition-based screening covers every category."
              if enabled else "No model is validated for predicting failures on this fleet. Condition-based screening is reported instead.")
    if demonstration:
        reason = (f"{len(demonstration)} of these predictions come from demonstration models trained on simulated SAMPLE- vehicles, "
                  "not real fleet outcomes; do not base maintenance decisions on them. " + reason)
    return {
        "status": "success",
        "assessment_id": payload["assessment_id"],
        "vehicle_id": payload["vehicle"]["id"],
        "processing": {
            "status": "completed" if not knowledge_error else "completed_without_knowledge",
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "catalog_version": catalog["version"],
            "knowledge_index": "available" if not knowledge_error else "unavailable",
            "message": None if not knowledge_error else "Fault-code and symptom lookup were skipped because the knowledge index is unavailable.",
        },
        "predictive_capability": {
            "status": "partial" if enabled else "unavailable",
            "label": (f"Available for {len(enabled)} of {len(results)} categories" if enabled else INSUFFICIENT_DATA)
                     + (" (demonstration models)" if demonstration and len(demonstration) == len(enabled) else ""),
            "reason": reason,
            "categories": enabled,
            "demonstration_categories": demonstration,
            "registered_model_status": model_status.get("status"),
        },
        "safety_warnings": warnings,
        "components": results,
    }
