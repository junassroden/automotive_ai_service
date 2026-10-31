from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from django.conf import settings


MAINTENANCE_LABELS = {
    "oil_filter": "Oil filter",
    "engine_oil": "Engine oil",
    "washer_plug_drain": "Washer plug / drain",
    "dust_and_pollen_filter": "Cabin dust and pollen filter",
    "whell_alignment_and_balancing": "Wheel alignment and balancing",
    "air_clean_filter": "Air filter",
    "fuel_filter": "Fuel filter",
    "spark_plug": "Spark plugs",
    "brake_fluid": "Brake fluid",
    "brake_and_clutch_oil": "Brake and clutch fluid",
    "transmission_fluid": "Transmission fluid",
    "brake_pads": "Brake pads",
    "clutch": "Clutch inspection",
    "coolant": "Coolant"
}

HIGH_PRIORITY_COMPONENTS = {
    "brake_fluid",
    "brake_pads",
    "brake_and_clutch_oil",
    "clutch"
}

SERVICE_NEED_TASK = "service_need_at_visit"
ACTION_TEXT = {
    "replacement": "Check the manufacturer schedule and inspect the {label} at the next service; replace it only if the schedule or the inspection confirms it is due.",
    "inspection": "Inspect the {label} (wear, thickness, operation) at the next service and record the finding before deciding on repair.",
    "routine_service": "Include {label} in the next scheduled service if the manufacturer schedule or an inspection supports it.",
}


class MaintenancePredictionsUnavailableError(RuntimeError):
    pass


def maintenance_model_status():
    registry_path = Path(settings.AI_MAINTENANCE_MODEL_REGISTRY_PATH)
    if not registry_path.is_file():
        return {
            "status": "blocked",
            "eligible_for_inference": False,
            "reason": "No maintenance-model registry is available.",
        }

    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "failed",
            "eligible_for_inference": False,
            "reason": f"Maintenance-model registry is unreadable: {error}",
        }

    model = registry.get("maintenance_model") if isinstance(registry, dict) else None
    if not isinstance(model, dict):
        return {
            "status": "not_started",
            "eligible_for_inference": False,
            "reason": "No maintenance model is registered.",
        }
    evaluation = model.get("evaluation")
    if not isinstance(evaluation, dict):
        evaluation = {}
    if (
        model.get("status") != "evaluated_accepted"
        or model.get("eligible_for_inference") is not True
        or evaluation.get("acceptance_criteria_passed") is not True
    ):
        return {
            "status": model.get("status", "blocked"),
            "eligible_for_inference": False,
            "reason": model.get(
                "reason",
                "A maintenance model has not passed the registered acceptance criteria.",
            ),
        }

    registered_artifact = model.get("artifact_path")
    if not isinstance(registered_artifact, str) or not registered_artifact:
        return {
            "status": "failed",
            "eligible_for_inference": False,
            "reason": "The accepted registry entry has no artifact path.",
        }
    artifact_path = Path(registered_artifact)
    if not artifact_path.is_absolute():
        artifact_path = settings.BASE_DIR / artifact_path
    artifact_path = artifact_path.resolve()
    try:
        artifact_path.relative_to(settings.BASE_DIR.resolve())
    except ValueError:
        return {
            "status": "failed",
            "eligible_for_inference": False,
            "reason": "Registered model artifacts must be located inside the service directory.",
        }
    if not artifact_path.is_file():
        return {
            "status": "failed",
            "eligible_for_inference": False,
            "reason": "The accepted maintenance-model artifact is missing.",
        }

    expected_hash = model.get("artifact_sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        return {
            "status": "failed",
            "eligible_for_inference": False,
            "reason": "The registry has no artifact integrity hash.",
        }
    try:
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    except OSError as error:
        return {
            "status": "failed",
            "eligible_for_inference": False,
            "reason": f"Maintenance-model artifact is unreadable: {error}",
        }
    if digest != expected_hash:
        return {
            "status": "failed",
            "eligible_for_inference": False,
            "reason": "The maintenance-model artifact does not match its registry hash.",
        }

    registered_report = model.get("evaluation_report_path")
    if not isinstance(registered_report, str) or not registered_report:
        return {
            "status": "failed",
            "eligible_for_inference": False,
            "reason": "The accepted registry entry has no evaluation report path.",
        }
    report_path = Path(registered_report)
    if not report_path.is_absolute():
        report_path = settings.BASE_DIR / report_path
    report_path = report_path.resolve()
    try:
        report_path.relative_to(settings.BASE_DIR.resolve())
    except ValueError:
        return {
            "status": "failed",
            "eligible_for_inference": False,
            "reason": "Registered evaluation reports must be located inside the service directory.",
        }
    report_hash = model.get("evaluation_report_sha256")
    try:
        report_bytes = report_path.read_bytes()
        evaluation_report = json.loads(report_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return {
            "status": "failed",
            "eligible_for_inference": False,
            "reason": f"Maintenance evaluation report is missing or invalid: {error}",
        }
    if not isinstance(evaluation_report, dict):
        return {
            "status": "failed",
            "eligible_for_inference": False,
            "reason": "Maintenance evaluation report must contain a JSON object.",
        }
    if (
        not isinstance(report_hash, str)
        or hashlib.sha256(report_bytes).hexdigest() != report_hash
    ):
        return {
            "status": "failed",
            "eligible_for_inference": False,
            "reason": "The evaluation report does not match its registry hash.",
        }
    report_acceptance = evaluation_report.get("acceptance_criteria")
    metrics = evaluation_report.get("metrics")
    if not isinstance(report_acceptance, dict) or not isinstance(metrics, dict):
        return {
            "status": "failed",
            "eligible_for_inference": False,
            "reason": "The evaluation report is missing acceptance evidence or metrics.",
        }
    group_split = evaluation_report.get("group_split")
    if not isinstance(group_split, dict):
        return {
            "status": "failed",
            "eligible_for_inference": False,
            "reason": "The evaluation report does not document a group-disjoint split.",
        }
    try:
        model_balanced_accuracy = float(metrics["balanced_accuracy"])
        baseline_balanced_accuracy = float(metrics["baseline_balanced_accuracy"])
        model_macro_f1 = float(metrics["macro_f1"])
        baseline_macro_f1 = float(metrics["baseline_macro_f1"])
        train_groups = int(group_split["train_group_count"])
        test_groups = int(group_split["test_group_count"])
        group_overlap = int(group_split["group_overlap"])
    except (KeyError, TypeError, ValueError):
        return {
            "status": "failed",
            "eligible_for_inference": False,
            "reason": "The evaluation report is missing required metrics or split counts.",
        }
    thresholds_passed = (
        all(
            math.isfinite(value) and 0 <= value <= 1
            for value in (
                model_balanced_accuracy,
                baseline_balanced_accuracy,
                model_macro_f1,
                baseline_macro_f1,
            )
        )
        and train_groups >= settings.AI_MAINTENANCE_EVALUATION_MIN_GROUPS_PER_SPLIT
        and test_groups >= settings.AI_MAINTENANCE_EVALUATION_MIN_GROUPS_PER_SPLIT
        and group_overlap == 0
        and model_balanced_accuracy >= settings.AI_MAINTENANCE_MIN_BALANCED_ACCURACY
        and model_macro_f1 >= settings.AI_MAINTENANCE_MIN_MACRO_F1
        and model_balanced_accuracy >= baseline_balanced_accuracy
        and model_macro_f1 >= baseline_macro_f1
    )
    if (
        evaluation_report.get("status") != "evaluated_accepted"
        or report_acceptance.get("acceptance_criteria_passed") is not True
        or not thresholds_passed
    ):
        return {
            "status": "evaluated_rejected",
            "eligible_for_inference": False,
            "reason": "Evaluation evidence failed the configured model acceptance criteria.",
        }
    probability_validation = evaluation_report.get("probability_validation")
    if (
        not isinstance(probability_validation, dict)
        or probability_validation.get("status") != "calibrated"
        or not probability_validation.get("method")
        or not isinstance(probability_validation.get("validation_records"), int)
        or probability_validation["validation_records"] < 2
        or not isinstance(probability_validation.get("independent_groups"), int)
        or probability_validation["independent_groups"] < 2
        or not isinstance(probability_validation.get("brier_score"), (int, float))
        or not math.isfinite(probability_validation["brier_score"])
        or not 0 <= probability_validation["brier_score"] <= 1
    ):
        return {
            "status": "evaluated_rejected",
            "eligible_for_inference": False,
            "reason": "Evaluation evidence does not establish calibrated probabilities on independent validation groups.",
        }

    feature_columns = model.get("feature_columns")
    if (
        not isinstance(feature_columns, list)
        or not feature_columns
        or not all(isinstance(column, str) and column for column in feature_columns)
        or len(set(feature_columns)) != len(feature_columns)
    ):
        return {
            "status": "failed",
            "eligible_for_inference": False,
            "reason": "The registry has an invalid feature schema.",
        }
    target_definitions = model.get("target_definitions")
    return {
        "status": "evaluated_accepted",
        "eligible_for_inference": True,
        "reason": "The registered maintenance model passed evaluation.",
        "artifact_path": str(artifact_path),
        "artifact_sha256": expected_hash,
        "evaluation_report_path": str(report_path),
        "evaluation_report_sha256": report_hash,
        "feature_columns": feature_columns,
        "model_version": model.get("model_version"),
        "task": model.get("task"),
        "target_definitions": target_definitions if isinstance(target_definitions, dict) else {},
        "metrics": {
            key: metrics.get(key)
            for key in ("balanced_accuracy", "baseline_balanced_accuracy", "macro_f1",
                        "baseline_macro_f1", "macro_pr_auc", "macro_brier")
            if key in metrics
        },
        "calibration": {
            "status": probability_validation["status"],
            "method": probability_validation["method"],
            "brier_score": probability_validation["brier_score"],
        },
    }


@lru_cache(maxsize=2)
def _load_registered_artifact(artifact_path, expected_hash):
    artifact_path = Path(artifact_path)
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise MaintenancePredictionsUnavailableError(
            "Maintenance-model artifact changed after eligibility validation."
        )
    try:
        artifact = joblib.load(artifact_path)
    except Exception as error:
        raise MaintenancePredictionsUnavailableError(
            f"Maintenance-model artifact could not be loaded: {error}"
        ) from error
    if (
        not isinstance(artifact, dict)
        or not isinstance(artifact.get("models"), dict)
        or not isinstance(artifact.get("target_columns"), list)
        or not isinstance(artifact.get("feature_columns"), list)
    ):
        raise MaintenancePredictionsUnavailableError(
            "Maintenance-model artifact has an unsupported structure."
        )
    return artifact


def load_maintenance_artifact():
    status = maintenance_model_status()
    if not status["eligible_for_inference"]:
        raise MaintenancePredictionsUnavailableError(status["reason"])
    artifact = _load_registered_artifact(
        status["artifact_path"],
        status["artifact_sha256"],
    )
    if artifact["feature_columns"] != status["feature_columns"]:
        raise MaintenancePredictionsUnavailableError(
            "Maintenance-model feature schema does not match its registry."
        )
    targets = artifact["target_columns"]
    models = artifact["models"]
    constants = artifact.get("constant_targets", {})
    thresholds = artifact.get("thresholds", {})
    if (
        not models
        or not targets
        or not all(isinstance(target, str) and target for target in targets)
        or len(set(targets)) != len(targets)
        or not isinstance(constants, dict)
        or not isinstance(thresholds, dict)
        or any(target not in models and target not in constants for target in targets)
        or any(target not in targets for target in models)
        or any(
            not callable(getattr(model, "predict_proba", None))
            or not hasattr(getattr(model, "named_steps", {}).get("classifier"), "classes_")
            or not {0, 1}.issubset(
                set(getattr(model.named_steps["classifier"], "classes_", []))
            )
            for model in models.values()
        )
        or any(
            not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
            for threshold in thresholds.values()
        )
    ):
        raise MaintenancePredictionsUnavailableError(
            "The registered maintenance-model artifact has invalid target or classifier structure."
        )
    return artifact


def validate_vehicle(vehicle, artifact):
    if not isinstance(vehicle, dict):
        raise ValueError("The vehicle field must be a JSON object.")
    if artifact.get("task") == SERVICE_NEED_TASK:
        return _validate_service_need_vehicle(vehicle, artifact)

    feature_columns = artifact.get("feature_columns", [])
    if not feature_columns:
        raise ValueError("The model artifact has no feature_columns.")

    missing = [
        column for column in feature_columns
        if column not in vehicle or vehicle[column] is None
        or str(vehicle[column]).strip() == ""
    ]
    if missing:
        raise ValueError(
            "Missing required vehicle fields: " + ", ".join(missing)
        )

    return {column: vehicle[column] for column in feature_columns}


def _validate_service_need_vehicle(vehicle, artifact):
    from data_pipeline.service_need import normalize_vehicle_input

    features = normalize_vehicle_input(vehicle)
    domain = artifact.get("feature_domain", {})
    if not features["vehicle_type"] or not features["fuel_type"] or features["mileage"] in (None, ""):
        missing = [name for name in ("vehicle_type", "fuel_type", "mileage") if features[name] in (None, "")]
        raise ValueError("Missing required vehicle fields: " + ", ".join(missing))
    try:
        mileage = float(features["mileage"])
    except (TypeError, ValueError):
        raise ValueError("mileage must be a number of kilometres.")
    if not math.isfinite(mileage) or mileage < 0:
        raise ValueError("mileage must be a non-negative number of kilometres.")
    if features["vehicle_type"] not in domain.get("vehicle_type", []):
        raise ValueError(
            f"Vehicle type '{features['vehicle_type']}' is outside the model's training data "
            f"({', '.join(domain.get('vehicle_type', []))}); no prediction is made for it."
        )
    if features["fuel_type"] not in domain.get("fuel_type", []):
        raise ValueError(
            f"Fuel type '{features['fuel_type']}' is outside the model's training data "
            f"({', '.join(domain.get('fuel_type', []))}); no prediction is made for it."
        )
    low, high = domain.get("mileage", [0, float("inf")])
    if not low <= mileage <= high:
        raise ValueError(
            f"Odometer {mileage:,.0f} km is outside the trained range ({low:,.0f}-{high:,.0f} km); "
            "the model does not extrapolate."
        )
    features["mileage"] = mileage
    return features


def _enrich_service_need(results, artifact, features):
    high_risk = float(getattr(settings, "AI_PREDICTION_HIGH_RISK_PROBABILITY", 0.70))
    milestone = int(round(features["mileage"] / 10_000) * 10_000)
    definitions = artifact.get("target_definitions", {})
    for item in results:
        target = item["component"]
        definition = definitions.get(target, {})
        evidence = artifact.get("evidence_tables", {}).get(target, {}).get(str(milestone))
        item["prediction_type"] = "service_need_at_visit"
        item["component_keys"] = definition.get("component_keys", [])
        item["calibrated_probability"] = item["score"]
        if item["score"] is None:
            item["risk_category"] = "not_assessed"
        elif item["recommendation"] and item["score"] >= max(item["threshold"], high_risk):
            item["risk_category"] = "high"
        elif item["recommendation"]:
            item["risk_category"] = "medium"
        else:
            item["risk_category"] = "low"
        item["recommended_action"] = (
            ACTION_TEXT.get(definition.get("action"), ACTION_TEXT["inspection"]).format(label=item["label"].lower())
            if item["recommendation"] else None
        )
        item["supporting_evidence"] = (
            f"In the training service records this item was performed at {evidence['performed']} of "
            f"{evidence['visits']} visits near {milestone:,} km."
            if evidence else f"The training service records have no visits near {milestone:,} km."
        )
        item["contributing_factors"] = [
            f"Odometer {features['mileage']:,.0f} km (nearest 10,000 km service milestone: {milestone:,} km)",
            f"Body type: {features['vehicle_type']}",
            f"Fuel type: {features['fuel_type']}",
        ]
    return results


def predict_vehicle(vehicle):
    artifact = load_maintenance_artifact()
    features = validate_vehicle(vehicle, artifact)
    row = pd.DataFrame([features])
    service_need = artifact.get("task") == SERVICE_NEED_TASK

    results = []
    models = artifact.get("models", {})
    thresholds = artifact.get("thresholds", {})
    constant_targets = artifact.get("constant_targets", {})

    for target in artifact.get("target_columns", []):
        label = MAINTENANCE_LABELS.get(
            target, target.replace("_", " ").title()
        )

        if target in models:
            model = models[target]
            probabilities = np.asarray(model.predict_proba(row)[0], dtype=float)
            classes = list(model.named_steps["classifier"].classes_)
            if (
                probabilities.ndim != 1
                or len(probabilities) != len(classes)
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or np.any(probabilities > 1)
                or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-6)
            ):
                raise MaintenancePredictionsUnavailableError(
                    f"Model probabilities for {target} are invalid."
                )
            positive_index = classes.index(1) if 1 in classes else None

            if positive_index is None:
                probability = 0.0
            else:
                probability = float(probabilities[positive_index])

            threshold = float(thresholds.get(target, 0.5))
            recommended = probability >= threshold

            if recommended and target in HIGH_PRIORITY_COMPONENTS:
                priority = "high"
            elif recommended and probability >= max(threshold, 0.70):
                priority = "high"
            elif recommended:
                priority = "medium"
            else:
                priority = "low"

            results.append({
                "component": target,
                "label": label,
                "recommendation": recommended,
                "score": round(probability, 4),
                "threshold": threshold,
                "priority": priority,
                "reason": (
                    f"The model score ({probability:.0%}) "
                    f"{'meets or exceeds' if recommended else 'is below'} "
                    f"the configured recommendation threshold "
                    f"({threshold:.0%})."
                )
            })

        elif target in constant_targets:
            constant = constant_targets[target]
            results.append({
                "component": target,
                "label": label,
                "recommendation": None,
                "score": None,
                "threshold": None,
                "priority": "unknown",
                "reason": (
                    "This category has no usable binary classifier: "
                    + constant.get("status", "training data is insufficient")
                )
            })

        else:
            results.append({
                "component": target,
                "label": label,
                "recommendation": None,
                "score": None,
                "threshold": None,
                "priority": "unknown",
                "reason": "No trained model is available for this category."
            })

    if service_need:
        _enrich_service_need(results, artifact, features)

    recommended = [
        item for item in results if item["recommendation"] is True
    ]
    recommended.sort(
        key=lambda item: (
            {"high": 3, "medium": 2, "low": 1}.get(item["priority"], 0),
            item["score"] or 0
        ),
        reverse=True
    )

    return {
        "status": "success",
        "vehicle": {
            key: vehicle.get(key)
            for key in ("id", "name", "brand", "model", "vehicle_type", "fuel_type", "make_year", "mileage")
            if key in vehicle
        },
        "summary": {
            "trained_categories": len(models),
            "recommendations_count": len(recommended),
            "unknown_categories_count": sum(
                item["recommendation"] is None for item in results
            )
        },
        "recommendations": recommended,
        "all_categories": results,
        **({
            "model_version": artifact.get("model_version"),
            "prediction_type": "service_need_at_visit",
            "prediction_type_meaning": (
                "Probability that each service item would be performed at a workshop "
                "service visit at this odometer reading. It is a scheduled-service need, "
                "not an observed fault or a failure forecast; no failure date is estimated."
            ),
            "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
            "model_inputs": features,
        } if service_need else {}),
        "disclaimer": (
            "These are model-based maintenance recommendations, not "
            "confirmed mechanical diagnoses. Inspect the vehicle and "
            "check its service history before making repair decisions."
        )
    }
