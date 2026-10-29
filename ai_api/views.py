import json

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .chatbot_service import answer_chat
from .fleet_service import analyze_fleet
from .prediction_service import (
    MaintenancePredictionsUnavailableError,
    load_maintenance_artifact,
    maintenance_model_status,
    predict_vehicle,
)
from .security import require_api_key


@require_GET
@require_api_key
def capabilities(request):
    model_status = maintenance_model_status()
    predictive_models = []
    if model_status["eligible_for_inference"]:
        predictive_models.append({
            "name": "registered_maintenance_model",
            "status": model_status["status"],
        })
    return JsonResponse({
        "status": "ok",
        "service": "custom-automotive-ai",
        "evidence_backed_predictive_models": predictive_models,
        "available_assistance": [
            {
                "name": "local_automotive_knowledge_retrieval",
                "status": "available",
                "limitation": (
                    "General reference guidance only; not a vehicle-specific "
                    "diagnosis or model prediction."
                ),
            }
        ],
        "training_readiness": {
            "engine_fault_classification": "blocked",
            "reason": (
                "The local EngineFaultDB copy lacks acquisition-run identifiers "
                "and a verified mapping from its numeric classes to fault names."
            ),
            "vehicle_maintenance_risk": "not_supported_by_verified_labels",
            "remaining_useful_life": "not_supported_for_fleet_vehicles",
        },
        "maintenance_model": model_status,
    })


def parse_json(request):
    try:
        return json.loads(request.body.decode("utf-8") or "{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("Request body must contain valid JSON.")


@require_GET
@require_api_key
def health(request):
    model_loaded = False
    model_error = None

    maintenance_status = maintenance_model_status()
    if maintenance_status["eligible_for_inference"]:
        try:
            artifact = load_maintenance_artifact()
            model_loaded = bool(artifact.get("models"))
        except MaintenancePredictionsUnavailableError as error:
            model_error = str(error)
    else:
        model_error = maintenance_status["reason"]

    return JsonResponse({
        "status": "ok",
        "service": "custom-automotive-ai",
        "maintenance_model_loaded": model_loaded,
        "maintenance_model_error": model_error,
        "maintenance_predictions_enabled": model_loaded,
        "maintenance_model_evidence_verified": model_loaded,
        "maintenance_model_status": maintenance_status["status"],
        "chatbot_model_exists": settings.CHATBOT_MODEL_PATH.exists(),
        "external_ai_provider": False
    })


@csrf_exempt
@require_POST
@require_api_key
def predict(request):
    try:
        payload = parse_json(request)
        result = predict_vehicle(payload.get("vehicle"))
        return JsonResponse(result)
    except MaintenancePredictionsUnavailableError as error:
        return JsonResponse(
            {
                "status": "unavailable",
                "code": "model_not_eligible",
                "error": str(error),
            },
            status=503,
        )
    except FileNotFoundError as error:
        return JsonResponse({"error": str(error)}, status=503)
    except ValueError as error:
        return JsonResponse({"error": str(error)}, status=400)
    except Exception:
        return JsonResponse(
            {"error": "Prediction failed. Check the service logs and model artifact."},
            status=500
        )


@csrf_exempt
@require_POST
@require_api_key
def analyze_fleet_view(request):
    try:
        payload = parse_json(request)
        result = analyze_fleet(payload.get("vehicles", []))
        return JsonResponse(result)
    except MaintenancePredictionsUnavailableError as error:
        return JsonResponse(
            {
                "status": "unavailable",
                "code": "model_not_eligible",
                "error": str(error),
            },
            status=503,
        )
    except FileNotFoundError as error:
        return JsonResponse({"error": str(error)}, status=503)
    except ValueError as error:
        return JsonResponse({"error": str(error)}, status=400)
    except Exception:
        return JsonResponse(
            {"error": "Fleet analysis failed. Check the service logs."},
            status=500
        )


@csrf_exempt
@require_POST
@require_api_key
def chat(request):
    try:
        payload = parse_json(request)
        rental_context = payload.get("rental_context")
        if rental_context is not None and not isinstance(rental_context, dict):
            raise ValueError("rental_context must be a JSON object.")
        diagnostic_context = (
            rental_context.get("diagnostic_context")
            if isinstance(rental_context, dict)
            else None
        )
        if diagnostic_context is not None and not isinstance(diagnostic_context, dict):
            raise ValueError("rental_context.diagnostic_context must be an object.")
        result = answer_chat(
            message=payload.get("message", ""),
            vehicle=payload.get("vehicle"),
            vehicles=payload.get("vehicles"),
            rental_context=rental_context,
            diagnostic_context=diagnostic_context,
        )
        return JsonResponse(result)
    except FileNotFoundError as error:
        return JsonResponse({"error": str(error)}, status=503)
    except ValueError as error:
        return JsonResponse({"error": str(error)}, status=400)
    except Exception:
        return JsonResponse(
            {"error": "Chat request failed. Check the service logs."},
            status=500
        )
