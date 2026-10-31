import json
import logging
import re
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .assistant import resolve_role
from .chatbot_service import answer_chat
from .rag import KnowledgeIndexUnavailable, retrieve
from .fleet_service import analyze_fleet
from .prediction_service import (
    MaintenancePredictionsUnavailableError,
    load_maintenance_artifact,
    maintenance_model_status,
    predict_vehicle,
)
from .security import require_api_key
from .apps import WARM_START


logger = logging.getLogger(__name__)
MAX_CHAT_MESSAGE_LENGTH = 4000
UUID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)


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
                "name": "hybrid_knowledge_retrieval",
                "status": "available" if (Path(settings.KNOWLEDGE_INDEX_PATH) / "index.joblib").exists() else "not_built",
                "sources": ["OBDex generic DTC/PID reference (CC0)", "curated automotive knowledge base",
                            "diagnostic flowcharts", "DriveRent system guide", "recorded workshop service statistics"],
            },
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
            "vehicle_maintenance_risk": (
                "service_need_model_accepted" if model_status["eligible_for_inference"]
                else "no_accepted_model"
            ),
            "component_failure_prediction": "not_supported_by_verified_labels",
            "remaining_useful_life": "not_supported_for_fleet_vehicles",
        },
        "maintenance_model": model_status,
    })


def parse_json(request):
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("Request body must contain valid JSON.")
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    return payload


STARTING = "The AI assistant is still starting up. Please try again in a minute."


@require_GET
@require_api_key
def ready(request):
    """200 once the warm start has loaded the models and indexes; 503 while it is still loading."""
    state = WARM_START["state"]
    return JsonResponse({"status": "starting" if state == "loading" else "ready", "warm_start": state,
                         "components": WARM_START["components"]}, status=503 if state == "loading" else 200)


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
        "knowledge_index_exists": (Path(settings.KNOWLEDGE_INDEX_PATH) / "index.joblib").exists(),
        "maintenance_model_version": maintenance_status.get("model_version"),
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
        logger.exception("prediction failed")
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
    if WARM_START["state"] == "loading":
        # The Back-End shows this message to the user instead of a generic "unavailable".
        return JsonResponse({"error": STARTING, "status": "starting"}, status=503)
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
        message = payload.get("message", "")
        if isinstance(message, str) and len(message) > MAX_CHAT_MESSAGE_LENGTH:
            raise ValueError(f"message must be at most {MAX_CHAT_MESSAGE_LENGTH} characters.")
        viewer_role = payload.get("viewer_role")
        if viewer_role is not None and viewer_role not in {"admin", "customer"}:
            raise ValueError("viewer_role must be admin or customer.")
        conversation_id = payload.get("conversation_id")
        if conversation_id is not None and not (isinstance(conversation_id, str) and UUID_PATTERN.fullmatch(conversation_id)):
            raise ValueError("conversation_id must be a UUID.")
        conversation_scope = payload.get("conversation_scope")
        if conversation_scope is not None and not isinstance(conversation_scope, str):
            raise ValueError("conversation_scope must be a string.")
        retrieval_protocol = payload.get("retrieval_protocol")
        if retrieval_protocol not in (None, 1):
            raise ValueError("retrieval_protocol must be 1.")
        vocabulary = payload.get("vocabulary")
        if vocabulary is not None and not isinstance(vocabulary, dict):
            raise ValueError("vocabulary must be an object.")
        retrieval = payload.get("retrieval")
        if retrieval is not None and not (isinstance(retrieval, dict) and isinstance(retrieval.get("results"), list)):
            raise ValueError("retrieval must be an object with a results list.")
        result = answer_chat(
            message=message,
            vehicle=payload.get("vehicle"),
            vehicles=payload.get("vehicles"),
            rental_context=rental_context,
            diagnostic_context=diagnostic_context,
            viewer_role=viewer_role,
            conversation_id=conversation_id,
            conversation_scope=conversation_scope,
            retrieval_protocol=retrieval_protocol,
            vocabulary=vocabulary,
            retrieval=retrieval,
        )
        return JsonResponse(result)
    except FileNotFoundError:
        # The error names local file paths; those stay in the service log.
        logger.exception("chat model or reference file missing")
        return JsonResponse(
            {"error": "The chat assistant is not ready: its trained model or reference data is missing on the AI service."},
            status=503,
        )
    except ValueError as error:
        return JsonResponse({"error": str(error)}, status=400)
    except Exception:
        logger.exception("chat request failed")
        return JsonResponse(
            {"error": "Chat request failed. Check the service logs."},
            status=500
        )


@csrf_exempt
@require_POST
@require_api_key
def knowledge_search(request):
    try:
        payload = parse_json(request)
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_CHAT_MESSAGE_LENGTH:
            raise ValueError("query must be a non-empty string of at most 4000 characters.")
        top_k = payload.get("top_k", 5)
        if not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError("top_k must be an integer between 1 and 20.")
        role = resolve_role(payload.get("viewer_role"))
        return JsonResponse({"status": "success", "viewer_role": role, "results": retrieve(query.strip(), role, top_k=top_k)})
    except KnowledgeIndexUnavailable as error:
        return JsonResponse({"status": "unavailable", "error": str(error)}, status=503)
    except ValueError as error:
        return JsonResponse({"error": str(error)}, status=400)


@require_GET
@require_api_key
def model_inventory(request):
    registry_path = Path(settings.AI_MAINTENANCE_MODEL_REGISTRY_PATH)
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        registry = {}
    experiments = registry.get("experiments", {}) if isinstance(registry, dict) else {}
    return JsonResponse({
        "status": "ok",
        "served_maintenance_model": maintenance_model_status(),
        "experiments": {
            name: {key: entry.get(key) for key in ("status", "task", "dataset", "eligible_for_inference", "reason", "metrics")}
            for name, entry in experiments.items() if isinstance(entry, dict)
        },
    })
