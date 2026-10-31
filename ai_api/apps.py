import logging
import os
import sys
import threading

from django.apps import AppConfig


logger = logging.getLogger(__name__)
# not_started (tests, management commands: components load on first use), loading, ready.
WARM_START = {"state": "not_started", "components": {}}


class AiApiConfig(AppConfig):
    name = "ai_api"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self):
        # Load models and the knowledge index once at startup instead of on the
        # first request. Skipped for tests and management commands other than runserver.
        serving = "runserver" in sys.argv or os.environ.get("AI_WARM_START") == "1"
        autoreload_parent = "runserver" in sys.argv and "--noreload" not in sys.argv and os.environ.get("RUN_MAIN") != "true"
        if not serving or autoreload_parent:
            return
        WARM_START["state"] = "loading"
        threading.Thread(target=_warm_start, name="ai-warm-start", daemon=True).start()


def _warm_start():
    from .automotive_knowledge import _load_index as load_curated_knowledge
    from .chatbot_service import load_chatbot
    from .prediction_service import MaintenancePredictionsUnavailableError, load_maintenance_artifact
    from .rag import KnowledgeIndexUnavailable, load_index, load_parts_lookup

    for name, loader in (("chatbot", load_chatbot), ("knowledge_index", load_index), ("curated_knowledge", load_curated_knowledge),
                         ("parts_lookup", load_parts_lookup), ("maintenance_model", load_maintenance_artifact)):
        try:
            loader()
            WARM_START["components"][name] = "loaded"
            logger.info("warm_start component=%s status=loaded", name)
        except (FileNotFoundError, ValueError, KnowledgeIndexUnavailable, MaintenancePredictionsUnavailableError) as error:
            # The maintenance model is expected to be unavailable until one passes its acceptance gate.
            WARM_START["components"][name] = "unavailable"
            logger.warning("warm_start component=%s status=unavailable reason=%s", name, error)
    WARM_START["state"] = "ready"
