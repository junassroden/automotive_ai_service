import os
import sys
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_env_file(path):
    """KEY=VALUE lines from .env (no extra dependency); variables already set in the environment win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_env_file(BASE_DIR / ".env")

DEFAULT_SECRET_KEY = "local-development-only-change-this-secret"
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", DEFAULT_SECRET_KEY)
# Off unless DJANGO_DEBUG=True is set (the development .env sets it).
DEBUG = os.environ.get("DJANGO_DEBUG", "False").lower() == "true"
if not DEBUG and SECRET_KEY == DEFAULT_SECRET_KEY and "test" not in sys.argv[1:2]:
    raise ImproperlyConfigured("Set DJANGO_SECRET_KEY (a long random value) when DJANGO_DEBUG is not True.")
ALLOWED_HOSTS = [host.strip() for host in os.environ.get("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",") if host.strip()]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "ai_api",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    # The POST endpoints are API-key protected JSON views marked csrf_exempt; these keep any other page safe.
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
# The Back-End reaches this service over plain HTTP on localhost. Behind a TLS proxy, set these.
SECURE_SSL_REDIRECT = os.environ.get("DJANGO_SECURE_SSL_REDIRECT", "False").lower() == "true"
SECURE_HSTS_SECONDS = int(os.environ.get("DJANGO_SECURE_HSTS_SECONDS", "0"))
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https") if SECURE_SSL_REDIRECT else None
CSRF_COOKIE_SECURE = SESSION_COOKIE_SECURE = SECURE_SSL_REDIRECT

ROOT_URLCONF = "automotive_ai.urls"
TEMPLATES = []
WSGI_APPLICATION = "automotive_ai.wsgi.application"
ASGI_APPLICATION = "automotive_ai.asgi.application"

# Conversation memory lives in MySQL (Aiven in production) when AI_DB_HOST is set, so it survives
# redeploys; without it, in the local SQLite file. Aiven requires TLS: AI_DB_SSL_CA is the path to
# its CA certificate (on Render, a Secret File such as /etc/secrets/ca.pem).
if os.environ.get("AI_DB_HOST"):
    import pymysql

    pymysql.install_as_MySQLdb()
    _db_options = {"charset": "utf8mb4"}
    if os.environ.get("AI_DB_SSL_CA"):
        _db_options["ssl"] = {"ca": os.environ["AI_DB_SSL_CA"]}
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.mysql",
            "HOST": os.environ["AI_DB_HOST"],
            "PORT": os.environ.get("AI_DB_PORT", "3306"),
            "NAME": os.environ.get("AI_DB_NAME", "defaultdb"),
            "USER": os.environ.get("AI_DB_USER", ""),
            "PASSWORD": os.environ.get("AI_DB_PASSWORD", ""),
            "OPTIONS": _db_options,
            "CONN_MAX_AGE": 60,
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Manila"
USE_I18N = True
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

AI_SERVICE_API_KEY = os.environ.get("AI_SERVICE_API_KEY", "")
SERVED_MODEL_REGISTRY_PATH = BASE_DIR / "models" / "model_registry.json"
AI_MAINTENANCE_MODEL_REGISTRY_PATH = Path(os.environ.get(
    "AI_MAINTENANCE_MODEL_REGISTRY_PATH",
    # The test suite starts from a blocked registry; tests opt in to real artifacts.
    str(BASE_DIR / "ai_api" / "test_fixtures" / "blocked_registry.json")
    if sys.argv[1:2] == ["test"] else str(SERVED_MODEL_REGISTRY_PATH),
))
AI_MAINTENANCE_EVALUATION_MIN_GROUPS_PER_SPLIT = 2
AI_MAINTENANCE_MIN_BALANCED_ACCURACY = 0.60
AI_MAINTENANCE_MIN_MACRO_F1 = 0.60
CHATBOT_MODEL_PATH = Path(
    os.environ.get(
        "CHATBOT_MODEL_PATH",
        str(BASE_DIR / "trained_models" / "chatbot_nlp.joblib")
    )
)
AUTOMOTIVE_KNOWLEDGE_PATH = Path(
    os.environ.get(
        "AUTOMOTIVE_KNOWLEDGE_PATH",
        str(BASE_DIR / "data" / "automotive_knowledge.json")
    )
)
# Category-specific predictive models. Overridable so tests never touch the real ones.
CATEGORY_MODEL_ROOT = Path(os.environ.get("CATEGORY_MODEL_ROOT", str(BASE_DIR / "trained_models" / "category_models")))

KNOWLEDGE_INDEX_PATH = Path(
    os.environ.get(
        "KNOWLEDGE_INDEX_PATH",
        str(BASE_DIR / "data" / "processed" / "knowledge_index")
    )
)
SEMANTIC_EMBEDDING_MODEL_PATH = Path(os.environ.get(
    "SEMANTIC_EMBEDDING_MODEL_PATH",
    str(BASE_DIR / "trained_models" / "embeddings" / "multilingual-e5-small"),
))
SEMANTIC_EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-small"
SEMANTIC_EMBEDDING_MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
SEMANTIC_EMBEDDING_ONNX_FILE = os.environ.get(
    "SEMANTIC_EMBEDDING_ONNX_FILE", "onnx/model_qint8_avx512_vnni.onnx"
)
SEMANTIC_EMBEDDING_THREADS = int(os.environ.get("SEMANTIC_EMBEDDING_THREADS", "4"))
SEMANTIC_EMBEDDING_MAX_SEQUENCE_LENGTH = int(os.environ.get("SEMANTIC_EMBEDDING_MAX_SEQUENCE_LENGTH", "384"))
# Evidence thresholds for retrieved answers (heuristic support score, not a
# probability; tuned and checked by scripts/evaluate_chatbot.py).
CHATBOT_MIN_RETRIEVAL_SUPPORT = float(os.environ.get("CHATBOT_MIN_RETRIEVAL_SUPPORT", "0.45"))
CHATBOT_STRONG_RETRIEVAL_SUPPORT = float(os.environ.get("CHATBOT_STRONG_RETRIEVAL_SUPPORT", "0.70"))
# Conversation memory (names of vehicles discussed, last codes) is deleted after this many days without use.
CHAT_MEMORY_RETENTION_DAYS = int(os.environ.get("CHAT_MEMORY_RETENTION_DAYS", "30"))
# Service-need predictions below this calibrated probability are reported as low.
AI_PREDICTION_HIGH_RISK_PROBABILITY = float(os.environ.get("AI_PREDICTION_HIGH_RISK_PROBABILITY", "0.70"))

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"structured": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "structured"}},
    "loggers": {"ai_api": {"handlers": ["console"], "level": os.environ.get("AI_LOG_LEVEL", "INFO")}},
}
