import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "local-development-only-change-this-secret"
)
DEBUG = os.environ.get("DJANGO_DEBUG", "True").lower() == "true"
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver"]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "ai_api",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "automotive_ai.urls"
TEMPLATES = []
WSGI_APPLICATION = "automotive_ai.wsgi.application"
ASGI_APPLICATION = "automotive_ai.asgi.application"

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
AI_MAINTENANCE_MODEL_REGISTRY_PATH = (
    BASE_DIR / "models" / "model_registry.json"
)
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
