import hmac
from functools import wraps

from django.conf import settings
from django.http import JsonResponse


def require_api_key(view_func):
    @wraps(view_func)
    def wrapped(request, *args, **kwargs):
        expected = settings.AI_SERVICE_API_KEY
        if expected:
            supplied = request.headers.get("X-AI-Service-Key", "")
            if not hmac.compare_digest(supplied, expected):
                return JsonResponse(
                    {"error": "Unauthorized AI service request."},
                    status=401
                )
        return view_func(request, *args, **kwargs)
    return wrapped
