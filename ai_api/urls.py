from django.urls import path
from . import views

urlpatterns = [
    path("health/", views.health, name="ai-health"),
    path("capabilities/", views.capabilities, name="ai-capabilities"),
    path("predict/", views.predict, name="ai-predict"),
    path("fleet/analyze/", views.analyze_fleet_view, name="ai-fleet-analyze"),
    path("chat/", views.chat, name="ai-chat"),
]
