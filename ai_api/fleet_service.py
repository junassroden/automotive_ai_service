from .prediction_service import predict_vehicle


def analyze_fleet(vehicles):
    if not isinstance(vehicles, list):
        raise ValueError("The vehicles field must be a JSON array.")

    ranked = []
    rejected = []
    model_version = None

    for index, vehicle in enumerate(vehicles):
        vehicle_id = vehicle.get("id", index + 1) if isinstance(vehicle, dict) else index + 1
        try:
            prediction = predict_vehicle(vehicle)
            model_version = model_version or prediction.get("model_version")
            recommendations = prediction["recommendations"]

            priority_rank = {"high": 3, "medium": 2, "low": 1}
            top_priority = max(
                (item["priority"] for item in recommendations),
                key=lambda value: priority_rank.get(value, 0),
                default="low"
            )

            ranked.append({
                "vehicle_id": vehicle_id,
                "vehicle": prediction["vehicle"],
                "priority": top_priority,
                "recommendation_count": len(recommendations),
                "recommendations": recommendations,
                "prediction_summary": prediction["summary"]
            })
        except (ValueError, TypeError) as error:
            rejected.append({
                "vehicle_id": vehicle_id,
                "error": str(error)
            })

    ranked.sort(
        key=lambda item: (
            {"high": 3, "medium": 2, "low": 1}.get(item["priority"], 0),
            item["recommendation_count"]
        ),
        reverse=True
    )

    for position, item in enumerate(ranked, start=1):
        item["rank"] = position

    return {
        "status": "success",
        "fleet_summary": {
            "submitted_vehicles": len(vehicles),
            "analyzed_vehicles": len(ranked),
            "rejected_vehicles": len(rejected),
            "vehicles_with_recommendations": sum(
                item["recommendation_count"] > 0 for item in ranked
            ),
            "high_priority_vehicles": sum(
                item["priority"] == "high" for item in ranked
            ),
            "medium_priority_vehicles": sum(
                item["priority"] == "medium" for item in ranked
            )
        },
        "model_version": model_version,
        "prioritized_vehicles": ranked,
        "rejected_vehicles": rejected
    }
