
import re
from functools import lru_cache

import joblib
from django.conf import settings

from .fleet_service import analyze_fleet
from .automotive_knowledge import (
    format_knowledge_answer,
    retrieve_knowledge,
)
from .prediction_service import (
    load_maintenance_artifact,
    predict_vehicle,
    validate_vehicle,
)


LIVE_RENTAL_INTENTS = {
    "car_availability",
    "rental_pricing",
    "vehicle_features",
}
RENTAL_INTENTS = LIVE_RENTAL_INTENTS | {
    "booking_reservation",
    "cancellation_rescheduling",
    "deposits_payments",
    "pickup_return",
    "rental_duration",
    "rental_requirements",
    "rental_support",
}
SAFETY_CRITICAL_PHRASES = (
    "smell gasoline",
    "gasoline smell",
    "fuel smell",
    "gas smell",
    "fuel leak",
    "overheat",
    "overheating",
    "brakes failed",
    "brake failure",
    "brake warning",
    "abs warning",
    "reduced braking",
    "soft brake pedal",
    "oil pressure",
    "steering feels loose",
    "loose steering",
    "tire bulge",
    "tyre bulge",
    "exposed cords",
    "smoke from",
    "steam from",
)
MINIMUM_INTENT_CONFIDENCE = 0.40
AUTOMOTIVE_TERMS = {
    "automotive", "auto", "car", "cars", "vehicle", "vehicles", "rental",
    "rent", "reservation", "booking", "fleet", "maintenance", "engine",
    "brake", "brakes", "tire", "tires", "tyre", "tyres", "oil", "battery",
    "transmission", "steering", "suspension", "coolant", "diagnostic",
    "mechanic", "service", "alternator", "starter", "spark", "plug",
    "fuel", "radiator", "thermostat", "wheel", "clutch", "filter",
    "overheating", "misfire", "inspection", "repair",
    "available", "availability", "price", "prices", "cost", "rate", "rates",
    "document", "documents", "paper", "papers", "requirement", "requirements",
    "deposit", "payment", "payments", "pickup", "return", "cancel",
    "cancellation", "reschedule", "extend", "extension", "duration",
    "reservation", "book", "booking", "model", "prediction", "predict",
    "score", "risk", "warning", "symptom", "fault", "part", "parts",
    "support", "agent", "representative",
}


@lru_cache(maxsize=1)
def load_chatbot():
    path = settings.CHATBOT_MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Chatbot NLP model not found: {path}. "
            "Run python train_chatbot.py first."
        )

    artifact = joblib.load(path)

    if "pipeline" not in artifact:
        raise ValueError(
            "The chatbot model uses an outdated format. "
            "Run python train_chatbot.py to retrain it."
        )

    return artifact["pipeline"], artifact["responses"]


def build_response(intent, confidence, answer, data=None):
    return {
        "status": "success",
        "intent": intent,
        "confidence": round(float(confidence), 4),
        "answer": answer,
        "data": data
    }


def answer_chat(
    message,
    vehicle=None,
    vehicles=None,
    rental_context=None,
    diagnostic_context=None,
):
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string.")
    if rental_context is not None and not isinstance(rental_context, dict):
        raise ValueError("rental_context must be an object.")
    if diagnostic_context is not None and not isinstance(diagnostic_context, dict):
        raise ValueError("diagnostic_context must be an object.")

    pipeline, responses = load_chatbot()
    message = message.strip()

    probabilities = pipeline.predict_proba([message])[0]
    best_index = int(probabilities.argmax())
    intent = str(pipeline.classes_[best_index])
    confidence = float(probabilities[best_index])
    knowledge = retrieve_knowledge(message)

    database_context = (
        rental_context.get("database_context")
        if isinstance(rental_context, dict)
        else None
    )
    if (
        knowledge is not None
        and knowledge["document"].get("safety_critical") is True
        and _contains_safety_critical_phrase(message)
    ):
        return _knowledge_response(
            knowledge,
            confidence,
            diagnostic_context,
            message,
        )

    if isinstance(database_context, dict):
        answer, data = answer_database_context(
            database_context,
            message,
            rental_context,
        )
        return build_response(intent, confidence, answer, data)

    normalized_terms = set(re.findall(r"[a-z0-9]+", message.casefold()))
    capability_question = any(
        phrase in message.casefold()
        for phrase in (
            "what can you do",
            "what can this automotive ai do",
            "what are your capabilities",
            "what can this chatbot do",
            "what can you help with",
        )
    )
    unsupported_domain_query = (
        intent not in {"greeting", "goodbye"}
        and not capability_question
        and not normalized_terms.intersection(AUTOMOTIVE_TERMS)
        and knowledge is None
    )
    if (
        confidence < MINIMUM_INTENT_CONFIDENCE and knowledge is None
    ) or unsupported_domain_query:
        return build_response(
            "unknown",
            confidence,
            (
                "I am not confident that I understood your question. "
                "Try asking about rental availability, prices, requirements, "
                "bookings, vehicle features, maintenance recommendations, "
                "fleet priorities, or service schedules."
            )
        )

    if intent in LIVE_RENTAL_INTENTS:
        answer = answer_rental_question(intent, message, rental_context)
        return build_response(intent, confidence, answer)
    if intent in RENTAL_INTENTS:
        return build_response(
            intent,
            confidence,
            responses.get(intent, {}).get(
                "answer",
                "Please check with the rental team for the current policy."
            ),
        )

    if intent == "vehicle_prediction":
        if knowledge is not None and not _requests_model_prediction(message):
            return _knowledge_response(
                knowledge,
                confidence,
                diagnostic_context,
                message,
            )
        try:
            load_maintenance_artifact()
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            return build_response(
                intent,
                confidence,
                (
                    "Vehicle-specific maintenance predictions are unavailable: "
                    f"{error}"
                ),
                {"prediction_status": "unavailable", "reason": str(error)},
            )

        if vehicle is None and isinstance(diagnostic_context, dict):
            context_vehicle = diagnostic_context.get("vehicle")
            if isinstance(context_vehicle, dict):
                vehicle = _model_features_from_context(context_vehicle)

        if vehicle:
            try:
                validate_vehicle(vehicle, load_maintenance_artifact())
            except ValueError as error:
                if knowledge is not None and not _requests_model_prediction(message):
                    return _knowledge_response(
                        knowledge,
                        confidence,
                        diagnostic_context,
                        message,
                    )
                return build_response(
                    intent,
                    confidence,
                    (
                        "I cannot run the maintenance model because the supplied "
                        f"vehicle data is incomplete or invalid: {error}"
                    ),
                    {
                        "prediction_status": "not_run",
                        "diagnostic_context_available": diagnostic_context is not None,
                    },
                )
        elif knowledge is not None and not _requests_model_prediction(message):
            return _knowledge_response(
                knowledge,
                confidence,
                diagnostic_context,
                message,
            )
        else:
            return build_response(
                intent,
                confidence,
                (
                    "Please include a vehicle object containing the required "
                    "model features so I can run a maintenance prediction."
                )
            )

        result = predict_vehicle(vehicle)
        recommendations = result.get("recommendations", [])

        names = [
            item.get("label", "Unknown category")
            for item in recommendations
        ]

        if names:
            answer = (
                "The model recommends checking: "
                + ", ".join(names)
                + ". These are model-based recommendations, not confirmed faults."
            )
        else:
            answer = (
                "The model did not flag any trained maintenance category "
                "for recommendation at the configured thresholds. This does "
                "not guarantee that the vehicle is fault-free."
            )

        return build_response(
            intent,
            confidence,
            answer,
            result
        )

    if intent == "fleet_analysis":
        if not vehicles:
            if knowledge is not None:
                response = _knowledge_response(
                    knowledge,
                    confidence,
                    diagnostic_context,
                    message,
                )
                response["answer"] += (
                    " I cannot rank actual vehicles because no authorized "
                    "maintenance-model records were supplied, so no fleet "
                    "analysis was run."
                )
                response["data"]["fleet_analysis_status"] = "not_run"
                return response
            return build_response(
                intent,
                confidence,
                "No eligible fleet vehicle records were supplied, so no fleet analysis was run.",
                {"fleet_analysis_status": "not_run"},
            )

        try:
            load_maintenance_artifact()
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            return build_response(
                intent,
                confidence,
                (
                    "Fleet maintenance analysis is unavailable because the "
                    f"maintenance model is not eligible: {error}"
                ),
                {
                    "fleet_analysis_status": "unavailable",
                    "prediction_status": "unavailable",
                    "reason": str(error),
                },
            )

        result = analyze_fleet(vehicles)
        summary = result.get("fleet_summary", {})

        analyzed = summary.get("analyzed_vehicles", 0)
        high_priority = summary.get("high_priority_vehicles", 0)

        answer = (
            f"I analyzed {analyzed} vehicles. "
            f"{high_priority} are marked high priority based on "
            "the current model output."
        )

        return build_response(
            intent,
            confidence,
            answer,
            result
        )

    if knowledge is not None:
        return _knowledge_response(
            knowledge,
            confidence,
            diagnostic_context,
            message,
        )

    if intent == "explain_score":
        answer = (
            "The score estimates how strongly the model associates a "
            "vehicle's features with a maintenance label in its training "
            "data. The service compares that score with a configured "
            "threshold. It is not a calibrated guarantee of component failure."
        )

        return build_response(
            intent,
            confidence,
            answer
        )

    if intent == "safety":
        answer = (
            "AI recommendations cannot establish whether a vehicle is safe "
            "to drive. If braking, steering, tire, or engine-temperature "
            "problems occur, stop using the vehicle when appropriate and "
            "arrange an inspection by a qualified technician."
        )

        return build_response(
            intent,
            confidence,
            answer
        )

    answer = responses.get(intent, {}).get(
        "answer",
        (
            "I can help with rental questions, maintenance recommendations, "
            "and fleet priorities."
        )
    )

    return build_response(
        intent,
        confidence,
        answer
    )


def _model_features_from_context(context_vehicle):
    features = {
        key: value
        for key, value in context_vehicle.items()
        if key in {
            "vehicle_type", "brand", "model", "engine_type", "make_year",
            "region", "mileage_range", "mileage",
        }
    }
    for model_key, aliases in (
        ("brand", ("make",)),
        ("make_year", ("year",)),
        ("mileage", ("current_mileage_km",)),
    ):
        if model_key not in features:
            for alias in aliases:
                if context_vehicle.get(alias) is not None:
                    features[model_key] = context_vehicle[alias]
                    break
    return features


def _requests_model_prediction(message):
    normalized = message.casefold()
    return any(
        term in normalized
        for term in ("predict", "prediction", "model", "score", "run maintenance")
    )


def _contains_safety_critical_phrase(message):
    normalized = " ".join(re.findall(r"[a-z0-9]+", message.casefold()))
    return any(
        " ".join(re.findall(r"[a-z0-9]+", phrase)) in normalized
        for phrase in SAFETY_CRITICAL_PHRASES
    )


def _knowledge_response(knowledge, confidence, diagnostic_context, message):
    document = knowledge["document"]
    documents = knowledge.get("documents", [document])
    source_references = {
        (
            source.get("name"),
            source.get("reference"),
            source.get("url"),
        )
        for source_document in documents
        for source in source_document.get("sources", [])
        if isinstance(source, dict)
    }
    answer = format_knowledge_answer(
        knowledge,
        diagnostic_context=diagnostic_context,
        query=message,
    )
    return build_response(
        document["category"],
        confidence,
        answer,
        {
            "knowledge_id": document["id"],
            "knowledge_ids": [item["id"] for item in documents],
            "retrieval_score": knowledge["score"],
            "retrieved_passage_count": len(knowledge.get("passages", [])),
            "source_count": len(source_references),
            "source_references_fetched": 0,
            "retrieval_score_interpretation": (
                "Heuristic lexical/vector relevance score for this query; "
                "not a probability or calibrated confidence."
            ),
            "intent_confidence_interpretation": (
                "Top chatbot-classifier predict_proba output; not calibrated "
                "or validated as a probability of correctness."
            ),
        },
    )


def answer_rental_question(intent, message, rental_context):
    if not isinstance(rental_context, dict):
        return (
            "I cannot access current rental records in this request. "
            "Please check the vehicle listings or contact the rental team "
            "for verified information."
        )

    vehicles = rental_context.get("vehicles")
    if not isinstance(vehicles, list):
        vehicles = []

    if intent == "car_availability":
        period = rental_context.get("availability_period")
        checked_period = isinstance(period, dict) and period.get("status") == "checked"
        if isinstance(period, dict):
            if period.get("status") == "needs_details":
                missing = period.get("missing", "rental dates")
                return f"Please provide the {missing} so I can check reservation overlaps."
            if period.get("status") == "invalid_dates":
                return "Those dates are invalid or in the past. Please provide a valid future pickup and return date."

        authorized = rental_context.get("vehicle_status_authorized") is True
        available = []
        status_data_available = False
        price_limit = requested_price_limit(message)

        for item in vehicles:
            if not isinstance(item, dict):
                continue
            availability = item.get("is_available", item.get("available"))
            status = item.get("status")
            if isinstance(availability, bool):
                status_data_available = True
                duration_satisfied = item.get("minimum_duration_satisfied")
                if availability and duration_satisfied is not False:
                    price = valid_price(item.get("daily_rental_price"))
                    if price_limit is None or (price is not None and price <= price_limit):
                        available.append(item)
            elif authorized and not checked_period and isinstance(status, str):
                status_data_available = True
                if status.casefold() == "available":
                    available.append(item)

        if not status_data_available:
            if rental_context.get("vehicle_count") == 0:
                if checked_period:
                    return (
                        "No listed vehicle in the supplied records is available "
                        f"for {period.get('start_date')} through {period.get('end_date')} "
                        "(the return date is exclusive)."
                    )
                return "No vehicles are currently listed as available in the supplied records."
            return (
                "The current vehicle records do not include availability "
                "information I am authorized to report. Please check the "
                "vehicle listings for current availability. A listing also "
                "does not confirm availability for specific dates."
            )
        if not available:
            if price_limit is not None:
                return (
                    "No vehicle in the supplied availability records meets the "
                    f"daily-rate limit of {rental_context.get('currency', 'PHP')} {price_limit:,.2f}."
                )
            if checked_period:
                return (
                    "No listed vehicle in the supplied records is available "
                    f"for {period.get('start_date')} through {period.get('end_date')} "
                    "(the return date is exclusive)."
                )
            return (
                "The supplied listings do not show a vehicle marked available. "
                "Availability for a specific rental period requires pickup and return dates."
            )

        if "cheapest" in message.casefold():
            available.sort(
                key=lambda item: valid_price(item.get("daily_rental_price"))
                if valid_price(item.get("daily_rental_price")) is not None
                else float("inf")
            )
        include_rates = price_limit is not None or "cheapest" in message.casefold()
        names = []
        for item in available:
            name = vehicle_name(item)
            if not name:
                continue
            price = valid_price(item.get("daily_rental_price"))
            if include_rates and price is not None:
                names.append(f"{name} ({rental_context.get('currency', 'PHP')} {price:,.2f}/day)")
            else:
                names.append(name)
        if not names:
            return (
                "The supplied records identify available vehicles but do not "
                "include their names. Please check the vehicle listings."
            )
        if checked_period:
            answer = (
                f"Vehicles available in the supplied reservation records from "
                f"{period.get('start_date')} to {period.get('end_date')} "
                "(return date exclusive): " + ", ".join(names) + "."
            )
        else:
            answer = "Vehicles whose current listings are marked available: " + ", ".join(names) + "."
        if rental_context.get("vehicle_list_truncated") is True:
            answer += " This list may be incomplete."
        if not checked_period:
            answer += " This does not confirm availability for specific dates; provide pickup and return dates for an overlap check."
        return answer

    if intent == "rental_pricing":
        candidates = select_vehicle_candidates(vehicles, message)
        limit = requested_price_limit(message)
        if limit is not None:
            candidates = [
                item for item in candidates
                if valid_price(item.get("daily_rental_price")) is not None
                and float(item["daily_rental_price"]) <= limit
            ]
        if "cheapest" in message.casefold():
            candidates = sorted(
                candidates,
                key=lambda item: valid_price(item.get("daily_rental_price"))
                if valid_price(item.get("daily_rental_price")) is not None
                else float("inf"),
            )
        prices = []
        for item in candidates:
            price = valid_price(item.get("daily_rental_price"))
            if price is None:
                continue
            name = vehicle_name(item)
            if not name:
                continue
            currency = rental_context.get("currency", "PHP")
            amount = f"{price:,.2f}".rstrip("0").rstrip(".")
            duration = rental_context.get("rental_duration_days")
            minimum = item.get("minimum_rental_duration")
            if isinstance(duration, int) and duration > 0:
                if isinstance(minimum, int) and minimum > duration:
                    total = f"; requested {duration}-day period is below the listed {minimum}-day minimum"
                else:
                    total = f"; {currency} {price * duration:,.2f} base-rate estimate for {duration} days"
            else:
                total = ""
            deposit = valid_price(item.get("security_deposit"))
            deposit_text = f"; listed security deposit {currency} {deposit:,.2f}" if deposit is not None and deposit > 0 else ""
            prices.append(f"{name}: {currency} {amount} per day{total}{deposit_text}")
        if not prices:
            if limit is not None:
                return (
                    "No matching vehicle in the supplied listings has a verified "
                    f"daily rate at or below {rental_context.get('currency', 'PHP')} {limit:,.2f}."
                )
            if candidates == []:
                return "I could not match that vehicle or category to the supplied listings. Please check the exact model name."
            return (
                "I cannot verify a daily rental price from the current records. "
                "Please check the vehicle listing for its current rate."
            )
        answer = "Daily rates in the supplied vehicle records: " + "; ".join(prices) + "."
        if rental_context.get("vehicle_list_truncated") is True:
            answer += " The list may be incomplete."
        if isinstance(rental_context.get("rental_duration_days"), int):
            answer += " The total is an estimate based only on the listed daily rate; confirm fees, deposits, and final pricing with the rental team."
        else:
            answer += " Confirm the final quote and any fees with the rental team."
        return answer

    candidates = select_vehicle_candidates(vehicles, message)
    if not candidates and rental_context.get("requested_vehicle_id") is not None:
        return "No matching vehicle is present in the listings this account is authorized to view."
    details = []
    for item in candidates:
        name = vehicle_name(item)
        if not name:
            continue
        features = []
        for key, label in (
            ("year", "year"),
            ("transmission", "transmission"),
            ("fuel_type", "fuel"),
            ("seating_capacity", "seats"),
        ):
            value = item.get(key)
            if value is not None and value != "":
                features.append(f"{label}: {value}")
        for key, label in (
            ("air_conditioning", "air conditioning"),
            ("gps_navigation", "GPS navigation"),
        ):
            value = item.get(key)
            if isinstance(value, bool):
                features.append(f"{label}: {'yes' if value else 'no'}")
        additional = item.get("additional_features")
        if isinstance(additional, list):
            features.extend(str(feature) for feature in additional if isinstance(feature, str))
        if features:
            details.append(f"{name} ({', '.join(features)})")
    if not details:
        return (
            "The supplied vehicle records do not include verified feature "
            "details. Please check the vehicle listing for its specifications."
        )
    answer = "Vehicle details in the supplied records: " + "; ".join(details) + "."
    if rental_context.get("vehicle_list_truncated") is True:
        answer += " The list may be incomplete."
    return answer


def answer_database_context(database_context, message, rental_context):
    kind = database_context.get("kind")
    if kind == "vehicle_inventory":
        purpose = database_context.get("purpose")
        intent = (
            "car_availability"
            if purpose == "availability"
            else "rental_pricing"
            if purpose == "pricing"
            else "vehicle_features"
        )
        answer = answer_rental_question(intent, message, rental_context)
        return answer, {
            "data_source": "authenticated application vehicle records",
            "data_as_of": rental_context.get("data_as_of"),
            "vehicle_ids": [
                item.get("id")
                for item in rental_context.get("vehicles", [])
                if isinstance(item, dict) and item.get("id") is not None
            ],
            "availability_period": rental_context.get("availability_period"),
        }

    if kind == "customer_bookings":
        booking_data = database_context.get("booking_data")
        records = booking_data.get("reservations", []) if isinstance(booking_data, dict) else []
        if not isinstance(records, list) or not records:
            answer = "The authenticated account has no reservation records in the supplied database results."
        else:
            lines = []
            for record in records:
                if not isinstance(record, dict):
                    continue
                name = vehicle_name(record) or "Vehicle"
                line = (
                    f"Reservation #{record.get('id')}: {name}, "
                    f"{record.get('start_date')} to {record.get('end_date')}, "
                    f"status: {record.get('status')}, "
                    f"{record.get('rental_days')} day(s), "
                    f"{rental_context.get('currency', 'PHP')} {float(record.get('total_amount') or 0):,.2f} stored total."
                )
                lines.append(line)
            answer = "Your reservation records: " + " ".join(lines)
            if booking_data.get("total_count", len(records)) > len(records):
                answer += f" Showing the latest {len(records)} of {booking_data['total_count']} records."
        return answer, {
            "data_source": "reservations belonging to the authenticated account",
            "data_as_of": database_context.get("data_as_of"),
            "reservation_ids": [
                record.get("id") for record in records
                if isinstance(record, dict) and record.get("id") is not None
            ],
        }

    if kind == "admin_fleet":
        counts = rental_context.get("vehicle_counts_by_status")
        counts = counts if isinstance(counts, dict) else {}
        total = sum(value for value in counts.values() if isinstance(value, int))
        lines = [f"{status}: {count}" for status, count in sorted(counts.items())]
        answer = (
            f"The fleet database contains {total} vehicle(s). "
            + ("Status counts: " + ", ".join(lines) + "." if lines else "No status counts were returned.")
        )
        if rental_context.get("vehicle_list_truncated") is True:
            answer += " The vehicle detail list is limited to the newest 100 records."
        return answer, {
            "data_source": "administrator-authorized vehicle records",
            "data_as_of": rental_context.get("data_as_of"),
            "vehicle_counts_by_status": counts,
            "vehicle_ids": [
                item.get("id")
                for item in rental_context.get("vehicles", [])
                if isinstance(item, dict) and item.get("id") is not None
            ],
        }

    if kind == "admin_maintenance":
        records = database_context.get("records")
        records = records if isinstance(records, dict) else {}
        legacy = records.get("legacy_service_records", [])
        verified = records.get("verified_evidence_records", [])
        legacy = legacy if isinstance(legacy, list) else []
        verified = verified if isinstance(verified, list) else []
        lines = []
        for item in legacy:
            if not isinstance(item, dict):
                continue
            name = vehicle_name(item) or f"Vehicle #{item.get('vehicleID')}"
            description = str(item.get("description") or "").strip()
            lines.append(
                f"Legacy service log #{item.get('maintenanceID')} for {name}: "
                f"{item.get('maintenanceType')} on {item.get('maintenanceDate')} "
                f"(status {item.get('maintenanceStatus')})"
                + (f"; note: {description[:300]}" if description else "")
                + ". This is a service-log entry, not a verified fault label."
            )
        for item in verified:
            if not isinstance(item, dict):
                continue
            name = vehicle_name(item) or f"Vehicle #{item.get('vehicle_id')}"
            details = []
            for key, label in (
                ("component", "component"),
                ("symptoms", "symptoms"),
                ("inspection_findings", "findings"),
                ("dtc_codes", "DTCs"),
                ("confirmed_cause", "confirmed cause"),
                ("repair_action", "repair"),
                ("repair_outcome", "outcome"),
            ):
                value = str(item.get(key) or "").strip()
                if value:
                    details.append(f"{label}: {value[:250]}")
            lines.append(
                f"Verified evidence record #{item.get('id')} for {name}: "
                f"{item.get('record_type')} on {item.get('recorded_at')}"
                + ("; " + "; ".join(details) if details else "")
                + (f"; source: {str(item.get('source'))[:160]}" if item.get("source") else "")
                + "."
            )

        if lines:
            answer = "Maintenance records returned from the authorized database:\n- " + "\n- ".join(lines)
        elif records.get("verified_evidence_table_available") is False:
            answer = (
                "No legacy service records were returned. The verified maintenance-evidence "
                "table is not installed, so those records cannot be queried yet."
            )
        else:
            answer = "No matching maintenance records were returned from the authorized database."
        return answer, {
            "data_source": "administrator-authorized maintenance records",
            "data_as_of": database_context.get("data_as_of"),
            "legacy_record_ids": [
                item.get("maintenanceID") for item in legacy
                if isinstance(item, dict) and item.get("maintenanceID") is not None
            ],
            "verified_record_ids": [
                item.get("id") for item in verified
                if isinstance(item, dict) and item.get("id") is not None
            ],
            "verified_evidence_table_available": records.get("verified_evidence_table_available"),
        }

    return (
        "I could not identify an authorized database context for that request.",
        {"data_source": "none", "database_context_status": "unsupported"},
    )


def select_vehicle_candidates(vehicles, message):
    candidates = [item for item in vehicles if isinstance(item, dict)]
    exact_matches = [item for item in candidates if vehicle_matches(item, message)]
    if exact_matches:
        return exact_matches

    categories = {
        "sedan": ("sedan",),
        "suv": ("suv", "sport utility"),
        "van": ("van", "minivan", "minibus"),
        "minivan": ("minivan",),
        "pickup": ("pickup", "pick-up", "truck"),
        "truck": ("pickup", "truck"),
        "motorcycle": ("motorcycle", "motorbike"),
    }
    query = message.casefold()
    selected = next(
        (terms for category, terms in categories.items() if re.search(rf"\b{re.escape(category)}\b", query)),
        None,
    )
    if selected:
        category_matches = [
            item for item in candidates
            if any(
                term in str(item.get("category") or "").casefold()
                or term in vehicle_name(item).casefold()
                for term in selected
            )
        ]
        make_terms = [
            str(item.get("make") or "").casefold()
            for item in candidates
            if str(item.get("make") or "").casefold() in query
        ]
        if make_terms:
            category_matches = [
                item for item in category_matches
                if str(item.get("make") or "").casefold() in make_terms
            ]
        return category_matches

    make_matches = [
        item for item in candidates
        if str(item.get("make") or "").casefold()
        and str(item.get("make") or "").casefold() in query
    ]
    if make_matches:
        return make_matches

    model_matches = [
        item for item in candidates
        if str(item.get("model") or "").casefold()
        and re.search(rf"\b{re.escape(str(item.get('model')).casefold())}\b", query)
    ]
    if model_matches:
        return model_matches

    named_target = re.search(
        r"\b(?:a|an|the|is|for|of|does|about)\s+"
        r"([A-Z][A-Za-z0-9-]+(?:\s+[A-Z][A-Za-z0-9-]+)?)\b",
        message,
    )
    if named_target and not (
        "cheapest" in query
        or re.search(r"\b(?:all|any|available|vehicles|cars|fleet)\b", query)
    ):
        return []

    return candidates


def valid_price(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def requested_price_limit(message):
    match = re.search(
        r"\b(?:under|below|less than|at most|maximum|max)\s*(?:php|₱)?\s*([\d,]+(?:\.\d{1,2})?)",
        message,
        re.IGNORECASE,
    )
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def vehicle_name(vehicle):
    name = " ".join(
        str(vehicle.get(field, "")).strip()
        for field in ("make", "model")
        if vehicle.get(field) is not None
    ).strip()
    return name


def vehicle_matches(vehicle, message):
    name = vehicle_name(vehicle)
    return bool(name and name.casefold() in message.casefold())
