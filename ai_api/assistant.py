"""Question routing in front of the intent/knowledge chatbot core.

Order of evidence, highest first:
  1. database_record  - facts the authenticated Back-End supplied for this viewer
  2. model_prediction - output of a registered, evaluated model
  3. reference        - retrieved reference text (DTC/PID definitions, curated
                        knowledge, system guide, recorded service statistics)
  4. general_guidance - the core chatbot's fixed policy answers
Every response carries ``data.evidence`` naming which of these it used, the
cited sources, and a retrieval support score. ``confidence`` keeps its old
meaning (intent-classifier output) for API compatibility and is not a
correctness measure.

The viewer's role comes only from the Back-End (``viewer_role`` /
``conversation_scope``), never from message text. Administrator-only database
contexts and documents are refused to everyone else here as well as in the
Back-End.
"""
import logging
import re
import time

from django.conf import settings
from django.db import DatabaseError

from . import chatbot_service as core
from . import grounded_answers as grounded
from . import planner
from .fleet_service import analyze_fleet
from .language import correct_typos, normalize_message
from .prediction_service import MaintenancePredictionsUnavailableError
from .rag import KnowledgeIndexUnavailable, extract_entities, lookup_parts, retrieve


logger = logging.getLogger(__name__)
ADMIN_CONTEXT_KINDS = {"admin_fleet", "admin_maintenance", "admin_fleet_risk"}
SYSTEM_GUIDE_INTENTS = core.RENTAL_INTENTS - core.LIVE_RENTAL_INTENTS | {"capabilities"}
FOLLOW_UP = re.compile(
    r"^(?:and|what about|how about|also|then)\b|\b(?:it|that|this|those|these)\b.*\b(?:code|fault|mean|cause|causes|symptom|symptoms|fix|repair|cost|serious|safe)\b"
    r"|^(?:causes?|symptoms?|fix(?:es)?|how (?:do i|to) fix|is it (?:serious|safe))\??$",
    re.IGNORECASE,
)
# Free-text retrieval never answers from DTC definitions: fuzzy matches between a
# symptom description and code titles are unreliable. DTCs are looked up only
# when the code itself appears in the question.
FREE_TEXT_SOURCES = {"system_documentation", "pid_reference", "curated_reference", "diagnostic_flowchart", "service_statistics"}
SPECIFIC_SOURCE_MIN_SUPPORT = 0.60
ASKS_FOR_CODE = re.compile(r"\b(?:this|that|the|my)\s+(?:dtc|code|fault code|error code|trouble code)\b", re.IGNORECASE)
GREETING = re.compile(
    r"^(?:hi+|hiya|hello+|hey+|heya|good\s+(?:morning|afternoon|evening|day)|greetings|kumusta|kamusta"
    r"|magandang\s+(?:umaga|hapon|gabi|araw))\b(?P<rest>.*)$",
    re.IGNORECASE,
)
# Words that may follow a greeting without turning it into a question ("Hiya, quick question for you").
GREETING_FILLER = {
    "there", "po", "again", "quick", "question", "questions", "for", "you", "can", "could", "would", "help", "me",
    "i", "ask", "a", "something", "have", "got", "how", "are", "please", "hope", "doing", "well", "ok", "okay",
    "the", "assistant", "bot", "team", "everyone", "just", "wanted", "to", "one", "little", "need", "some",
}
# Greetings and fillers around a question ("Good morning, how much is the Vios po?") are not part of it.
COURTESY_LEAD = re.compile(
    r"^\s*(?:hi+|hiya|hello+|hey+|heya|good\s+(?:morning|afternoon|evening|day)|magandang\s+(?:umaga|hapon|gabi|araw)|kumusta|kamusta"
    r"|excuse\s+me|quick\s+question|question|ask\s+ko\s+lang|tanong\s+(?:ko\s+)?lang|tanong|uh+m*|um+|uh+|ah+|po|pls|plz|please|sir"
    r"|ma'?am|kuya|ate|boss|ok(?:ay)?|so|yo)"
    r"(?:\s*[,:;.!?-]+\s*|\s+)",
    re.IGNORECASE,
)
COURTESY_WORDS = {"hi", "hello", "hey", "hiya", "heya", "good", "morning", "afternoon", "evening", "day", "magandang", "umaga", "hapon",
                  "gabi", "araw", "kumusta", "kamusta", "excuse", "uhm", "um", "uh", "pls", "plz", "please", "sir", "ma'am", "maam",
                  "kuya", "ate", "boss", "yo"}
THANKS_WORDS = {"thanks", "thank", "you", "u", "thx", "ty", "salamat", "lot", "so", "much", "naman", "sana", "lang"}
FILLER_LEAD = re.compile(r"^\s*(?:(?:uh+m*|um+|uh+|ah+|po|pls|plz|please|ok(?:ay)?|so|excuse\s+me|quick\s+question|question|ask\s+ko\s+lang"
                         r"|tanong\s+(?:ko\s+)?lang|sir|ma'?am|kuya|ate|boss)(?:\s*[,:;.!?-]+\s*|\s+))+", re.IGNORECASE)
COURTESY_TAIL = re.compile(r"(?:[\s,]+(?:please|pls|plz|po|thanks|thank\s+you|thank\s+u|thx|ty|salamat|naman|sana|lang))+\s*([?!.]*)\s*$",
                           re.IGNORECASE)


def strip_courtesy(message):
    """The question without greetings or fillers around it; a message that is only a greeting or thanks is kept as typed."""
    text = message
    while True:
        match = COURTESY_LEAD.match(text)
        if not match or not re.search(r"[A-Za-z0-9]", text[match.end():]):
            break
        text = text[match.end():]
    tail = COURTESY_TAIL.search(text)
    # "budget ko 2000 lang" keeps its "lang": after a number it means "only".
    if tail and re.search(r"[A-Za-z0-9]", text[:tail.start()]) and not re.search(r"\d\s*$", text[:tail.start()]):
        text = text[:tail.start()] + tail.group(1)
    # "Hello there", "Hi, hi pls" or "hi, kumusta thanks" is still just a greeting (fillers before it dropped).
    if all(word in GREETING_FILLER | COURTESY_WORDS | THANKS_WORDS for word in re.findall(r"[a-z']+", text.casefold())):
        return FILLER_LEAD.sub("", message) or message
    return text.strip() or message


GREETING_ANSWERS = {
    "customer": ("Hello! I can help with vehicle prices and availability, your own reservations, how to book or cancel, "
                 "and general car questions such as warning lights or trouble codes. What would you like to know?"),
    "admin": ("Hello! I can help with fleet status, maintenance records, vehicle prices and availability, how to use the "
              "admin pages, and diagnostic trouble codes. What would you like to know?"),
}
# Rental-policy and how-to-use-the-system questions are answered from the system guide only, so they
# never pick up a mechanical note that happens to share words ("provide", "before", "car").
RENTAL_POLICY = re.compile(
    r"\b(?:rent|rental|renting|book|booking|bookings|reserve|reservation|reservations|deposit|deposits|payment|pay"
    r"|licen[cs]e|requirements?|provide|bring|documents?|cancel|reschedule|pickup|account|profile"
    r"|discounts?|promos?|promotions?|free\s+days?|fees?|cards?|credit|debit|gcash|g-cash|maya|paymaya|cash|online\s+payment|accept"
    r"|sign\s*up|signup|register|papers)\b",
    re.IGNORECASE,
)
SYSTEM_PROCEDURE = re.compile(
    r"\b(?:how\s+(?:do|can|could)\s+(?:i|we)|how\s+to|steps?\s+(?:to|for)|what\s+are\s+the\s+steps)\b"
    r".*\b(?:add|register|create|update|edit|change|delete|remove|upload)\b.*\b(?:vehicles?|cars?)\b",
    re.IGNORECASE,
)
# "How do I add a maintenance record?" is about the admin pages, although it contains a mechanical word.
RECORD_PROCEDURE = re.compile(
    r"\b(?:how\s+(?:do|can|could)\s+(?:i|we)|how\s+to|steps?\s+(?:to|for))\b.*\b(?:add|create|record|log|enter|upload|edit|update|delete)\b"
    r".*\b(?:maintenance|service|inspection|repair)\s+(?:records?|entry|entries|logs?|history|evidence|photos?)\b",
    re.IGNORECASE,
)
# Vehicle-inventory management that only administrators can do.
VEHICLE_ADMINISTRATION = re.compile(
    r"\b(?:how\s+(?:do|can|could)\s+(?:i|we)|how\s+to|steps?\s+(?:to|for)|what\s+are\s+the\s+steps)\b"
    r".*\b(?:add|register|create|delete|remove|upload)\b.*\b(?:vehicles?|cars?)\b",
    re.IGNORECASE,
)
MECHANICAL = re.compile(
    r"\b(?:engine|brakes?|braking|smoke|smoking|noise|squeak\w*|squeal\w*|grind\w*|leak\w*|overheat\w*|tires?|tyres?"
    r"|oil|battery|alternator|starter|misfir\w*|stall\w*|vibrat\w*|warning light|check engine|coolant|transmission"
    r"|exhaust|won'?t start|inspect\w*|parts?|repair\w*|maintenance|diagnos\w*|mechanic\w*|fault\w*|symptoms?)\b",
    re.IGNORECASE,
)
ADMIN_RECORDS_REQUEST = re.compile(
    r"\b(?:fleet\s+(?:vehicles?|records?|status|maintenance|data|risk)"
    r"|(?:maintenance|service|repair|inspection)\s+(?:records?|history|logs?|status|evidence|predictions?)"
    r"|due\s+for\s+(?:maintenance|service)|high[- ]risk|riskiest|(?:in|under)\s+(?:maintenance|repair))\b",
    re.IGNORECASE,
)
ADMIN_ONLY_ANSWER = (
    "Fleet and maintenance records are limited to administrators, so I can't show them for this account. "
    "I can help with vehicle listings, prices, availability, your own reservations and general car questions."
)
PRIVATE_DATA_ANSWER = (
    "I can't share other customers' reservations, names, phone numbers, emails or other personal details. "
    "Your role comes from your signed-in account, and the assistant only shows your own bookings. "
    "Administrators manage customer records in the admin pages, not in chat."
)
# "Hi, I need help with a booking": no detail yet, so ask what kind of help.
HELP_REQUEST = re.compile(
    r"^(?:(?:hi+|hello+|hey+|hiya|good\s+(?:morning|afternoon|evening)|kumusta|magandang\s+\w+)[\s,!.]*)?"
    r"(?:i\s+(?:need|want|would like)\s+(?:some\s+)?help|can\s+you\s+help(?:\s+me)?|help\s+me|i\s+have\s+a\s+question)"
    r"\s*(?:with|about|on|regarding)?\s*(?:a|an|my|the)?\s*(?P<topic>bookings?|reservations?|rentals?|renting|cars?|vehicles?)?[\s?!.]*$",
    re.IGNORECASE,
)
VEHICLE_MANAGEMENT_ANSWER = (
    "Vehicles are added, edited and removed by DriveRent administrators; customer accounts can browse vehicles, "
    "see their details and daily rates, and reserve them from Vehicles (/users/vehicles)."
)
UNRECORDED_POLICY = "This policy is not recorded in the system. Please confirm it with the rental team."
CREDENTIALS_ANSWER = ("I can't share passwords, keys or other account credentials, for any account. "
                      "If you can't sign in to your own account, contact the rental team.")
# General safety steps only: the system records no accident, insurance or liability procedure.
ACCIDENT_ANSWER = (
    "If you are in an accident:\n"
    "- Stop in a safe place if you can, switch on the hazard lights and check whether anyone is hurt.\n"
    "- Call emergency services if anyone is injured or the vehicle is a hazard to traffic.\n"
    "- Follow local requirements for reporting the accident, and note the time, place, other vehicles and any witnesses.\n"
    "- Notify the rental team as soon as you can, and follow your rental agreement.\n"
    "The system does not record accident procedures, insurance coverage or liability terms, so confirm those with the rental team."
)
GUIDE_NOT_FOUND = UNRECORDED_POLICY
OUT_OF_SCOPE_ANSWER = (
    "I can only answer from this system's data and its automotive references, and I found nothing "
    "relevant enough to answer that reliably. I can help with vehicle availability, prices and bookings, "
    "your own booking records, how to use the system, diagnostic trouble codes, maintenance and symptoms, and "
    "(for administrators) fleet maintenance records and model-based service-need predictions."
)


def resolve_role(viewer_role=None, conversation_scope=None):
    if viewer_role in {"admin", "customer"}:
        return viewer_role
    if isinstance(conversation_scope, str) and conversation_scope.startswith("admin:"):
        return "admin"
    return "customer"


def answer_chat(message, vehicle=None, vehicles=None, rental_context=None, diagnostic_context=None,
                viewer_role=None, conversation_id=None, conversation_scope=None,
                retrieval_protocol=None, vocabulary=None, retrieval=None):
    """Answers a message.

    With retrieval_protocol=1 (the Back-End's database-grounded mode), a first call
    without ``retrieval`` may return {"status": "needs_data", "plan": ...}; the Back-End
    runs the plan's read-only tools for the signed-in user and calls again with
    ``retrieval`` = {"plan", "results"}, and the answer is built from those results.
    """
    started = time.perf_counter()
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string.")
    role = resolve_role(viewer_role, conversation_scope)
    message = strip_courtesy(correct_typos(message.strip()))
    memory = _load_memory(conversation_id, conversation_scope)
    entities = extract_entities(message)
    resolved_message, used_memory = _resolve_follow_up(message, entities, memory)
    if used_memory:
        entities = extract_entities(resolved_message)
    grounded_vehicles = None

    memory_updates = {}
    database_mode = retrieval_protocol == 1 and rental_context is None and not diagnostic_context and not vehicle and not vehicles
    if database_mode and isinstance(retrieval, dict):
        response, grounded_vehicles, memory_updates = _grounded_response(message, role, retrieval, vocabulary, memory, entities)
    else:
        direct = None
        if database_mode:
            plan = planner.plan_message(message, role, vocabulary, memory)
            if plan and plan.get("retrievals"):
                return {"status": "needs_data", "plan": plan}
            if plan and plan.get("intent") == "multi":
                # Several questions, none needing records: answer each part in turn.
                response, _, memory_updates = _grounded_response(message, role, {"plan": plan, "results": []}, vocabulary, memory, entities)
                direct = response
            elif plan and plan.get("direct"):
                direct = _direct_response(plan["direct"], plan)
        # Typos and Tagalog terms are mapped before matching; entities come from the message as typed.
        response = direct or _route(normalize_message(resolved_message), entities, role, vehicle, vehicles, rental_context, diagnostic_context)
    response.setdefault("data", None)
    if response["data"] is None:
        response["data"] = {}
    response["data"].setdefault("evidence", {"evidence_type": "general_guidance", "sources": []})
    response["data"]["entities"] = {key: value for key, value in entities.items() if value}
    response["data"]["viewer_role"] = role
    if used_memory:
        response["data"]["resolved_from_conversation"] = used_memory
    response["data"]["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    _save_memory(conversation_id, conversation_scope, memory, entities, response.get("intent"), grounded_vehicles, memory_updates)
    logger.info("chat intent=%s evidence=%s role=%s latency_ms=%s", response.get("intent"),
                response["data"]["evidence"].get("evidence_type"), role, response["data"]["latency_ms"])
    return response


def _grounded_response(message, role, retrieval, vocabulary, memory=None, entities=None):
    """Answer from the Back-End's retrieval results. Returns (response, vehicles mentioned, memory updates)."""
    memory = memory or {}
    plan = retrieval.get("plan") if isinstance(retrieval.get("plan"), dict) else {}
    results = retrieval.get("results") if isinstance(retrieval.get("results"), list) else []
    shown = list(memory.get("notes_shown", []))
    parts = plan.get("parts") if plan.get("intent") == "multi" and isinstance(plan.get("parts"), list) else None
    pieces = [(message, plan, results)] if parts is None else [
        (part.get("text") or message, part.get("plan"), results[part.get("start", 0):part.get("start", 0) + part.get("count", 0)])
        for part in parts]

    answers, sources, all_status, vehicles_seen = [], [], {}, []
    for text, sub_plan, sub_results in pieces:
        if not sub_plan or not sub_plan.get("retrievals"):
            # A part that needs no records ("and can I pay with GCash?") is answered from the guide or references.
            reply = _direct_response(sub_plan["direct"], sub_plan) if sub_plan and sub_plan.get("direct") else _route(
                normalize_message(text), extract_entities(text), role, None, None, None, None)
            answers.append(reply["answer"])
            sources += ((reply.get("data") or {}).get("evidence") or {}).get("sources", [])
            continue

        def guide(text=text):
            found = _grounded_answer(normalize_message(text), role, {"intent": None, "confidence": 0.0},
                                     source_types={"system_documentation"})
            return (found["answer"], found["data"]["evidence"]["sources"]) if found else None

        def fleet_risk(vehicle_rows):
            return _fleet_risk_response({"vehicles": vehicle_rows, "data_as_of": retrieval.get("data_as_of")})["answer"]

        answer, data = grounded.compose(sub_plan, sub_results, role, vocabulary, guide=guide, fleet_risk=fleet_risk, shown_notes=shown)
        shown += data.get("notes_shown", [])
        answers.append(answer)
        all_status.update(data["retrieval_status"])
        sources += sub_plan.pop("_guide_sources", [])
        vehicles_seen += [{"make": v.get("make"), "model": v.get("model")}
                          for item in sub_results if isinstance(item, dict) and item.get("status") == "ok"
                          for v in (item.get("data") or {}).get("vehicles", []) if isinstance(v, dict) and v.get("model")]
    response = core.build_response(plan.get("intent") or "database_answer", 1.0, "\n\n".join(answers),
                                   {"plan_intent": plan.get("intent"), "retrieval_status": all_status})
    ok = any(status == "ok" for status in all_status.values())
    if ok:
        sources = [{"name": "DriveRent database (read-only, authorised for this account)", "tools": sorted(all_status),
                    "data_as_of": retrieval.get("data_as_of")}] + sources
    updates = {"notes_shown": sorted(set(shown))}
    period = next((r.get("args", {}).get("period") for r in results if isinstance(r, dict) and (r.get("args") or {}).get("period")), None)
    if period:
        updates["last_period"] = {"start_date": period["start_date"], "end_date": period["end_date"]}
    return _with_evidence(response, "database_record" if ok else ("reference" if sources else "none"), sources), vehicles_seen[:5], updates


def _direct_response(kind, plan=None):
    if kind == "policy_unrecorded":
        topic = (plan or {}).get("topic", "that")
        return _with_evidence(core.build_response("policy_unrecorded", 1.0,
            f"DriveRent has no information about {topic}. {UNRECORDED_POLICY}"), "none", [])
    if kind == "refuse_sql":
        return _with_evidence(core.build_response("permission_denied", 1.0,
            "I can't run database queries. I can only look up your own reservations and the vehicles available for rent, "
            "through the application's checks. Ask your question in words, for example \"What are my reservations?\""), "none", [])
    if kind == "clarify_vehicle":
        return _with_evidence(core.build_response("clarification_needed", 1.0,
            "Which vehicle do you mean? Tell me the make or model (for example Toyota Vios), and I'll look up its current rate."),
            "none", [], clarification="vehicle")
    if kind == "admin_no_bookings":
        return _with_evidence(core.build_response("clarification_needed", 1.0,
            "Administrator accounts don't make reservations, so there are none of your own to show. "
            "Ask \"show all reservations\" to see the customers' bookings."), "none", [])
    if kind == "refuse_private":
        return _with_evidence(core.build_response("permission_denied", 1.0, PRIVATE_DATA_ANSWER), "none", [])
    if kind == "refuse_credentials":
        return _with_evidence(core.build_response("permission_denied", 1.0, CREDENTIALS_ANSWER), "none", [])
    if kind == "accident_guidance":
        return _with_evidence(core.build_response("accident_guidance", 1.0, ACCIDENT_ANSWER), "none", [])
    return _with_evidence(core.build_response("permission_denied", 1.0, ADMIN_ONLY_ANSWER), "none", [])


def _route(message, entities, role, vehicle, vehicles, rental_context, diagnostic_context):
    database_context = rental_context.get("database_context") if isinstance(rental_context, dict) else None
    if isinstance(database_context, dict) and database_context.get("kind") in ADMIN_CONTEXT_KINDS and role != "admin":
        return _with_evidence(core.build_response("permission_denied", 1.0, (
            "That information is limited to administrators, so I can't show it for this account. "
            "I can help with vehicle listings, prices, availability and your own reservations."
        )), "none", [])

    if not isinstance(database_context, dict) and not diagnostic_context and _is_greeting(message):
        return _with_evidence(core.build_response("greeting", 1.0, GREETING_ANSWERS[role]), "general_guidance", [])
    help_request = HELP_REQUEST.match(message.strip()) if not isinstance(database_context, dict) else None
    if help_request:
        topic = (help_request.group("topic") or "").casefold()
        question = ("Sure. What do you need with your booking: make a new reservation, check or cancel an existing one, "
                    "or see prices and availability?" if topic.startswith(("book", "reserv", "rent"))
                    else "Sure. What would you like to know? For example vehicle prices or availability, your reservations, "
                         "how to book or cancel, or a car problem.")
        return _with_evidence(core.build_response("clarification_needed", 1.0, question), "none", [], clarification="help_topic")

    parts = lookup_parts(entities["part_numbers"])
    if parts:
        lines = [f"{number}: {item['description']} (supply group {item['supply_group']})" for number, item in parts.items()]
        return _with_evidence(core.build_response("parts_lookup", 1.0,
            "Honda (AHM) motorcycle master parts list, September 2025 update 19:\n- " + "\n- ".join(lines)
            + "\nThe list gives part descriptions only; it does not say whether a part fits a specific vehicle or is in stock."),
            "reference", [{"name": "Honda AHM master parts list (masterparts.xlsx)", "part_numbers": sorted(parts)}])

    if isinstance(database_context, dict) and database_context.get("kind") == "admin_fleet_risk":
        return _fleet_risk_response(database_context)

    if entities["dtc_codes"] and not isinstance(database_context, dict):
        return _dtc_response(entities["dtc_codes"], role)
    if ASKS_FOR_CODE.search(message) and not entities["dtc_codes"] and not isinstance(database_context, dict):
        return _with_evidence(core.build_response("clarification_needed", 1.0,
            "Which diagnostic trouble code do you mean? Please type it exactly as the scanner shows it, for example P0420."),
            "none", [], clarification="dtc_code")

    if not isinstance(database_context, dict) and not diagnostic_context and not vehicle and not vehicles:
        if role != "admin" and ADMIN_RECORDS_REQUEST.search(message):
            return _with_evidence(core.build_response("permission_denied", 1.0, ADMIN_ONLY_ANSWER), "none", [])
        if role != "admin" and VEHICLE_ADMINISTRATION.search(message):
            return _with_evidence(core.build_response("system_support", 1.0, VEHICLE_MANAGEMENT_ANSWER), "reference",
                                  [{"id": "sys:sys_overview_features", "name": "DriveRent system guide",
                                    "file": "data/knowledge/system_guide.json"}])
        if _is_guide_question(message):
            guide = _grounded_answer(message, role, {"intent": None, "confidence": 0.0}, source_types={"system_documentation"})
            if guide is not None:
                return guide
            abstained = _with_evidence(core.build_response("unknown", 0.0, GUIDE_NOT_FOUND), "none", [])
            abstained["data"]["abstained"] = True
            abstained["data"]["abstain_reason"] = "not_in_system_guide"
            return abstained
        specific = _specific_reference(message, role)
        if specific is not None:
            return specific

    response = core._answer_chat_core(message, vehicle=vehicle, vehicles=vehicles,
                                      rental_context=rental_context, diagnostic_context=diagnostic_context)
    data = response.get("data") or {}
    if isinstance(database_context, dict):
        if entities["dtc_codes"]:
            response = _append_dtc_definitions(response, entities["dtc_codes"], role)
        sources = [{"name": data.get("data_source"), "data_as_of": data.get("data_as_of")}]
        if database_context.get("kind") == "customer_bookings" and database_context.get("purpose") == "procedure":
            # "How do I cancel my booking?": the steps from the guide, then the customer's own bookings.
            guide = _grounded_answer(message, role, {"intent": None, "confidence": 0.0}, source_types={"system_documentation"})
            if guide is not None:
                response["answer"] = guide["answer"] + "\n\n" + response["answer"]
                sources += guide["data"]["evidence"]["sources"]
        return _with_evidence(response, "database_record", sources)
    if data.get("knowledge_id"):
        return _with_evidence(response, "reference", [{"name": "Curated automotive knowledge base", "ids": data.get("knowledge_ids")}],
                              support=None)
    if (data.get("prediction_status") or data.get("fleet_analysis_status")
            or (response.get("intent") in {"vehicle_prediction", "fleet_analysis"}
                and (data.get("all_categories") or data.get("prioritized_vehicles")))):
        return _with_evidence(response, "model_prediction" if data.get("all_categories") or data.get("prioritized_vehicles") else "none", [])
    if response.get("intent") in core.LIVE_RENTAL_INTENTS and isinstance(rental_context, dict):
        return _with_evidence(response, "database_record", [{"name": "authenticated application vehicle records", "data_as_of": rental_context.get("data_as_of")}])

    static_answer = response["answer"] == core.load_chatbot()[1].get(response.get("intent"), {}).get("answer")
    if response.get("intent") == "unknown" or (static_answer and response.get("intent") in SYSTEM_GUIDE_INTENTS) or static_answer:
        grounded = _grounded_answer(message, role, response)
        if grounded is not None:
            return grounded
        if response.get("intent") == "unknown":
            abstained = _with_evidence(core.build_response("unknown", response["confidence"], OUT_OF_SCOPE_ANSWER), "none", [])
            abstained["data"]["abstained"] = True
            abstained["data"]["abstain_reason"] = "no_relevant_evidence"
            return abstained
    return response


def _with_evidence(response, evidence_type, sources, support=None, clarification=None):
    data = response.get("data") or {}
    data["evidence"] = {"evidence_type": evidence_type, "sources": sources}
    if support is not None:
        data["evidence"]["retrieval_support"] = support
        data["evidence"]["support_interpretation"] = (
            "Heuristic evidence-strength score from term coverage and retrieval margin; not a probability of correctness."
        )
    if clarification:
        data["clarification_needed"] = clarification
    response["data"] = data
    return response


def _citation(passage):
    citation = dict(passage["citation"])
    citation.update(id=passage["id"], title=passage["title"], trust=passage["trust"])
    return citation


def _dtc_response(codes, role):
    try:
        passages = {p["metadata"].get("dtc_code"): p for code in codes for p in retrieve(code, role, top_k=1, source_types={"dtc_reference"})}
    except KnowledgeIndexUnavailable as error:
        return _with_evidence(core.build_response("dtc_lookup", 0.0, f"The diagnostic code reference is unavailable: {error}"), "none", [])
    lines, citations, missing = [], [], []
    for code in codes:
        passage = passages.get(code)
        if passage is None:
            missing.append(code)
            continue
        metadata = passage["metadata"]
        line = passage["text"]
        if metadata.get("mil"):
            line += " This code normally turns on the check-engine (MIL) light."
        if metadata.get("limp_mode_possible"):
            line += " Reduced-power (limp) mode is possible."
        lines.append(line)
        citations.append(_citation(passage))
    answer = "\n\n".join(lines)
    if missing:
        answer += ("\n\n" if answer else "") + (
            f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not in the generic SAE J2012 code reference I use. "
            "Codes like P1xxx, B1xxx or U1xxx are usually manufacturer-specific: tell me the make, model and year, "
            "or check the manufacturer's service information or a scan tool that supports that brand."
        )
    if lines:
        answer += ("\n\nThese are generic code definitions, not a diagnosis of your vehicle. A code shows which circuit or "
                   "test failed; confirm the cause with an inspection before replacing parts.")
    return _with_evidence(core.build_response("dtc_lookup", 1.0, answer.strip(), {"dtc_codes": codes, "dtc_not_found": missing}),
                          "reference" if lines else "none", citations, support=1.0 if lines else 0.0)


def _append_dtc_definitions(response, codes, role):
    try:
        passages = [p for code in codes for p in retrieve(code, role, top_k=1, source_types={"dtc_reference"})]
    except KnowledgeIndexUnavailable:
        return response
    if passages:
        response["answer"] += "\n\nGeneric code definitions (reference, not a diagnosis):\n- " + "\n- ".join(p["title"] for p in passages)
        (response.setdefault("data", {}) or {}).setdefault("dtc_references", [_citation(p) for p in passages])
    return response


def _specific_reference(message, role):
    """System-guide and PID answers are more specific than the general curated notes, so they go first."""
    try:
        passages = retrieve(message, role, top_k=3, source_types=FREE_TEXT_SOURCES)
    except KnowledgeIndexUnavailable:
        return None
    if not passages or passages[0]["support"] < SPECIFIC_SOURCE_MIN_SUPPORT:
        return None
    top = passages[0]["source_type"]
    if top == "system_documentation" or (top == "pid_reference" and re.search(r"\bpids?\b", message, re.IGNORECASE)):
        return _grounded_answer(message, role, {"intent": None, "confidence": 0.0}, passages=passages)
    return None


def _is_greeting(message):
    match = GREETING.match(message.strip())
    if not match:
        return False
    rest = re.findall(r"[a-z']+", match.group("rest").casefold())
    return all(word in GREETING_FILLER | COURTESY_WORDS | THANKS_WORDS for word in rest)


def _is_guide_question(message):
    """Rental-policy or system how-to questions with no car-problem words."""
    if RECORD_PROCEDURE.search(message):
        return True
    if MECHANICAL.search(message) or extract_entities(message)["dtc_codes"]:
        return False
    return bool(RENTAL_POLICY.search(message) or SYSTEM_PROCEDURE.search(message))


def _grounded_answer(message, role, core_response, passages=None, source_types=FREE_TEXT_SOURCES):
    try:
        if passages is None:
            passages = retrieve(message, role, top_k=3, source_types=source_types)
    except KnowledgeIndexUnavailable as error:
        logger.warning("knowledge index unavailable: %s", error)
        return None
    if not passages or passages[0]["support"] < settings.CHATBOT_MIN_RETRIEVAL_SUPPORT:
        return None
    best = passages[0]
    used = [best] + [p for p in passages[1:] if p["support"] >= settings.CHATBOT_STRONG_RETRIEVAL_SUPPORT
                     and p["source_type"] == best["source_type"]][:1]
    label = {
        "system_documentation": "From the DriveRent system guide",
        "dtc_reference": "From the generic OBD-II code reference",
        "pid_reference": "From the OBD-II PID reference",
        "curated_reference": "From the automotive reference notes",
        "diagnostic_flowchart": "From a diagnostic flowchart (source not independently verified)",
        "service_statistics": "From recorded workshop service data",
    }.get(best["source_type"], "From the reference material")
    answer = f"{label}: " + " ".join(f"{p['title']}. {p['text']}" if p is not best else p["text"] for p in used)
    if best["support"] < settings.CHATBOT_STRONG_RETRIEVAL_SUPPORT:
        answer += " (This is the closest match I found; if it doesn't answer your question, please add more detail.)"
    intent = core_response.get("intent") if core_response.get("intent") not in {"unknown", None} else best["metadata"].get("category", "reference")
    response = core.build_response(intent, core_response.get("confidence", 0.0), answer,
                                   {"retrieved_ids": [p["id"] for p in passages], "retrieval_scores": [p["score"] for p in passages]})
    return _with_evidence(response, "reference", [_citation(p) for p in used], support=best["support"])


def _fleet_risk_response(database_context):
    vehicles = database_context.get("vehicles")
    if not isinstance(vehicles, list) or not vehicles:
        return _with_evidence(core.build_response("fleet_analysis", 1.0, "No fleet vehicle records were supplied, so no prediction was run."), "none", [])
    try:
        result = analyze_fleet(vehicles)
    except MaintenancePredictionsUnavailableError as error:
        return _with_evidence(core.build_response("fleet_analysis", 1.0,
            f"Model-based maintenance predictions are unavailable: {error} The fleet's recorded maintenance status is still available on the Maintenance Records page."),
            "none", [])
    ranked = [item for item in result["prioritized_vehicles"] if item["recommendation_count"]]
    lines = []
    for item in ranked[:10]:
        info = item["vehicle"]
        name = " ".join(str(info.get(key)) for key in ("brand", "model") if info.get(key)) or "Vehicle"
        items = ", ".join(f"{rec['label']} ({rec['score']:.0%}, {rec.get('risk_category', rec['priority'])})" for rec in item["recommendations"][:4])
        lines.append(f"Vehicle #{item['vehicle_id']} {name} at {float(info.get('mileage') or 0):,.0f} km: {items}")
    rejected = result["rejected_vehicles"]
    reasons = {}
    for item in rejected:
        key = re.sub(r"'[^']*'|\d[\d,\.]*", "…", item["error"])[:140]
        reasons.setdefault(key, []).append(item["vehicle_id"])
    answer = (
        f"Model-based service-need results for {result['fleet_summary']['analyzed_vehicles']} of "
        f"{result['fleet_summary']['submitted_vehicles']} vehicle(s)"
        + (f" (data as of {database_context.get('data_as_of')})" if database_context.get("data_as_of") else "") + ". "
    )
    answer += ("Items likely due at the next service:\n- " + "\n- ".join(lines)) if lines else "No vehicle has a service item above its recommendation threshold."
    if rejected:
        answer += "\n\nNot scored: " + "; ".join(f"{len(ids)} vehicle(s) ({', '.join('#' + str(i) for i in ids[:8])}): {reason}" for reason, ids in reasons.items())
    answer += ("\n\nThese are probabilities that each item would be performed at a service visit at the current odometer, "
               "learned from workshop records; they are not detected faults or failure dates. Confirm with the manufacturer schedule and an inspection.")
    version = result.get("model_version")
    return _with_evidence(core.build_response("fleet_analysis", 1.0, answer, result), "model_prediction",
                          [{"name": "registered maintenance model", "model_version": version or "see /api/capabilities/"},
                           {"name": "administrator-authorized vehicle records", "data_as_of": database_context.get("data_as_of")}])


def _resolve_follow_up(message, entities, memory):
    if entities["dtc_codes"] or not memory.get("last_dtc_codes"):
        return message, None
    if FOLLOW_UP.search(message) or ASKS_FOR_CODE.search(message):
        codes = memory["last_dtc_codes"]
        return f"{message} {' '.join(codes)}", {"dtc_codes": codes}
    return message, None


def _load_memory(conversation_id, scope):
    if not conversation_id or not scope:
        return {}
    from .models import Conversation
    try:
        conversation = Conversation.objects.filter(id=conversation_id).first()
    except (DatabaseError, ValueError):
        return {}
    if conversation is None or conversation.memory.get("scope") != scope:
        # A conversation is only resumed by the same viewer scope that created it.
        return {}
    if conversation.updated_at < _retention_cutoff():
        return {}
    return conversation.memory


_last_purge = 0.0


def _retention_cutoff():
    from django.utils import timezone
    from datetime import timedelta
    return timezone.now() - timedelta(days=settings.CHAT_MEMORY_RETENTION_DAYS)


def purge_expired_conversations():
    """Deletes conversation memory unused for CHAT_MEMORY_RETENTION_DAYS. Returns the number deleted."""
    from .models import Conversation
    deleted, _ = Conversation.objects.filter(updated_at__lt=_retention_cutoff()).delete()
    return deleted


def _save_memory(conversation_id, scope, memory, entities, intent, vehicles=None, updates=None):
    if not conversation_id or not scope:
        return
    from .models import Conversation
    # Only names of vehicles already shown to this viewer are kept; every follow-up is
    # answered from a fresh database retrieval, never from remembered prices or statuses.
    memory = {"scope": scope, "last_intent": intent,
              "last_dtc_codes": entities["dtc_codes"] or memory.get("last_dtc_codes", []),
              "last_vehicle_ids": entities["vehicle_ids"] or memory.get("last_vehicle_ids", []),
              "last_vehicles": vehicles or memory.get("last_vehicles", []),
              # Explanations already shown in this conversation, and the last date range asked about.
              "notes_shown": (updates or {}).get("notes_shown", memory.get("notes_shown", [])),
              "last_period": (updates or {}).get("last_period", memory.get("last_period"))}
    try:
        conversation = Conversation.objects.filter(id=conversation_id).first()
        if conversation is not None and conversation.memory.get("scope") not in {None, scope}:
            return
        Conversation.objects.update_or_create(id=conversation_id, defaults={"memory": memory})
        global _last_purge
        if time.time() - _last_purge > 3600:
            # Expired memory is removed at most once an hour (also: manage.py purge_conversations).
            _last_purge = time.time()
            purge_expired_conversations()
    except (DatabaseError, ValueError) as error:
        logger.warning("conversation memory not saved: %s", error)
