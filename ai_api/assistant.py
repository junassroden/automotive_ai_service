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
from .automotive_knowledge import (
    _terms_match, compose_chat_answer, compose_follow_up, has_chat_fields, knowledge_document, mentions, retrieve_knowledge, triage_reply,
)
from .fleet_service import analyze_fleet
from .language import correct_typos, normalize_message
from .prediction_service import MaintenancePredictionsUnavailableError
from .rag import KnowledgeIndexUnavailable, extract_entities, lookup_parts, retrieve


logger = logging.getLogger(__name__)
ADMIN_CONTEXT_KINDS = {"admin_fleet", "admin_maintenance", "admin_fleet_risk"}
SYSTEM_GUIDE_INTENTS = core.RENTAL_INTENTS - core.LIVE_RENTAL_INTENTS | {"capabilities"}
FOLLOW_UP = re.compile(
    r"^(?:and|what about|how about|also|then)\b|\b(?:it|that|this|those|these)\b.*\b(?:code|fault|mean|cause|causes|symptom|symptoms|fix|repair|cost|serious|safe)\b"
    r"|^(?:causes?|symptoms?|fix(?:es)?|how (?:do i|to) fix|is it (?:serious|safe))\??$"
    # "How much to fix?", "magkano ipagawa?", "can I still drive?" right after a code.
    r"|^(?:so |and |ok )?(?:how much(?: (?:to|will it|would it|does it|is it))?(?: cost)?(?: to)? (?:fix|repair)\w*|magkano(?: ang)? (?:ipagawa|paayos|pagawa)|can i (?:still )?drive(?: it)?|(?:is it )?(?:ok|okay) to drive)\b.*$"
    r"|^(?:what|why|how)\b[\w\s']{0,30}\b(?:it|that|this)\s*\??$",
    re.IGNORECASE,
)
RENTAL_OPERATION_TOPICS = {"pre_rental_inspection", "rental_damage_reporting", "rental_fault_reporting"}
RENTAL_CAR = re.compile(r"\b(?:rental|rented|hired)\b", re.IGNORECASE)
# Booking, price and inventory words: these questions belong to the database planner even when they name a car part.
RENTAL_REQUEST = re.compile(
    r"\b(?:price|prices|rate|rates|per day|daily|cheap|cheapest|available|availability|book|booking|bookings|reserve|reservation|reservations"
    r"|deposit|payment|pay|show me|list|cars\b|vehicles\b|for rent|rent a|rent an|seats?|seater)\b",
    re.IGNORECASE,
)
# "Does the Vios have automatic transmission?": a listed vehicle's equipment, so the planner looks first; it falls back to the
# references when no vehicle of that name is listed. "Does my car have ...?" and "does the engine ...?" are about the customer's car.
NAMED_VEHICLE_SPEC = re.compile(
    r"^(?:does|do|is|has|have)\s+(?:the|a|an)\s+(?!car\b|vehicle\b|engine\b|battery\b|brakes?\b|tires?\b|transmission\b|it\b)"
    r"[a-z0-9-]+(?:\s+[a-z0-9-]+)?\s+(?:have|has|come with|comes with|use|uses|run on|runs on)\b",
    re.IGNORECASE,
)
# Follow-ups about the car problem being discussed ("what if the coolant is full?", "is it safe to drive?").
TOPIC_FOLLOW_UP = re.compile(
    r"^(?:and|but|so|then|ok(?:ay)?|hmm+)?[\s,]*(?:what if|what about|how about|and if|if it|if the|if my|even if|is it|is that|is this|can i|could i|should i"
    r"|could it|could that|would it|what else|anything else|any ?more|more\W*$|tell me more|more details?|more info|explain more|ano pa|iba pa|meron pa|go on|continue"
    r"|why|what causes|what caused|what could cause"
    r"|how (?:do|can|should) i|how much|i (?:already|just|have) |it'?s not|it is not|not the |what (?:should|do|can) i do|what now|now what"
    r"|the [a-z ]{2,30} (?:is|are|was|were|works?|seems?|looks?) )",
    re.IGNORECASE,
)
CONDITION = re.compile(
    r"(?:what if|and if|but if|even if|if|suppose|assuming)?\s*(?:the |my |its )?(?P<subject>[a-z][a-z -]{1,40}?)\s+(?P<verb>is|are|was|were|looks?|seems?|works?|working)\s+"
    r"(?P<state>full|fine|ok|okay|good|normal|new|topped up|not low|correct|working|alright|all right|clean|fixed|replaced|charged|tight)\b",
    re.IGNORECASE,
)
CHECKED = re.compile(r"\bi (?:already |just |have )?(?P<verb>checked|replaced|changed|topped up|filled|cleaned|charged|tightened)\s+(?:the |my )?"
                     r"(?P<subject>[a-z][a-z -]{1,40}?)(?:\s+(?:and|but|already|too|yesterday)\b.*)?[.?!]*$", re.IGNORECASE)
NOT_THE = re.compile(r"\b(?:it'?s|it is|that'?s|that is)?\s*not (?:the |my )(?P<subject>[a-z][a-z -]{1,40}?)[.?!]*$", re.IGNORECASE)
ASKS_SAFE = re.compile(r"\b(?:safe|ok(?:ay)? to drive|still drive|keep driving|drive it|drive with it|drive like this|dangerous|serious|urgent|emergency)\b", re.IGNORECASE)
ASKS_COST = re.compile(r"\b(?:how much|cost|costs|price|expensive|cheap)\b", re.IGNORECASE)
ASKS_FIX = re.compile(r"\b(?:fix|repair|what (?:should|do|can) i do|what now|how (?:do|can|should) i)\b", re.IGNORECASE)
ASKS_WHY = re.compile(r"\b(?:why|cause|causes|caused|reason)\b", re.IGNORECASE)
TOPIC_SWITCH = re.compile(r"^(?:what|how) about (?:the |my )?(?P<subject>[a-z][a-z -]{1,40}?)[.?!]*$", re.IGNORECASE)
# A message made only of these words names a kind of symptom but nothing specific about it.
VAGUE_WORDS = {
    "my", "car", "the", "a", "an", "is", "it", "its", "it's", "makes", "making", "made", "has", "have", "having", "had", "there", "weird", "strange",
    "funny", "odd", "some", "something", "kind", "of", "sort", "bit", "little", "new", "started", "starting", "keeps", "from", "coming", "came", "come",
    "with", "vehicle", "i", "hear", "heard", "hearing", "on", "what", "do", "should", "help", "me", "please", "can", "you", "wrong", "problem", "problems",
    "issue", "issues", "trouble", "noise", "noises", "sound", "sounds", "smell", "smells", "smelling", "vibration", "vibrates", "vibrating", "shaking",
    "shakes", "light", "lights", "warning", "dashboard", "leak", "leaking", "leaks", "fluid", "under", "this", "that", "getting", "get", "got", "am",
    "ive", "i've", "think", "rental", "now", "today", "again", "lately", "recently", "when", "just",
    # "My steering wheel is shaking" says where, not when: braking, high speed and idle have different causes.
    "steering", "wheel",
}
VAGUE_SYMPTOMS = (
    (r"\b(?:noises?|sounds?)\b", "noise", "What kind of noise is it, and when does it happen? For example: squealing when braking, clicking when turning, "
     "knocking from the engine, or a hum that gets louder with speed."),
    (r"\bsmell", "smell", "What does it smell like, and where is it coming from? For example: burning plastic, fuel, a sweet coolant smell or burnt oil. "
     "If it's a burning or fuel smell, stop somewhere safe first."),
    (r"\b(?:vibrat|shak)", "vibration", "When do you feel the shaking? For example: only when braking, at high speed, when accelerating, or at idle. "
     "If it's sudden and severe, slow down and stop somewhere safe first."),
    (r"\bleak|\bfluid\b", "leak", "What colour is the fluid, and where under the car is it dripping? For example: green or pink near the front, "
     "brown or black under the engine, or red near the middle."),
    (r"\blights?\b", "warning_light", "Which warning light is it, and what colour? For example: the check-engine light (steady or flashing), "
     "the battery light, the oil-pressure light, or the brake or ABS light."),
    (r"\b(?:wrong|problems?|issues?|trouble)\b", "problem", "What's happening with the car? Tell me the symptom, for example a noise, a warning light, "
     "a smell, a leak, or trouble starting, steering or braking."),
)
# Free-text retrieval never answers from DTC definitions: fuzzy matches between a
# symptom description and code titles are unreliable. DTCs are looked up only
# when the code itself appears in the question.
FREE_TEXT_SOURCES = {"system_documentation", "pid_reference", "curated_reference", "diagnostic_flowchart", "service_statistics"}
SPECIFIC_SOURCE_MIN_SUPPORT = 0.60
DETAIL_REQUEST = re.compile(r"\b(?:in detail|detailed|details|in depth|technical|how (?:is|does) (?:it|the code|that) (?:get )?(?:detected|set|work))\b", re.IGNORECASE)
ASKS_FOR_CODE = re.compile(r"\b(?:this|that|the|my)\s+(?:dtc|code|fault code|error code|trouble code)\b", re.IGNORECASE)
GREETING = re.compile(
    r"^(?:hi+|hiya|hello+|hey+|heya|good\s+(?:morning|afternoon|evening|day|noon|am|pm)|greetings|kumusta|kamusta|musta|yo+|sup|wassup|what'?s\s+up"
    r"|magandang\s+(?:umaga|hapon|gabi|araw|tanghali))\b(?P<rest>.*)$",
    re.IGNORECASE,
)
# Words that may follow a greeting without turning it into a question ("Hiya, quick question for you").
GREETING_FILLER = {
    "there", "po", "again", "quick", "question", "questions", "for", "you", "can", "could", "would", "help", "me",
    "i", "ask", "a", "something", "have", "got", "how", "are", "please", "hope", "doing", "well", "ok", "okay",
    "the", "assistant", "bot", "team", "everyone", "just", "wanted", "to", "one", "little", "need", "some", "kayo", "ka",
}
# Greetings and fillers around a question ("Good morning, how much is the Vios po?") are not part of it.
COURTESY_LEAD = re.compile(
    r"^\s*(?:hi+|hiya|hello+|hey+|heya|good\s+(?:morning|afternoon|evening|day|noon|am|pm)|magandang\s+(?:umaga|hapon|gabi|araw|tanghali)|(?:kumusta|kamusta)(?!\s+(?:ang|yung|ng|na)\b)"
    r"|excuse\s+me|quick\s+question|question|ask\s+ko\s+lang|tanong\s+(?:ko\s+)?lang|tanong|uh+m*|um+|uh+|ah+|po|pls\s+help|please\s+help|pa\s+help|pls|plz|please|sir"
    r"|can\s+i\s+ask|may\s+i\s+ask|could\s+i\s+ask"
    r"|ma'?am|kuya|ate|boss|ok(?:ay)?|so|yo)"
    r"(?:\s*[,:;.!?-]+\s*|\s+)",
    re.IGNORECASE,
)
COURTESY_WORDS = {"hi", "hello", "hey", "hiya", "heya", "good", "morning", "afternoon", "evening", "day", "noon", "am", "pm", "magandang", "umaga", "hapon",
                  "tanghali", "gabi", "araw", "kumusta", "kamusta", "musta", "excuse", "uhm", "um", "uh", "pls", "plz", "please", "sir", "ma'am", "maam",
                  "kuya", "ate", "boss", "yo", "sup", "wassup", "asap", "help"}
THANKS_WORDS = {"thanks", "thank", "you", "u", "thx", "tnx", "ty", "salamat", "maraming", "marami", "many", "a", "lot", "so", "much", "naman", "sana", "lang", "again"}
# A message made only of these acknowledges or closes the chat; it asks for nothing, so no records or references are looked up.
ACK_WORDS = {"ok", "okay", "okey", "k", "kk", "sige", "noted", "got", "it", "understood", "alright", "cool", "nice", "great", "perfect", "awesome", "fine",
             "copy", "gotcha", "good", "ayos", "oo", "yes", "yup", "yep", "that", "that's", "helps", "helped", "for", "the", "help", "your", "time", "all", "ng", "po", "and", "one", "very"}
GOODBYE_WORDS = {"bye", "goodbye", "paalam", "ingat", "see", "ya", "later", "soon", "take", "care", "now", "byebye"}
FILLER_LEAD = re.compile(r"^\s*(?:(?:uh+m*|um+|uh+|ah+|po|pls\s+help|please\s+help|pls|plz|please|ok(?:ay)?|so|excuse\s+me|quick\s+question|question|ask\s+ko\s+lang|can\s+i\s+ask|may\s+i\s+ask"
                         r"|tanong\s+(?:ko\s+)?lang|sir|ma'?am|kuya|ate|boss)(?:\s*[,:;.!?-]+\s*|\s+))+", re.IGNORECASE)
COURTESY_TAIL = re.compile(r"(?:[\s,]+(?:please|pls|plz|po|thanks|thank\s+you|thank\s+u|thx|tnx|ty|salamat|naman|sana|lang|asap))+\s*([?!.]*)\s*$",
                           re.IGNORECASE)


def strip_courtesy(message):
    """The question without greetings or fillers around it; a message that is only a greeting or thanks is kept as typed."""
    if _small_talk_close(message):
        return message
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
    r"|licen[cs]e|requirements?|provide|bring|documents?|id|ids|cancel|reschedule|pickup|pick\s+up|return|returns|account|profile"
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
    r"|exhaust|won'?t start|inspect\w*|parts?|repair\w*|maintenance|diagnos\w*|mechanic\w*|fault\w*|symptoms?|downtime|breakdowns?)\b",
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
# Unrecorded topics that are vehicle facts rather than policies get their own second sentence.
UNRECORDED_NOTES = {
    "fuel-consumption figures for its vehicles": (
        "The listings record each vehicle's fuel type but not its consumption. As a rough guide, smaller cars and diesel "
        "engines usually use less fuel; ask the rental team for a specific vehicle."),
}
CREDENTIALS_ANSWER = ("I can't share passwords, keys or other account credentials, for any account. "
                      "If you can't sign in to your own account, contact the rental team.")
BREAKDOWN_ANSWER = (
    "If the rental car breaks down:\n"
    "- Move to a safe place if you can, switch on the hazard lights, and stay clear of traffic.\n"
    "- Do not keep driving if there is smoke, a red warning light, overheating, or a brake or steering problem.\n"
    "- Contact the rental team as soon as you can and tell them what happened and where you are; ask them before arranging any repair.\n"
    "The system does not record a breakdown or roadside-assistance procedure, so confirm the next steps with the rental team."
)
# General safety steps only: the system records no accident, insurance or liability procedure.
ACCIDENT_ANSWER = (
    "If you are in an accident:\n"
    "- Stop in a safe place if you can, switch on the hazard lights and check whether anyone is hurt.\n"
    "- Call emergency services if anyone is injured or the vehicle is a hazard to traffic.\n"
    "- Follow local requirements for reporting the accident, and note the time, place, other vehicles and any witnesses.\n"
    "- Notify the rental team as soon as you can, and follow your rental agreement.\n"
    "The system does not record accident procedures, insurance coverage or liability terms, so confirm those with the rental team."
)
# The first sentence stays UNRECORDED_POLICY: evaluations recognise an abstention by it.
GUIDE_NOT_FOUND = (UNRECORDED_POLICY + " I can answer questions about booking, payment, cancellation, rental length and prices, "
                   "deposits, and pickup and return.")
OUT_OF_SCOPE_ANSWER = (
    "I can only answer from this system's data and its automotive references, and I don't have anything reliable on that. "
    "I can help with car problems, warning lights, trouble codes and maintenance, and with vehicles, prices and your bookings."
)


# "Who are you?", "Are you a real person?", "What can you do?", "Sino ka?"
ABOUT_YOU = re.compile(
    r"^(?:(?:hi|hello|hey)[\s,!.]*)?(?:who|what) (?:are|r) (?:you|u)\b|^are (?:you|u) (?:a |an )?(?:real|human|person|bot|robot|ai|chatbot|machine)\b"
    r"|^am i (?:talking|chatting) (?:to|with) (?:a )?(?:real |human|person|bot|robot|ai)|^what (?:can|do) (?:you|u) do\b"
    r"|^what (?:can|could) (?:you|u) help (?:me )?with\b|^(?:sino|who) ka\b|^ano(?:ng)? (?:kaya|magagawa) mo\b|^tao ka ba\b|^bot ka ba\b",
    re.IGNORECASE)
# An off-topic word in a message that names no vehicle ("is it going to rain tomorrow?") is not a car question.
CAR_NOUN = re.compile(r"\b(?:car|cars|vehicle|vehicles|kotse|sasakyan|unit|rental|windshield|sunroof|wipers?|cabin)\b", re.IGNORECASE)
ABOUT_YOU_ANSWER = "I'm DriveRent's virtual assistant, an automated chatbot rather than a person. "
ACK_ANSWER = "You're welcome! Is there anything else I can help you with?"
OKAY_ANSWER = "Alright! Is there anything else I can help you with?"
GOODBYE_ANSWER = "Goodbye, and thank you! Message me anytime you need help with a rental."


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
    no_context = rental_context is None and not diagnostic_context and not vehicle and not vehicles
    database_mode = retrieval_protocol == 1 and no_context
    follow_up = _topic_follow_up(message, entities, memory) if no_context and not isinstance(retrieval, dict) else None
    if follow_up is not None:
        response, memory_updates = follow_up
    elif database_mode and isinstance(retrieval, dict):
        response, grounded_vehicles, memory_updates = _grounded_response(message, role, retrieval, vocabulary, memory, entities)
    else:
        direct = None
        if database_mode and not _is_car_problem(message, entities):
            plan = planner.plan_message(message, role, vocabulary, memory)
            if plan and plan.get("retrievals") and _policy_question_over_vague_plan(message, plan):
                plan = None  # the rules only guessed a listing; the classifier is sure it is a policy question
            if plan and plan.get("retrievals"):
                return {"status": "needs_data", "plan": plan}
            if plan and plan.get("intent") == "multi":
                # Several questions, none needing records: answer each part in turn.
                response, _, memory_updates = _grounded_response(message, role, {"plan": plan, "results": []}, vocabulary, memory, entities)
                direct = response
            elif plan and plan.get("direct"):
                direct = _direct_response(plan["direct"], plan)
        # Typos and Tagalog terms are mapped before matching; entities come from the message as typed.
        # "Is it serious?" / "What causes it?" about a code from earlier in the conversation answers just that.
        dtc_focus = None
        if used_memory and used_memory.get("dtc_codes"):
            dtc_focus = "safety" if ASKS_SAFE.search(message) or re.search(r"\bcan i (?:still )?drive\b", message, re.IGNORECASE) \
                else "cost" if ASKS_COST.search(message) or re.search(r"\bmagkano\b", message, re.IGNORECASE) \
                else "causes" if ASKS_WHY.search(message) else None
        response = direct or _route(normalize_message(resolved_message), entities, role, vehicle, vehicles, rental_context, diagnostic_context,
                                    dtc_focus=dtc_focus)
    response.setdefault("data", None)
    if response["data"] is None:
        response["data"] = {}
    response["data"].setdefault("evidence", {"evidence_type": "general_guidance", "sources": []})
    response["data"]["entities"] = {key: value for key, value in entities.items() if value}
    response["data"]["viewer_role"] = role
    if used_memory:
        response["data"]["resolved_from_conversation"] = used_memory
    topic_id = response["data"].get("knowledge_id")
    # Fleet-management knowledge (pm_*) is for the business, not a renter with a problem car.
    if topic_id and topic_id not in RENTAL_OPERATION_TOPICS and not topic_id.startswith("pm_") and RENTAL_CAR.search(message) \
            and "rental team" not in response["answer"]:
        response["answer"] += "\n\nAs it's a rental car, tell the rental team before arranging any repairs."
    response["data"]["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    if "last_topic" not in memory_updates:
        # The car problem being discussed, so a follow-up can build on it; anything else ends that topic.
        topic_id = response["data"].get("knowledge_id")
        memory_updates["last_topic"] = {"id": topic_id, "shown": response["data"].get("shown_parts", [])} if topic_id else None
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
    if plan.get("pending_pickup"):
        updates["pending_pickup"] = plan["pending_pickup"]
    period = next((r.get("args", {}).get("period") for r in results if isinstance(r, dict) and (r.get("args") or {}).get("period")), None)
    if period:
        updates["last_period"] = {"start_date": period["start_date"], "end_date": period["end_date"]}
        if (memory or {}).get("pending_pickup") == period["start_date"]:
            updates["pending_pickup"] = None  # the waiting pickup date has its return date now
    return _with_evidence(response, "database_record" if ok else ("reference" if sources else "none"), sources), vehicles_seen[:5], updates


def _direct_response(kind, plan=None):
    if kind == "policy_unrecorded":
        topic = (plan or {}).get("topic", "that")
        return _with_evidence(core.build_response("policy_unrecorded", 1.0,
            f"DriveRent has no information about {topic}. {UNRECORDED_NOTES.get(topic, UNRECORDED_POLICY)}"), "none", [])
    if kind == "refuse_sql":
        return _with_evidence(core.build_response("permission_denied", 1.0,
            "I can't run database queries. I can only look up your own reservations and the vehicles available for rent, "
            "through the application's checks. Ask your question in words, for example \"What are my reservations?\""), "none", [])
    if kind == "clarify_quote":
        return _with_evidence(core.build_response("clarification_needed", 1.0,
            "Which vehicle do you mean, and for how many days? Tell me both (for example Toyota Vios for 3 days) and I'll work out the estimate from its current rate."),
            "none", [], clarification="vehicle_and_days")
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
    if kind == "breakdown_guidance":
        return _with_evidence(core.build_response("breakdown_guidance", 1.0, BREAKDOWN_ANSWER), "none", [])
    return _with_evidence(core.build_response("permission_denied", 1.0, ADMIN_ONLY_ANSWER), "none", [])


def _route(message, entities, role, vehicle, vehicles, rental_context, diagnostic_context, dtc_focus=None):
    database_context = rental_context.get("database_context") if isinstance(rental_context, dict) else None
    if isinstance(database_context, dict) and database_context.get("kind") in ADMIN_CONTEXT_KINDS and role != "admin":
        return _with_evidence(core.build_response("permission_denied", 1.0, (
            "That information is limited to administrators, so I can't show it for this account. "
            "I can help with vehicle listings, prices, availability and your own reservations."
        )), "none", [])

    if not isinstance(database_context, dict) and not diagnostic_context and _is_greeting(message):
        return _with_evidence(core.build_response("greeting", 1.0, GREETING_ANSWERS[role]), "general_guidance", [])
    closing = _small_talk_close(message) if not isinstance(database_context, dict) and not diagnostic_context else None
    if closing:
        # Both use the existing "goodbye" intent, which the rest of the service already treats as small talk.
        return _with_evidence(core.build_response("goodbye", 1.0, GOODBYE_ANSWER if closing == "goodbye" else ACK_ANSWER if re.search(
            r"\b(?:thanks?|thx|tnx|ty|salamat)\b", message, re.IGNORECASE) else OKAY_ANSWER), "general_guidance", [])
    if not isinstance(database_context, dict) and not diagnostic_context and planner.MODEL_QUESTION.search(message.casefold()) \
            and not MECHANICAL.search(message.replace("maintenance", "")):
        # "How accurate is the maintenance model?", "what data do you use?": from the model registry, as it is now.
        return _with_evidence(core.build_response("model_overview", 1.0, _models_answer(role, message)), "reference",
                              [{"name": "model registry", "file": "models/model_registry.json"}])
    if not isinstance(database_context, dict) and not diagnostic_context and ABOUT_YOU.match(message.strip()):
        return _with_evidence(core.build_response("capabilities", 1.0, ABOUT_YOU_ANSWER + GREETING_ANSWERS[role].replace("Hello! ", "", 1)),
                              "general_guidance", [])
    if not isinstance(database_context, dict) and not diagnostic_context and planner.OFF_TOPIC.search(message.casefold()) \
            and not MECHANICAL.search(message) and (not CAR_NOUN.search(message) or retrieve_knowledge(message) is None):
        # "What's the weather today?" shares words with the booking guide ("today") but is not about this system.
        abstained = _with_evidence(core.build_response("unknown", 0.0, OUT_OF_SCOPE_ANSWER), "none", [])
        abstained["data"].update(abstained=True, abstain_reason="off_topic")
        return abstained
    help_request = HELP_REQUEST.match(message.strip()) if not isinstance(database_context, dict) else None
    if help_request:
        topic = (help_request.group("topic") or "").casefold()
        question = ("Sure. What do you need with your booking: make a new reservation, check or cancel an existing one, "
                    "or see prices and availability?" if topic.startswith(("book", "reserv", "rent"))
                    else "Sure. What would you like to know? For example vehicle prices or availability, your reservations, "
                         "how to book or cancel, or a car problem.")
        return _with_evidence(core.build_response("clarification_needed", 1.0, question), "none", [], clarification="help_topic")

    vague = _vague_symptom(message) if not isinstance(database_context, dict) and not diagnostic_context else None
    if vague:
        # "My car makes a noise": asking beats guessing which system it is.
        return _with_evidence(core.build_response("clarification_needed", 1.0, vague[1]), "none", [], clarification=f"symptom_{vague[0]}")

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
        return _dtc_response(entities["dtc_codes"], role, detailed=bool(DETAIL_REQUEST.search(message)), focus=dtc_focus)
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


MODEL_ACCURACY = re.compile(r"\b(?:accura\w*|precision|recall|reliab\w*|how good|how well|performance|evaluat\w*|trust)\b", re.IGNORECASE)
MODEL_DATA = re.compile(r"\b(?:what data|which data|data do you|data does|trained on|training data|learn\w* from|inputs?)\b", re.IGNORECASE)
MODEL_STATUS = re.compile(r"\b(?:approved|accepted|ready|status|unavailable|not available|switched off|disabled|turned off|in use)\b", re.IGNORECASE)
NEXT_STEPS = ("To make the predictions real: keep recording monthly assessments of the real fleet and the verified outcome of every "
              "repair (what was found, what was replaced, the date and odometer). Once there are a few hundred real outcomes, retrain "
              "the models from Model Management (/admin/maintenance/model-management).")
SIMULATED_WARNING = ("All the fault models were trained on the simulated SAMPLE- vehicles, so treat them as a demonstration: the "
                     "numbers show the method works, not how your real cars will behave.")


def _models_answer(role, message=""):
    """How the maintenance predictions work, how good each model is, what data they use, or whether they're in use:
    whichever the question asks, read from the registry and evaluation reports as they are now."""
    import json
    from pathlib import Path
    from .category_models import models_overview

    if role != "admin":
        return ("Maintenance predictions are an administrator tool: vehicles are assessed monthly, and the ratings, findings and "
                "verified repair outcomes decide which cars get serviced first. As a customer you only rent vehicles marked available.")
    models = models_overview()
    in_use = [m for m in models if m["status"] == "in_use"]
    others = [m for m in models if m["status"] != "in_use"]
    horizon = next((m["horizon_days"] for m in models if m.get("horizon_days")), 90)
    simulated = any(m.get("training_source") in ("simulated", "mixed") for m in models)
    try:
        registry = json.loads(Path(settings.AI_MAINTENANCE_MODEL_REGISTRY_PATH).read_text(encoding="utf-8"))
        main = registry.get("maintenance_model") or {}
        report = main.get("evaluation_report_path")
        evaluation = json.loads((settings.BASE_DIR / report).read_text(encoding="utf-8")) if report else {}
    except (OSError, ValueError, TypeError):
        main, evaluation = {}, {}
    metrics = evaluation.get("metrics") or {}
    visits = (evaluation.get("dataset") or {}).get("rows_used")
    main_accepted = main.get("status") == "evaluated_accepted"

    def model_line(m):
        if m["status"] == "in_use":
            return f"- {m['name']}: in use; catches {m['recall']:.0%} of faults, and {m['precision']:.0%} of its alerts are right."
        if m["status"] == "rejected" and m.get("recall") is not None:
            return f"- {m['name']}: rejected; it misses too many faults (catches {m['recall']:.0%}, not reliably enough to pass)."
        return f"- {m['name']}: {str(m['status']).replace('_', ' ')}."

    def service_need_line():
        if not main:
            return None
        if main_accepted:
            return "Service-need model (which service items are due, from vehicle type, fuel type and mileage): approved and in use."
        detail = (f": it ranks cars sensibly (ROC-AUC {metrics['macro_roc_auc']:.2f}) but picks the right service items only about "
                  f"half the time (F1 {metrics['macro_f1']:.2f})" if metrics.get("macro_roc_auc") and metrics.get("macro_f1") else "")
        return ("Service-need model (which service items are due, from vehicle type, fuel type and mileage): switched off" + detail
                + ". Three inputs aren't enough to know what a particular car needs.")

    text = message or ""
    if MODEL_ACCURACY.search(text):
        lines = [f"How accurate each model is, tested on assessments it never saw in training (fault within {horizon} days):"]
        lines += [model_line(m) for m in in_use + others]
        if service_need_line():
            lines.append(service_need_line())
        lines.append("\"Catches\" (recall) is the share of real faults it flags; \"alerts right\" (precision) is the share of its "
                     "flags that were real faults. For safety systems catching faults matters more, so some false alarms are accepted.")
        if simulated:
            lines.append(SIMULATED_WARNING)
        return "\n".join(lines)
    if MODEL_DATA.search(text):
        lines = ["The maintenance answers use four kinds of data from your records:",
                 "1. Monthly assessments: each system's rating, the checklist answers, measurements (pad thickness, tread depth and "
                 "so on), warning lights and fault codes.",
                 "2. Verified outcomes: for each recommendation, whether a mechanic confirmed a fault, repaired it, or found nothing.",
                 "3. Maintenance records: date, odometer, the work done and its cost.",
                 "4. The vehicle profile: type, fuel, year, mileage and how often it's rented."]
        if models:
            rows = sum(m.get("rows") or 0 for m in models) // max(1, len(models))
            lines.append(f"The per-system fault models learned from about {rows} assessments each, of "
                         f"{max((m.get('vehicles') or 0) for m in models)} vehicles"
                         + (" (the simulated SAMPLE- fleet)" if simulated else "") + ".")
        if visits:
            lines.append(f"The service-need model learned from {visits:,} workshop visits in a public dataset, not from your cars.")
        lines.append(NEXT_STEPS)
        return "\n".join(lines)
    if MODEL_STATUS.search(text):
        lines = [f"{len(in_use)} of {len(models)} per-system fault models are in use"
                 + (f" ({_join_names(m['name'] for m in in_use)})" if in_use else "") + "."]
        lines += [model_line(m) for m in others]
        if service_need_line():
            lines.append(service_need_line())
        if simulated:
            lines.append(SIMULATED_WARNING)
        lines.append("The condition screening (assessment ratings, findings and their trend) is always on, whatever the models' status.")
        return "\n".join(lines)
    lines = ["Maintenance predictions here work in three layers:",
             "1. Condition screening (always on): every vehicle is assessed monthly and each system gets a rating. The ratings, the "
             "findings and how they changed since the previous assessment decide which cars are at risk; that's what I use when "
             "you ask which cars or components need attention.",
             f"2. Per-system fault models: each predicts whether a system will have a verified fault within {horizon} days of an "
             f"assessment, from the assessment's answers. {len(in_use)} of {len(models)} are in use"
             + (f" ({_join_names(m['name'] for m in in_use)})" if in_use else "") + "; ask \"how accurate is the maintenance model\" "
             "for their figures."]
    if service_need_line():
        lines.append("3. " + service_need_line())
    if simulated:
        lines.append(SIMULATED_WARNING)
    lines.append(NEXT_STEPS)
    return "\n".join(lines)


def _join_names(names):
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1] if names else ""


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


def _short_dtc(text):
    """The code's name, its three most likely causes and typical signs; the detection details stay out of a chat answer."""
    head = re.match(r"^[A-Z]\d{4}: [^.]+\.", text)
    if not head:
        return text
    parts = [head.group(0)]
    causes = re.search(r"Common causes: (.*?)(?= Symptoms:| Repair difficulty:|$)", text)
    if causes:
        items = [re.sub(r"\s*\((?:high|medium|low) likelihood\)", "", item).strip(" .") for item in causes.group(1).split(";")]
        items = [item if item[:2].isupper() else item[0].lower() + item[1:] for item in items if item]
        parts.append("Common causes: " + "; ".join(items[:3]) + ".")
    symptoms = re.search(r"Symptoms: (.*?)(?= Repair difficulty:|$)", text)
    if symptoms:
        items = [item.strip(" .") for item in symptoms.group(1).split(";") if item.strip(" .")]
        if items:
            parts.append("Typical signs: " + "; ".join(item[0].lower() + item[1:] for item in items[:2]) + ".")
    return " ".join(parts)


def _dtc_causes(text):
    causes = re.search(r"Common causes: (.*?)(?= Symptoms:| Repair difficulty:|$)", text)
    if not causes:
        return []
    items = [re.sub(r"\s*\((?:high|medium|low) likelihood\)", "", item).strip(" .") for item in causes.group(1).split(";")]
    return [item if item[:2].isupper() else item[0].lower() + item[1:] for item in items if item]


def _dtc_focused(codes, passages, focus):
    """A short answer to "is it serious?" or "what causes it?" about codes already defined."""
    found = [code for code in codes if code in passages]
    if focus == "cost":
        # "How much to fix it?": what the bill depends on, without inventing prices.
        lines = []
        for code in found:
            causes = _dtc_causes(passages[code]["text"])
            lines.append(f"I can't quote an exact price for {code} without the car being checked: it depends on what's behind it"
                         + (f", usually one of these: {'; '.join(causes)}." if causes else "."))
        return ("\n\n".join(lines) + "\n\nHave a shop confirm the cause with tests before replacing anything; with fault codes, guessing "
                "at parts is how a cheap fix becomes an expensive one. Reading the codes costs little, and ask for a written quote "
                "after the diagnosis.") if lines else None
    if focus == "causes":
        lines = [f"The usual causes of {code} are: " + "; ".join(_dtc_causes(passages[code]["text"])) + "."
                 for code in found if _dtc_causes(passages[code]["text"])]
        return "\n\n".join(lines) + "\n\nConfirm the cause with an inspection or tests before replacing parts." if lines else None
    notes = []
    for code in found:
        metadata = passages[code]["metadata"]
        note = f"{code} normally turns on the check-engine light" if metadata.get("mil") else f"{code} is a stored fault code"
        if metadata.get("limp_mode_possible"):
            note += ", and the car may go into reduced-power (limp) mode"
        notes.append(note + ".")
    return " ".join(notes) + (
        " A steady check-engine light usually means have it checked soon rather than stop straight away. If the light starts "
        "flashing, or the car overheats, loses power or drives abnormally, stop when it's safe and arrange an inspection.") if notes else None


def _dtc_response(codes, role, detailed=False, focus=None):
    try:
        passages = {p["metadata"].get("dtc_code"): p for code in codes for p in retrieve(code, role, top_k=1, source_types={"dtc_reference"})}
    except KnowledgeIndexUnavailable as error:
        return _with_evidence(core.build_response("dtc_lookup", 0.0, f"The diagnostic code reference is unavailable: {error}"), "none", [])
    focused = _dtc_focused(codes, passages, focus) if focus else None
    if focused:
        return _with_evidence(core.build_response("dtc_lookup", 1.0, focused, {"dtc_codes": codes, "dtc_not_found": [], "focus": focus}),
                              "reference", [_citation(passages[code]) for code in codes if code in passages], support=1.0)
    lines, citations, missing = [], [], []
    for code in codes:
        passage = passages.get(code)
        if passage is None:
            missing.append(code)
            continue
        metadata = passage["metadata"]
        line = passage["text"] if detailed else _short_dtc(passage["text"])
        if metadata.get("mil"):
            line += " This code normally turns on the check-engine (MIL) light."
        if metadata.get("limp_mode_possible"):
            line += " Reduced-power (limp) mode is possible."
        lines.append(line)
        citations.append(_citation(passage))
    answer = "\n\n".join(lines)
    if missing:
        answer += ("\n\n" if answer else "") + (
            f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not in the generic OBD-II code list. "
            "Codes like P1xxx, B1xxx or U1xxx are usually manufacturer-specific: tell me the make, model and year, "
            "or check the manufacturer's service information or a scan tool that supports that brand."
        )
    if lines:
        answer += "\n\nA code shows which circuit or test failed, so confirm the cause with an inspection before replacing parts."
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
    if top == "system_documentation" and _rental_operation_topic(message):
        return None  # Inspecting the car or reporting damage: the vehicle guidance answers, not the booking guide.
    if top == "system_documentation" or (top == "pid_reference" and re.search(r"\bpids?\b", message, re.IGNORECASE)):
        return _grounded_answer(message, role, {"intent": None, "confidence": 0.0}, passages=passages)
    return None


def _is_greeting(message):
    match = GREETING.match(message.strip())
    if not match:
        return False
    rest = re.findall(r"[a-z']+", match.group("rest").casefold())
    return all(word in GREETING_FILLER | COURTESY_WORDS | THANKS_WORDS for word in rest)


def _small_talk_close(message):
    """ "goodbye" or "acknowledgement" for a message that only thanks, acknowledges or says goodbye ("Got it, thanks.", "sige po"); else None."""
    words = re.findall(r"[a-z']+", message.casefold())
    if not words or not all(word in ACK_WORDS | GOODBYE_WORDS | THANKS_WORDS for word in words):
        return None
    # "you" or "it" alone is not an acknowledgement.
    if not set(words) & (ACK_WORDS | GOODBYE_WORDS | {"thanks", "thank", "thx", "tnx", "ty", "salamat"}) - {"it", "that", "that's", "for", "the", "your", "all", "ng", "po", "and", "time", "help", "one", "very"}:
        return None
    return "goodbye" if set(words) & GOODBYE_WORDS - {"now", "soon", "take"} else "acknowledgement"


def _is_guide_question(message):
    """Rental-policy or system how-to questions with no car-problem words."""
    if RECORD_PROCEDURE.search(message):
        return True
    if MECHANICAL.search(message) or extract_entities(message)["dtc_codes"]:
        return False
    if _rental_operation_topic(message):
        # Inspecting the car, or reporting damage or a fault, is vehicle guidance rather than booking policy.
        return False
    return bool(RENTAL_POLICY.search(message) or SYSTEM_PROCEDURE.search(message))


def _grounded_answer(message, role, core_response, passages=None, source_types=FREE_TEXT_SOURCES):
    try:
        if passages is None:
            passages = retrieve(message, role, top_k=3, source_types=source_types)
    except KnowledgeIndexUnavailable as error:
        logger.warning("knowledge index unavailable: %s", error)
        return None
    if passages and passages[0]["source_type"] == "curated_reference":
        # The knowledge retrieval has stricter topic checks; a curated entry it rejects for this message
        # ("my car caught fire" vs. brake grinding) is not used from this index either.
        agreed = retrieve_knowledge(message)
        agreed_id = f"kb:{agreed['document']['id']}" if agreed is not None else None
        passages = [p for p in passages if p["source_type"] != "curated_reference" or p["id"] == agreed_id]
    if not passages or passages[0]["support"] < settings.CHATBOT_MIN_RETRIEVAL_SUPPORT:
        return None
    best = passages[0]
    used = [best] + [p for p in passages[1:] if p["support"] >= settings.CHATBOT_STRONG_RETRIEVAL_SUPPORT
                     and p["source_type"] == best["source_type"]][:1]
    # Which source answered stays in data.evidence; the text itself carries no source labels.
    curated = knowledge_document(best["id"][3:]) if best["source_type"] == "curated_reference" and best["id"].startswith("kb:") else None
    shown_parts = []
    if curated is not None and has_chat_fields(curated):
        used = [best]
        answer, shown_parts = compose_chat_answer(curated, message)
    elif best["source_type"] == "system_documentation":
        used = [best]
        answer = _focused_guide_text(best["text"], message)
    elif best["source_type"] == "diagnostic_flowchart" and _flowchart_sentences(best["text"]):
        used = [best]
        answer = _flowchart_sentences(best["text"])
    else:
        answer = " ".join(f"{p['title']}. {p['text']}" if p is not best else p["text"] for p in used)
    if best["support"] < settings.CHATBOT_STRONG_RETRIEVAL_SUPPORT:
        answer += " If that isn't what you meant, tell me a bit more."
    intent = core_response.get("intent") if core_response.get("intent") not in {"unknown", None} else best["metadata"].get("category", "reference")
    data = {"retrieved_ids": [p["id"] for p in passages], "retrieval_scores": [p["score"] for p in passages]}
    if curated is not None:
        data.update(knowledge_id=curated["id"], knowledge_ids=[curated["id"]], shown_parts=shown_parts)
    response = core.build_response(intent, core_response.get("confidence", 0.0), answer, data)
    return _with_evidence(response, "reference", [_citation(p) for p in used], support=best["support"])


# How-to and requirements questions keep the whole entry: every step or requirement matters. ("how do extend": normalised text.)
GUIDE_PROCEDURE = re.compile(r"\b(?:how (?:do|can|should)(?: i| we)?|how to|steps?|procedure|process|paano|walk me through"
                             r"|requirements?|required|needed|what do i need|kailangan)\b", re.IGNORECASE)
GUIDE_FOCUS_MIN_WORDS = 60
GUIDE_STOP_WORDS = {
    "a", "an", "the", "i", "me", "my", "you", "your", "we", "is", "are", "was", "be", "do", "does", "did", "can", "could", "will", "would",
    "should", "to", "of", "for", "in", "on", "at", "with", "and", "or", "it", "this", "that", "there", "what", "which", "who", "when", "where",
    "why", "how", "any", "if", "about", "need", "have", "has", "get", "please", "po", "ba", "system", "driverent", "rent", "rental", "car",
    "vehicle", "booking", "reservation", "reserve", "book",
}
# Question words that the guide words differently ("contract" is written "rental agreement").
GUIDE_SYNONYMS = {
    "contract": {"agreement"}, "sign": {"accept", "agreement"}, "longest": {"most", "maximum", "longer"}, "maximum": {"most", "longer"},
    "max": {"most", "longer"}, "shortest": {"least", "minimum"}, "minimum": {"least"}, "refund": {"refund", "refunded", "refunds"},
    "late": {"late-return", "return"}, "fee": {"fees", "charge", "charges"}, "fees": {"fee", "charge", "charges"},
    "approval": {"approve", "automatically"}, "approved": {"approve", "automatically"}, "airport": {"location"},
    "hours": {"hourly"}, "hour": {"hourly"}, "age": {"minimum-age", "age"}, "old": {"minimum-age", "age"},
    "charge": {"fee", "fees"}, "charges": {"fee", "fees"}, "penalty": {"fee", "fees"},
    "cancelling": {"cancellation"}, "canceling": {"cancellation"}, "cancel": {"cancellation"},
    "refundable": {"refund", "refunds", "returned"}, "gcash": {"methods", "xendit"}, "card": {"methods", "xendit"},
    "credit": {"methods", "xendit"}, "license": {"licence"}, "licence": {"license"}, "documents": {"licence", "document"},
    "id": {"licence", "document"}, "expire": {"expires", "lasts"}, "expired": {"expires", "lasts"}, "paid": {"verified", "verifies"},
}


def _focused_guide_text(text, message):
    """The sentences of a guide entry that answer this question ("what's the longest I can rent?" gets the limit, not
    the whole booking procedure). The full entry is kept for how-to questions, short entries, or when no sentence
    clearly matches better than the rest."""
    words = text.split()
    if len(words) < GUIDE_FOCUS_MIN_WORDS or GUIDE_PROCEDURE.search(message):
        return text
    sentences = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text.strip()) if s]
    if len(sentences) < 3:
        return text
    terms = {w for w in re.findall(r"[a-z0-9-]+", message.casefold()) if w not in GUIDE_STOP_WORDS and len(w) > 2}
    terms |= {synonym for term in list(terms) for synonym in GUIDE_SYNONYMS.get(term, ())}
    if not terms:
        return text

    def score(sentence):
        # Hyphenated words count whole and by part ("advance-booking" matches "advance").
        whole = re.findall(r"[a-z0-9-]+", sentence.casefold())
        tokens = set(whole) | {part for word in whole for part in word.split("-") if part}
        return sum(1 for term in terms if any(_terms_match(term, token) for token in tokens))

    scores = [score(sentence) for sentence in sentences]
    best = max(scores)
    kept = [sentence for sentence, value in zip(sentences, scores) if value == best and value > 0]
    # Only when the match is clear: a few sentences stand out and most of the entry is left out.
    if not kept or len(kept) > 2 or sum(len(s.split()) for s in kept) > 0.6 * len(words):
        return text
    return " ".join(kept)


FLOWCHART = re.compile(r"^(?P<system>[^–.]+?)\s+–\s+(?P<component>[^.]+)\.\s+Symptoms:\s+(?P<symptoms>[^.]+)\.\s+Diagnostic steps:\s+(?P<steps>.+?)\.?\s*$")


def _flowchart_sentences(text):
    """A diagnostic flowchart entry as plain sentences, or None if it is not in the flowchart shape.

    "Fuel System – Fuel Injector. Symptoms: A; B. Diagnostic steps: X (possible results: ...); Y (...)." becomes
    "This can point to the fuel injector (fuel system), which typically causes a and b. A mechanic would usually x, then y.";
    the lists of possible results are left out."""
    match = FLOWCHART.match(text.strip())
    if not match:
        return None

    def lower(item):
        # Lower case except acronyms ("ABS", "ECU", "M.A.P.").
        return " ".join(word if sum(c.isupper() for c in word) > 1 else word.lower() for word in item.strip().split())

    symptoms = [lower(item) for item in match.group("symptoms").split(";") if item.strip()]
    steps = [lower(re.sub(r"\s*\(possible results:[^)]*\)", "", item)) for item in match.group("steps").split(";") if item.strip()]
    if not symptoms or not steps:
        return None
    component, system = lower(match.group("component")), lower(match.group("system"))
    step_text = steps[0] if len(steps) == 1 else ", then ".join(steps)
    return (f"This can point to the {component} in the {system}. Typical signs: {'; '.join(symptoms)}. "
            f"A mechanic would usually {step_text}.")


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


def _topic_follow_up(message, entities, memory):
    """(response, memory updates) when the message follows up on the car problem just discussed, else None.

    Only short follow-ups are handled here; a message that clearly names another topic, a
    trouble code, or anything not covered below goes through normal routing."""
    topic = memory.get("last_topic") if isinstance(memory, dict) else None
    if not isinstance(topic, dict) or not topic.get("id") or entities["dtc_codes"]:
        return None
    text = message.strip()
    document = knowledge_document(topic["id"])
    if document is None or not has_chat_fields(document):
        return None
    if len(re.findall(r"[a-z0-9']+", text.casefold())) <= 12:
        # A short reply to the narrowing question just asked ("in traffic", "the coolant is low", "oo, umiikot").
        other = retrieve_knowledge(text)
        if other is None or other["document"]["id"] == document["id"] or other.get("phrase_length", 0) == 0:
            reply = triage_reply(document, normalize_message(text), topic.get("shown"))
            if reply is not None:
                answer, shown = reply
                response = core.build_response(document["category"], 1.0, answer,
                                               {"knowledge_id": document["id"], "knowledge_ids": [document["id"]], "follow_up": "triage"})
                response = _with_evidence(response, "reference", [{"name": "Curated automotive knowledge base", "ids": [document["id"]]}])
                response["data"]["resolved_from_conversation"] = {"topic": document["id"]}
                return response, {"last_topic": {"id": document["id"], "shown": sorted(set(topic.get("shown") or []) | set(shown))}}
            asked = any(str(part).startswith("triage_asked:") for part in topic.get("shown") or [])
            if asked and other is not None and other["document"]["id"] == document["id"] and not TOPIC_FOLLOW_UP.search(text):
                # A reply about the same problem that none of the narrowing answers covers: carry on from what was
                # said instead of repeating the first answer.
                answer, shown = compose_follow_up(document, "more", topic.get("shown"))
                response = core.build_response(document["category"], 1.0, answer,
                                               {"knowledge_id": document["id"], "knowledge_ids": [document["id"]], "follow_up": "more"})
                response = _with_evidence(response, "reference", [{"name": "Curated automotive knowledge base", "ids": [document["id"]]}])
                response["data"]["resolved_from_conversation"] = {"topic": document["id"]}
                return response, {"last_topic": {"id": document["id"], "shown": sorted(set(topic.get("shown") or []) | set(shown))}}
    if len(re.findall(r"[a-z0-9']+", text.casefold())) > 16 or not TOPIC_FOLLOW_UP.search(text):
        return None
    document_text = " ".join([document["answer"], document["title"], " ".join(document.get("key_causes") or []),
                              " ".join(document.get("actions") or []), " ".join(document.get("keywords") or []),
                              " ".join(document.get("condition_hints") or {})])
    ruled_out = CONDITION.search(text) or CHECKED.search(text) or NOT_THE.search(text)
    about_this_topic = ruled_out is not None and mentions(document_text, [
        term for term in re.findall(r"[a-z]+", ruled_out.group("subject").casefold()) if term not in {"the", "my", "a", "an", "it", "that", "this"}])
    other = retrieve_knowledge(text)
    if not about_this_topic and other is not None and other["document"]["id"] != document["id"] and other.get("phrase_length", 0) > 0:
        return None  # "what about a flat tire?" after overheating: a new topic. "I replaced the spark plugs" after a misfire is not.
    switch = TOPIC_SWITCH.match(text)
    if switch:
        subject_terms = [term for term in re.findall(r"[a-z]+", switch.group("subject").casefold()) if term not in {"the", "my", "a", "an", "if", "it"}]
        if not subject_terms or not mentions(document_text, subject_terms):
            return None

    kind, subject, lead = None, None, None
    condition, checked, not_the = CONDITION.search(text), CHECKED.search(text), NOT_THE.search(text)
    if ASKS_SAFE.search(text) and not condition:
        kind = "safety"
    elif ASKS_COST.search(text):
        kind = "cost"
    elif condition and condition.group("subject").casefold() not in {"it", "that", "this"}:
        kind, subject = "condition", condition.group("subject")
        lead = f"If the {subject} {condition.group('verb')} {condition.group('state')}"
    elif checked:
        kind, subject = "condition", checked.group("subject")
        lead = f"With the {subject} already {checked.group('verb')}"
    elif not_the:
        kind, subject = "condition", not_the.group("subject")
        lead = f"If it isn't the {subject}"
    elif ASKS_WHY.search(text):
        kind = "causes"
    elif ASKS_FIX.search(text):
        kind = "fix"
    elif re.search(r"\b(?:what else|anything else|any ?more|more|tell me more|more details?|more info|explain more|ano pa|iba pa|meron pa|go on|continue)\b",
                   text, re.IGNORECASE):
        kind = "more"
    if kind is None:
        return None

    answer, shown = compose_follow_up(document, kind, topic.get("shown"), subject=subject, lead=lead)
    response = core.build_response(document["category"], 1.0, answer,
                                   {"knowledge_id": document["id"], "knowledge_ids": [document["id"]], "follow_up": kind})
    response = _with_evidence(response, "reference", [{"name": "Curated automotive knowledge base", "ids": [document["id"]]}])
    response["data"]["resolved_from_conversation"] = {"topic": document["id"]}
    updates = {"last_topic": {"id": document["id"], "shown": sorted(set(topic.get("shown") or []) | set(shown))}}
    return response, updates


# Classifier intents answered from the system guide, and how sure it must be to override a vague rule-based plan.
POLICY_INTENTS = {"rental_duration", "rental_requirements", "booking_reservation", "deposits_payments", "pickup_return",
                  "cancellation_rescheduling"}
POLICY_OVERRIDE_CONFIDENCE = 0.5


def _policy_question_over_vague_plan(message, plan):
    """True when the rules made only a vague plan (every listed vehicle, or every booking, with no filter or specific
    purpose) and the trained classifier is confident the question is about a rental policy. That question is then
    answered from the guide ("ilang araw minimum ng rental nyo?" is not a request to list the fleet)."""
    retrievals = plan.get("retrievals") or []
    if len(retrievals) != 1 or plan.get("follow_up"):
        return False
    tool, args = retrievals[0].get("tool"), retrievals[0].get("args") or {}
    requested = plan.get("requested") or {}
    named = any(requested.get(key) for key in ("makes", "models", "categories", "unknown_names"))
    vague_listing = tool == "vehicles.search" and not named and not any(
        value for key, value in args.items() if key not in ("limit",)) and plan.get("purpose") in (None, "listing", "price", "details")
    vague_bookings = tool == "bookings.mine" and not args and plan.get("purpose") in (None, "list") and not plan.get("guide")
    if not (vague_listing or vague_bookings):
        return False
    try:
        pipeline, _ = core.load_chatbot()
        probabilities = pipeline.predict_proba([normalize_message(message)])[0]
    except (FileNotFoundError, ValueError):
        return False
    best = int(probabilities.argmax())
    return str(pipeline.classes_[best]) in POLICY_INTENTS and float(probabilities[best]) >= POLICY_OVERRIDE_CONFIDENCE


def _is_car_problem(message, entities):
    """True for a question about a car problem or maintenance topic, which needs no rental records.

    "I smell gasoline near the car" names a fuel type and "hard to shift gears in my manual car" a
    transmission, but neither is an inventory search: a keyword phrase of a knowledge topic
    matched and no booking, price or inventory words are present."""
    if entities["dtc_codes"] or RENTAL_REQUEST.search(message) or NAMED_VEHICLE_SPEC.search(message):
        return False
    knowledge = retrieve_knowledge(message)
    return knowledge is not None and knowledge.get("phrase_length", 0) > 0


def _rental_operation_topic(message):
    """True when a keyword phrase of a rental-operations topic (inspection, damage or fault reporting) matches."""
    knowledge = retrieve_knowledge(message)
    return knowledge is not None and knowledge.get("phrase_length", 0) > 0 and knowledge["document"]["id"] in RENTAL_OPERATION_TOPICS


def _vague_symptom(message):
    """(kind, clarifying question) when the message names a kind of symptom and nothing specific, else None."""
    words = re.findall(r"[a-z']+", message.casefold())
    if not words or len(words) > 14 or any(word not in VAGUE_WORDS for word in words):
        return None
    for pattern, kind, question in VAGUE_SYMPTOMS:
        if re.search(pattern, message, re.IGNORECASE):
            return kind, question
    return None


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


PENDING_PICKUP_TURNS = 3


def _pending_pickup(memory, updates):
    """The pickup date still waiting for a return date, and how many more messages it is kept for."""
    if "pending_pickup" in updates:
        date_value = updates["pending_pickup"]
        return {"pending_pickup": date_value, "pending_pickup_left": PENDING_PICKUP_TURNS if date_value else 0}
    left = int(memory.get("pending_pickup_left") or 0) - 1
    if memory.get("pending_pickup") and left > 0:
        return {"pending_pickup": memory["pending_pickup"], "pending_pickup_left": left}
    return {"pending_pickup": None, "pending_pickup_left": 0}


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
              "last_period": (updates or {}).get("last_period", memory.get("last_period")),
              # A pickup date waiting for its return date: kept for up to three more messages, cleared once used.
              **_pending_pickup(memory, updates or {}),
              "last_topic": (updates or {})["last_topic"] if "last_topic" in (updates or {}) else memory.get("last_topic")}
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
