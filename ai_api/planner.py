"""Understands a chat message and plans which database facts are needed to answer it.

The planner never touches the database. It returns a plan made only of the read-only
tools the Back-End's ChatRetrieval accepts; the Back-End validates the plan, applies
the signed-in user's role and identity, runs it, and sends the results back for
grounded_answers.compose(). A plan of None means no database facts are needed (car
problems, policies, how-to questions), and the existing guide/knowledge routing answers.

Understanding is built from:
  - the message normalised by language.normalize_message (typos, Tagalog/Taglish);
  - concept groups (price, deposit, booking, ...) matched as words or phrases, with
    typo-tolerant fuzzy matching for longer words;
  - vehicle names, makes and categories taken from the live database vocabulary the
    Back-End sends with every message, so new vehicles need no code change;
  - numbers, durations, dates, seat counts and budgets;
  - the conversation memory, for follow-ups such as "how much for 3 days?".
"""
import difflib
import re
from datetime import date, timedelta

from django.utils import timezone

from .language import normalize_message
from .rag import extract_entities


CONCEPTS = {
    "price": {
        "words": {"price", "prices", "pricing", "rate", "rates", "cost", "costs", "fee", "fees", "charge", "charges",
                  "budget", "cheap", "cheaper", "cheapest", "mura", "pinakamura", "expensive", "mahal", "pinakamahal",
                  "afford", "total", "quote", "presyo", "halaga"},
        "phrases": ["how much", "per day", "a day", "daily rate", "per night"],
    },
    "deposit": {"words": {"deposit", "deposits", "refundable", "refund", "refunds"}, "phrases": ["security deposit"]},
    "availability": {
        "words": {"available", "availability", "bakante", "vacant", "free", "rentable"},
        "phrases": ["for rent", "can i rent", "can i book", "can i reserve", "is there any", "are there any"],
    },
    "listing": {
        "words": {"what", "which", "list", "show", "anong", "ano", "lahat", "all", "options", "choices", "meron", "offer", "units"},
        "phrases": ["do you have", "may available", "what cars", "what vehicles", "any cars", "kinds of", "see the cars",
                    "see your cars", "see the vehicles", "see all", "see available", "your fleet", "your cars",
                    "your vehicles", "your units", "what you have", "what do you have", "what have you got"],
    },
    "specs": {
        "words": {"features", "feature", "specs", "spec", "specifications", "seats", "seat", "seater", "seating", "transmission",
                  "automatic", "manual", "fuel", "diesel", "gasoline", "gas", "petrol", "gps", "aircon", "airconditioning",
                  "color", "colour", "year", "mileage", "details", "detail", "info", "information", "describe", "model",
                  "plate", "plaka", "minimum", "kulay", "upuan", "dashcam", "camera", "bluetooth", "usb", "sunroof", "carplay",
                  "equipment", "equipped", "keyless", "touchscreen", "sensors", "navigation", "nav", "ac", "transmission",
                  "odometer", "km", "kilometers", "conditioned", "airconditioned", "gearbox", "matic"},
        "phrases": ["tell me about", "how many seats", "how many people", "what kind", "dash cam", "child seat", "baby seat",
                    "cruise control", "android auto", "roof rack"],
    },
    "vehicle_noun": {"words": {"car", "cars", "vehicle", "vehicles", "unit", "units", "auto", "suv", "suvs", "sedan", "sedans",
                               "van", "vans", "pickup", "truck", "rental", "rentals", "rent", "renting", "fleet"}},
    "own": {"words": {"my", "mine", "ko", "ako", "akin", "namin", "our"},
            "phrases": ["do i have", "have i", "i booked", "i reserved", "i rented", "did i", "i spent", "days are left", "days left"]},
    "booking": {"words": {"booking", "bookings", "reservation", "reservations", "reserved", "booked", "rental", "rentals",
                          "pickup", "trip", "trips", "book"},
                "phrases": ["rental history", "return date", "pick up", "days left", "days are left", "spent so far",
                            "have i spent", "did i spend", "next pickup", "my pickup", "my return"]},
    "upcoming": {"words": {"upcoming", "next", "current", "active", "susunod", "future", "pending", "confirmed"}},
    "how_to": {"words": {"paano", "steps", "procedure", "process"},
               "phrases": ["how do i", "how can i", "how to", "can i", "where can i", "what are the steps", "is it possible"]},
    "action": {"words": {"extend", "extension", "cancel", "change", "reschedule", "modify", "move", "pay", "register",
                         "signup", "approve", "confirm", "reject", "add", "delete", "remove", "update", "edit", "upload",
                         "create"},
               "phrases": ["sign up", "log in", "create an account"]},
    "policy": {"words": {"policy", "policies", "rule", "rules", "allowed", "require", "required", "requirement", "requirements",
                         "license", "licence", "age", "old", "discount", "discounts", "promo", "promos", "promotion", "penalty",
                         "late", "insurance", "refund", "refundable", "payment", "payments", "pay", "gcash", "maya", "paymaya",
                         "cash", "card", "credit", "debit", "bank", "documents", "document", "provide", "bring",
                         "password", "insurance", "insured", "delivery", "deliver", "hours", "contact"},
               "phrases": ["free day", "free days", "is it true", "any promo"]},
    "maintenance": {"words": {"maintenance", "service", "serviced", "servicing", "repair", "repairs", "repaired", "inspection",
                              "inspections", "inspected", "due", "overdue", "recurring", "repeated", "predictive", "prediction",
                              "predictions", "risk", "risky", "diagnostic", "diagnostics", "dtc", "condition", "history", "records",
                              "record", "logs", "activities", "activity", "worked", "work", "maintain", "breakdown", "breakdowns"},
                    "phrases": ["trouble code", "trouble codes", "fault code", "fault codes", "needs service", "need service",
                                "break down", "breaks down"]},
    "status": {"words": {"status", "rented", "unavailable", "reserved", "booked", "maintenance", "utilization"},
               "phrases": ["how many", "in maintenance", "under maintenance", "on rent", "not available", "out of service",
                           "out on rent", "in the shop", "being repaired", "under repair"]},
    "other_people": {"words": {"customers", "users", "everyone", "everybody", "others", "people's",
                               "customer's", "customers'", "phone", "phones", "email", "emails", "address", "addresses",
                               "contact", "contacts"},
                     "phrases": ["other customers", "other users", "every customer", "each customer", "all customers",
                                 "all reservations", "every reservation", "all bookings", "everyone's", "personal information",
                                 "phone number", "phone numbers", "in the system", "someone else"]},
    "explain": {"words": {"accurate", "accuracy", "limits", "limitations", "reliable", "explain", "meaning"},
                "phrases": ["what does", "how does", "how is the", "when is a", "how are"]},
    "symptom": {"words": {"engine", "brake", "brakes", "braking", "smoke", "smoking", "noise", "noises", "squeak", "squeaks",
                          "squeaking", "squeal", "squealing", "grind", "grinding", "leak", "leaking", "overheat", "overheating",
                          "tire", "tires", "tyre", "tyres", "battery", "alternator", "starter", "misfire", "misfiring", "stall",
                          "stalls", "stalling", "vibration", "vibrates", "vibrating", "shake", "shakes", "shaking", "pulls",
                          "pulling", "drifts", "steering", "wobble", "coolant", "transmission", "exhaust", "clunk", "rattle",
                          "smell", "smells", "flashing", "blinking", "warning", "aircon", "hesitates", "jerks", "knocking"},
                "phrases": ["won't start", "wont start", "check engine", "warning light", "does not start", "doesn't start"]},
}
# Words that are numbers in English or Tagalog.
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "isa": 1, "isang": 1, "dalawa": 2, "dalawang": 2, "tatlo": 3, "tatlong": 3, "apat": 4, "lima": 5, "limang": 5,
    "anim": 6, "pito": 7, "pitong": 7, "walo": 8, "siyam": 9, "sampu": 10, "sampung": 10,
}
CATEGORY_SYNONYMS = {
    "suv": {"suv", "suvs", "crossover"}, "sedan": {"sedan", "sedans", "saloon"}, "van": {"van", "vans", "minivan", "minibus"},
    "pickup": {"pickup", "pickups", "pick-up", "truck", "trucks"}, "mpv": {"mpv", "mpvs"}, "hatchback": {"hatchback", "hatchbacks"},
    "motorcycle": {"motorcycle", "motorcycles", "motorbike", "scooter"},
}
# General automotive knowledge, not database contents: lets the assistant say that a brand
# the fleet does not carry is not listed, instead of answering about other vehicles.
CAR_BRANDS = {
    "toyota", "honda", "mitsubishi", "nissan", "ford", "hyundai", "kia", "suzuki", "mazda", "isuzu", "subaru", "chevrolet",
    "chevy", "geely", "chery", "mg", "byd", "volkswagen", "vw", "bmw", "mercedes", "mercedes-benz", "benz", "audi", "lexus",
    "volvo", "peugeot", "renault", "jeep", "tesla", "porsche", "ferrari", "lamborghini", "foton", "jac", "gac", "changan",
    "haval", "jetour", "yamaha", "kawasaki", "lexus", "land rover", "rover", "mini", "dodge", "ram", "infiniti", "acura",
}
BRAND_NAMES = {"bmw": "BMW", "mg": "MG", "byd": "BYD", "gac": "GAC", "jac": "JAC", "vw": "VW", "mercedes-benz": "Mercedes-Benz",
               "kia": "Kia", "mini": "MINI", "ram": "RAM", "land rover": "Land Rover"}
# Policies and services the application has no record of; the answer says so instead of guessing.
UNRECORDED_TOPICS = [
    # Age is in the system guide (no minimum-age rule is stored); licence types are not.
    ("which licence types are accepted", re.compile(
        r"\b(?:student|learner'?s?|foreign|international|provisional|temporary)\s+(?:driving\s+|driver'?s?\s+)?(?:permit|license|licence)\b")),
    ("a driver or chauffeur service", re.compile(
        r"\bchauffeur\w*\b|\bdriver service\b|\b(?:with|include|includes|including|provide|provides|offer|offers|hire|rent) (?:a |an |your )?(?:own )?driver\b"
        r"|\bmay (?:kasamang )?driver\b|\bkasama(?:ng)? driver\b")),
    ("smoking in the vehicle", re.compile(
        r"\b(?:can|may) (?:i|we) (?:smoke|vape)\b|\b(?:smoking|vaping) (?:allowed|policy|inside|in the (?:car|vehicle))\b|\bno smoking\b"
        r"|\b(?:is it ok|allowed) to (?:smoke|vape)\b|\b(?:pwede|bawal)(?: ba)?(?: mag)?(?:-| )?(?:yosi|manigarilyo|smoke|vape)\b"
        r"|\b(?:manigarilyo|magsigarilyo|magyosi|mag yosi|mag-yosi)\b")),
    ("pets in the vehicle", re.compile(
        r"\b(?:bring|take|travel with|allowed|pwede|isakay|isama)\b.*\b(?:pets?|dogs?|cats?|aso|pusa)\b"
        r"|\b(?:pets?|dogs?|cats?) (?:are |is )?(?:allowed|friendly|policy|ok|okay|welcome)\b")),
    ("additional drivers", re.compile(r"\b(?:someone else|another person|another driver|additional drivers?|second driver|other drivers?|my (?:friend|wife|husband|brother|sister|kapatid|asawa)|ibang (?:tao|driver))\b.*\b(?:drive|driver|magmaneho|mag-drive)\b|\badditional drivers?\b")),
    ("insurance", re.compile(r"\b(?:insurance|insured|insure|seguro)\b")),
    ("travel outside the area", re.compile(r"\b(?:another|other|ibang) (?:province|provinces|city|region|island|probinsya)\b|"
                                           r"\b(?:out of town|outside (?:the )?(?:city|metro manila|province|area|region)|cross[- ]country|road trip)\b")),
    ("home delivery", re.compile(r"\b(?:deliver|delivery|drop[- ]?off at|bring the car to (?:me|my|our)|ihatid|i-deliver)\b")),
    ("business hours", re.compile(r"\b(?:business hours|opening hours|office hours|what time do you (?:open|close)|are you open|open (?:on|at|today)|anong oras)\b")),
    ("late returns and late fees", re.compile(r"\blate (?:return|returns|fee|fees|charge|penalty)\b|\breturn\w* (?:the car |it )?late\b|\bhuli(?:ng)? (?:pagbalik|balik)\b")),
    ("contact details", re.compile(r"\b(?:contact (?:you|the rental team|number|details|info)|your (?:phone|number|email|address)|how (?:do|can) i (?:contact|reach|call)|hotline|where are you located|your location)\b")),
]
VEHICLE_STATUS_WORDS = {"rented": "rented", "maintenance": "maintenance", "repair": "maintenance", "unavailable": "unavailable",
                        "available": "available"}
STOP_AFTER_BRAND = {"per", "for", "the", "na", "po", "ba", "car", "cars", "vehicle", "vehicles", "price", "prices", "rate",
                    "rates", "cost", "is", "are", "available", "rental", "rent", "deposit", "and", "or", "unit", "units",
                    "today", "now", "tomorrow", "please", "shows", "has", "have", "with"}
TOKEN = re.compile(r"[a-z0-9']+")
FUEL_WORDS = {"diesel": "Diesel", "gasoline": "Gasoline", "gas": "Gasoline", "petrol": "Gasoline", "hybrid": "Hybrid",
              "electric": "Electric", "ev": "Electric", "gasolina": "Gasoline"}
TRANSMISSION_WORDS = {"automatic": "Automatic", "matic": "Automatic", "auto transmission": "Automatic", "manual": "Manual",
                      "stick": "Manual", "cvt": "CVT"}
COLOR_WORDS = {"white", "black", "red", "silver", "gray", "grey", "blue", "green", "yellow", "orange", "brown", "beige", "gold", "maroon",
               "puti", "itim", "pula", "asul"}
COLOR_CANONICAL = {"grey": "Gray", "puti": "White", "itim": "Black", "pula": "Red", "asul": "Blue"}
MONTHS = {name: number for number, names in enumerate([
    ("january", "jan", "enero"), ("february", "feb", "pebrero"), ("march", "mar", "marso"), ("april", "apr", "abril"),
    ("may", "mayo"), ("june", "jun", "hunyo"), ("july", "jul", "hulyo"), ("august", "aug", "agosto"),
    ("september", "sep", "sept", "setyembre"), ("october", "oct", "oktubre"), ("november", "nov", "nobyembre"),
    ("december", "dec", "disyembre")], start=1) for name in names}
WEEKDAYS = {name: index for index, names in enumerate([("monday", "lunes"), ("tuesday", "martes"), ("wednesday", "miyerkules"),
            ("thursday", "huwebes"), ("friday", "biyernes"), ("saturday", "sabado"), ("sunday", "linggo")]) for name in names}
YES_NO = re.compile(r"^(?:does|do|is|are|has|have|may|meron|can)\b|\b(?:ba|po ba)\b|\b(?:is|does) it\b"
                    r"|\b(?:gps|navigation|aircon|a/c|ac|automatic|manual|diesel|gasoline)\s*\?"
                    r"|\b(?:has|have|with|got)\s+(?:a |an )?(?:gps|navigation|nav|ac|a/c|aircon|air ?con\w*|air[- ]?condition\w*)\b")
# Words that ask for a broader list, so the question is not about the vehicle discussed before.
BROADER_LIST = re.compile(r"\b(?:all|list|lahat|options|others?|every|any other|ibang|mga)\b")
ELLIPTICAL = re.compile(r"^(?:and|what about|how about|eh|yung|paano yung|e di|so)\b")
PRONOUN = re.compile(r"\b(?:it(?! (?:true|possible|ok|okay|allowed|safe|normal|required)\b)|this one|that one|this car|that car|yan|iyan|yun|iyon|ito"
                     r"|(?:the )?(?:vehicle|car|one) i (?:selected|chose|picked|clicked)|selected (?:vehicle|car))\b")
# Questions about the assistant's own access are answered from the guide.
ABOUT_ASSISTANT = re.compile(r"\bcan (?:the )?(?:chatbot|assistant|bot|ai|you) (?:see|access|view|read|show)\b")
# Price questions about repairs or services, not rentals.
SERVICE_COST = re.compile(r"\b(?:oil change|repair|repairs|servicing|replacement|replace|parts?|tune[- ]?up|alignment|labou?r|fix|fixing)\b")
OFF_TOPIC = re.compile(r"\b(?:stock|shares?|stock market|net worth|ceo|founder|headquarters|weather|news|election|joke|jokes|poem|poems|song|songs"
                       r"|sing|movie|movies|bitcoin|crypto\w*|recipe|cook|cooking|basketball|football|lotto|horoscope)\b")
BARE_ATTRIBUTES = [
    ("transmission", re.compile(r"\btransmission\b|\bgear ?box\b")),
    ("color", re.compile(r"\bcolou?rs?\b|\bkulay\b")),
    ("seating_capacity", re.compile(r"\bseating(?: capacity)?\b|\bcapacity\b|\bseats?\b|\bpassengers?\b|\bupuan\b")),
    ("fuel_type", re.compile(r"\bfuel\b")),
    ("year", re.compile(r"\byear\b")),
    ("current_mileage", re.compile(r"\bmileage\b|\bodometer\b|\bkilomet(?:er|re)s?\b|\bkm\b")),
]
# "How much air in my tires?" asks about car care, not a price.
CARE_AMOUNT = re.compile(r"\bhow much (?:air|oil|coolant|water|fluid|pressure|psi)\b|\bpsi\b")
# A fee for cancelling, changing or returning late is a policy question for the guide.
POLICY_FEE = re.compile(r"\b(?:cancel\w*|late|penalty|penalties|change|changing|reschedul\w*|rebook\w*)\b.*\b(?:fee|fees|charge|charges|penalty)\b"
                        r"|\b(?:fee|fees|charge|charges)\b.*\b(?:cancel\w*|late|chang\w*)\b")
# "What do I need to rent a car?" asks for the requirements.
NEEDS_TO_RENT = re.compile(r"\bwhat (?:\w+ )?(?:do|does|would) (?:i|one|we|you) need\b(?!.*\b(?:\d+|people|persons|seats?|seater)\b)"
                           r"|\brequirements? (?:for|to) (?:rent|book|reserv)\w*")
# Asking how to do something in the system, as opposed to asking for records.
HOW_TO_DO = re.compile(r"\b(?:how (?:do|can|should|would) (?:i|we|an admin)|how to|steps|procedure|paano|how do i)\b")
# Asking about the fleet's records rather than describing one's own car ("DTC stored: P0999").
STORED_RECORDS = re.compile(r"\b(?:in (?:the|our) (?:records|fleet|system)|which (?:vehicles|cars|units)|our (?:vehicles|cars|fleet|units)"
                            r"|for (?:the|our|all) (?:vehicles|cars|units)|fleet|maintenance records?|service records?"
                            r"|(?:vehicle|car|unit)\s*(?:id\s*)?#?\s*\d+)\b")
OWN_PICKUP_RETURN = re.compile(r"\bwhen (?:do|should|must|will) i (?:return|pick up|pickup|get)\b")
OTHER_PERSON_RECORDS = re.compile(r"\b(?!(?:my|what|that|it|where|who|how|here|there|when|which|let)'s\b)(?!my\b)[a-z]+'s\s+"
                                  r"(?:bookings?|reservations?|rentals?|records?|details|account)\b")
# "How can I reserve the Vios?" asks for the booking steps, even with a vehicle named.
HOW_TO_BOOK = re.compile(r"\bhow (?:can|do|to|would|should) (?:i |we )?(?:book|reserve|rent|make a (?:booking|reservation))\b"
                         r"|\b(?:process|steps|procedure) (?:to|for) (?:book|reserv|rent)\w*")
# Common words that typo-tolerant matching must not read as concept words ("feels" is not "fees").
FUZZY_SKIP = {"feels", "feeling", "files", "fills", "fails", "falls", "seems", "sells", "tells", "wheels"}
FUZZY_CUTOFF = 0.86
# Text that is a database query, never something to run.
SQL_TEXT = re.compile(r"\b(?:select\s+[\w*,\s]+\s+from|insert\s+into|update\s+\w+\s+set|delete\s+from|drop\s+(?:table|database)|union\s+select)\b"
                      r"|'\s*(?:or|and)\s+'?\w+'?\s*=\s*'?\w+|\b(?:or|and)\s+1\s*=\s*1\b|;\s*--", re.IGNORECASE)
# Rental rules that share words with car symptoms ("can I smoke in the car?" is not about exhaust smoke).
SYMPTOM_SAFE_TOPICS = {"smoking in the vehicle", "pets in the vehicle"}
# Being in an accident: general safety steps, not company procedures.
ACCIDENT = re.compile(r"\b(?:accident|accidents|crash|crashed|collision|collided|fender bender|naaksidente|aksidente|nabangga|bumangga|banggaan)\b")
# Someone's password or other credentials (not "how do I change my password?").
CREDENTIALS = re.compile(r"\b(?:password|passwords|passcode|credentials?|login details|api key|secret key|access token)\b")
OWN_CREDENTIAL_HELP = re.compile(r"\b(?:my|change|reset|forgot|forgotten|update|new)\b")
# Equipment stored in the vehicle's additional_features list: canonical name -> phrasings.
EQUIPMENT = {
    "dash camera": ("dash cam", "dashcam", "dash camera", "dashboard camera"),
    "backup camera": ("backup camera", "reverse camera", "reversing camera", "rear camera", "rearview camera", "back camera", "parking camera"),
    "parking sensors": ("parking sensor", "parking sensors", "reverse sensor", "reverse sensors"),
    "Bluetooth": ("bluetooth",),
    "USB charging": ("usb", "usb port", "usb charger"),
    "child seat": ("child seat", "baby seat", "car seat", "booster seat"),
    "sunroof": ("sunroof", "moonroof"),
    "cruise control": ("cruise control",),
    "Apple CarPlay": ("carplay", "apple carplay"),
    "Android Auto": ("android auto",),
    "touchscreen": ("touchscreen", "touch screen", "infotainment"),
    "roof rack": ("roof rack",),
    "keyless entry": ("keyless", "push start", "push button start"),
}
# One requested attribute of a vehicle record ("what year is the Fortuner?").
ATTRIBUTES = [
    ("plate_number", re.compile(r"\bplate(?:\s*(?:number|no\.?|#))?\b|\bplaka\b|\bregistration (?:number|no\.?)\b|\blicen[cs]e plate\b")),
    ("minimum_rental_duration", re.compile(r"\bminimum (?:rental|rent|number of days|days?|duration|period|booking|stay)\b|\bmin(?:imum)? days?\b"
                                           r"|\bshortest (?:rental|booking|period)\b|\bat least how many days\b|\bminimum na araw\b|\bilang araw (?:ang )?minimum\b")),
    ("year", re.compile(r"\bwhat year\b|\bwhich year\b|\byear model\b|\bmodel year\b|\banong taon\b")),
    ("color", re.compile(r"\bwhat colou?r\b|\bwhich colou?r\b|\bcolou?r (?:of|is)\b|\bkulay\b")),
    ("seating_capacity", re.compile(r"\bhow many (?:seats|people|passengers|persons|pax)\b|\bseating capacity\b|\bilang (?:ang )?(?:upuan|tao|katao|seats)\b"
                                    r"|\bhow many (?:can|could|will) (?:ride|fit|sit|go)\b|\bnumber of (?:seats|passengers)\b|\bkasya\b")),
    ("fuel_type", re.compile(r"\b(?:diesel|gasoline|gas|petrol) or (?:diesel|gasoline|gas|petrol)\b|\bwhat (?:kind of |type of )?fuel\b|\bfuel type\b|\banong (?:gas|fuel)\b")),
    ("transmission", re.compile(r"\b(?:manual|automatic) or (?:manual|automatic)\b|\bwhat (?:kind of |type of )?transmission\b|\btransmission type\b")),
]
# "Which car has the most seats?" / "best for a big family": rank the listed cars by seating capacity.
SEATS_RANKING = re.compile(r"\b(?:most|more|biggest|largest|maximum) (?:seats|seating|people|passengers|space)\b|\bbig(?:gest)? (?:family|group)\b"
                           r"|\b(?:family|group|barkada) (?:trip|outing|car)\b|\bpinakamaluwag\b|\bmaraming (?:tao|upuan)\b|\bbiggest car\b|\blargest car\b")
# Weekly or monthly rates (only daily rates are stored).
RATE_PERIOD = re.compile(r"\b(?:weekly|monthly|per week|per month|a week rate|week rates?|month rates?|lingguhan|buwanan)\b")
# The customer's own past rentals.
PAST_RENTAL = re.compile(r"\b(?:did i|have i|i have|i've)\s+(?:rent|rented|book|booked|reserve|reserved|use|used)\b"
                         r"|\bi (?:rented|booked|reserved|used)\b|\blast time\b"
                         r"|\b(?:past|previous|last|recent|huling) (?:rentals?|bookings?|reservations?|trips?)\b|\brental history\b|\bnirentahan ko\b")
OWE = re.compile(r"\b(?:owe|balance|still (?:need|have) to pay|need to pay|babayaran|bayaran ko|utang)\b")
# Another person's records by name ("Ana Santos booking details") or by account id ("user_id=3").
NAMED_RECORDS = re.compile(r"\b(?!My\b)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:'s)?\s+(?:bookings?|reservations?|rentals?|booking details|reservation details|account)\b")
ACCOUNT_ID = re.compile(r"\b(?:user_?id|user|customer_?id|customer|account)\s*(?:=|:|#|no\.?|number)?\s*\d+\b")


class Understanding:
    def __init__(self, message, vocabulary):
        self.original = message
        # "air-conditioned" is the aircon, not the maintenance word "condition".
        self.text = re.sub(r"\bair[- ]?condition(?:ed|ing|er)?\b", "aircon", normalize_message(message).casefold())
        self.tokens = TOKEN.findall(self.text)
        self.concepts = {name for name, lexicon in CONCEPTS.items() if self._matches(lexicon)}
        self.dtc_codes = extract_entities(message)["dtc_codes"]
        self._numbers()
        self._vehicles(vocabulary or {})
        # "Toyota Fortuner 2021 lang" is a model year after a vehicle name, not a PHP 2,021 budget.
        if self._year_like_budget and (self.models or self.makes):
            self.max_daily_rate = None
            if self._price_from_budget:
                self.concepts.discard("price")

    def has(self, *names):
        return any(name in self.concepts for name in names)

    def _matches(self, lexicon):
        # Punctuation next to a phrase ("do I have??", "dash cam?") must not hide it.
        padded = " " + re.sub(r"[^\w'/-]+", " ", self.text) + " "
        if any(f" {phrase} " in padded for phrase in lexicon.get("phrases", [])):
            return True
        words = lexicon.get("words", set())
        for token in self.tokens:
            if token in words:
                return True
            if len(token) >= 5 and token not in FUZZY_SKIP and _fuzzy(token, words):
                return True
        return False

    def _vehicles(self, vocabulary):
        """Makes, models and categories from the database vocabulary named in the message."""
        self.makes, self.models, self.categories = [], [], []
        joined = " " + " ".join(self.tokens) + " "
        for make in vocabulary.get("makes", []):
            if _names_word(make, self.tokens, joined):
                self.makes.append(make)
        for entry in vocabulary.get("models", []):
            if _names_word(entry.get("model", ""), self.tokens, joined):
                self.models.append(entry)
        listed_makes = {str(make).casefold() for make in vocabulary.get("makes", [])}
        self.unlisted_brands = []
        for index, token in enumerate(self.tokens):
            if token in CAR_BRANDS and token not in listed_makes:
                following = self.tokens[index + 1] if index + 1 < len(self.tokens) else ""
                model = following if re.fullmatch(r"[a-z][a-z0-9-]{1,}", following) and following not in STOP_AFTER_BRAND else ""
                brand = BRAND_NAMES.get(token, token.title())
                self.unlisted_brands.append(f"{brand} {model.title() if not model.isupper() else model}".strip())
        known = {str(category).casefold(): category for category in vocabulary.get("categories", [])}
        self.unlisted_categories = []
        for key, synonyms in CATEGORY_SYNONYMS.items():
            if any(token in synonyms for token in self.tokens):
                if key in known:
                    self.categories.append(known[key])
                else:
                    self.unlisted_categories.append(key)
        for key, category in known.items():
            if category not in self.categories and _names_word(category, self.tokens, joined):
                self.categories.append(category)
        # A model name can be an ordinary word ("City"); without a make, a vehicle word or a
        # rental question it is not a vehicle reference.
        signal = (self.makes or self.has("price", "deposit", "availability", "specs", "vehicle_noun", "booking", "status", "maintenance")
                  or len(self.tokens) <= 3 or self.start_date or self.rental_days or len(self.models) >= 2
                  or self.equipment or self.attribute or self.colors or self.fuel_types or self.transmissions
                  or re.match(r"(?:and|what about|how about|eh|yung|e ang|paano ang)\b", self.text))
        if not signal:
            self.models = []

    def _numbers(self):
        text = self.text
        self.rental_days = None
        match = re.search(r"\b(\d{1,3})\s*(?:-\s*)?(?:days?|nights?|d)\b", text)
        if match:
            self.rental_days = int(match.group(1))
        else:
            match = re.search(r"\b(" + "|".join(NUMBER_WORDS) + r")\s+(?:na\s+)?(?:days?|nights?)\b", text)
            if match:
                self.rental_days = NUMBER_WORDS[match.group(1)]
        weeks = re.search(r"\b(\d{1,2}|" + "|".join(NUMBER_WORDS) + r")\s+weeks?\b", text)
        if self.rental_days is None and weeks:
            value = weeks.group(1)
            self.rental_days = 7 * (int(value) if value.isdigit() else NUMBER_WORDS[value])
        dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", self.original)
        self.start_date, self.end_date = (dates[0], dates[1]) if len(dates) >= 2 else (None, None)
        self.single_date = dates[0] if len(dates) == 1 else None
        seats = (re.search(r"\b(\d{1,2})\s*(?:-\s*)?(?:people|persons|passengers|pax|seats?|seater|katao|tao)\b", text)
                 or re.search(r"\b(?:group|family|party|barkada) of (\d{1,2})\b|\bwe are (\d{1,2})\b", text))
        self.min_seats = int(next(g for g in seats.groups() if g)) if seats else None
        money = r"(?:php|p|₱)?\s*([\d][\d,]{2,})(?:\s*(?:php|pesos?))?"
        budget = (re.search(r"\b(?:under|below|less than|at most|max(?:imum)?|budget(?: of| is| ko| ay)?|badyet(?: ko)?|hanggang|not more than|only have|i have|meron akong|may)\s*" + money, text)
                  or re.search(money + r"\s*(?:budget|badyet|ang budget|lang)\b", text)
                  or re.search(money + r"\s*(?:ako|lang ako)\b", text)
                  or (re.search(r"\b(?:with|for)\s*" + money + r"\s*(?:pesos?\s*)?(?:per day|a day|/day|daily|kada araw)\b", text)
                      if not re.search(r"\b\d+\s*(?:days?|nights?)\b", text) else None))
        self.max_daily_rate = float(budget.group(1).replace(",", "")) if budget else None
        self._year_like_budget = bool(budget and "," not in budget.group(1) and 1990 <= self.max_daily_rate <= 2035
                                      and not re.search(r"\b(?:budget|badyet|php|pesos?|under|below|less than|at most|max\w*|hanggang"
                                                        r"|only have|i have)\b|₱", text))
        self._price_from_budget = self.max_daily_rate is not None and "price" not in self.concepts
        if self.max_daily_rate is not None:
            self.concepts.add("price")
        if re.search(r"\b(?:cheapest|pinakamura|lowest|least expensive|most affordable)\b", text):
            self.sort = "price_asc"
        elif re.search(r"\b(?:most expensive|pinakamahal|highest price|priciest)\b", text):
            self.sort = "price_desc"
        elif re.search(r"\b(?:highest|most|biggest|largest|pinakamataas na) (?:mileage|odometer|km|kilometers)\b|\bmost (?:driven|used km)\b", text):
            self.sort = "mileage_desc"
        elif re.search(r"\b(?:lowest|least|smallest) (?:mileage|odometer|km|kilometers)\b", text):
            self.sort = "mileage_asc"
        elif re.search(r"\b(?:newest|latest model|most recent model|pinakabago)\b", text):
            self.sort = "year_desc"
        elif re.search(r"\b(?:oldest|pinakaluma)\b", text):
            self.sort = "year_asc"
        else:
            self.sort = None
        self.fuel_types = sorted({value for word, value in FUEL_WORDS.items() if re.search(rf"\b{word}\b", text)})
        self.transmissions = sorted({value for word, value in TRANSMISSION_WORDS.items() if re.search(rf"\b{word}\b", text)})
        self.colors = sorted({COLOR_CANONICAL.get(word, word.title()) for word in COLOR_WORDS if re.search(rf"\b{word}\b", text)})
        self.has_gps = (False if re.search(r"\b(?:without|no|walang) (?:gps|navigation)\b", text)
                        else True if re.search(r"\b(?:gps|navigation|nav|navi)\b", text) else None)
        self.has_aircon = True if re.search(r"\b(?:aircon|air ?con|air[- ]?condition\w*|a/c|ac)\b", text) else None
        self.equipment = next(({"name": item, "terms": list(terms)} for item, terms in EQUIPMENT.items()
                               if any(re.search(rf"\b{re.escape(term)}s?\b", text) for term in terms)), None)
        self.attribute = next((key for key, pattern in ATTRIBUTES if pattern.search(text)), None)
        self.seats_ranking = bool(SEATS_RANKING.search(text))
        self.rate_period = bool(RATE_PERIOD.search(text))
        year = re.search(r"\b(20[0-3]\d|19[89]\d)\s*(?:or )?(?:newer|and (?:newer|up|above)|pataas|onwards)\b", text)
        self.min_year = int(year.group(1)) if year else None
        self.reservation_statuses = sorted({status for status in ("pending", "confirmed", "cancelled", "rejected", "completed")
                                            if re.search(rf"\b{status}\b", text)}
                                           | ({"pending"} if re.search(r"\b(?:waiting|awaiting|for|need(?:ing)?) (?:for )?approval\b|\bunapproved\b", text)
                                              else set()))
        self.reservation_id = None
        match = re.search(r"(?:#|\b(?:reservation|booking)\s*(?:no\.?|number|#)?\s*)(\d{1,8})\b", text)
        if match:
            self.reservation_id = int(match.group(1))
        self._dates()
        self.vehicle_ids = [int(value) for value in re.findall(r"\b(?:vehicle|car|unit)\s*(?:id)?\s*#?\s*(\d{1,6})\b", text)][:5]
        self.vehicle_statuses = sorted({status for word, status in VEHICLE_STATUS_WORDS.items()
                                        if re.search(rf"\b{word}\b", text) and word != "available"}
                                       | ({"rented"} if re.search(r"\b(?:on rent|out on rent|currently rented|nirentahan)\b", text) else set())
                                       | ({"maintenance"} if re.search(r"\b(?:in the (?:shop|garage)|being (?:repaired|serviced|fixed)|under repair"
                                                                       r"|out of service)\b", text) else set()))
        # A bare field name next to a vehicle name ("Fortuner color?", "vios transmission").
        self.bare_attribute = next((key for key, pattern in BARE_ATTRIBUTES if pattern.search(text)), None)


    def _dates(self):
        """Relative and named dates ("tomorrow", "this weekend", "next Friday", "November 5 to 8", "bukas")."""
        if self.start_date:
            return
        today = timezone.localdate()
        text = self.text
        start = end = None
        if re.search(r"\b(?:this|ngayong|sa) (?:weekend|sabado at linggo)\b|\bweekend\b", text):
            start = today + timedelta(days=(5 - today.weekday()) % 7)
            end = start + timedelta(days=2)
        elif re.search(r"\b(?:day after tomorrow|sa makalawa|makalawa)\b", text):
            start = today + timedelta(days=2)
        elif re.search(r"\b(?:tomorrow|bukas)\b", text):
            start = today + timedelta(days=1)
        elif re.search(r"\b(?:today|ngayon|ngayong araw)\b", text) and self.has("availability", "price", "booking"):
            start = today
        else:
            weekday = re.search(r"\b(?:next|this|on|sa|ngayong|sa susunod na)?\s*(" + "|".join(WEEKDAYS) + r")\b", text)
            months = "|".join(sorted(MONTHS, key=len, reverse=True))
            named = (re.search(rf"\b({months})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:-|to|until|hanggang)\s*(?:({months})\.?\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\b", text)
                     or re.search(rf"\b(\d{{1,2}})\s*(?:-|to|until|hanggang)\s*(\d{{1,2}})\s+({months})\b", text)
                     or re.search(rf"\b({months})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", text))
            if named:
                groups = named.groups()
                try:
                    if len(groups) == 4:
                        start = self._month_day(MONTHS[groups[0]], int(groups[1]), today)
                        end = self._month_day(MONTHS[groups[2] or groups[0]], int(groups[3]), today, after=start)
                    elif len(groups) == 3:
                        start = self._month_day(MONTHS[groups[2]], int(groups[0]), today)
                        end = self._month_day(MONTHS[groups[2]], int(groups[1]), today, after=start)
                    else:
                        start = self._month_day(MONTHS[groups[0]], int(groups[1]), today)
                except ValueError:
                    start = end = None
            elif weekday and (weekday.group(0).strip() != weekday.group(1) or self.has("availability", "price", "booking")):
                days_ahead = (WEEKDAYS[weekday.group(1)] - today.weekday()) % 7 or 7
                start = today + timedelta(days=days_ahead)
        if start is None:
            return
        if end is None:
            end = start + timedelta(days=self.rental_days or 1)
        if end <= start:
            return
        self.start_date, self.end_date = start.isoformat(), end.isoformat()
        self.relative_period = True

    @staticmethod
    def _month_day(month, day, today, after=None):
        year = today.year
        value = date(year, month, day)
        if value < (after or today):
            value = date(year + 1, month, day)
        return value


def _fuzzy(token, words):
    candidates = [word for word in words if word[:1] == token[:1] and abs(len(word) - len(token)) <= 2]
    return bool(difflib.get_close_matches(token, candidates, n=1, cutoff=FUZZY_CUTOFF))


def _names_word(name, tokens, joined):
    """A database name in the message, as whole words, or a close misspelling of a one-word name."""
    words = TOKEN.findall(str(name).casefold())
    if not words:
        return False
    if f" {' '.join(words)} " in joined or f" {''.join(words)} " in joined:
        return True
    return len(words) == 1 and len(words[0]) >= 4 and any(
        token[:1] == words[0][:1] and difflib.SequenceMatcher(None, token, words[0]).ratio() >= 0.8 for token in tokens if len(token) >= 4)


def plan(message, role, vocabulary=None, memory=None):
    """A retrieval plan, a direct answer type, or None (answered from the guide and references)."""
    u = Understanding(message, vocabulary)
    memory = memory or {}
    admin = role == "admin"

    # Other people's records or contact details.
    contact = re.search(r"\b(?:phone|phones|email|emails|address|addresses|contact|contacts|personal information)\b", u.text)
    other_person = OTHER_PERSON_RECORDS.search(u.text) and not re.search(r"\b(?:my|mine|ko)\b", u.text)
    if ABOUT_ASSISTANT.search(u.text) or OFF_TOPIC.search(u.text):
        return None
    if SQL_TEXT.search(u.original):
        return _direct("refuse_sql", u)
    if CREDENTIALS.search(u.text) and not OWN_CREDENTIAL_HELP.search(u.text):
        return _direct("refuse_credentials", u)
    if ACCIDENT.search(u.text) and not u.has("maintenance"):
        return _direct("accident_guidance", u)
    named = NAMED_RECORDS.search(u.original)
    # "Toyota Vios rental" names a vehicle, not a person.
    named_person = named and not re.search(r"\b(?:my|mine|ko)\b", u.text) and not (
        u.makes or u.models or set(TOKEN.findall(named.group(0).casefold())) & CAR_BRANDS)
    # "What's your phone number?" asks for the rental company's contact details (not recorded), not a person's.
    business_contact = re.search(r"\byour (?:phone|number|email|address|contact)", u.text) and not u.has("booking") \
        or re.search(r"\bdrive\b", u.text) and not contact  # "can someone else drive the rental?" is a driver policy
    if u.has("other_people") and (contact or u.has("booking")) and not business_contact or other_person or named_person or ACCOUNT_ID.search(u.text):
        if not admin or contact:
            return _direct("refuse_private", u)
        return _plan("reservation_summary", u, [("reservations.summary", {"upcoming": u.has("upcoming")})])

    if u.dtc_codes and not (admin and STORED_RECORDS.search(u.text) and not u.unlisted_brands):
        return None

    for topic, pattern in UNRECORDED_TOPICS:
        if pattern.search(u.text) and (not u.has("symptom") or topic in SYMPTOM_SAFE_TOPICS):
            result = _direct("policy_unrecorded", u)
            result["topic"] = topic
            return result

    if admin:
        admin_plan = _admin_lookup_plan(u)
        if admin_plan:
            return admin_plan

    known_vehicle = bool(u.makes or u.models or u.categories or u.min_seats or u.max_daily_rate or u.sort
                         or ((u.fuel_types or u.transmissions or u.has_gps is not None) and u.has("vehicle_noun", "listing")))
    unlisted = bool(u.unlisted_brands or u.unlisted_categories)
    rental_question = u.has("price", "deposit", "availability") or known_vehicle
    # A brand or type the fleet does not have is only a rental question when renting is asked about.
    asks_vehicle = known_vehicle or unlisted and u.has("price", "deposit", "availability", "listing", "specs", "vehicle_noun")
    if u.has("symptom") and not rental_question and not (admin and u.has("maintenance", "status")):
        return None
    if CARE_AMOUNT.search(u.text) and not asks_vehicle:
        return None
    if POLICY_FEE.search(u.text) and not u.reservation_id:
        return None
    if NEEDS_TO_RENT.search(u.text) and not asks_vehicle:
        return None
    if HOW_TO_BOOK.search(u.text) and not (u.has("price", "deposit") or u.rental_days or u.start_date or u.has("own")):
        return None
    if u.has("price") and SERVICE_COST.search(u.text) and not (u.has("deposit", "availability") or re.search(r"\b(?:rent|rental|per day|daily)\b", u.text)):
        return None

    # The signed-in customer's own bookings.
    own_booking = u.has("own") and u.has("booking") or re.search(r"\b(?:any|ba)\s+(?:bookings?|reservations?)\b", u.text) and u.has("own") \
        or OWN_PICKUP_RETURN.search(u.text) and re.search(r"\b(?:my|the) (?:car|vehicle|rental|unit)\b", u.text) and not asks_vehicle \
        or PAST_RENTAL.search(u.text) and u.has("own") \
        or u.has("own") and re.search(r"\b(?:spent|nagastos|gastos|nagbayad)\b", u.text) and not asks_vehicle
    # "Show reservation 41": a customer can only look up reservations on their own account.
    numbered = not admin and u.reservation_id and re.search(r"\b(?:reservations?|bookings?)\b|#\s*\d", u.text)
    if (own_booking or numbered) and not (u.has("price") and asks_vehicle and not numbered):
        if admin:
            return _direct("admin_no_bookings", u)
        # Viewing or checking bookings is not a procedure; changing them is.
        procedure = bool(re.search(r"\b(?:cancel|extend|extension|change|reschedule|modify|move|pay|payment)\b"
                                   r"|\badd (?:more |extra )?days\b|\bmore days\b|\blonger\b", u.text))
        purpose, args = "list", {}
        if u.reservation_id:
            purpose, args = "details", {"reservation_id": u.reservation_id}
        elif re.search(r"\b(?:days? (?:are )?left|remaining|natitira|ilang araw pa)\b", u.text):
            purpose, args = "days_left", {"upcoming": True}
        elif PAST_RENTAL.search(u.text) and re.search(r"\b(?:last|previous|past|recent|huling|rented|history|before)\b", u.text) \
                and not u.has("upcoming"):
            purpose = "last"
        elif re.search(r"\b(?:spent|spend|gastos|nagastos|paid|total so far|how much (?:have|did) i)\b", u.text) or OWE.search(u.text):
            purpose = "spent"
        elif re.search(r"\b(?:next|upcoming|susunod) (?:pickup|pick up|return|booking|reservation|rental|trip)\b|\bwhen is my (?:pickup|return)\b", u.text) \
                or OWN_PICKUP_RETURN.search(u.text) or re.search(r"\bwhen\b.*\b(?:pick ?up|pickup|get)\b", u.text):
            purpose, args = "next", {"upcoming": True}
        elif re.search(r"\b(?:status|confirmed|approved|pending|accepted|rejected|cancelled)\b", u.text) and not procedure:
            purpose = "status"
        elif u.has("upcoming") and not procedure:
            args = {"upcoming": True}
        result = _plan("own_bookings", u, [("bookings.mine", args)], purpose=purpose, guide=procedure)
        if u.reservation_statuses:
            result["asked_statuses"] = u.reservation_statuses
        if OWE.search(u.text):
            result["owe"] = True
        return result

    # Fleet-wide status for administrators ("fleet status", "how many cars do we have").
    if admin and not u.has("maintenance") and (
            re.search(r"\bfleet\b", u.text) and (u.has("status") or re.search(r"\b(?:overview|summary|size|status|how many)\b", u.text))
            or re.search(r"\bhow many (?:cars|vehicles|units)\b|\bnumber of (?:cars|vehicles|units)\b|\bhow big is (?:our|the) fleet\b", u.text)):
        return _plan("fleet_status", u, [("fleet.status_counts", {})])

    # Fleet and maintenance records.
    records_question = u.has("maintenance") or (u.has("status") and (u.vehicle_statuses or u.has("listing")) and not u.has("availability"))
    general_advice = re.search(r"\b(?:how often|should i|when should|what is a|what does|tips?)\b", u.text)
    if records_question and not general_advice and not u.has("explain") and not HOW_TO_DO.search(u.text):
        if not admin:
            if u.has("maintenance", "status") and (asks_vehicle or u.has("listing", "vehicle_noun", "status")):
                return _direct("admin_only", u)
        else:
            return _maintenance_plan(u)

    if u.has("explain") and not asks_vehicle:
        return None
    # Policies and how-to questions are answered from the guide unless they name a vehicle.
    if (u.has("policy") or u.has("how_to") and u.has("action")) and not asks_vehicle:
        return None

    filters = _vehicle_filters(u)
    follow_up = False
    elliptical = ELLIPTICAL.search(u.text) or PRONOUN.search(u.text) or len(u.tokens) <= 6
    if not asks_vehicle and memory.get("last_vehicles") and (u.rental_days or u.has("price", "deposit", "specs")) \
            and elliptical and not BROADER_LIST.search(u.text):
        # "How much for 3 days?" after asking about a vehicle refers to that vehicle.
        filters["makes"] = sorted({item["make"] for item in memory["last_vehicles"] if item.get("make")})
        filters["models"] = sorted({item["model"] for item in memory["last_vehicles"] if item.get("model")})
        follow_up = True

    if not asks_vehicle and not follow_up and PRONOUN.search(u.text) and u.has("price", "deposit", "specs"):
        # "How much is it?" with nothing discussed before: ask which vehicle instead of guessing.
        return _direct("clarify_vehicle", u)

    if not (asks_vehicle or follow_up or u.has("price", "deposit", "availability") or u.rental_days
            or u.has("vehicle_noun") and u.has("listing", "specs") or u.equipment and u.has("vehicle_noun", "listing")
            or re.search(r"^(?:show me |so )?what (?:do )?you have(?: available| for rent)?\W*$", u.text)):
        return None

    if u.start_date:
        filters.update(start_date=u.start_date, end_date=u.end_date)
    elif u.single_date and u.rental_days:
        try:
            end = date.fromisoformat(u.single_date) + timedelta(days=u.rental_days)
            filters.update(start_date=u.single_date, end_date=end.isoformat())
        except ValueError:
            filters.update(start_date=u.single_date, end_date=u.single_date)  # rejected as invalid by the Back-End
    if admin and u.vehicle_statuses:
        filters["statuses"] = u.vehicle_statuses

    feature_check = None
    if u.equipment and not u.has("price", "deposit"):
        # "Does the Fortuner have a dash cam?": checked against the vehicle's listed features.
        feature_check = {"feature": "equipment", "value": u.equipment["name"], "terms": u.equipment["terms"]}
    elif u.attribute and not (u.has("price", "deposit") or u.rental_days or u.start_date):
        # One field asked for ("what year is the Fortuner?"): answer that field only.
        filters = {k: v for k, v in filters.items() if k not in ("fuel_types", "transmissions", "colors", "has_gps", "has_aircon")}
        result = _plan("vehicles", u, [("vehicles.search", filters)], purpose="attribute", follow_up=follow_up)
        result["attribute"] = u.attribute
        return result
    elif u.models and YES_NO.search(u.text) and not u.has("price", "deposit", "availability"):
        # "Does the Vios have GPS?" / "Automatic ba ang Fortuner?": answer yes or no for the named car.
        for key, value in (("has_gps", u.has_gps), ("has_aircon", u.has_aircon), ("transmissions", u.transmissions),
                           ("fuel_types", u.fuel_types), ("colors", u.colors)):
            if value not in (None, []):
                feature_check = {"feature": key, "value": value}
                filters.pop(key, None)
                break
    if feature_check is None and u.bare_attribute and len(u.models) == 1 and not (
            u.has("price", "deposit", "availability") or u.rental_days or u.start_date or u.sort or u.min_seats):
        filters = {k: v for k, v in filters.items() if k not in ("fuel_types", "transmissions", "colors", "has_gps", "has_aircon")}
        result = _plan("vehicles", u, [("vehicles.search", filters)], purpose="attribute", follow_up=follow_up)
        result["attribute"] = u.bare_attribute
        return result
    if feature_check is None and len(u.models) >= 2 and re.search(r"\b(?:or|vs\.?|versus|compare\w*|difference|mas|than|kaysa)\b", u.text):
        purpose = "compare"
    elif feature_check:
        purpose = "feature_check"
    elif u.has("deposit") and not (u.rental_days or u.start_date):
        purpose = "deposit"
    elif u.rental_days or u.start_date:
        purpose = "quote" if (u.rental_days and not u.start_date) or u.has("price", "deposit") else "availability"
    elif u.sort in ("mileage_desc", "mileage_asc", "year_desc", "year_asc") or (u.seats_ranking and len(u.models) < 2 and not u.min_seats):
        purpose = "ranking"
    elif u.has("price") or u.sort in ("price_asc", "price_desc"):
        purpose = "price"
    elif u.has("availability"):
        purpose = "availability"
    elif u.has("specs") or u.min_seats:
        purpose = "details"
    else:
        purpose = "listing"

    if u.rental_days and not u.start_date:
        filters["rental_days"] = u.rental_days
    tool = "vehicles.quote" if purpose == "quote" and filters.get("rental_days") else "vehicles.search"
    result = _plan("vehicles", u, [(tool, filters)], purpose=purpose, follow_up=follow_up)
    if feature_check:
        result["feature_check"] = feature_check
    if purpose == "ranking" and not u.sort:
        result["rank_by"] = "seats"
    if purpose == "quote" and u.has("deposit"):
        result["include_deposit"] = True
    if purpose in ("price", "listing") and u.rate_period:
        result["rate_period"] = True
    if purpose == "compare":
        result["compare"] = {"by": "price" if u.has("price") or re.search(r"\b(?:expensive|cheaper|mahal|mura)\b", u.text)
                             else "seats" if re.search(r"\b(?:bigger|larger|seats?|more people|maluwag)\b", u.text) else "all"}
    if getattr(u, "relative_period", False):
        result["period_from_words"] = True
    if u.single_date and not u.rental_days and not u.start_date:
        result["clarification"] = "return_date"
    return result


def _admin_lookup_plan(u):
    """Administrator questions about one car, reservations, costs, revenue and utilisation."""
    text = u.text
    if HOW_TO_DO.search(text):
        return None  # "How do I mark a reservation as completed?" is a how-to question for the guide.
    vehicle = {k: v for k, v in _vehicle_filters(u).items() if k in ("makes", "models", "categories")}
    if u.vehicle_ids:
        vehicle["vehicle_ids"] = u.vehicle_ids
    period = {k: v for k, v in (("from", u.start_date), ("to", u.end_date)) if v}
    period.update(_report_period(text))
    if re.search(r"\b(?:revenue|income|earnings?|earned|sales|kita|kinita)\b|\bbook(?:ed|ing|ings)? (?:value|amount|total)\b"
                 r"|\b(?:bookings|reservations) (?:are )?worth\b|\bvalue of (?:all )?(?:the )?(?:bookings|reservations)\b", text):
        result = _plan("revenue", u, [("reservations.revenue", period)])
        if u.reservation_statuses:
            result["focus_statuses"] = u.reservation_statuses
        return result
    if re.search(r"\b(?:utili[sz]ation|utili[sz]ed|most (?:rented|booked|used)|least (?:rented|booked|used)|busiest|idle)\b", text):
        return _plan("utilization", u, [("fleet.utilization", period)])
    if re.search(r"\b(?:maintenance|maintain|service|repair)\b", text) and re.search(r"\b(?:spent|spend|cost|costs|total|expenses?|gastos)\b", text) \
            and not re.search(r"\b(?:how much (?:is|does) (?:an?|the) )", text) and not re.search(r"\b(?:last|latest|recent|previous)\b", text):
        return _plan("maintenance_costs", u, [("maintenance.costs", {**vehicle, **period})])
    reservation_word = re.search(r"\b(?:reservations?|bookings?|rentals?|pickups?|pick-ups?|returns?)\b", text)
    # "My reservations" from an administrator is about their own (none), handled by the bookings rule.
    # "What's pending?" from an administrator asks about reservations.
    status_only = u.reservation_statuses and not (u.models or u.makes or u.has("maintenance"))
    if (reservation_word or status_only) and not re.search(r"\b(?:my|mine|ko|akin)\b", text) and not u.has("explain"):
        args = {}
        if u.reservation_statuses:
            args["statuses"] = u.reservation_statuses
        if u.reservation_id:
            args["reservation_id"] = u.reservation_id
        today = timezone.localdate()
        if re.search(r"\bpick ?-?ups?\b|\bpicking up\b", text) and re.search(r"\btoday|ngayon\b", text):
            args["pickup_on"] = today.isoformat()
        elif re.search(r"\b(?:returns?|returning|due back)\b", text) and re.search(r"\btoday|ngayon\b", text):
            args["return_on"] = today.isoformat()
        elif re.search(r"\bpick ?-?ups?\b", text) and re.search(r"\btomorrow|bukas\b", text):
            args["pickup_on"] = (today + timedelta(days=1)).isoformat()
        elif u.start_date:
            args.update({"from": u.start_date, "to": u.end_date})
        name = re.search(r"\b(?:of|for|by|ni|ng customer)\s+([a-z][a-z]+(?:\s+[a-z][a-z]+)?)\s*(?:'s)?\s*$", text) \
            or re.search(r"\b([a-z]+(?:\s+[a-z]+)?)'s\s+(?:reservations?|bookings?)", text)
        if name and name.group(1) not in {"the", "my", "our", "all", "today", "tomorrow", "this week", "next week", "pending",
                                          "approval", "confirmation", "now", "me", "us", "review", "later"}:
            args["customer_name"] = name.group(1).title()
        if args or re.search(r"\b(?:all|show|list|any|are there|which)\b", text):
            return _plan("reservation_summary", u, [("reservations.summary", args)])
    asks_status = re.search(r"\b(?:status|where is|how is|mileage|odometer|km|kilometers|kilometres|last (?:service|serviced|maintenance)|next (?:service|maintenance)"
                            r"|currently (?:booked|rented)|is (?:it|the \w+) (?:booked|rented|available)|kumusta|nasaan)\b", text)
    if (u.models or u.vehicle_ids) and asks_status and not u.has("price", "deposit") and not u.sort:
        return _plan("vehicle_status", u, [("vehicles.status", vehicle)])
    return None


def _report_period(text):
    """Report windows: "this month", "last month", "this year", "in October"."""
    today = timezone.localdate()
    if re.search(r"\bthis month\b|\bngayong buwan\b", text):
        return {"from": today.replace(day=1).isoformat(), "to": today.isoformat()}
    if re.search(r"\blast month\b|\bnakaraang buwan\b", text):
        first = today.replace(day=1)
        start = (first - timedelta(days=1)).replace(day=1)
        return {"from": start.isoformat(), "to": (first - timedelta(days=1)).isoformat()}
    if re.search(r"\bthis year\b|\bngayong taon\b", text):
        return {"from": today.replace(month=1, day=1).isoformat(), "to": today.isoformat()}
    month = re.search(r"\bin (" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b", text)
    if month:
        number = MONTHS[month.group(1)]
        year = today.year if number <= today.month else today.year - 1
        start = date(year, number, 1)
        end = (date(year + (number == 12), number % 12 + 1, 1) - timedelta(days=1))
        return {"from": start.isoformat(), "to": end.isoformat()}
    return {}


def _maintenance_plan(u):
    filters = {"vehicle_ids": u.vehicle_ids, **{k: v for k, v in _vehicle_filters(u).items() if k in ("makes", "models")}}
    text = u.text
    if re.search(r"\b(?:recurring|repeated|repeat|again and again|keeps? (?:coming|happening)|same (?:issue|problem))\b", text):
        return _plan("maintenance_recurring", u, [("maintenance.recurring", dict(filters))])
    if re.search(r"\b(?:predict\w*|risk\w*|forecast\w*|break\s?downs?|breaks down|fail\w*)\b", text):
        return _plan("maintenance_prediction", u, [("maintenance.model_inputs", {}), ("maintenance.due", dict(filters))])
    if u.dtc_codes or re.search(r"\b(?:dtc|trouble codes?|fault codes?|diagnostic\w*|inspection\w*|inspected|condition)\b", text):
        return _plan("maintenance_inspections", u, [("maintenance.inspections", dict(filters)), ("maintenance.records", dict(filters))])
    if re.search(r"\b(?:due|overdue|need\w*|upcoming|soon|schedule\w*|next service)\b", text):
        return _plan("maintenance_due", u, [("maintenance.due", dict(filters))])
    if re.search(r"\b(?:records?|history|histories|logs?|latest|last|recent|activit\w+|cost\w*|spent|serviced|repaired|done)\b", text):
        return _plan("maintenance_records", u, [("maintenance.records", {"limit": 20, **filters})])
    if u.vehicle_statuses or u.has("status"):
        if u.vehicle_statuses:
            return _plan("vehicles", u, [("vehicles.search", {"statuses": u.vehicle_statuses, **filters})], purpose="status")
        return _plan("fleet_status", u, [("fleet.status_counts", {})])
    return _plan("maintenance_due", u, [("maintenance.due", dict(filters))])


def _vehicle_filters(u):
    filters = {}
    if u.models:
        filters["models"] = sorted({entry["model"] for entry in u.models})
        makes = {entry["make"] for entry in u.models}
        if u.makes and not set(u.makes) <= makes:
            filters["makes"] = sorted(set(u.makes) | makes)
    elif u.makes:
        filters["makes"] = sorted(set(u.makes))
    if u.categories:
        filters["categories"] = sorted(set(u.categories))
    if u.min_seats:
        filters["min_seats"] = u.min_seats
    if u.max_daily_rate is not None:
        filters["max_daily_rate"] = u.max_daily_rate
    if u.sort:
        filters["sort"] = u.sort
    for key in ("fuel_types", "transmissions", "colors"):
        if getattr(u, key):
            filters[key] = getattr(u, key)
    if u.has_gps is not None:
        filters["has_gps"] = u.has_gps
    if u.has_aircon is not None:
        filters["has_aircon"] = u.has_aircon
    if u.min_year:
        filters["min_year"] = u.min_year
    return filters


def _requested(u):
    return {"makes": list(u.makes), "models": [entry["model"] for entry in u.models], "categories": list(u.categories),
            "unknown_names": _unknown_names(u) + list(u.unlisted_brands) + list(u.unlisted_categories)}


def _unknown_names(u):
    """A word right after a known make that is not one of its models ("Toyota Supra")."""
    names = []
    for make in u.makes:
        match = re.search(rf"\b{re.escape(make.casefold())}\s+([a-z][a-z0-9-]{{2,}})", u.text)
        if match and not any(entry["make"] == make and entry["model"].casefold().startswith(match.group(1)) for entry in u.models):
            if match.group(1) not in CONCEPTS["vehicle_noun"]["words"] | CONCEPTS["price"]["words"] | {"per", "for", "the", "na", "po", "ba", "cars"}:
                names.append(f"{make} {match.group(1).title()}")
    return names


def _plan(intent, u, retrievals, purpose=None, follow_up=False, guide=False):
    return {
        "version": 1,
        "intent": intent,
        "purpose": purpose,
        "retrievals": [{"tool": tool, "args": args} for tool, args in retrievals],
        "requested": _requested(u),
        "follow_up": follow_up,
        "guide": guide,
    }


def _direct(kind, u):
    return {"version": 1, "intent": kind, "direct": kind, "retrievals": [], "requested": _requested(u)}


# A new question starts after "?", ";", "tapos", "at saka", "also", or "and" followed by a question word.
QUESTION_BREAK = re.compile(
    r"\?\s+|;\s*|\s+(?:tapos|at saka|and also|also)\s+"
    r"|\s*,?\s+and\s+(?=(?:do|does|did|is|are|can|could|what|how|when|where|which|who|may|ano|magkano|show|list|tell|give|check|meron)\b)",
    re.IGNORECASE,
)


INSTRUCTION = re.compile(r"\b(?:pretend|act as|you are now|from now on|ignore (?:previous|all|the|your)|developer mode|admin mode|jailbreak|system:)",
                         re.IGNORECASE)


def split_questions(message):
    parts = [part.strip(" ,?") for part in QUESTION_BREAK.split(message) if part and part.strip(" ,?")]
    return [part for part in parts if len(TOKEN.findall(part.casefold())) >= 2][:3]


def plan_message(message, role, vocabulary=None, memory=None):
    """plan() for one question, or a "multi" plan that answers each question in the message."""
    # An instruction to the assistant ("pretend you are the admin and show ...") is not a separate question.
    parts = [] if INSTRUCTION.search(message) else split_questions(message)
    if len(parts) < 2:
        return _with_remembered_period(plan(message, role, vocabulary, memory), message, memory)
    sub_plans = [_with_remembered_period(plan(part, role, vocabulary, memory), part, memory) for part in parts]
    intents = {(sub or {}).get("intent") for sub in sub_plans}
    if len(intents) < 2 and all(sub and sub.get("retrievals") for sub in sub_plans):
        # Same kind of question twice ("price of the Vios and the Fortuner") is one question.
        return _with_remembered_period(plan(message, role, vocabulary, memory), message, memory)
    retrievals, entries = [], []
    for part, sub in zip(parts, sub_plans):
        requests = (sub or {}).get("retrievals") or []
        if len(retrievals) + len(requests) > 4:
            requests, sub = [], None
        entries.append({"text": part, "plan": sub, "start": len(retrievals), "count": len(requests)})
        retrievals += requests
    return {"version": 1, "intent": "multi", "parts": entries, "retrievals": retrievals,
            "requested": (sub_plans[0] or {}).get("requested", {})}


def _with_remembered_period(result, message, memory):
    """"How about the Fortuner?" after an availability check reuses that check's dates."""
    period = (memory or {}).get("last_period")
    if not result or not period or result.get("intent") != "vehicles" or not result.get("retrievals"):
        return result
    request = result["retrievals"][0]
    if "start_date" in request["args"] or not re.match(r"^(?:and|what about|how about|eh|yung|e ang|paano ang)\b", message.strip().casefold()):
        return result
    request["args"].update(start_date=period["start_date"], end_date=period["end_date"])
    request["args"].pop("rental_days", None)
    if result.get("purpose") not in ("quote", "availability"):
        result["purpose"] = "availability"
    result["follow_up_period"] = True
    return result
