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
    "deposit": {"words": {"deposit", "deposits", "refundable", "refund", "refunds"}, "phrases": ["security deposit", "security on", "security for the", "security of the"]},
    "availability": {
        "words": {"available", "availables", "availability", "bakante", "vacant", "free", "rentable"},
        "phrases": ["for rent", "can i rent", "can i book", "can i reserve", "is there any", "are there any", "ready to go", "be around",
                    "be rented", "available to rent", "available for rent", "pwedeng rentahan", "pwede rentahan"],
    },
    "listing": {
        "words": {"what", "which", "list", "show", "anong", "ano", "lahat", "all", "options", "choices", "meron", "offer", "units"},
        "phrases": ["do you have", "may available", "what cars", "what vehicles", "any cars", "kinds of", "see the cars",
                    "see your cars", "see the vehicles", "see the car", "see all", "see available", "your fleet", "your cars",
                    "your vehicles", "your units", "what you have", "what do you have", "what have you got", "on offer", "lineup", "line up",
                    "choose from"],
    },
    "specs": {
        "words": {"features", "feature", "specs", "spec", "specifications", "seats", "seat", "seater", "seating", "transmission",
                  "automatic", "manual", "fuel", "diesel", "gasoline", "gas", "petrol", "gps", "aircon", "airconditioning",
                  "color", "colour", "year", "mileage", "details", "detail", "info", "information", "describe", "model",
                  "plate", "plaka", "minimum", "kulay", "upuan", "dashcam", "camera", "bluetooth", "usb", "sunroof", "carplay",
                  "equipment", "equipped", "keyless", "touchscreen", "sensors", "navigation", "nav", "ac", "transmission",
                  "odometer", "km", "kilometers", "conditioned", "airconditioned", "gearbox", "matic", "capacity", "passengers", "passenger", "riders"},
        "phrases": ["tell me about", "how many seats", "how many people", "what kind", "dash cam", "child seat", "baby seat",
                    "cruise control", "android auto", "roof rack"],
    },
    "vehicle_noun": {"words": {"car", "cars", "vehicle", "vehicles", "unit", "units", "auto", "suv", "suvs", "sedan", "sedans",
                               "van", "vans", "pickup", "truck", "rental", "rentals", "rent", "renting", "fleet"}},
    "own": {"words": {"my", "mine", "ko", "ako", "akin", "sakin", "namin", "our"},
            "phrases": ["do i have", "have i", "i booked", "i reserved", "i rented", "did i", "i spent", "days are left", "days left", "am i", "under me", "i have a", "have i got"]},
    "booking": {"words": {"booking", "bookings", "reservation", "reservations", "reserved", "booked", "rental", "rentals",
                          "pickup", "trip", "trips", "book"},
                "phrases": ["rental history", "return date", "pick up", "days left", "days are left", "spent so far",
                            "have i spent", "did i spend", "next pickup", "my pickup", "my return", "mine approved", "am i renting"]},
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
                                "break down", "breaks down", "been done", "done to", "work done", "was done"]},
    "status": {"words": {"status", "rented", "unavailable", "reserved", "booked", "maintenance", "utilization", "how's"},
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
                          "smell", "smells", "flashing", "blinking", "warning", "aircon", "hesitates", "jerks", "knocking", "clicks", "clicking", "wobbles",
                          # Found by fresh questions on 2026-11-13: "water dripping after using AC" is a car problem, not an aircon search.
                          "dripping", "drips", "drip", "rattling", "rattles", "whining", "whine", "humming", "clunking", "bounces",
                          "flicker", "flickers", "dying", "dies", "loud", "squeaky", "burning"},
                "phrases": ["won't start", "wont start", "check engine", "warning light", "does not start", "doesn't start"]},
}
# Words that are numbers in English or Tagalog.
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "isa": 1, "isang": 1, "dalawa": 2, "dalawang": 2, "tatlo": 3, "tatlong": 3, "apat": 4, "lima": 5, "limang": 5,
    "anim": 6, "pito": 7, "pitong": 7, "walo": 8, "walong": 8, "siyam": 9, "sampu": 10, "sampung": 10,
    "eleven": 11, "twelve": 12, "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30,
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
# Common model names (general knowledge, not database contents). "City" and similar ordinary words are left out.
KNOWN_MODELS = {name.casefold(): label for name, label in (
    ("hiace", "Toyota Hiace"), ("innova", "Toyota Innova"), ("avanza", "Toyota Avanza"), ("wigo", "Toyota Wigo"), ("raize", "Toyota Raize"),
    ("hilux", "Toyota Hilux"), ("fortuner", "Toyota Fortuner"), ("vios", "Toyota Vios"), ("corolla", "Toyota Corolla"),
    ("montero", "Mitsubishi Montero"), ("xpander", "Mitsubishi Xpander"), ("mirage", "Mitsubishi Mirage"), ("strada", "Mitsubishi Strada"),
    ("navara", "Nissan Navara"), ("almera", "Nissan Almera"), ("terra", "Nissan Terra"), ("ertiga", "Suzuki Ertiga"),
    ("civic", "Honda Civic"), ("brv", "Honda BR-V"), ("crv", "Honda CR-V"), ("ranger", "Ford Ranger"), ("everest", "Ford Everest"),
    ("tucson", "Hyundai Tucson"), ("stargazer", "Hyundai Stargazer"), ("picanto", "Kia Picanto"), ("rush", "Toyota Rush"), ("veloz", "Toyota Veloz"),
    ("zenix", "Toyota Innova Zenix"), ("yaris", "Toyota Yaris"), ("camry", "Toyota Camry"), ("traviz", "Isuzu Traviz"), ("dmax", "Isuzu D-Max"), ("mux", "Isuzu mu-X"),
    ("jimny", "Suzuki Jimny"), ("swift", "Suzuki Swift"), ("dzire", "Suzuki Dzire"), ("celerio", "Suzuki Celerio"), ("creta", "Hyundai Creta"), ("accent", "Hyundai Accent"),
    ("seltos", "Kia Seltos"), ("stonic", "Kia Stonic"), ("soluto", "Kia Soluto"), ("sorento", "Kia Sorento"), ("coolray", "Geely Coolray"), ("tiggo", "Chery Tiggo"),
    ("livina", "Nissan Livina"), ("kicks", "Nissan Kicks"), ("patrol", "Nissan Patrol"), ("jazz", "Honda Jazz"), ("brio", "Honda Brio"), ("hrv", "Honda HR-V"),
    ("raptor", "Ford Raptor"), ("territory", "Ford Territory"), ("forester", "Subaru Forester"), ("cx5", "Mazda CX-5"), ("supra", "Toyota Supra"))}
# Model names that are also ordinary words: without a rental signal in the message they are not read as a vehicle.
ORDINARY_MODEL_WORDS = {"city", "accent", "mirage", "ranger", "everest", "territory", "fit", "jazz", "swift", "rush", "patrol", "forester", "kicks", "focus",
                        "escape", "explorer", "soul", "pilot", "element", "cross", "sport", "terra", "strada", "almera"}
BRAND_NAMES = {"bmw": "BMW", "mg": "MG", "byd": "BYD", "gac": "GAC", "jac": "JAC", "vw": "VW", "mercedes-benz": "Mercedes-Benz",
               "kia": "Kia", "mini": "MINI", "ram": "RAM", "land rover": "Land Rover"}
# Policies and services the application has no record of; the answer says so instead of guessing.
UNRECORDED_TOPICS = [
    # Age is in the system guide (no minimum-age rule is stored); licence types are not.
    ("which licence types are accepted", re.compile(
        r"\b(?:student|learner'?s?|foreign|international|provisional|temporary)\s+(?:driving\s+|driver'?s?\s+)?(?:permit|license|licence)\b")),
    ("a driver or chauffeur service", re.compile(
        r"\bchauffeur\w*\b|\bdriver service\b|\bdriver for hire\b|\b(?:with|include|includes|including|provide|provides|offer|offers|hire|rent|have) (?:a |an |your )?(?:own )?driver\b(?!'s|s)"
        r"|\bmay (?:kasamang )?driver\b|\bkasama(?:ng)? driver\b")),
    ("smoking in the vehicle", re.compile(
        r"\b(?:can|may) (?:i|we) (?:smoke|vape)\b|\b(?:smoking|vaping) (?:ok\b|okay\b|fine\b|allowed|policy|inside|in the (?:car|vehicle))\b|\bno smoking\b"
        r"|\b(?:is|are) (?:smoking|vaping)\b"
        r"|\b(?:is it ok|allowed) to (?:smoke|vape)\b|\b(?:pwede|bawal)(?: ba)?(?: mag)?(?:-| )?(?:yosi|manigarilyo|smoke|vape)\b"
        r"|\b(?:manigarilyo|magsigarilyo|magyosi|mag yosi|mag-yosi)\b")),
    ("pets in the vehicle", re.compile(
        r"\b(?:bring|take|travel with|allowed|pwede|isakay|isama)\b.*\b(?:pets?|dogs?|cats?|aso|pusa)\b"
        r"|\b(?:pets?|dogs?|cats?)\b.*\b(?:ride|come|along|join)\b"
        r"|\b(?:pets?|dogs?|cats?) (?:are |is )?(?:allowed|friendly|policy|ok|okay|welcome)\b")),
    ("additional drivers", re.compile(r"\b(?:someone else|another person|another driver|additional drivers?|second driver|other drivers?|my (?:friend|wife|husband|brother|sister|kapatid|asawa)|ibang (?:tao|driver))\b.*\b(?:drive|driver|magmaneho|mag-drive)\b|\badditional drivers?\b")),
    ("insurance", re.compile(r"\b(?:insurance|insured|insure|seguro)\b")),
    ("travel outside the area", re.compile(r"\b(?:drive|take|bring) (?:it|the car|the vehicle|the unit) to (?!my\b|me\b|our\b|the\b|a\b|your\b)[a-z]+\b|"
                                           # "Can I drive to Baguio?", "pwede ba dalhin sa Batangas?", "biyahe papuntang Ilocos"
                                           r"\b(?:can|may|could) (?:i|we) (?:drive|go|travel|take a trip) (?:up )?to (?!the\b|a\b|my\b|your\b|work\b|school\b)[a-z]+\b|"
                                           r"\b(?:biyahe|byahe|dalhin|gamitin)\b.*\b(?:papunta|papuntang|hanggang|sa probinsya)\b|"
                                           r"\bdrive\b.*\b(?:province|provinces|probinsya)\b|\b(?:another|other|ibang) (?:province|provinces|city|region|island|probinsya)\b|"
                                           r"\b(?:out of town|outside (?:the )?(?:city|metro manila|province|area|region)|cross[- ]country|road trip)\b")),
    ("mileage or distance limits", re.compile(r"\b(?:mileage|distance|km|kilometer|kilometre)s? (?:limit|cap|restriction|allowance)s?\b"
                                              r"|\b(?:limit|cap)s? (?:on|to|for) (?:the )?(?:mileage|distance|km|kilometers?|kilometres?)\b"
                                              r"|\bunlimited (?:mileage|km|kilometers)\b|\bhow far can (?:i|we) drive\b")),
    ("home delivery", re.compile(r"\b(?:deliver|delivery|drop[- ]?off at|drop the car (?:at|to|off)|bring the car to (?:me|my|our)|ihatid|i-deliver)\b")),
    ("business hours", re.compile(r"\b(?:business hours|opening hours|office hours|what time do you (?:open|close)|are you open|open ba kayo|open kayo|open (?:on|at|today|sa)|anong oras)\b")),
    ("late returns and late fees", re.compile(r"\blate (?:return|returns|fee|fees|charge|penalty)\b|\breturn\w* (?:the car |it )?late\b|\bhuli(?:ng)? (?:pagbalik|balik)\b"
                                              r"|\bpenalt(?:y|ies)\b.*\b(?:late|return\w*)\b|\bif i'?m late\b")),
    ("cleaning and sanitising of vehicles", re.compile(
        r"\b(?:clean|cleaned|cleaning|sanitized|sanitised|sanitizing|sanitising|disinfected|hygiene|malinis|nililinis)\b"
        r"(?!.*\b(?:engine|filter|injector|throttle|battery|terminals?|brakes?|vents?|leak)\b)")),
    # Vehicle records store fuel type but no consumption figures; "my car uses too much fuel" is a car problem, not this.
    ("fuel-consumption figures for its vehicles", re.compile(
        r"^(?!.*\bmy\b).*\b(?:fuel (?:economy|consumption|efficiency)|km ?(?:per|/) ?(?:liter|litre|l)\b|kmpl|mpg|matipid(?: sa)?(?: gas| gasolina)?"
        r"|tipid sa (?:gas|gasolina|fuel)|fuel[- ]efficient|most economical|least fuel|uses less fuel)")),
    # The system guide: no pickup location and no fuel or handover rules are stored.
    ("the pickup and return location", re.compile(
        r"\b(?:pickup|return|drop[- ]?off)\s+(?:location|place|point|address|branch|area|site)\b"
        r"|\bwhere (?:do|can|will|should|would) (?:i|we) (?:pickup|pick up|get|collect|claim|return|drop off)\b"
        r"|\bwhere (?:is|are) (?:the |your )?(?:pickup|office|branch|garage|shop)\b|\bsaan (?:kukunin|ang pickup|ibabalik|isasauli)\b")),
    ("fuel rules when returning the car", re.compile(
        r"\b(?:refill|refuel|full tank|fill (?:it |the tank )?up|fill the tank|gas|fuel|gasoline|gasolina)\b.*\b(?:before|when|upon|after) "
        r"(?:i |we )?(?:return|returning|give it back|bring it back|ibalik|isauli)\b"
        r"|\bfuel (?:policy|rules?)\b|\bfull[- ]to[- ]full\b"
        r"|\breturn\w*\b.*\b(?:empty|full|half|low)(?: a)? (?:tank|gas|fuel)\b|\b(?:empty|full|half|low) (?:tank|gas|fuel)\b.*\breturn\w*\b")),
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
YES_NO = re.compile(r"^(?:does|do|is|are|has|have|may|meron|can|would|could|with)\b|\b(?:ba|po ba|right)\b|\b(?:is|does) it\b|\bcome with\b|\binclude\b"
                    r"|\b(?:gps|navigation|aircon|a/c|ac|automatic|manual|diesel|gasoline)\s*\?"
                    r"|\b(?:has|have|with|got)\s+(?:a |an )?(?:gps|navigation|nav|ac|a/c|aircon|air ?con\w*|air[- ]?condition\w*)\b")
# Words that ask for a broader list, so the question is not about the vehicle discussed before.
BROADER_LIST = re.compile(r"\b(?:all|list|lahat|options|others?|every|any other|ibang|mga)\b"
                          # "Available cars" after a list asks about the fleet, not the cars just shown ("are they available?" does).
                          r"|^(?!.*\b(?:they|them|these|those|it|sila)\b).*\b(?:cars|vehicles|units)\b")
# "Suggest a car", "what's the best car for 4 people", "which car should I rent", "anong magandang sasakyan".
RECOMMEND = re.compile(r"\b(?:suggest\w*|recommend\w*|irekomenda)\b"
                       r"|\b(?:best|ideal|perfect|suitable|good) (?:car|cars|vehicle|vehicles|unit|units|sasakyan|kotse|option|choice)\b"
                       r"|\bwhich (?:car|vehicle|unit|one) (?:should|would|do you think) i\b|\bwhat (?:car|vehicle) should i (?:rent|get|choose|take)\b"
                       r"|\b(?:ano|anong|alin|what|which)(?: ang| the)? (?:magandang|maganda|best|pinakamagandang?) (?:sasakyan|kotse|car|unit)\b"
                       r"|\b(?:magandang|maganda|pinakamagandang?) (?:sasakyan|kotse|car|unit)\b")
# "Are they all rented?", "is everything available?" (administrators): the fleet's status totals, answered yes or no.
STATUS_ALL = re.compile(r"\b(?:are|is) (?:they|these|those|all|all of them|the cars|the vehicles|everything|every car|lahat)\b.*?"
                        r"\b(?P<status>rented|available|unavailable|in maintenance|under maintenance|booked)\b"
                        r"|\b(?:all|lahat) (?:of them |ng sasakyan )?(?:are|ay) (?P<status2>rented|available|unavailable|booked)\b")
ELLIPTICAL = re.compile(r"^(?:and|what about|how about|eh|yung|paano yung|e di|so)\b")
# "Compare them", "which one is cheaper?", "alin ang mas mura?": a comparison of the vehicles shown in the previous answer.
CROSS_TURN_COMPARE = re.compile(
    r"\bcompare (?:them|these|those|both|the two|the (?:cars|vehicles))\b|\b(?:between|of) (?:them|these|those|the two)\b|\bboth of them\b"
    r"|\b(?:which|alin|ano)(?: one| sa (?:kanila|dalawa))?(?: ang| is| the)? (?:cheaper|cheapest|bigger|biggest|better|best|more expensive|roomier"
    r"|mas (?:mura|mahal|malaki|maluwag|maganda))\b|\bdifference between them\b|\bhow do they compare\b")
# A reply that is only dates ("2026-12-09", "from 2026-12-10 to 2026-12-14") continues the vehicle being discussed.
DATES_ONLY = re.compile(r"(?:\s*(?:\d{4}-\d{2}-\d{2}|from|to|until|till|til|hanggang|mula|sa|on|the|and|return(?:ing)?(?: on)?|pickup(?: on)?"
                        r"|pick up(?: on)?|po|lang|[-,.?!])\s*)+")
PRONOUN = re.compile(r"\b(?:it(?! (?:true|possible|ok|okay|allowed|safe|normal|required)\b)|this one|that one|this car|that car|yan|iyan|yun|iyon|ito"
                     r"|(?:the )?(?:vehicle|car|one) i (?:selected|chose|picked|clicked)|selected (?:vehicle|car))\b")
# Questions about the assistant's own access are answered from the guide.
ABOUT_ASSISTANT = re.compile(r"\bcan (?:the )?(?:chatbot|assistant|bot|ai|you) (?:see|access|view|read|show)\b(?!.*\b(?:my|mine|ko|cars?|vehicles?|units?)\b)")
# Price questions about repairs or services, not rentals.
SERVICE_COST = re.compile(r"\b(?:oil change|repair|repairs|servicing|replacement|replace|parts?|tune[- ]?up|alignment|labou?r|fix|fixing)\b")
OFF_TOPIC = re.compile(r"\b(?:stock|shares?|stock market|net worth|ceo|founder|headquarters|weather|forecast|typhoon|bagyo|uulan|going to rain|will it rain"
                       r"|news|election|joke|jokes|poem|poems|song|songs"
                       r"|sing|movie|movies|bitcoin|crypto\w*|recipe|cook|cooking|bake|baking|basketball|football|lotto|horoscope|nba|laptop|ulam|president"
                       r"|capital of|essay|love letter)\b")
BARE_ATTRIBUTES = [
    ("transmission", re.compile(r"\btransmission\b|\bgear ?box\b")),
    ("color", re.compile(r"\bcolou?r(?:s|'s)?\b|\bkulay\b|\bpaint\b")),
    ("seating_capacity", re.compile(r"\bseating(?: capacity)?\b|\bcapacity\b|\bseats?\b|\bpassengers?\b|\bupuan\b|\bseater\b")),
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
                           r"|\brequirements? (?:for|to) (?:rent|book|reserv)\w*|\bneed\b.*\b(?:to|para|sa|the)\s+(?:rent|book|reserve)\b|\brequired to (?:rent|book|reserve)\b")
# Asking how to do something in the system, as opposed to asking for records.
HOW_TO_DO = re.compile(r"\b(?:how (?:do|can|should|would) (?:i|we|an admin)|how to|steps|procedure|paano|how do i)\b")
# Asking about the fleet's records rather than describing one's own car ("DTC stored: P0999").
STORED_RECORDS = re.compile(r"\b(?:in (?:the|our) (?:records|fleet|system)|which (?:vehicles|cars|units)|our (?:vehicles|cars|fleet|units)"
                            r"|for (?:the|our|all) (?:vehicles|cars|units)|fleet|maintenance records?|service records?"
                            r"|(?:vehicle|car|unit)\s*(?:id\s*)?#?\s*\d+)\b")
OWN_PICKUP_RETURN = re.compile(r"\bwhen (?:do|should|must|will|can|am) i (?:have to |need to )?(?:return|pick up|pickup|picking up|get|collect|give|bring)\b"
                               r"|\bwhen ko (?:return|pick up)\b|\bwhen pick up\b|\bwhen return\b"
                               r"|\bwhen (?:the )?(?:return|pick up|pickup) (?:of )?(?:the )?my (?:car|unit|rental)\b"
                               r"|\b(?:what|which) (?:day|date) (?:do|will|can|should) i (?:get|pick up|pickup|collect|return|bring)\b"
                               r"|\b(?:what|which) (?:day|date) is (?:the|my) (?:car|vehicle|unit|rental) due\b"
                               r"|\bwhen(?:'s| is| does| will) (?:the|my) (?:car|vehicle|unit|rental)\b.*\b(?:due|back|return)\b"
                               r"|\bwhen will i pick\b|\bwhen do i bring\b")
OTHER_PERSON_RECORDS = re.compile(r"\b(?!(?:my|what|that|it|where|who|how|here|there|when|which|let)'s\b)(?!my\b)[a-z]+'s\s+"
                                  r"(?:bookings?|reservations?|rentals?|records?|details|account)\b")
DEPOSIT_REFUND = re.compile(r"\bdeposits?\b.*\b(?:back|return(?:ed)?|refund\w*)\b|\b(?:back|return(?:ed)?|refund\w*)\b.*\bdeposits?\b")
# Company money. Only administrators may see it, however it is worded.
COMPANY_REVENUE = re.compile(r"\b(?:revenue|income|earnings?|sales|kita|kinita|grossing)\b|\b(?:cars?|vehicles?|units?|fleet)\b.*\b(?:earns?|earning|earned)\b"
                             r"|\bhow much (?:money )?(?:have|did|do) (?:you|the business|the company)(?: \w+)? (?:make|made|earn|earned)\b"
                             r"|\b(?:the business|the company)\b.*\b(?:make|made|earn|earned)\b")
# Asking what something would cost is a quote, not a look-up of an existing booking ("how much would 5 days with the Vios cost me").
HYPOTHETICAL = re.compile(r"\b(?:would|will|if i|kung|going to|gonna|planning|thinking|suppose|supposing|say i|i'?ll|might|could i)\b")
# "Do you have ...?", "May Stargazer ba kayo?": is this vehicle in stock.
STOCK_QUESTION = re.compile(r"\b(?:do|did) (?:you|u)(?: guys| all)?(?: still)? (?:have|carry|got|stock|offer|rent out)\b|\b(?:have|got) you\b|\byou(?: guys)? (?:have|got)\b|\bgot (?:any|a|an|anything)\b"
                            r"|^anything\b|\blooking for an?\b|\bi (?:need|want|prefer) an? \w+\b|\bnaghahanap\b|\bmeron\b|^any\b|^may\b.*\b(?:ba|po|kayo)\b|\bkayong?\b|\bis there an?\b|\bany \w+(?: \w+)? available\b|\bany (?:\w+ )?for rent\b")
# "How can I reserve the Vios?" asks for the booking steps, even with a vehicle named.
HOW_TO_BOOK = re.compile(r"\bhow (?:can|do|to|would|should) (?:i |we )?(?:book|reserve|rent|make a (?:booking|reservation))\b"
                         r"|\b(?:process|steps|procedure) (?:to|for) (?:book|reserv|rent)\w*")
# Common words that typo-tolerant matching must not read as concept words ("feels" is not "fees").
FUZZY_SKIP = {"feels", "feeling", "files", "fills", "fails", "falls", "seems", "sells", "tells", "wheels"}
# Ordinary words that must not be read as a misspelt database name ("pick up the car" is not the Pickup category).
NAME_FUZZY_SKIP = {"pick", "picks", "picked"}
# "How is the Territory?", "tell me about the City": a listed vehicle named at the end of a question about it.
ABOUT_VEHICLE = re.compile(r"\b(?:how(?:'s| is| was| about)|tell me (?:more )?about|describe|info(?:rmation)? (?:on|about)|details? (?:of|on|about)"
                           r"|review of|thoughts on|is it good|any good)\s+(?:the\s+)?[a-z0-9-]+(?:\s+[a-z0-9-]+)?\s*(?:po|ba)?\W*$")
# "Mga sasakyan nyo po" (normalised to "car your"), "your cars", "what vehicles do you have": the listings.
FLEET_LISTING = re.compile(r"^(?:what are |show me |list )?(?:all |the )?(?:cars?|vehicles?|units?) (?:your|you have|you got)(?: po| ba)?\W*$"
                           r"|^(?:your|ur) (?:cars|vehicles|units)\b|\bwhat (?:cars|vehicles|units) (?:are )?(?:yours|you have)\b"
                           r"|^(?:show|list|tell)(?: me)? what (?:i can|can i) (?:book|rent|reserve)\W*$|^what (?:can i|i can) (?:book|rent|reserve)\W*$")
FUZZY_CUTOFF = 0.86
# Text that is a database query, never something to run.
SQL_TEXT = re.compile(r"\b(?:select\s+[\w*,\s]+\s+from|insert\s+into|update\s+\w+\s+set|delete\s+from|drop\s+(?:table|database)|union\s+select)\b"
                      r"|'\s*(?:or|and)\s+'?\w+'?\s*=\s*'?\w+|\b(?:or|and)\s+1\s*=\s*1\b|;\s*--", re.IGNORECASE)
# Rental rules that share words with car symptoms ("can I smoke in the car?" is not about exhaust smoke).
SYMPTOM_SAFE_TOPICS = {"smoking in the vehicle", "pets in the vehicle"}
# Being in an accident: general safety steps, not company procedures.
ACCIDENT = re.compile(r"\b(?:accident|accidents|crash|crashed|collision|collided|fender bender|naaksidente|aksidente|nabangga|bumangga|banggaan|hit (?:a|an|the|another) \w+)\b")
# A model name that is also an ordinary word ("Ranger", "Mirage", "City") is the vehicle in these phrasings.
FLEET_CONDITION_SIGNAL = re.compile(
    r"\b(?:condition (?:report|of)|report (?:on|of|for)|worth (?:keeping|fixing|repairing|it)|retire|sell (?:the|it|our)|(?:problems?|issues?) with"
    r"|wrong with|needs? (?:service|repair|maintenance|work|fixing)|overdue|due for|assessments? (?:of|for)|in (?:bad|good|poor) (?:shape|condition))\b")
# Maintenance questions answered from vehicle assessments (administrators).
MAINTENANCE_RETIRE = re.compile(
    r"\b(?:retire|retiring|retirement|sell (?:it|the|this|that|our)|dispose|phase (?:it )?out|decommission\w*|end of (?:its )?(?:life|service)"
    r"|worth (?:keeping|fixing|repairing|it)|keep or (?:sell|replace|retire)|repair or (?:replace|retire|sell)|replace (?:the|this|that) (?:car|vehicle|unit)"
    r"|ibenta|itapon)\b")
MAINTENANCE_PRIORITY = re.compile(
    r"\b(?:first|priorit\w*|most urgent|urgently|urgent|which (?:car|vehicle|unit)s? (?:should|to) (?:we |i )?(?:service|fix|repair|check)"
    r"|what (?:should|do) (?:we|i) (?:service|fix|repair|check|do)|this week|today|top (?:\d+|three|five)|unahin|uunahin)\b")
MAINTENANCE_CONDITION = re.compile(
    r"\b(?:predict\w*|risk\w*|at[- ]risk|forecast\w*|break\s?downs?|breaks? down|fail\w*|likely|weakest|worst|component\w*|systems?|parts?"
    r"|condition|health|healthy|worsen\w*|getting worse|deteriorat\w*|declin\w*|wear\w*|assess\w*|screening|red flags?|warning signs?"
    r"|bad shape|poor shape|bad condition|sira|sirang|may sira)\b")
RENTAL_DAMAGE = re.compile(r"\b(?:damage|damaged|damaging|scratch|scratched|dent|dented|nasira|masira|gasgas|nagasgas)\b"
                           r"(?=.*\b(?:car|rental|vehicle|unit|it|sasakyan|kotse)\b)")
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
    ("plate_number", re.compile(r"\bplate(?:\s*(?:number|no\.?|#))?\b|\bplaka\b|\bregistration(?: (?:number|no\.?|of))?\b|\blicen[cs]e plate\b")),
    ("minimum_rental_duration", re.compile(r"\bminimum (?:rental|rent|number of days|days?|duration|period|booking|stay)\b|\bmin(?:imum)? days?\b"
                                           r"|\bshortest (?:rental|booking|period)\b|\bat least how many days\b|\bminimum na araw\b|\bilang araw (?:ang )?minimum\b")),
    ("year", re.compile(r"\bwhat year(?:'s)?\b|\bwhich (?:model )?year\b|\byear model\b|\bmodel year\b|\banong (?:model )?(?:taon|year)\b|\btaon\b|\bhow (?:old|new)\b|\byears? old\b"
                        r"|\byear of\b|\bage of\b|\bwhat model year\b")),
    ("color", re.compile(r"\bwhat colou?r\b|\bwhich colou?r\b|\bcolou?r (?:of|is)\b|\bcolou?r's\b|\bkulay\b")),
    ("seating_capacity", re.compile(r"\bhow many (?:\w+ )?(?:seats?|people|passengers|persons|pax|riders)\b|\b(?:seating|seat|passenger) (?:capacity|count)\b|\bilan[g]? (?:ang )?(?:upuan|tao|katao|seats?|seater|passengers|maisasakay)\b"
                                    r"|\bhow many (?:can|could|will|does|do)\b.*\b(?:ride|fit|sit|go|seat|hold|carry|take)\b|\bnumber of (?:seats|passengers)\b|\bkasya\b(?!.*\d)|\bseats? how many\b"
                                    r"|\b\d{1,2}(?:\s*-?\s*seaters?)?\s+or\s+(?:an?\s+)?\d{1,2}\s*-?\s*seaters?\b|\btotal seats\b|\bhow many are\b")),
    ("fuel_type", re.compile(r"\b(?:diesel|gasoline|gas|petrol) or (?:diesel|gasoline|gas|petrol)\b|\b(?:what|which) (?:kind of |type of )?(?:fuel|gas|gasoline)\b|\bfuel (?:type|of)\b"
                             r"|\b(?:run|runs|running) on\b|\bwhat do i fill\b|\bfill .* with\b|\btakes? what fuel\b|\bwhat (?:gas|fuel) does\b|\buses? what\b")),
    ("transmission", re.compile(r"\b(?:manual|automatic|stick(?: shift)?) or (?:a |an )?(?:manual|automatic|stick(?: shift)?)\b|\b(?:what|anong) (?:kind of |type of )?(?:transmission|gearbox)\b|\btransmission type\b")),
]
# "Which car has the most seats?" / "best for a big family": rank the listed cars by seating capacity.
SEATS_RANKING = re.compile(r"\b(?:most|more|biggest|largest|maximum) (?:seats|seating|people|passengers|space)\b|\bbig(?:gest)? (?:family|group)\b"
                           r"|\b(?:family|group|barkada) (?:trip|outing|car)\b|\bpinakamaluwag\b|\bpinakamalaki\b|\bmaraming (?:tao|upuan)\b|\bbiggest\b|\blargest\b"
                           r"|\b(?:holds?|carr(?:y|ies)|fits?|takes?) the most\b")
# "Which car is good for a long drive?": no comfort rating is stored, so the roomiest vehicles are listed first, with that said.
LONG_TRIP = re.compile(r"\b(?:long|longer|mahabang) (?:drives?|trips?|road trips?|distance|travel|biyahe)\b|\blong-distance\b|\bmalayuan\b"
                       r"|\bbiyaheng? malayo\b")
# Weekly or monthly rates (only daily rates are stored).
RATE_PERIOD = re.compile(r"\b(?:weekly|monthly|per week|per month|a week rate|week rates?|month rates?|lingguhan|buwanan)\b")
HOURLY = re.compile(r"\b(?:hourly|per hour|an hour|\d+\s*(?:hours?|hrs?|oras)|(?:one|two|three|four|five|six|few|couple of) hours?)\b")
# The customer's own past rentals.
PAST_RENTAL = re.compile(r"\b(?:did i|have i|i have|i've)\s+(?:rent|rented|book|booked|reserve|reserved|use|used)\b"
                         r"|\bi (?:rented|booked|reserved|used)\b|\blast time\b"
                         r"|\b(?:past|previous|last|recent|huling|dati) (?:rentals?|bookings?|reservations?|trips?)\b|\brental history\b|\bnirentahan ko\b")
# The rental car breaking down: safety steps and the rental team, not records.
BREAKDOWN = re.compile(r"\b(?:breaks? down|broke down|broken down|breakdown|stalled on the (?:road|highway)|tumirik|nasiraan|(?:rental|rented) (?:car |unit |vehicle )?(?:stopped working|isn't working|not working)"
                       # "The car stopped working" is a breakdown; "my power windows stopped working" is a part, answered as a car problem.
                       r"|(?:car|vehicle|unit|engine|it) (?:just |suddenly )?stopped working|^stopped working)\b")
# The customer's own booking of a named vehicle ("how much did I pay for the Honda City rental?").
OWN_VEHICLE_RECORD = re.compile(r"\b(?:did i pay|i paid|did i spend|did \w+ .*\bcost me|did (?:my|the) .*\b(?:rental|booking|reservation) cost"
                                r"|my (?:\w+ ){0,3}(?:rental|booking|reservation)s?\b)")
# "Who booked the Fortuner?" (administrators).
WHO_BOOKED = re.compile(r"\bwho(?! can\b| may\b| could\b| should\b| do\b| is allowed\b)(?:'s| is| has| have)?\s+(?:got\s+)?(?:the\s+)?(?:[\w-]+\s+){0,3}?(?:booked|book|rented|rent|reserved|reserve|renting|renter|booking|reservation)\b"
                        r"|\bwhose\b.*\b(?:booked|book|booking|reservation|reserved|rented|renting)\b|\bbooked by\b"
                        r"|\b(?:customer|renter|person)\b.*\b(?:booked|reserved|renting|rented)\b|\b(?:customer|renter) (?:of|on) the\b"
                        r"|\b(?:booked|reserved|rented)\b.*\bwho\b|\banyone (?:booked|reserved|renting)\b"
                        r"|\b(?:reservations?|bookings?) (?:for|of|on) the\b")
# Words that are never a customer's name in "bookings of ...".
NOT_A_NAME = {"the", "my", "our", "all", "today", "tomorrow", "this", "next", "week", "pending", "approval", "confirmation", "now", "me", "us", "review", "later", "reservations",
              "reservation", "bookings", "booking", "rentals", "rental", "vehicles", "cars", "units", "price", "rate", "maintenance", "what", "who", "that", "it", "there", "here", "where", "how",
              "when", "which", "let", "system", "everyone", "customers", "approving", "me", "completed", "cancelled", "confirmed", "rejected"}
# A reservation of a named status, in any common wording.
NEEDS_APPROVAL = re.compile(r"\b(?:should|do|must|need to) i (?:approve|review)\b|\bi (?:need|have) to (?:approve|review)\b|\b(?:waiting|awaiting)\b|\bneed\w* (?:\w+ )?(?:approv\w*|review\w*)\b|\bfor approval\b|\b(?:to|for) (?:approve|review)\b|\bunapproved\b"
                            r"|\bnot (?:yet )?(?:been )?approved\b|\b(?:haven'?t|have not|hasn'?t) (?:been )?approved\b|\bhindi pa (?:na)?approved\b|\bpending approval\b")
# "Did my payment go through?", "is my booking paid?", "nagbayad na ba ako?" (normalised to "pay na ba ako"): the
# customer's own payment status, not how to pay.
PAYMENT_STATUS = re.compile(r"\b(?:did|has|have|was|is) (?:my |the )?payments? (?:go(?:ne)? through|went through|been (?:received|verified|confirmed)|received|verified|confirmed|successful|ok(?:ay)?)\b"
                            r"|\bpayment (?:status|went through|go through|failed|didn'?t go through|not (?:confirmed|showing|reflected|verified))\b"
                            r"|\b(?:have|did) i (?:already )?paid\b|\bam i (?:already )?paid\b|\bi (?:already )?paid\b.*\b(?:still|but|why|pending|not)\b"
                            r"|\bis (?:my |the )?(?:booking|reservation|rental)(?: #?\d+)? (?:already )?paid\b|\b(?:booking|reservation) (?:is )?(?:not )?paid\b"
                            r"|\bpay na (?:ba )?(?:ako|ko)\b|\bbayad na (?:ba )?(?:ako|ko|yung|the)\b|\bnabayaran\b|\bpumasok (?:na )?(?:ba )?(?:ang |yung |the )?pay")
# "Which cars need reassessment?", "overdue assessments", "never assessed" (administrators): the monthly vehicle assessment.
REASSESSMENT = re.compile(r"\bre-?assess\w*\b|\bassessments? (?:are |is )?(?:due|overdue|late)\b|\b(?:due|overdue|need\w*|late) (?:for )?(?:an? |the )?(?:monthly )?assessments?\b"
                          r"|\bnever (?:been )?assessed\b|\bnot (?:yet |been )?assessed\b|\bmonthly (?:vehicle )?(?:assessments?|checks?)\b"
                          r"|\b(?:last|latest) assess(?:ed|ment)\b|\bwhen (?:was|were) .*\bassessed\b")
# "Who has the most bookings?", "top customers" (administrators): customers ranked by reservation count.
CUSTOMER_RANKING = re.compile(r"\b(?:who|which customers?|which renters?|sino)\b.*\b(?:most|pinakamaraming|pinakamadami)\b"
                              r"|\btop (?:customers?|renters?|clients?)\b|\b(?:best|frequent|loyal|regular) (?:customers?|renters?|clients?)\b"
                              r"|\bcustomers? with the most\b|\bmost (?:bookings|reservations) by\b")
OWE = re.compile(r"\b(?:owe|balance|still (?:need|have) to pay|need to pay|babayaran|bayaran ko|utang)\b")
# Another person's records by name ("Ana Santos booking details") or by account id ("user_id=3").
NAMED_RECORDS = re.compile(r"\b(?!My\b)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:'s)?\s+(?:bookings?|reservations?|rentals?|booking details|reservation details|account)\b")
ACCOUNT_ID = re.compile(r"\b(?:user_?id|user|customer_?id|customer|account)\s*(?:=|:|#|no\.?|number)?\s*\d+\b")


class Understanding:
    def __init__(self, message, vocabulary):
        self.original = message
        # "air-conditioned" is the aircon, not the maintenance word "condition".
        self.text = re.sub(r"\bair[- ]?condition(?:ed|ing|er)?\b|\bair[- ]con\b", "aircon", normalize_message(message).casefold())
        self.tokens = TOKEN.findall(self.text)
        self.concepts = {name for name, lexicon in CONCEPTS.items() if self._matches(lexicon)}
        if re.search(r"\b(?:battery|alternator|baterya)\b", self.text) and not re.search(r"\b(?:how much|magkano|price|cost|rent|rental)\b", self.text):
            self.concepts.discard("price")
        self.stock_question = bool(STOCK_QUESTION.search(self.text))
        if self.stock_question or FLEET_LISTING.search(self.text):
            self.concepts.add("listing")
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
            if _names_word(entry.get("model", ""), self.tokens, joined) or _names_first_word(entry.get("model", ""), self.tokens, self.text):
                self.models.append(entry)
        listed_makes = {str(make).casefold() for make in vocabulary.get("makes", [])}
        self.unlisted_brands = []
        for index, token in enumerate(self.tokens):
            if token in CAR_BRANDS and token not in listed_makes:
                # The word after the brand, with its hyphen kept ("Isuzu mu-X", "Mazda CX-5").
                after = re.search(rf"\b{re.escape(token)}\s+([a-z0-9][a-z0-9-]*)", self.text)
                following = after.group(1) if after else ""
                model = following if re.fullmatch(r"[a-z0-9][a-z0-9-]{1,}", following) and not following.isdigit() and following not in STOP_AFTER_BRAND else ""
                brand = BRAND_NAMES.get(token, token.title())
                self.unlisted_brands.append(f"{brand} {model.title() if not model.isupper() else model}".strip())
        # A known model name this viewer cannot rent ("is the Hiace available?" from a customer) is named, not ignored.
        listed_models = {word for entry in vocabulary.get("models", []) for word in TOKEN.findall(str(entry.get("model", "")).casefold())}
        listed_models |= {"".join(TOKEN.findall(str(entry.get("model", "")).casefold())) for entry in vocabulary.get("models", [])}
        named_brands = " ".join(self.unlisted_brands).casefold()
        squeezed = self.tokens + re.findall(r"\b([a-z]{2}-[a-z0-9]+)\b", self.text)
        for token in squeezed:
            token = token.replace("-", "")
            if token in KNOWN_MODELS and token not in listed_models and token not in named_brands and KNOWN_MODELS[token].casefold() not in named_brands:
                if token not in ORDINARY_MODEL_WORDS or self.stock_question or self.has("price", "availability"):
                    self.unlisted_brands.append(KNOWN_MODELS[token])
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
        # A model name can be an ordinary word ("City"), and "my Vios is overheating" is a car problem. Without a make, a vehicle
        # word or a rental question, such a mention is not a question about the listed vehicle.
        signal = (self.makes or self.has("price", "deposit", "availability", "specs", "vehicle_noun", "booking", "status", "maintenance")
                  or len(self.tokens) <= 3 or self.start_date or self.rental_days or len(self.models) >= 2
                  or self.equipment or self.attribute or self.colors or self.fuel_types or self.transmissions
                  or self.stock_question or self.min_seats or ABOUT_VEHICLE.search(self.text) or FLEET_CONDITION_SIGNAL.search(self.text)
                  or re.match(r"(?:and|what about|how about|eh|yung|e ang|paano ang)\b", self.text))
        if self.models and re.search(r"\b(?:open|taken)\b", self.text):
            self.concepts.add("availability")
        if not signal and (self.has("symptom") or any(str(entry.get("model", "")).casefold() in ORDINARY_MODEL_WORDS for entry in self.models)):
            self.models = []

    def _numbers(self):
        text = self.text
        self.rental_days = None
        match = re.search(r"\b(\d{1,3})\s*(?:-\s*)?(?:days?|nights?|d)\b", text.replace(",", " "))
        if match:
            self.rental_days = int(match.group(1))
        else:
            match = re.search(r"\b(" + "|".join(NUMBER_WORDS) + r")[\s-]+(?:na\s+)?(?:days?|nights?)\b", text)
            if match:
                self.rental_days = NUMBER_WORDS[match.group(1)]
        weeks = re.search(r"\b(\d{1,2}|" + "|".join(NUMBER_WORDS) + r")[\s-]+(?:na\s+)?(?:weeks?|linggo)\b", text)
        if self.rental_days is None and weeks:
            value = weeks.group(1)
            self.rental_days = 7 * (int(value) if value.isdigit() else NUMBER_WORDS[value])
        dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", self.original)
        self.start_date, self.end_date = (dates[0], dates[1]) if len(dates) >= 2 else (None, None)
        self.single_date = dates[0] if len(dates) == 1 else None
        count = r"(\d{1,2}|" + "|".join(NUMBER_WORDS) + r")"
        seats = (re.search(rf"\b{count}\s*(?:-\s*)?(?:na\s+)?(?:people|persons?|passengers|pax|seats?|seaters?|katao|tao|kami|of us)\b", text)
                 or re.search(rf"\b(?:group|family|party|barkada) of {count}\b|\bwe(?:'re| are) {count}\b|\bseats? {count}\b|\bseating {count}\b|\bfits? {count}\b"
                              rf"|\bfor {count}(?: people)?\s*$|\bgood for {count}\b(?!\s*(?:-\s*)?(?:na\s+)?(?:days?|nights?|weeks?|araw|linggo))|\bkasya(?: ba)?(?: ang| the)? {count}\b|\broom for {count}\b", text))
        value = next((g for g in seats.groups() if g), None) if seats else None
        self.min_seats = (int(value) if value.isdigit() else NUMBER_WORDS[value]) if value else None
        # "A 5 or 7 seater?" asks which one it is, so neither number is a requirement.
        self.seat_choice = bool(re.search(r"\b\d{1,2}(?:\s*-?\s*seaters?)?\s+or\s+(?:an?\s+)?\d{1,2}\s*-?\s*seaters?\b", text))
        if self.seat_choice:
            self.min_seats = None
        money = r"(?:php|p|₱)?\s*([\d][\d,]{2,})(?:\s*(?:php|pesos?))?"
        budget = (re.search(r"\b(?:under|below|less than|at most|max(?:imum)?|budget(?: of| is| ko| ay)?|badyet(?: ko)?|hanggang|not more than|only have|i have|meron akong|up to|spend)\s*(?:up to\s*)?" + money, text)
                  or re.search(money + r"\s*(?:pesos?\s*)?(?:pababa|or less|or below|or under|and below|and under|or cheaper)\b", text)
                  or re.search(r"\banything for\s*" + money, text)
                  or re.search(r"\bsa\s*" + money + r"\s*(?:na\s*)?(?:budget|badyet)\b", text)
                  or re.search(money + r"\s*(?:budget|badyet|ang budget|lang)\b", text)
                  or re.search(money + r"\s*(?:ako|lang ako)\b", text)
                  or (re.search(r"\b(?:with|for)\s*" + money + r"\s*(?:pesos?\s*)?(?:per day|a day|/day|daily|kada araw)\b", text)
                      if not re.search(r"\b\d+\s*(?:days?|nights?)\b", text) else None))
        self.max_daily_rate = float(budget.group(1).replace(",", "")) if budget else None
        self._year_like_budget = bool(budget and "," not in budget.group(1) and 1990 <= self.max_daily_rate <= 2035
                                      and not re.search(r"\b(?:budget|badyet|php|pesos?|under|below|less than|at most|max\w*|hanggang"
                                                        r"|only have|i have|pababa|up to|spend|or less)\b|₱", text))
        self._price_from_budget = self.max_daily_rate is not None and "price" not in self.concepts
        if self.max_daily_rate is not None:
            self.concepts.add("price")
        # Mileage first: "lowest mileage" is not "lowest price".
        if re.search(r"\b(?:highest|most|biggest|largest|pinakamataas na) (?:mileage|odometer|km|kilometers)\b|\bmost (?:driven|used km)\b", text):
            self.sort = "mileage_desc"
        elif re.search(r"\b(?:lowest|least|smallest) (?:mileage|odometer|km|kilometers)\b", text):
            self.sort = "mileage_asc"
        elif re.search(r"\b(?:cheapest|pinakamura\w*|pinaka ?mura|lowest|least expensive|least costly|most affordable|most budget friendly|costs? the least)\b", text):
            self.sort = "price_asc"
        elif re.search(r"\b(?:most expensive|most costly|pinakamahal\w*|pinaka ?mahal|highest (?:price|priced|rate|daily rate)|priciest|top priced|costs? the most)\b", text):
            self.sort = "price_desc"
        elif re.search(r"\b(?:newest|latest model|most recent model|pinakabago\w*)\b", text):
            self.sort = "year_desc"
        elif re.search(r"\b(?:oldest|pinakaluma\w*)\b", text):
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
        self.long_trip = bool(LONG_TRIP.search(text))
        self.seats_ranking = bool(SEATS_RANKING.search(text)) or self.long_trip
        self.rate_period = bool(RATE_PERIOD.search(text))
        self.hourly = bool(HOURLY.search(text))
        year = re.search(r"\b(20[0-3]\d|19[89]\d)\s*(?:or )?(?:newer|and (?:newer|up|above)|pataas|onwards)\b", text)
        self.min_year = int(year.group(1)) if year else None
        self.needs_approval = bool(NEEDS_APPROVAL.search(text))
        self.reservation_statuses = sorted({status for status in ("pending", "confirmed", "cancelled", "rejected", "completed")
                                            if re.search(rf"\b{status}\b", text)}
                                           | ({"cancelled"} if re.search(r"\bcanceled\b", text) else set())
                                           | ({"rejected"} if re.search(r"\b(?:declined|denied)\b", text) else set())
                                           | ({"completed"} if re.search(r"\b(?:finished|(?:are|is|were) done|done (?:reservations?|bookings?|rentals?))\b", text) else set())
                                           | ({"pending"} if self.needs_approval
                                              else {"confirmed"} if re.search(r"\b(?:approved|accepted)\b|\b(?:did we|we) (?:confirm|approve)\b", text) else set()))
        if self.needs_approval:
            self.reservation_statuses = sorted(set(self.reservation_statuses) - {"confirmed"})
        self.reservation_id = None
        match = re.search(r"(?:#\s*|\b(?:reservation|booking)s?\s*(?:no\.?|number|#)?\s*)(\d{1,8})\b", text) or re.search(r"\b(\d{1,8})\s+(?:reservation|booking)\b", text)
        if match:
            self.reservation_id = int(match.group(1))
        self._dates()
        self.vehicle_ids = [int(value) for value in re.findall(r"\b(?:vehicle|car|unit)\s*(?:id)?\s*#?\s*(\d{1,6})\b", text)][:5]
        self.vehicle_statuses = sorted({status for word, status in VEHICLE_STATUS_WORDS.items()
                                        # "Which cars can be rented?" asks what is available, not which cars are out on rent.
                                        if re.search(rf"\b{word}\b", re.sub(r"\b(?:(?:i|we) rented|(?:can|could|to|will|may) be rented)\b", "", text))
                                        and word != "available"}
                                       | ({"rented"} if re.search(r"\b(?:on rent|out on rent|currently rented|nirentahan|rented out|on hire|out right now|units? out|(?:is|are) out|currently out|what'?s out|naka-rent)\b", text)
                                          and not re.search(r"\bthe [\w-]+ out\b", text) else set())
                                       | ({"maintenance"} if re.search(r"\b(?:(?:in|at) the (?:shop|garage)|being (?:repaired|serviced|fixed)|under repair"
                                                                       r"|out of service)\b", text) else set()))
        if self.vehicle_statuses:
            self.concepts.add("status")
        elif re.search(r"\b(?:i|we) rented\b", text):
            self.concepts.discard("status")
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
            # "dalawang linggo" is two weeks, not a Sunday.
            without_weeks = re.sub(r"\b(?:\d{1,2}|" + "|".join(NUMBER_WORDS) + r")[\s-]+(?:na\s+)?linggo\b", " ", text)
            weekday = re.search(r"\b(?:next|this|on|sa|ngayong|sa susunod na)?\s*(" + "|".join(WEEKDAYS) + r")\b", without_weeks)
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
        token[:1] == words[0][:1] and difflib.SequenceMatcher(None, token, words[0]).ratio() >= 0.8
        for token in tokens if len(token) >= 4 and token not in NAME_FUZZY_SKIP)


def _names_first_word(name, tokens, text=""):
    """The first word of a longer model name ("montero" for "Montero Sport"), unless it is an ordinary word; an ordinary
    word counts too when the message is plainly about a vehicle's condition ("condition report of the mirage")."""
    words = TOKEN.findall(str(name).casefold())
    ordinary_ok = words and words[0] in ORDINARY_MODEL_WORDS and bool(FLEET_CONDITION_SIGNAL.search(text)) \
        and re.search(rf"\bthe {re.escape(words[0])}\b", text)
    return len(words) > 1 and len(words[0]) >= 4 and (words[0] not in ORDINARY_MODEL_WORDS or bool(ordinary_ok)) and words[0] in tokens


def plan(message, role, vocabulary=None, memory=None):
    """A retrieval plan, a direct answer type, or None (answered from the guide and references)."""
    u = Understanding(message, vocabulary)
    memory = memory or {}
    admin = role == "admin"
    pending = memory.get("pending_pickup")
    if pending and u.single_date and not u.rental_days and not u.start_date and u.single_date > pending:
        # The pickup date came in the previous message ("available on 2026-12-05?"); this one is the return date.
        u.start_date, u.end_date, u.single_date = pending, u.single_date, None

    # Other people's records or contact details.
    contact = re.search(r"\b(?:phone|phones|email|emails|address|addresses|contact|contacts|personal information)\b", u.text)
    other_person = OTHER_PERSON_RECORDS.search(u.text) and not re.search(r"\b(?:my|mine|ko)\b", u.text)
    if ABOUT_ASSISTANT.search(u.text) or OFF_TOPIC.search(u.text):
        return None
    if (MODEL_QUESTION.search(u.text) or GENERAL_REASSESSMENT.search(u.text)) and not (u.makes or u.models or u.vehicle_ids):
        return None  # how the models work, or what a reassessment is: the assistant explains, no records needed
    if not admin and COMPANY_REVENUE.search(u.text) and not u.has("own"):
        return _direct("admin_only", u)
    if DEPOSIT_REFUND.search(u.text) and not (u.makes or u.models or u.reservation_id):
        return None
    if SQL_TEXT.search(u.original):
        return _direct("refuse_sql", u)
    if CREDENTIALS.search(u.text) and not OWN_CREDENTIAL_HELP.search(u.text):
        return _direct("refuse_credentials", u)
    if ACCIDENT.search(u.text) and not u.has("maintenance"):
        return _direct("accident_guidance", u)
    if not admin and RENTAL_DAMAGE.search(u.text) and not (u.makes or u.models or u.reservation_id or u.has("deposit", "price")):
        return None  # "what if I damage the car?": the damage-reporting guidance, not a vehicle listing
    if not admin and BREAKDOWN.search(u.text) and not re.search(r"\b(?:predict\w*|risk\w*|forecast\w*|will)\b", u.text):
        return _direct("breakdown_guidance", u)
    if not admin and WHO_BOOKED.search(u.text) and not u.has("own"):
        return _direct("refuse_private", u)  # who booked a vehicle is another customer's information
    named = NAMED_RECORDS.search(u.original)
    # "Toyota Vios rental" names a vehicle, not a person.
    named_person = named and not re.search(r"\b(?:my|mine|ko)\b", u.text) and not (
        u.makes or u.models or set(TOKEN.findall(named.group(0).casefold())) & CAR_BRANDS)
    # "What's your phone number?" asks for the rental company's contact details (not recorded), not a person's.
    business_contact = re.search(r"\byour (?:phone|number|email|address|contact)", u.text) and not u.has("booking") \
        or re.search(r"\b(?:contact|reach|call|message|email) (?:you|u|us|the rental team|driverent)\b", u.text) \
        or re.search(r"\bdrive\b", u.text) and not contact  # "can someone else drive the rental?" is a driver policy
    if u.has("other_people") and (contact or u.has("booking")) and not business_contact or other_person or named_person or ACCOUNT_ID.search(u.text):
        if not admin or contact:
            return _direct("refuse_private", u)
        return _admin_lookup_plan(u) or _plan("reservation_summary", u, [("reservations.summary", {"upcoming": u.has("upcoming")})])

    if u.dtc_codes and not (admin and STORED_RECORDS.search(u.text) and not u.unlisted_brands):
        return None

    for topic, pattern in UNRECORDED_TOPICS:
        if topic == "business hours" and (u.models or u.makes):
            continue
        if pattern.search(u.text) and (not u.has("symptom") or topic in SYMPTOM_SAFE_TOPICS):
            result = _direct("policy_unrecorded", u)
            result["topic"] = topic
            return result

    if not admin and REASSESSMENT.search(u.text) and not HOW_TO_DO.search(u.text):
        return _direct("admin_only", u)  # vehicle assessments are fleet records
    if not admin and re.search(r"\bhow many (?:cars|vehicles|units|sasakyan)\b(?!.*\b(?:rented|maintenance|repair|broken|booked)\b)", u.text):
        # "How many cars do you have?": public, answered with the cars a customer can rent.
        result = _plan("vehicles", u, [("vehicles.search", _vehicle_filters(u))], purpose="listing")
        result["count_question"] = True
        return result
    if admin and _fleet_maintenance_question(u):
        return _maintenance_plan(u)
    if admin:
        status_all = STATUS_ALL.search(u.text)
        # With dates ("are they available tomorrow?") it is a booking check of the cars just discussed, not the fleet totals.
        if status_all and not (u.models or u.makes or u.start_date or u.single_date):
            # "Are they all rented?" after a list: the fleet's totals, with a yes or no first.
            result = _plan("fleet_status", u, [("fleet.status_counts", {})])
            result["asked_status"] = (status_all.group("status") or status_all.group("status2")).replace("under ", "in ")
            return result
        admin_plan = _admin_lookup_plan(u)
        if admin_plan:
            return admin_plan

    known_vehicle = bool(u.makes or u.models or u.categories or u.min_seats or u.max_daily_rate is not None or u.sort or u.seats_ranking
                         or ((u.fuel_types or u.transmissions or u.has_gps is not None) and u.has("vehicle_noun", "listing"))
                         # "Is GPS included?", "kasama ba ang dash cam?": which rentable cars have that equipment.
                         or ((u.has_gps is not None or u.has_aircon is not None or u.equipment)
                             and re.search(r"\b(?:included|include|includes|kasama|come with|comes with|equipped)\b", u.text)
                             and not u.has("symptom")))
    unlisted = bool(u.unlisted_brands or u.unlisted_categories)
    rental_question = u.has("price", "deposit", "availability") or known_vehicle
    # A brand or type the fleet does not have is only a rental question when renting is asked about.
    asks_vehicle = known_vehicle or unlisted and u.has("price", "deposit", "availability", "listing", "specs", "vehicle_noun") \
        or bool(u.unlisted_brands) and u.has("booking") and not HOW_TO_BOOK.search(u.text)  # "I want to book the Vios tomorrow"
    if unlisted and asks_vehicle and not re.search(r"\b(?:my|ko|problem|issue|noise|leak\w*|overheat\w*|smok\w*|won't|slipping|shifting)\b", u.text):
        rental_question = True
    if u.has("symptom") and not rental_question and not (admin and u.has("maintenance", "status")):
        return None
    if CARE_AMOUNT.search(u.text) and not asks_vehicle:
        return None
    # "Difference between diesel and gasoline?" is general knowledge unless listed vehicles are named.
    if re.search(r"\b(?:difference between|differences between|vs\.?|versus)\b", u.text) and not (u.makes or u.models or u.categories or u.unlisted_brands):
        return None
    if POLICY_FEE.search(u.text) and not u.reservation_id:
        return None
    if NEEDS_TO_RENT.search(u.text) and not asks_vehicle:
        return None
    # "Kelangan ko ba ng ID pag kukunin yung kotse?", "do I need my licence at pickup?": a requirement, not my bookings.
    if re.search(r"\b(?:need|needs|kailangan|kelangan|require\w*|bring|dalhin|dadalhin|present)\b.*\b(?<!user )(?<!account )(?<!customer )"
                 r"(?:id|ids|license|licence|lisensya|documents?|requirements?)\b", u.text) \
            and not ACCOUNT_ID.search(u.text) and not u.reservation_id and not asks_vehicle:
        return None
    # "What if I need the car longer?", "can I keep it longer?": the extension rules in the guide.
    if re.search(r"\b(?:longer than (?:planned|expected|booked)|need (?:it|the car|the vehicle|the unit) longer|keep (?:it|the car|the vehicle) longer"
                 r"|(?:add|extra) (?:a few |more |one more )?days?\b.*\b(?:rental|booking|reservation)|mas matagal)\b", u.text) and not u.reservation_id:
        return None
    if HOW_TO_BOOK.search(u.text) and not (u.has("price", "deposit") or u.rental_days or u.start_date or u.has("own")):
        return None
    if u.has("price") and SERVICE_COST.search(u.text) and not (u.has("deposit", "availability") or re.search(r"\b(?:rent|rental|per day|daily)\b", u.text)):
        return None

    if not admin and PAYMENT_STATUS.search(u.text) and not HOW_TO_DO.search(u.text):
        args = {"reservation_id": u.reservation_id} if u.reservation_id else {}
        return _plan("own_bookings", u, [("bookings.mine", args)], purpose="payment")

    # The signed-in customer's own bookings.
    own_booking =u.has("own") and u.has("booking") or re.search(r"\b(?:any|ba)\s+(?:bookings?|reservations?)\b", u.text) and u.has("own") \
        or OWN_PICKUP_RETURN.search(u.text) and re.search(r"\b(?:my|the) (?:car|vehicle|rental|unit)\b", u.text) and not asks_vehicle \
        or PAST_RENTAL.search(u.text) and u.has("own") \
        or u.has("own") and re.search(r"\b(?:spent|nagastos|gastos|nagbayad)\b", u.text) and not asks_vehicle
    # "How much did I pay for the Honda City rental?" / "When do I pick up the Vios?": the customer's booking of that car.
    named_vehicles = [entry["model"] for entry in u.models] + u.unlisted_brands
    hypothetical = bool(HYPOTHETICAL.search(u.text)) or bool(u.rental_days)
    vehicle_record = not admin and bool(named_vehicles) and bool(
        OWN_VEHICLE_RECORD.search(u.text) and not hypothetical or OWN_PICKUP_RETURN.search(u.text) and not u.has("price"))
    # "Previous bookings", "When do I bring back the car?": first-person questions about a rental that need no vehicle named.
    own_booking = own_booking or vehicle_record or (not admin and not asks_vehicle and not u.has("deposit") and (
        PAST_RENTAL.search(u.text) or OWN_PICKUP_RETURN.search(u.text) or re.search(r"\buntil when\b.*\b(?:my|booking|reservation|rental)\b", u.text)))
    if own_booking and hypothetical and (u.has("price") or u.rental_days or re.search(r"\bpay\b", u.text)) and not vehicle_record and not re.search(r"\b(?:my|mine|ko)\b", u.text):
        own_booking = False
    # "Show reservation 41": a customer can only look up reservations on their own account.
    numbered = not admin and u.reservation_id and re.search(r"\b(?:reservations?|bookings?)\b|#\s*\d", u.text)
    if (own_booking or numbered) and not (u.has("price") and asks_vehicle and not numbered and not vehicle_record):
        if admin:
            return _direct("admin_no_bookings", u)
        # Viewing or checking bookings is not a procedure; changing them is ("how much did I pay" is a lookup).
        procedure = bool(re.search(r"\b(?:cancel|extend|extension|change|reschedule|modify|move|pay|payment)\b"
                                   r"|\badd (?:more |extra )?days\b|\bmore days\b|\blonger\b", u.text)) \
            and not re.search(r"\b(?:did i pay|i paid|how much (?:did|do) i (?:pay|owe))\b", u.text)
        asks_total = re.search(r"\b(?:total|cost|costs|paid|pay for|how much|magkano)\b", u.text)
        purpose, args = "list", {}
        if u.reservation_id:
            purpose, args = "details", {"reservation_id": u.reservation_id}
        elif re.search(r"\b(?:days? (?:are )?left|remaining|natitira|ilang araw pa)\b", u.text):
            purpose, args = "days_left", {"upcoming": True}
        elif re.search(r"\bhow (?:many days|long) (?:is|was|will be) (?:my|the)\b|\bhow long is my\b|\bilang araw (?:ang )?(?:booking|rental|reservation)\b", u.text):
            purpose = "length"
        elif PAST_RENTAL.search(u.text) and re.search(r"\b(?:last|previous|past|recent|huling|rented|history|before)\b", u.text) \
                and not u.has("upcoming") and not asks_total:
            purpose = "last"
        elif vehicle_record and asks_total or asks_total and re.search(r"\b(?:my|the) (?:\w+ )?(?:booking|reservation|rental)(?:'s)? total\b"
                                                                      r"|\btotal (?:of|for) my\b|\bmy (?:\w+ )?(?:booking|reservation|rental)\b", u.text) \
                and not re.search(r"\b(?:spent|spend|gastos|nagastos|owe)\b", u.text):
            purpose = "total"
        elif re.search(r"\b(?:spent|spend|gastos|nagastos|paid|total so far|how much (?:have|did) i)\b", u.text) or OWE.search(u.text):
            purpose = "spent"
        elif re.search(r"\b(?:next|upcoming|susunod) (?:pickup|pick up|return|booking|reservation|rental|trip)\b|\bwhen(?:'s| is) my (?:pickup|return)\b"
                       r"|\b(?:pickup|pick up|return|start|end) date\b|\bwhen (?:does|will) my\b|\bwhen am i\b|\buntil when\b", u.text) \
                or OWN_PICKUP_RETURN.search(u.text) or re.search(r"\bwhen\b.*\b(?:pick ?up|pickup|get)\b", u.text):
            purpose, args = "next", {"upcoming": True}
        elif re.search(r"\b(?:status|confirmed|approved|approval|pending|accepted|rejected|cancelled)\b", u.text) and not procedure:
            purpose = "status"
        elif u.has("upcoming") and not procedure:
            args = {"upcoming": True}
        result = _plan("own_bookings", u, [("bookings.mine", args)], purpose=purpose, guide=procedure)
        if vehicle_record or named_vehicles and purpose in ("next", "total", "status"):
            result["vehicle_names"] = named_vehicles
        if u.reservation_statuses:
            result["asked_statuses"] = u.reservation_statuses
        if OWE.search(u.text):
            result["owe"] = True
        return result

    # Fleet-wide status for administrators ("fleet status", "how many cars do we have").
    if admin and not u.has("maintenance") and (
            re.search(r"\bfleet\b", u.text) and (u.has("status") or re.search(r"\b(?:overview|summary|size|status|how many)\b", u.text))
            or re.search(r"\bfleet (?:count|size)\b|\bhow big is (?:our|the) fleet\b", u.text)
            or not u.vehicle_statuses and not u.has("availability") and not u.models and re.search(
                r"\bhow many\b.*\b(?:cars?|vehicles?|units?)\b|\b(?:number|count) of (?:cars|vehicles|units)\b|\b(?:cars?|vehicles?|units?) (?:count|total)\b|\btotal (?:number of )?(?:cars|vehicles|units)\b", u.text)):
        return _plan("fleet_status", u, [("fleet.status_counts", {})])

    # Fleet and maintenance records.
    records_question = u.has("maintenance") or (u.has("status") and (u.vehicle_statuses or u.has("listing")) and not u.has("availability"))
    general_advice = re.search(r"\b(?:how often|should i|when should|what is a|what does|tips?)\b", u.text) \
        or GENERAL_MAINTENANCE_QUESTION.search(u.text)
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

    # "My car is using too much gas" is about the customer's own car, not the rental listings.
    if not (u.makes or u.models or u.categories) and not u.has("price", "deposit") and re.search(r"\bmy (?:car|vehicle|engine|auto)\b|\bcar ko\b", u.text):
        return None

    filters = _vehicle_filters(u)
    follow_up = False
    compare_earlier = False
    remembered = [item for item in memory.get("last_vehicles") or [] if item.get("model")]
    if len(u.models) < 2 and len(remembered) >= 2 and CROSS_TURN_COMPARE.search(u.text):
        # "Compare them", "which one is cheaper?" after two or more vehicles were shown.
        filters["makes"] = sorted({item["make"] for item in remembered if item.get("make")})
        filters["models"] = sorted({item["model"] for item in remembered})
        follow_up = compare_earlier = True
    elliptical = ELLIPTICAL.search(u.text) or PRONOUN.search(u.text) or len(u.tokens) <= 6 or DATES_ONLY.fullmatch(u.text)
    if not compare_earlier and not asks_vehicle and memory.get("last_vehicles") and (u.rental_days or u.has("price", "deposit", "specs", "availability") or u.start_date
                                                                                   or u.single_date and DATES_ONLY.fullmatch(u.text)) \
            and elliptical and not BROADER_LIST.search(u.text):
        # "How much for 3 days?" after asking about a vehicle refers to that vehicle.
        filters["makes"] = sorted({item["make"] for item in memory["last_vehicles"] if item.get("make")})
        filters["models"] = sorted({item["model"] for item in memory["last_vehicles"] if item.get("model")})
        follow_up = True

    if not asks_vehicle and not follow_up and PRONOUN.search(u.text) and u.has("price", "deposit", "specs") and not u.rental_days:
        # "How much is it?" with nothing discussed before: ask which vehicle instead of guessing.
        return _direct("clarify_vehicle", u)
    if not asks_vehicle and not follow_up and u.has("price") and not (u.rental_days or u.start_date or u.has("deposit") or u.rate_period) \
            and not re.search(r"\b(?:cars|vehicles|units|rates|prices|fleet|all|each|every|list|lahat|daily)\b", u.text):
        # "How much?" or "Can you calculate the total?": neither the vehicle nor the days are known yet.
        return _direct("clarify_quote", u)

    # "I want to rent a car tomorrow until Sunday": dates and wanting a car is an availability check.
    wants_car_on_dates = bool(u.start_date) and bool(re.search(r"\b(?:rent|renting|book|booking|reserve|need|want|hire|rentahan|magrent|mag-rent)\b", u.text)) \
        and (u.has("vehicle_noun") or re.search(r"\b(?:rent|renting|rentahan|magrent|mag-rent)\b", u.text))
    if wants_car_on_dates:
        u.concepts.add("availability")
    if not (asks_vehicle or follow_up or u.has("price", "deposit", "availability") or u.rental_days
            or u.has("vehicle_noun") and u.has("listing", "specs") or u.equipment and u.has("vehicle_noun", "listing")
            or re.search(r"^(?:show me |so )?what (?:do )?you have(?: available| for rent)?\W*$", u.text)
            or re.search(r"\b(?:on offer|lineup|line up|choose from|what have you got|got anything)\b", u.text) or FLEET_LISTING.search(u.text)):
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
    elif (u.models or follow_up) and u.min_seats and not (u.has("price", "deposit", "availability") or u.rental_days or u.start_date):
        # "Is the Vios a 7 seater?" / "Can 7 people fit in the Vios?": a question about this car, not a filter that hides it.
        filters.pop("min_seats", None)
        feature_check = {"feature": "seats", "value": u.min_seats}
    elif (u.models or follow_up) and (YES_NO.search(u.text) or u.has_gps is not None or u.has_aircon is not None) and not u.has("price", "deposit", "availability"):
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
    if feature_check is None and (compare_earlier or len(u.models) >= 2 and re.search(r"\b(?:or|o|vs\.?|versus|compare\w*|difference|differ|mas|than|kaysa)\b", u.text)):
        purpose = "compare"
    elif feature_check:
        purpose = "feature_check"
    elif u.has("deposit") and not (u.rental_days or u.start_date):
        purpose = "deposit"
    elif u.rental_days or u.start_date:
        purpose = "quote" if (u.rental_days and not u.start_date) or u.has("price", "deposit") else "availability"
    elif RECOMMEND.search(u.text) and len(u.models) < 2 and not u.sort:
        # "What car can you suggest?", "best car for 4 people": a recommendation from what can be rented now.
        purpose = "recommend"
        if admin and not filters.get("statuses"):
            filters["statuses"] = ["available"]
    elif u.sort in ("mileage_desc", "mileage_asc", "year_desc", "year_asc") or (u.seats_ranking and len(u.models) < 2 and not u.min_seats):
        purpose = "ranking"
    elif u.has("price") or u.sort in ("price_asc", "price_desc"):
        purpose = "price"
    elif u.has("availability"):
        purpose = "availability"
    elif u.has("specs") or u.min_seats or (len(u.models) == 1 and not u.has("listing")) or (u.models and u.has("explain")):
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
    if purpose == "deposit":
        # "Which car has the biggest deposit?": the answer first, then the list from highest to lowest.
        if re.search(r"\b(?:biggest|highest|largest|most|max(?:imum)?|pinakamalaki|pinakamataas)\b", u.text):
            result["deposit_rank"] = "desc"
        elif re.search(r"\b(?:smallest|lowest|least|min(?:imum)?|cheapest|pinakamaliit|pinakamababa)\b", u.text):
            result["deposit_rank"] = "asc"
    if purpose in ("price", "listing") and u.rate_period:
        result["rate_period"] = True
    if u.hourly and not u.rental_days:
        result["hourly"] = True
        if purpose in ("listing", "availability", "details"):
            result["purpose"] = "price"
    if u.long_trip and result.get("rank_by") == "seats":
        result["long_trip"] = True
    if u.stock_question and (u.models or u.makes or u.categories) and purpose in ("price", "listing", "details"):
        result["stock_check"] = True
    if purpose == "compare":
        result["compare"] = {"by": "price" if u.has("price") or re.search(r"\b(?:expensive|cheaper|pricier|costs? more|costs? less|mahal|mura)\b", u.text)
                             else "seats" if re.search(r"\b(?:bigger|larger|seats?|more people|maluwag)\b", u.text) else "all"}
        if re.search(r"\b(?:cheaper|cheapest|costs? less|less expensive|mas mura|pinakamura|mura)\b", u.text):
            result["compare"]["lead"] = "cheaper"
    if getattr(u, "relative_period", False):
        result["period_from_words"] = True
    if u.single_date and not u.rental_days and not u.start_date:
        result["clarification"] = "return_date"
        result["pending_pickup"] = u.single_date  # remembered for a few messages: the next date is the return date
        if result.get("purpose") in ("price", "listing", "details"):
            result["purpose"] = "availability"  # a date alone asks whether the vehicle is free, so ask for the return date
    return result


def _admin_lookup_plan(u):
    """Administrator questions about one car, reservations, costs, revenue and utilisation."""
    text = u.text
    if HOW_TO_DO.search(text):
        return None  # "How do I mark a reservation as completed?" is a how-to question for the guide.
    vehicle = {k: v for k, v in _vehicle_filters(u).items() if k in ("makes", "models", "categories")}
    if u.vehicle_ids:
        vehicle["vehicle_ids"] = u.vehicle_ids
    if REASSESSMENT.search(text):
        return _plan("reassessment_due", u, [("assessments.due", vehicle)])
    period = {k: v for k, v in (("from", u.start_date), ("to", u.end_date)) if v}
    period.update(_report_period(text))
    if re.search(r"\b(?:revenue|income|earnings?|earns?|earned|earning|grossing|sales|kita|kinita)\b|\bbook(?:ed|ing|ings)? (?:value|amount|total)\b"
                 r"|\b(?:bookings|reservations) (?:are )?worth\b|\bvalue of (?:all )?(?:the )?(?:bookings|reservations)\b"
                 r"|\bhow much (?:money )?(?:have|did|do) (?:we|the business|the company)(?: \w+)? (?:make|made|earn|earned)\b|\bmoney (?:have |did )?we (?:make|made|earn|earned)\b", text):
        result = _plan("revenue", u, [("reservations.revenue", period)])
        if u.reservation_statuses:
            result["focus_statuses"] = u.reservation_statuses
        # "Revenue per car", "which car earns the most", "top customers by revenue".
        if re.search(r"\b(?:per|by|each|every|for each) (?:car|vehicle|unit|model)s?\b|\bwhich (?:car|vehicle|unit)s?\b|\b(?:car|vehicle)s? (?:earn|earning|make|made)\b", text):
            result["breakdown"] = "vehicle"
        elif re.search(r"\b(?:per|by|each|every) (?:customer|renter|client)s?\b|\b(?:which|top|best) (?:customer|renter|client)s?\b|\bwho (?:spent|spends|paid|brings)\b", text):
            result["breakdown"] = "customer"
        return result
    if re.search(r"\b(?:utili[sz]ation|utili[sz]ed|most (?:rented|booked|used)|least (?:rented|booked|used)|busiest|idle)\b", text):
        return _plan("utilization", u, [("fleet.utilization", period)])
    named = [entry["model"] for entry in u.models]
    if WHO_BOOKED.search(text) and named:
        # Reservations of one vehicle, with the customer's name (administrators only).
        result = _plan("reservation_summary", u, [("reservations.summary", {})])
        result["vehicle_names"] = named
        return result
    spending = re.search(r"\b(?:spent|spend|spending|cost|costs|total|expenses?|gastos|nagastos|bill|paying|pay|paid)\b", text)
    if spending and re.search(r"\b(?:maintenance|maintain|services?|repairs?|fix|fixing|fixes)\b", text) \
            and not re.search(r"\b(?:how much (?:is|does) (?:an?|the) )", text) and not re.search(r"\b(?:last|latest|recent|previous)\b", text):
        return _plan("maintenance_costs", u, [("maintenance.costs", {**vehicle, **period})])
    # "What have we spent on the Hiace?": money spent on a named car is its maintenance cost.
    if named and re.search(r"\b(?:spent|spend|nagastos|gastos)\b", text):
        return _plan("maintenance_costs", u, [("maintenance.costs", {**vehicle, **period})])
    reservation_word = re.search(r"\b(?:reservations?|bookings?|pickups?|pick-ups?|returns?)\b|^what(?:'s| is) booked\b", text) or (
        re.search(r"\brentals?\b", text) and not (u.has("price", "deposit") or u.models))  # "rental price of the Vios" is a vehicle's price
    # "My reservations" from an administrator is about their own (none), handled by the bookings rule.
    # "What's pending?" from an administrator asks about reservations.
    status_only = u.reservation_statuses and not (u.models or u.makes or u.has("maintenance"))
    if (reservation_word or status_only or CUSTOMER_RANKING.search(text)) and not re.search(r"\b(?:my|mine|ko|akin)\b(?! approv)", text) and not u.has("explain") and not u.has("policy"):
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
        elif period:
            # "Reservations this month": the whole month, since bookings later this month count too.
            args.update(period)
            if re.search(r"\bthis month\b|\bngayong buwan\b", text):
                args["to"] = (date(today.year + (today.month == 12), today.month % 12 + 1, 1) - timedelta(days=1)).isoformat()
        if CUSTOMER_RANKING.search(text):
            result = _plan("reservation_summary", u, [("reservations.summary", args)])
            result["rank_customers"] = True
            return result
        name = re.search(r"\b(?:of|for|by|ni|ng customer)\s+([a-z][a-z]+(?:\s+[a-z][a-z]+)?)\s*(?:'s)?\s*$", text) \
            or re.search(r"\b([a-z]+(?:\s+[a-z]+)?)'s\s+(?:reservations?|bookings?)", text)
        # A vehicle or an ordinary word is not a person ("rental price of the Vios", "what's reservation 52", "complete list of reservations").
        if name and not set(TOKEN.findall(name.group(1))) & NOT_A_NAME and not u.models and not u.makes and not u.reservation_id:
            args["customer_name"] = name.group(1).title()
        return _plan("reservation_summary", u, [("reservations.summary", args)])
    asks_status = re.search(r"\b(?:status|where is|where's|how is|how's|how far|mileage|odometer|km|kilometers|kilometres|kilometro|last (?:service|serviced|maintenance)|next (?:service|maintenance)"
                            r"|currently (?:booked|rented)|is (?:it|the [\w-]+(?: [\w-]+){0,2}) (?:booked|rented|available|out|in the shop)|kumusta|nasaan)\b", text)
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


def _fleet_maintenance_question(u):
    """An administrator asking about the fleet's condition, priorities or retirement, not a general explanation
    ("what is predictive maintenance?" is answered from the knowledge base)."""
    text = u.text
    if GENERAL_MAINTENANCE_QUESTION.search(text) or MODEL_QUESTION.search(text) or REASSESSMENT.search(text) \
            or HOW_TO_DO.search(text) or re.search(r"\b[0-9]{5}[a-z0-9]{3,7}\b", text):
        # Explanations, the models themselves, the reassessment schedule, how-to questions (the guide) and part numbers
        # (the parts list) are answered elsewhere.
        return False
    named = bool(u.vehicle_ids or u.makes or u.models)
    if named and NAMED_CAR_NEEDS.search(text) and not u.has("price", "deposit", "booking", "availability") \
            and not re.search(r"\b(?:rent|renting|book|license|licence|requirements?)\b", text):
        return True
    scope = named or re.search(
        r"\b(?:cars?|vehicles?|units?|fleet|we|our|ours|component\w*|systems?|parts?|sasakyan|kotse)\b", text)
    if not scope:
        return False
    work = re.search(r"\b(?:service\w*|maint\w*|fix\w*|repair\w*|check\w*|inspect\w*|attention|work|shop|garage|mechanic|talyer"
                     r"|ipaayos|paayos|ayusin|ipagawa|pagawa)\b", text)
    return bool(MAINTENANCE_RETIRE.search(text) or MAINTENANCE_CONDITION.search(text) and not u.has("price", "booking", "availability")
                or MAINTENANCE_PRIORITY.search(text) and work)


# "What does the Hiace need?", "what's wrong with the Vios?", "any problems with the Innova?"
NAMED_CAR_NEEDS = re.compile(r"\b(?:needs?|needed|wrong|issues?|problems?|condition|health|check on|report|fix|repair)\b")
# Questions about the prediction models themselves: answered from the model registry, not from fleet records.
MODEL_QUESTION = re.compile(
    r"\b(?:how (?:accurate|reliable|good|well)|accuracy|precision|recall|(?:how|what) (?:does|do) (?:the |your )?(?:prediction|maintenance|predictive|ml|ai)? ?(?:model|models|ai|system) (?:work|use|learn)"
    r"|what data (?:do|does|did) (?:you|it|the models?)|trained on|training data|is the model (?:approved|accepted|trained|working|ready|reliable)"
    r"|why (?:are|is) (?:the )?(?:model|predictions?)\b.*\b(?:unavailable|not available|off|disabled|not approved|not working)"
    r"|model (?:status|accuracy|performance|evaluation)|machine learning"
    r"|(?:prediction|predictive|maintenance|fault) models?\b.*\b(?:approved|accepted|ready|in use|working|trained|active)"
    r"|(?:are|is) (?:the |your )?(?:models?|predictions?) (?:approved|accepted|ready|in use|working|trained|active|reliable|accurate))\b")
# "What is a reassessment?": the guide, not the reassessment list.
GENERAL_REASSESSMENT = re.compile(r"^(?:what(?:'s| is| are) (?:a |the )?reassessments?|what does reassessment mean|explain (?:the )?reassessments?"
                                  r"|how (?:do|does) (?:the )?reassessments? work|why (?:do|does) .*reassess)")
# "What is predictive maintenance?", "how long do brake pads last?": knowledge, not fleet records.
GENERAL_MAINTENANCE_QUESTION = re.compile(
    r"^(?:what(?:'s| is| are) (?:a |an |the )?(?:predictive|preventive|preventative|condition[- ]based|reactive|corrective|proactive|planned|scheduled)\b"
    r"|what (?:is|are) (?:the )?difference|difference between|explain|define|what does .* mean|how (?:long|often) (?:do|does|should|can)"
    r"|how (?:do|does|can|should) (?:fleets?|companies|you|i|we) (?:predict|prioriti[sz]e|plan|reduce|decide)|why (?:do|does|is|are) .*(?:predictive|preventive)"
    r"|what are (?:the )?(?:signs|warning signs|early signs|symptoms)|tips?\b|best practices?)")


def _maintenance_plan(u):
    filters = {"vehicle_ids": u.vehicle_ids, **{k: v for k, v in _vehicle_filters(u).items() if k in ("makes", "models")}}
    text = u.text
    named = bool(u.vehicle_ids or filters.get("makes") or filters.get("models"))
    if re.search(r"\b(?:recurring|repeated|repeat|again and again|keeps? (?:coming|happening)|same (?:issue|problem))\b", text):
        return _plan("maintenance_recurring", u, [("maintenance.recurring", dict(filters))])
    # "Should we retire the Hiace?", "is it worth fixing?": the facts that decide repair or replace.
    if MAINTENANCE_RETIRE.search(text):
        return _plan("maintenance_retire", u, [("assessments.condition", dict(filters)), ("maintenance.costs", dict(filters)),
                                                ("maintenance.due", dict(filters))])
    # "Any vehicle with poor condition in inspections?": the component-inspection records, which are their own records.
    if re.search(r"\b(?:component )?inspections?\b|\binspected\b", text) and not re.search(r"\bassess\w*\b", text):
        return _plan("maintenance_inspections", u, [("maintenance.inspections", dict(filters))])
    # "What should we service first this week?": one ranked list from assessments and the service schedule.
    if MAINTENANCE_PRIORITY.search(text):
        return _plan("maintenance_priority", u, [("assessments.condition", dict(filters)), ("maintenance.due", dict(filters))])
    # "Which component is most likely to fail?", "which cars are at risk?": the latest assessments, their trend and the
    # verified fault history, plus the registered model when one is accepted.
    if MAINTENANCE_CONDITION.search(text) and not u.dtc_codes:
        purpose = "vehicle" if named else "fleet"
        result = _plan("maintenance_condition", u, [("assessments.condition", dict(filters)), ("maintenance.due", dict(filters)),
                                                     ("maintenance.model_inputs", {})], purpose=purpose)
        # What the question is about decides what comes first: systems, the trend, or the vehicles at risk.
        if re.search(r"\b(?:worsen\w*|getting worse|deteriorat\w*|declin\w*|trend\w*|dropp\w*)\b", text):
            result["focus"] = "trend"
        elif re.search(r"\b(?:component\w*|systems?|parts?|weakest|weak spots?)\b", text):
            result["focus"] = "systems"
        return result
    # "What does the Hiace need?", "what's wrong with the Vios?": one car's condition, service status and recent work.
    if named and NAMED_CAR_NEEDS.search(text) and not re.search(r"\b(?:records?|history|logs?|cost\w*|spent)\b", text):
        return _plan("maintenance_condition", u, [("assessments.condition", dict(filters)), ("maintenance.due", dict(filters)),
                                                   ("maintenance.records", {"limit": 5, **filters})], purpose="vehicle")
    if u.dtc_codes or re.search(r"\b(?:dtc|trouble codes?|fault codes?|diagnostic\w*|inspection\w*|inspected|condition)\b", text):
        return _plan("maintenance_inspections", u, [("maintenance.inspections", dict(filters))])
    if re.search(r"\b(?:due|overdue|need\w*|upcoming|soon|schedule\w*|next service)\b", text):
        retrievals = [("maintenance.due", dict(filters))]
        if re.search(r"\b(?:need\w*|attention|work)\b", text):
            # "Which cars need maintenance?": the schedule alone misses what the assessments found.
            retrievals.append(("assessments.condition", dict(filters)))
        result = _plan("maintenance_due", u, retrievals)
        if re.search(r"\boverdue\b", text) and named:
            result["asked_overdue"] = True  # "Why is the Hilux overdue?" is answered yes or no first
        return result
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
            "unknown_names": list(dict.fromkeys(_unknown_names(u) + list(u.unlisted_brands) + list(u.unlisted_categories)))}


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
    plan = {
        "version": 1,
        "intent": intent,
        "purpose": purpose,
        "retrievals": [{"tool": tool, "args": args} for tool, args in retrievals],
        "requested": _requested(u),
        "follow_up": follow_up,
        "guide": guide,
    }
    if intent == "vehicles" and re.search(r"\b(?:why|bakit)\b", u.text) and u.vehicle_statuses:
        plan["asks_why_status"] = True  # "Why are the sample cars unavailable?": no reason is stored; say so first
    return plan


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
    sub_plans = []
    for part in parts:
        named_before = [{"model": name} for sub in sub_plans for name in ((sub or {}).get("requested") or {}).get("models", [])]
        context = {**(memory or {}), "last_vehicles": named_before} if named_before else memory
        sub_plans.append(_with_remembered_period(plan(part, role, vocabulary, context), part, memory))
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
    text = message.strip().casefold()
    # "How much is that?" / "what's the total?" right after a dated check prices those dates.
    price_follow_up = result.get("follow_up") and "rental_days" not in request["args"] and re.search(
        r"^(?:so )?(?:how much(?: is| would)? (?:it|that|this|those dates|for those dates)|what(?:'s| is) the total|total|magkano (?:yan|iyan|yun|iyon|lahat)|how much (?:in total|all in all))\b", text)
    if "start_date" in request["args"] or not (re.match(r"^(?:and|what about|how about|eh|yung|e ang|paano ang)\b", text) or price_follow_up):
        return result
    request["args"].update(start_date=period["start_date"], end_date=period["end_date"])
    request["args"].pop("rental_days", None)
    if result.get("purpose") not in ("quote", "availability"):
        result["purpose"] = "availability"
    result["follow_up_period"] = True
    return result
