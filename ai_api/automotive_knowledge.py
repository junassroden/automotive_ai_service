import json
import re
from functools import lru_cache
from pathlib import Path

from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer


MINIMUM_RETRIEVAL_SCORE = 0.12
# Function words that say nothing about the topic; they don't count when checking how much of a question matched.
COVERAGE_FILLER = {"did", "going", "goes", "went", "up", "down", "get", "gets", "getting", "got", "keeps", "really", "very", "just", "still",
                   "now", "been", "being", "was", "were", "will", "would", "there", "these", "those", "again", "lately", "today", "always"}
SYMPTOM_FILLER = {"not", "working", "work", "works", "won", "wont", "t", "go", "goes", "into", "like", "smell", "smells", "smelling", "start", "starts",
                  "turn", "turns", "stop", "stops", "feel", "feels", "make", "makes", "making", "noise", "noises", "sound", "sounds", "light", "lights",
                  "on", "off", "came", "come", "after", "out", "does", "can", "with", "caught", "suddenly", "keeps"}
GENERIC_ENGLISH = {"no", "symptom", "symptoms", "open", "safety", "safe", "specific", "verify", "whether", "details", "final", "finals", "new", "old",
                   "good", "bad", "issue", "issues", "trouble", "help", "need", "needs", "question", "thing", "things", "way", "time", "times",
                   "conditions", "condition", "today", "road", "roads", "difference", "differences", "between", "versus", "vs",
                   # Vehicle types and makes say which vehicle, not what is wrong with it.
                   "motorcycle", "motorbike", "bike", "van", "truck", "pickup", "suv", "sedan", "hatchback",
                   "toyota", "honda", "ford", "nissan", "mitsubishi", "hyundai", "kia", "suzuki", "mazda", "isuzu", "chevrolet", "subaru"}
NEAR_TIE_RATIO = 0.9
COVERAGE_TIE_RATIO = 0.8
CLEARLY_BETTER_COVERAGE = 0.5
# One specific word in common is enough only with a reasonable retrieval score ("wheel" alone in a lug-nut torque question is not).
MIN_SINGLE_TERM_SCORE = 0.15
# Words that ask about a topic without naming one.
QUESTION_ONLY_TERMS = {"cause", "causes", "caused", "causing", "reason", "reasons", "happen", "happens", "happening", "mean", "means",
                       "serious", "safe", "fix", "fixed", "repair", "cost", "costs", "wrong", "else", "more", "bad", "dangerous",
                       "normal", "drive", "driving", "still", "now", "so", "then", "really", "much", "will", "would", "go", "away"}
MAX_RETRIEVED_PASSAGES = 6
RELATED_PASSAGE_SCORE_RATIO = 0.45
GENERIC_QUERY_TERMS = {
    "automotive", "car", "cars", "engine", "fault", "maintenance",
    "problem", "symptom", "system", "vehicle", "vehicles",
}
QUERY_STOP_WORDS = {
    "a", "about", "am", "an", "and", "are", "as", "at", "be", "before",
    "between", "can", "could", "do", "does", "for", "from", "get", "give",
    "has", "have", "help", "how", "i", "if", "in", "is", "it", "keep",
    "me", "my", "of", "on", "or", "please", "should", "tell", "that",
    "the", "this", "to", "vehicle", "what", "when", "where", "which",
    "while", "why", "with", "you", "your", "car", "automotive",
    "maintenance", "problem", "issue", "symptom", "fault", "part",
    "component", "system", "check", "inspect", "need", "first",
    "make", "weather", "traffic", "last", "bulletin", "vin", "explain",
}


# Ranking compares every question word with every passage and keyword word, so these pure helpers see
# the same inputs millions of times per question; caching them keeps a chat answer fast.
@lru_cache(maxsize=16384)
def _normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


@lru_cache(maxsize=65536)
def _term_variants(term):
    variants = {term}
    if len(term) > 5 and term.endswith("ing"):
        stem = term[:-3]
        variants.update((stem, stem + "e"))
    if len(term) > 4 and term.endswith("ed"):
        variants.add(term[:-2])
        if len(term) > 5 and term[-3] == term[-4]:
            variants.add(term[:-3])  # "flagged" -> "flag"
    if len(term) > 4 and term.endswith("ies"):
        variants.add(term[:-3] + "y")
    if len(term) > 4 and term.endswith("es"):
        variants.add(term[:-2])
    if len(term) > 3 and term.endswith("s"):
        variants.add(term[:-1])
    return frozenset(variants)


@lru_cache(maxsize=262144)
def _terms_match(left, right):
    return not _term_variants(left).isdisjoint(_term_variants(right))


def _document_passages(document):
    passages = []
    title = document.get("title", "")
    for field in (
        "overview",
        "component_functions",
        "possible_causes",
        "symptoms",
        "checks",
        "diagnostic_steps",
        "professional_diagnostics",
        "vehicle_variations",
        "known_unknowns",
        "urgency",
        "needed_information",
    ):
        value = document.get(field)
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, str) and item.strip():
                passages.append({
                    "field": field,
                    "text": item.strip(),
                    "search_text": " ".join(
                        (title, " ".join(document.get("keywords", [])), item)
                    ),
                    "lexical_text": " ".join((title, item)),
                })
    return passages


@lru_cache(maxsize=1)
def _load_index():
    path = Path(settings.AUTOMOTIVE_KNOWLEDGE_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Automotive knowledge base not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Automotive knowledge base is invalid JSON at line "
            f"{error.lineno}, column {error.colno}: {error.msg}"
        ) from error

    documents = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(documents, list) or not documents:
        raise ValueError("Automotive knowledge base must contain documents.")

    document_ids = set()
    for document in documents:
        required = (
            "id", "title", "category", "keywords", "overview",
            "possible_causes", "symptoms", "checks", "urgency",
            "needed_information", "safety_critical", "sources",
        )
        if not isinstance(document, dict) or any(
            field not in document for field in required
        ):
            raise ValueError(
                "Every automotive knowledge document must contain the "
                "required structured fields."
            )
        if not isinstance(document["id"], str) or not document["id"].strip():
            raise ValueError("Automotive knowledge document IDs must be strings.")
        if document["id"] in document_ids:
            raise ValueError(
                f"Duplicate automotive knowledge document ID: {document['id']}"
            )
        document_ids.add(document["id"])
        if not isinstance(document["keywords"], list) or not document["keywords"]:
            raise ValueError(
                f"Knowledge document '{document['id']}' needs keywords."
            )

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words=sorted(QUERY_STOP_WORDS),
    )
    character_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
    )
    passages = [
        (document_index, passage)
        for document_index, document in enumerate(documents)
        for passage in _document_passages(document)
    ]
    texts = [passage["search_text"] for _, passage in passages]
    word_vectors = word_vectorizer.fit_transform(texts)
    character_vectors = character_vectorizer.fit_transform(texts)
    return (
        documents,
        passages,
        word_vectorizer,
        word_vectors,
        character_vectorizer,
        character_vectors,
    )


def knowledge_document(document_id):
    """The knowledge document with this id, or None."""
    return next((document for document in _load_index()[0] if document["id"] == document_id), None)


def rank_knowledge(query):
    if not isinstance(query, str) or not query.strip():
        return []

    (
        documents,
        passages,
        word_vectorizer,
        word_vectors,
        character_vectorizer,
        character_vectors,
    ) = _load_index()
    query = query.strip()
    query_terms = [
        term for term in _normalize(query).split()
        if term not in QUERY_STOP_WORDS
    ]
    if not query_terms:
        return []

    retrieval_query = " ".join(query_terms)
    word_query = word_vectorizer.transform([retrieval_query])
    character_query = character_vectorizer.transform([retrieval_query])
    word_scores = (word_vectors @ word_query.T).toarray().ravel()
    character_scores = (character_vectors @ character_query.T).toarray().ravel()
    normalized_query = _normalize(query)
    query_term_set = set(query_terms)
    query_words = normalized_query.split()
    ranked = []
    coverage_cache = {}

    def topic_coverage(document_index):
        """Share of the question's words named by this document's keywords or title."""
        if document_index not in coverage_cache:
            document = documents[document_index]
            names = set(_normalize(" ".join(document.get("keywords", []) + [document.get("title", "")])).split())
            coverage_cache[document_index] = round(sum(
                any(_terms_match(term, name) for name in names) for term in query_term_set) / len(query_term_set), 4)
        return coverage_cache[document_index]

    phrase_cache = {}

    def keyword_phrase(document_index):
        """(score, length) of the document's best keyword phrase in the question; the same for all its passages."""
        if document_index in phrase_cache:
            return phrase_cache[document_index]
        phrase_score = 0.0
        phrase_length = 0
        for keyword in documents[document_index].get("keywords", []):
            normalized_keyword = _normalize(keyword)
            if not normalized_keyword:
                continue
            keyword_words = normalized_keyword.split()
            keyword_length = len(keyword_words)
            if normalized_keyword in normalized_query:
                weight = 1.0
            elif keyword_length >= 3 and all(
                any(_terms_match(word, term) for term in query_words) for word in keyword_words
            ):
                # Every word of the keyword in another order ("check engine light is blinking"); on equal
                # scores the longer, more specific keyword wins (see the sort below).
                weight = 1.0
            else:
                continue
            phrase_length = max(phrase_length, keyword_length)
            phrase_score = max(
                phrase_score,
                weight * min(1.0, keyword_length / max(1, len(query_terms)) + 0.35),
            )
        phrase_cache[document_index] = (phrase_score, phrase_length)
        return phrase_cache[document_index]

    for index, (document_index, passage) in enumerate(passages):
        document = documents[document_index]
        phrase_score, phrase_length = keyword_phrase(document_index)

        keyword_terms = set(_normalize(passage["lexical_text"]).split())
        matched_terms = 0
        for term in query_term_set:
            if any(_terms_match(term, candidate) for candidate in keyword_terms):
                matched_terms += 1
        keyword_coverage = (
            matched_terms / len(query_term_set) if query_term_set else 0.0
        )
        vector_score = (
            0.65 * float(word_scores[index])
            + 0.35 * float(character_scores[index])
        )
        lexical_score = max(phrase_score * 0.9, keyword_coverage * 0.48)
        score = max(vector_score, lexical_score)
        ranked.append({
            "document": document,
            "field": passage["field"],
            "text": passage["text"],
            "score": round(score, 4),
            "phrase_length": phrase_length,
            "topic_coverage": topic_coverage(document_index),
        })

    # On equal scores the document whose keyword phrase matched more of the question wins
    # ("white smoke from exhaust" over "smoke from exhaust"), then the one whose keywords and title
    # name more of the question's words ("leaking oil": the oil-leak topic over overheating).
    ranked.sort(key=lambda item: (item["score"], item["phrase_length"], item["topic_coverage"]), reverse=True)
    return ranked


def retrieve_knowledge(
    query,
    minimum_score=MINIMUM_RETRIEVAL_SCORE,
    max_passages=MAX_RETRIEVED_PASSAGES,
):
    normalized_query = _normalize(query)
    if "service bulletin" in normalized_query and "vin" in normalized_query:
        return None
    if set(normalized_query.split()) - QUERY_STOP_WORDS <= QUESTION_ONLY_TERMS:
        # "What causes it?" names no topic; only the conversation can say what "it" is.
        return None
    ranked = rank_knowledge(query)
    if not ranked or ranked[0]["score"] < minimum_score:
        return None
    # Among near-equal scores, a document that matched a keyword phrase, or whose keywords and title name
    # clearly more of the question, wins (a document can score well only because its safety note mentions
    # another topic: "the red oil pressure light came on" vs. the oil-level entry).
    near = [item for item in ranked if item["score"] >= ranked[0]["score"] * NEAR_TIE_RATIO]
    preferred = max(near, key=lambda item: (item["phrase_length"], item["topic_coverage"], item["score"]))
    if preferred["phrase_length"] > ranked[0]["phrase_length"] or (
            preferred["phrase_length"] == ranked[0]["phrase_length"]
            and preferred["topic_coverage"] >= ranked[0]["topic_coverage"] + CLEARLY_BETTER_COVERAGE):
        ranked = [preferred] + [item for item in ranked if item is not preferred]
    elif ranked[0]["phrase_length"] == 0:
        # With no keyword phrase on top, a document whose keywords and title name clearly more of the question
        # also wins from a little further back ("I smell gas in my car": the fuel-leak entry over a rotten-egg
        # smell entry that shares only "smell").
        wider = [item for item in ranked if item["score"] >= ranked[0]["score"] * COVERAGE_TIE_RATIO]
        covering = max(wider, key=lambda item: (item["topic_coverage"], item["score"]))
        if covering["topic_coverage"] >= ranked[0]["topic_coverage"] + CLEARLY_BETTER_COVERAGE:
            ranked = [covering] + [item for item in ranked if item is not covering]
    query_terms = [
        term for term in _normalize(query).split()
        if term not in QUERY_STOP_WORDS
    ]

    def matching_terms(document):
        keywords = {
            term
            for keyword in document.get("keywords", [])
            for term in _normalize(keyword).split()
        }
        specific_terms = set(query_terms) - GENERIC_QUERY_TERMS
        matching = {
            term
            for term in specific_terms or query_terms
            if term in keywords
            or any(_terms_match(term, candidate) for candidate in keywords)
        }
        return matching

    primary = ranked[0]["document"]
    if not matching_terms(primary):
        return None
    if ranked[0]["phrase_length"] == 0:
        # Without a keyword phrase, one stray word in common ("who won the NBA finals" / "won't start")
        # is not a topic match: at least two specific words, or half of them, must match.
        # Without a keyword phrase, at least one matched word must be specific to cars: generic words
        # ("not working", "won't go", "smells like", "no symptoms", "open safety recall") don't identify a topic.
        specific_matches = matching_terms(primary) - SYMPTOM_FILLER - COVERAGE_FILLER - GENERIC_ENGLISH
        if not specific_matches or (len(specific_matches) == 1 and ranked[0]["score"] < MIN_SINGLE_TERM_SCORE):
            return None
    top_score = ranked[0]["score"]
    score_floor = max(minimum_score, top_score * RELATED_PASSAGE_SCORE_RATIO)
    selected = []
    seen = set()
    for result in ranked:
        if (
            result["document"]["id"] != primary["id"]
            and not matching_terms(result["document"])
        ):
            continue
        identity = (result["document"]["id"], result["field"])
        if result["score"] < score_floor or identity in seen:
            continue
        seen.add(identity)
        selected.append(result)
        if len(selected) >= max_passages:
            break
    best = dict(selected[0])
    best["passages"] = selected
    best["documents"] = list({
        item["document"]["id"]: item["document"]
        for item in selected
    }.values())
    return best


def format_knowledge_answer(document, diagnostic_context=None, query=None, concise=False):
    is_retrieval = "document" in document
    if is_retrieval:
        retrieval = document
        primary = retrieval["document"]
        documents = retrieval.get("documents", [primary])
        passages = retrieval.get("passages", [])
    else:
        primary = document
        documents = [document]
        passages = []

    component_functions = primary.get("component_functions")
    asks_component_function = (
        isinstance(query, str)
        and re.search(
            r"\b(what does|what is the function|what is the role|purpose of)\b",
            query.casefold(),
        )
    )
    if asks_component_function and isinstance(component_functions, list):
        sections = ["Component roles: " + " ".join(component_functions)]
    else:
        sections = ["Direct answer: " + primary["overview"]]
        if component_functions:
            sections.append("Relevant system operation: " + " ".join(component_functions))

        passages_by_field = {}
        for passage in passages:
            passages_by_field.setdefault(passage["field"], [])
            if passage["text"] not in passages_by_field[passage["field"]]:
                passages_by_field[passage["field"]].append(passage["text"])

        def retrieved_section(field, fallback):
            if field not in passages_by_field:
                return fallback
            selected_text = []
            primary_value = primary.get(field)
            if isinstance(primary_value, list):
                selected_text.extend(primary_value)
            elif isinstance(primary_value, str):
                selected_text.append(primary_value)
            selected_text.extend(
                text
                for passage in passages
                if passage["field"] == field
                and passage["document"]["id"] != primary["id"]
                for text in [passage["text"]]
            )
            return " ".join(dict.fromkeys(selected_text))

        causes = passages_by_field.get("possible_causes")
        if causes:
            sections.append(
                "Possible causes and mechanisms: "
                + retrieved_section("possible_causes", " ".join(causes))
            )
        else:
            sections.append(
                "Possible causes: " + " ".join(primary["possible_causes"])
            )

        symptoms = passages_by_field.get("symptoms")
        if symptoms:
            sections.append(
                "Clues that help distinguish causes: "
                + retrieved_section("symptoms", " ".join(symptoms))
            )
        else:
            sections.append("Clues to note: " + " ".join(primary["symptoms"]))

        checks = passages_by_field.get("checks")
        if checks:
            sections.append(
                "Safe preliminary checks: "
                + retrieved_section("checks", " ".join(checks))
            )
        else:
            sections.append("Safe preliminary checks: " + " ".join(primary["checks"]))

        diagnostic_steps = primary.get("diagnostic_steps", [])
        professional = primary.get("professional_diagnostics", [])
        if diagnostic_steps or professional:
            sections.append(
                "Workshop diagnostic approach: "
                + " ".join(diagnostic_steps + professional)
            )

        variations = primary.get("vehicle_variations", [])
        if variations:
            sections.append(
                "Vehicle-specific differences: " + " ".join(variations)
            )

        urgency = passages_by_field.get("urgency") or [primary["urgency"]]
        sections.append("Urgency: " + " ".join(urgency))

        unknowns = primary.get("known_unknowns", [])
        if unknowns:
            sections.append("What remains unknown: " + " ".join(unknowns))

        needed_information = passages_by_field.get("needed_information")
        sections.append(
            "Useful follow-up details: "
            + " ".join(needed_information or primary["needed_information"])
        )

    if isinstance(diagnostic_context, dict):
        vehicle = diagnostic_context.get("vehicle")
        if isinstance(vehicle, dict):
            vehicle_name = " ".join(
                str(vehicle.get(field, "")).strip()
                for field in ("make", "model", "year")
                if vehicle.get(field) not in (None, "")
            )
            details = []
            if vehicle_name:
                details.append(vehicle_name)
            mileage = vehicle.get(
                "current_mileage_km",
                vehicle.get("mileage"),
            )
            if mileage is not None:
                details.append(f"recorded mileage: {mileage} km")
            component = diagnostic_context.get("selected_component")
            if isinstance(component, str) and component.strip():
                details.append(f"selected component: {component.strip()}")
            if details:
                sections.append(
                    "Backend-provided diagnostic context (recorded facts, "
                    "not a new diagnosis): " + "; ".join(details) + "."
                )

        history = diagnostic_context.get("maintenance_history")
        if isinstance(history, list):
            sections.append(
                f"The backend supplied {len(history)} maintenance-history "
                "record(s); records are not proof of current component condition."
            )

        inspections = diagnostic_context.get("component_inspections")
        if isinstance(inspections, list):
            if not inspections:
                sections.append(
                    "No inspection is recorded for the selected component, so "
                    "its condition is unknown."
                )
            elif isinstance(inspections[0], dict):
                latest = inspections[0]
                condition = str(latest.get("recorded_condition") or "not assessed")
                sections.append(
                    f"The backend supplied {len(inspections)} inspection record(s) "
                    f"for the selected component; the latest ({latest.get('date')}) "
                    f"recorded its condition as {condition.replace('_', ' ')}. "
                    "This is the recorded inspection result, not a new diagnosis."
                )

    references = []
    for source_document in documents:
        for source in source_document.get("sources", []):
            if not isinstance(source, dict):
                continue
            name = source.get("name")
            reference = source.get("reference")
            url = source.get("url")
            if isinstance(name, str) and isinstance(reference, str):
                citation = f"{name}: {reference}"
                if isinstance(url, str) and url.startswith("https://"):
                    citation += f" ({url})"
                if citation not in references:
                    references.append(citation)
    if references:
        sections.append(
            "References: knowledge-base pointers (not fetched for this response): "
            + " ".join(references)
        )
    source_gaps = []
    for source_document in documents:
        gap = source_document.get("source_gap")
        if isinstance(gap, str) and gap and gap not in source_gaps:
            source_gaps.append(gap)
    if source_gaps:
        sections.append("Source limitation: " + " ".join(source_gaps))

    sections.append(
        "This is general information, not a confirmed diagnosis or a "
        "determination that the vehicle is safe to drive."
    )
    if concise and has_chat_fields(primary):
        # Chat answers come from the matched document only; the backend's recorded facts are kept after it.
        context = [section for section in sections if section.startswith(BACKEND_CONTEXT_SECTIONS)]
        return "\n\n".join([compose_chat_answer(primary, query)[0]] + context)
    if concise:
        # Chat answers: the direct answer, causes, signs, safe checks and next steps; background sections are left out.
        sections = [section for section in sections if not section.startswith(CONCISE_OMIT)]
        sections = [section.replace("Urgency: ", "Next steps: ", 1) for section in sections]
    return "\n\n".join(sections)


CONCISE_OMIT = ("Relevant system operation:", "Workshop diagnostic approach:", "Vehicle-specific differences:",
                "What remains unknown:", "Useful follow-up details:", "References:")
BACKEND_CONTEXT_SECTIONS = ("Backend-provided diagnostic context", "The backend supplied", "No inspection is recorded")


# ---------------------------------------------------------------------------
# Chat answers: short, from the matched document only, shaped by the question.
# ---------------------------------------------------------------------------

ASKS_CAUSE = re.compile(r"\b(?:why|causes?|caused|causing|reasons?|what'?s wrong|what is wrong)\b", re.IGNORECASE)
ASKS_ACTION = re.compile(
    r"\b(?:what (?:should|do|can|must) i do|what to do|what now|how (?:do|can|should) i (?:fix|repair|handle|deal|check|change|top)"
    r"|how to (?:fix|repair|check|change|top up|handle)|can i fix|should i)\b", re.IGNORECASE)
ASKS_DEFINITION = re.compile(r"^\s*(?:what (?:is|are|does|do)|what's|whats|define|meaning of|explain)\b", re.IGNORECASE)
# Something is happening now: the safety instruction goes first.
EMERGENCY = re.compile(
    r"\b(?:while driving|right now|suddenly|on the (?:highway|road|expressway|freeway)|fire|flames?"
    r"|failed|failure|not working|stopped working|won'?t stop|can'?t stop|cannot stop|blew|blowout|burst)\b",
    re.IGNORECASE)


FUNCTION_QUESTION = re.compile(
    r"\b(?:what does (?:an? |the )?[a-z -]+ do|what do (?:the )?[a-z -]+ do|what is the (?:function|role|purpose)|purpose of|function of|role of)\b",
    re.IGNORECASE)
FUNCTION_FILLER = {"does", "do", "function", "role", "purpose", "work", "works", "an", "the"}


def has_chat_fields(document):
    return isinstance(document.get("answer"), str) and document["answer"].strip() != ""


def _words(text):
    return set(_normalize(text).split())


def mentions(text, terms):
    """True when any of ``terms`` (or a simple plural/verb variant) appears in ``text``."""
    words = _words(text)
    return any(_terms_match(term, word) for term in terms for word in words)


def _join(items, conjunction="or"):
    items = [item.rstrip(".") for item in items if item]
    if len(items) <= 1:
        return "".join(items)
    if any("," in item or f" {conjunction} " in item for item in items):
        # Items that contain commas or "or" themselves are separated with semicolons so the list stays readable.
        return "; ".join(items[:-1]) + f"; {conjunction} " + items[-1]
    return ", ".join(items[:-1]) + f" {conjunction} " + items[-1]


def _sentence(text):
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else text + "."


def causes_sentence(causes, lead="Common causes include"):
    return _sentence(f"{lead} {_join(causes)}") if causes else ""


def likely_causes(causes):
    """The causes as a mechanic says them. Entries don't rank their causes reliably, so no order is claimed."""
    if not causes:
        return ""
    return _sentence(f"Usually it's {_join(causes)}")


# "Can I still drive it?", "is it safe to drive?", "pwede pa ba i-drive?"
ASKS_DRIVE = re.compile(
    r"\b(?:can i (?:still |keep )?drive|could i (?:still )?drive|can i (?:still |keep )?use (?:it|the car|my car|the vehicle)\b(?! (?:instead|for))|safe to drive|ok(?:ay)? to drive|is it safe|still drive|keep driving"
    r"|drive (?:it )?with|drive on it|pwede (?:pa )?(?:ba )?(?:i-?|ma)?(?:drive|gamitin|byahe)|delikado)\b", re.IGNORECASE)
# "How much does it cost to fix?", "magkano ipagawa?"
ASKS_PRICE = re.compile(r"\b(?:how much (?:does|do|will|would|is|to|for)|cost|costs|price|magkano|expensive|mahal ba)\b", re.IGNORECASE)
COST_TIP = "Get a written quote before the work starts."
# Words in "how often should X be replaced" that don't name the component.
INTERVAL_FILLER = {"often", "long", "last", "lasts", "replace", "replaced", "replacing", "change", "changed", "should", "be", "interval",
                   "intervals", "life", "lifespan", "typical", "every", "when", "much", "km", "car", "parts", "part"}


def cost_answer(document):
    """What decides the bill for this problem, the way a shop explains it before quoting. No prices are invented:
    the entry's cost note when it has one, otherwise the likely causes the price depends on."""
    if document.get("cost_note"):
        return f"No exact price without seeing the car. What decides it: {document['cost_note']} {COST_TIP}"
    causes = list(document.get("key_causes") or [])[:3]
    if causes:
        return (f"No exact price without seeing the car: it depends on which it turns out to be: {_join(causes)}. "
                f"Have it diagnosed first so you only pay for the part that's bad. {COST_TIP}")
    return f"No exact price without seeing the car; it depends on what the workshop finds. {COST_TIP}"


def compose_chat_answer(document, query=""):
    """(answer, parts shown) for a chat question, from this document only.

    Simple and definition questions get the direct answer and one next step; "why" questions
    get the causes; "what should I do" questions get the steps; a problem happening right now
    gets the safety instruction first. Internal source names are never part of the text."""
    query = query or ""
    if FUNCTION_QUESTION.search(query):
        terms = [term for term in _normalize(query).split() if term not in QUERY_STOP_WORDS and term not in FUNCTION_FILLER]
        roles = [item for item in document.get("component_functions") or [] if terms and mentions(item, terms)]
        if roles:
            return " ".join(roles[:2]), ["component_functions"]
    causes = list(document.get("key_causes") or [])
    actions = list(document.get("actions") or [])
    stop = document.get("stop")
    emergency = bool(stop) and bool(EMERGENCY.search(query))
    wants_action = bool(ASKS_ACTION.search(query))
    wants_cause = bool(ASKS_CAUSE.search(query))
    definition = bool(ASKS_DEFINITION.search(query)) and not wants_cause and not wants_action

    if document.get("interval_items"):
        # "How often should brake pads be replaced?": the item asked about, not the whole table.
        terms = [term for term in _normalize(query).split() if term not in QUERY_STOP_WORDS and term not in INTERVAL_FILLER]
        scores = {item: sum(1 for term in terms if mentions(item.split(":")[0], [term])) for item in document["interval_items"]}
        best = max(scores.values(), default=0)
        matched = [item for item, score in scores.items() if best and score == best]
        if matched and len(matched) <= 3:
            return ("Typical interval: " + "; ".join(matched) + ". The owner's manual decides for each car, and city traffic and heat "
                    "put rental cars at the short end of the range."), ["interval_items"]
    if ASKS_DRIVE.search(query) and (document.get("drive_safety") or stop) and not emergency:
        # "Can I still drive with the check-engine light on?": the yes or no first, like a mechanic would.
        verdict = document.get("drive_safety") or stop
        rest = [document["answer"]] + ([_sentence(actions[0])] if actions else [])
        return "\n\n".join([verdict] + rest), ["drive_safety" if document.get("drive_safety") else "stop", "answer", "actions:1"]
    if ASKS_PRICE.search(query) and not definition:
        # "How much does a brake pad replacement cost?": what decides the bill, without inventing prices.
        return "\n\n".join([document["answer"], cost_answer(document)]), ["answer", "cost"]

    paragraphs, shown = [], ["answer"]
    if emergency:
        paragraphs.append(stop)
        shown.append("stop")
    paragraphs.append(document["answer"])
    if definition:
        cause_count, action_count = 0, 1
    elif wants_action:
        cause_count, action_count = 3, 3
    elif wants_cause:
        cause_count, action_count = 5, 1
    else:
        cause_count, action_count = 4, 2
    if document.get("triage") and not definition:
        # The narrowing question takes the place of the other suggestions; the next steps follow from the reply.
        action_count = min(action_count, 1)
    if causes and cause_count:
        paragraphs.append(likely_causes(causes[:cause_count]))
        shown.append(f"causes:{min(cause_count, len(causes))}")
    if actions and action_count:
        paragraphs.append(" ".join(_sentence(item) for item in actions[:action_count]))
        shown.append(f"actions:{min(action_count, len(actions))}")
    if stop and not emergency:
        paragraphs.append(stop)
        shown.append("stop")
    triage = document.get("triage") or []
    if triage and not definition:
        already = triage_reply(document, query)
        if already is not None:
            # The question already says what a narrowing question would ask ("green fluid near the front", "only in traffic").
            narrowed, asked = already
            # On a first message the closing "have a mechanic test those parts" line is not needed yet.
            narrowed = narrowed.replace("\n\n" + TRIAGE_CLOSING, "")
            return "\n\n".join([document["answer"], narrowed]), ["answer"] + asked
        # Like a mechanic: one question that splits the likely causes; the reply is handled by triage_reply().
        paragraphs.append(f"To narrow it down: {triage[0]['ask']}")
        shown.append("triage_asked:0")
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph), shown


TRIAGE_CLOSING = "If that doesn't settle it, have a mechanic test those parts before replacing anything."


def triage_reply(document, message, shown=()):
    """(answer, parts shown) when the message answers one of the topic's narrowing questions, else None.

    The answer says what the reply points to and asks the next question that hasn't been asked yet."""
    triage = document.get("triage") or []
    if not triage:
        return None
    shown = set(shown or ())
    asked = sorted(int(item.split(":")[1]) for item in shown if item.startswith("triage_asked:"))
    answered = {int(item.split(":")[1]) for item in shown if item.startswith("triage_answered:")}
    # The question just asked is checked first, then any other question the reply happens to answer.
    order = list(reversed(asked)) + [index for index in range(len(triage)) if index not in asked]
    text = message.casefold()
    for index in order:
        if index in answered:
            continue
        hits = [(option_position, match.group(0))
                for option_position, option in enumerate(triage[index]["answers"])
                for match in re.finditer(option["match"], text, re.IGNORECASE)]
        if not hits:
            continue
        # The most specific answer wins: "cold while driving but warm in traffic" hits the traffic answer twice.
        counts = {}
        for position, _ in hits:
            counts[position] = counts.get(position, 0) + 1
        position = max(counts, key=lambda key: (counts[key], -key))
        option = triage[index]["answers"][position]
        parts, new = [option["narrow"]], [f"triage_answered:{index}"]
        upcoming = [i for i in range(len(triage)) if i not in answered and i != index and i not in asked]
        if upcoming:
            parts.append(f"Next: {triage[upcoming[0]]['ask']}")
            new.append(f"triage_asked:{upcoming[0]}")
        else:
            parts.append(TRIAGE_CLOSING)
        return "\n\n".join(parts), new
    return None


def compose_follow_up(document, kind, shown=(), subject=None, lead=None):
    """(answer, parts shown) for a follow-up about the document already being discussed.

    ``kind``: "condition" (the user ruled something out: "what if the coolant is full?"),
    "safety", "cost", "causes", "fix" or "more". Parts already shown are not repeated."""
    shown = set(shown or ())
    causes = list(document.get("key_causes") or [])
    actions = list(document.get("actions") or [])
    stop = document.get("stop")
    professional = document.get("professional")

    if kind == "condition":
        excluded = [term for term in _normalize(subject or "").split() if term not in CONDITION_SUBJECT_FILLER]
        remaining = [cause for cause in causes if not _names_part(cause, excluded)]
        next_steps = [action for action in actions if not _names_part(action, excluded)]
        hint = next((text for part, text in (document.get("condition_hints") or {}).items()
                     if any(_terms_match(word, term) for word in _normalize(part).split() for term in excluded)), None)
        if hint:
            # What ruling this part out tells a technician (bright lights mean the battery still has charge).
            return " ".join([_sentence(hint)] + ([_sentence(next_steps[0])] if next_steps else [])), ["condition"]
        if not remaining:
            text = _sentence(f"{lead or 'With that ruled out'}, I don't have a more specific cause to suggest") + \
                " It needs an inspection by a technician to find what's wrong."
            return text, ["condition"]
        parts = [causes_sentence(remaining[:4], f"{lead}, the more likely causes are" if lead else "The more likely causes are")]
        if next_steps:
            parts.append(_sentence(next_steps[0]))
        elif professional:
            parts.append(_sentence(professional))
        return " ".join(parts), ["condition"]
    if kind == "safety":
        if document.get("drive_safety"):
            return document["drive_safety"], ["drive_safety"]
        if stop:
            return stop, ["stop"]
        return _sentence(document.get("urgency") or "Have it checked by a technician before relying on the car"), ["urgency"]
    if kind == "cost":
        return cost_answer(document), ["cost"]
    if kind == "causes":
        if not causes:
            return _sentence(document["answer"]), ["answer"]
        return causes_sentence(causes[:5], "The usual causes are"), ["causes:5"]
    if kind == "fix":
        # The steps not given yet, then what needs a technician.
        given = max((int(item.split(":")[1]) for item in shown if item.startswith("actions:")), default=0)
        steps = actions[given:] or actions
        parts = [" ".join(_sentence(item) for item in steps)] if steps else []
        if professional:
            parts.append(_sentence(professional))
        if stop and "stop" not in shown:
            parts.append(stop)
        return "\n\n".join(part for part in parts if part) or _sentence(document["answer"]), ["actions", "professional"]
    # "more": what hasn't been said yet, then what would narrow it down.
    already = max((int(item.split(":")[1]) for item in shown if item.startswith("causes:")), default=0)
    parts = []
    if causes[already:]:
        parts.append(causes_sentence(causes[already:], "Other possible causes are"))
    if professional:
        parts.append(_sentence(professional))
    needed = document.get("needed_information") or []
    if needed:
        parts.append(_sentence("To narrow it down, tell me: " + needed[0].rstrip(".").lower()))
    return " ".join(parts) or _sentence(document["answer"]), ["more"]


CONDITION_SUBJECT_FILLER = {"the", "my", "a", "an", "its", "level", "levels", "one", "new", "old", "it", "is", "are", "already"}
# "battery terminals" is a different part from "the battery": ruling out the battery keeps its terminals.
SEPARATE_PART_WORDS = {"terminal", "terminals", "cable", "cables", "connection", "connections", "clamp", "clamps", "tray", "light", "lights",
                       "sensor", "sensors", "switch", "fuse", "fuses", "relay", "relays", "cap", "hose", "hoses", "pump", "fan", "belt", "wiring"}


def _names_part(text, terms):
    """True when ``text`` names the part ``terms`` itself, not a separate part named after it."""
    words = _normalize(text).split()
    for index, word in enumerate(words):
        if any(_terms_match(term, word) for term in terms):
            following = words[index + 1] if index + 1 < len(words) else ""
            if following not in SEPARATE_PART_WORDS:
                return True
    return False
