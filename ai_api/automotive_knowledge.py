import hashlib
import json
import logging
import math
import re
from functools import lru_cache
from pathlib import Path

from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer


logger = logging.getLogger(__name__)
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
NAMED_TOPIC_RATIO = 0.7
NAMED_TOPIC_COVERAGE = 0.75
NAMED_TOPIC_GAIN = 0.25
SEMANTIC_ONLY_MIN_SIMILARITY = 0.84
# Phrase-to-question similarity runs about this much higher than whole-entry similarity for the same right topic
# (measured on development questions on 2026-11-17).
PHRASE_SIMILARITY_SHIFT = 0.05
# How much closer in meaning another topic must be before it replaces a topic that only matched by shared words.
SEMANTIC_OVERRIDE_MARGIN = 0.03
# A topic answers on meaning alone only when one of its own short descriptions is at least this close too (real
# paraphrases scored 0.90 to 0.95 on development questions; fleet questions that only resemble a topic, 0.83 to 0.87).
SEMANTIC_ONLY_MIN_PHRASE_SIMILARITY = 0.88
SEMANTIC_SINGLE_TERM_MIN_SIMILARITY = 0.84
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
    "include", "included", "includes", "including",
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


def _phrase_terms_match(left, right):
    # Phrase matching permits plurals without conflating distinct forms such as "brakes" and "braking".
    return left == right or (left.endswith("s") and left[:-1] == right) or (right.endswith("s") and right[:-1] == left)


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
    from .rag import KnowledgeIndexUnavailable, phrase_similarity, retrieve
    try:
        semantic_results = retrieve(query, top_k=50, source_types={"curated_reference"})
        phrases = phrase_similarity(query)
    except KnowledgeIndexUnavailable as error:
        logger.warning("semantic retrieval unavailable; using curated lexical matching: %s", error)
        semantic_results, phrases = [], {}
    semantic_by_id = {
        result["id"][3:]: result["semantic_similarity"]
        for result in semantic_results
        if result["semantic_similarity"] is not None
    }
    # Whole-entry similarity alone, for the checks that decide whether a topic may answer at all with no words in
    # common ("which car has the lowest mileage" must not reach an engine topic by meaning alone).
    entry_by_id = dict(semantic_by_id)
    # What the question means is compared with each topic's own short descriptions as well as with the whole entry:
    # the closer of the two counts for ranking. Short texts compare higher, so phrase similarity is shifted onto the
    # whole-entry scale.
    for document_id, (similarity, _) in phrases.items():
        shifted = round(similarity - PHRASE_SIMILARITY_SHIFT, 4)
        if shifted > semantic_by_id.get(document_id, 0.0):
            semantic_by_id[document_id] = shifted
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
            elif keyword_length >= 2 and all(
                any(_phrase_terms_match(word, term) for term in query_words) for word in keyword_words
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
        semantic_score = semantic_by_id.get(document["id"], 0.0)
        if semantic_by_id:
            semantic_weight = 0.5 if phrase_length else 0.8
            score = (1 - semantic_weight) * max(vector_score, lexical_score) + semantic_weight * semantic_score
        else:
            score = max(vector_score, lexical_score)
        ranked.append({
            "document": document,
            "field": passage["field"],
            "text": passage["text"],
            "score": round(score, 4),
            "semantic_similarity": round(semantic_score, 4),
            "entry_similarity": round(entry_by_id.get(document["id"], 0.0), 4),
            "phrase_similarity": round(phrases.get(document["id"], (0.0,))[0], 4),
            "phrase_length": phrase_length,
            "topic_coverage": topic_coverage(document_index),
        })

    # On equal scores the document whose keyword phrase matched more of the question wins
    # ("white smoke from exhaust" over "smoke from exhaust"), then the one whose keywords and title
    # name more of the question's words ("leaking oil": the oil-leak topic over overheating).
    ranked.sort(
        key=lambda item: (
            item["score"], item["phrase_length"], item["semantic_similarity"], item["topic_coverage"]
        ),
        reverse=True,
    )
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
    preferred = max(
        near,
        key=lambda item: (
            item["phrase_length"], item["semantic_similarity"], item["topic_coverage"], item["score"]
        ),
    )
    if preferred["phrase_length"] > ranked[0]["phrase_length"] or (
            preferred["phrase_length"] == ranked[0]["phrase_length"]
            and preferred["semantic_similarity"] >= ranked[0]["semantic_similarity"]
            and preferred["topic_coverage"] >= ranked[0]["topic_coverage"] + CLEARLY_BETTER_COVERAGE):
        ranked = [preferred] + [item for item in ranked if item is not preferred]
    elif ranked[0]["phrase_length"] == 0:
        # With no keyword phrase on top, a document whose keywords and title name clearly more of the question
        # also wins from a little further back ("I smell gas in my car": the fuel-leak entry over a rotten-egg
        # smell entry that shares only "smell").
        wider = [item for item in ranked if item["score"] >= ranked[0]["score"] * COVERAGE_TIE_RATIO]
        covering = max(
            wider, key=lambda item: (item["semantic_similarity"], item["topic_coverage"], item["score"])
        )
        if (covering["semantic_similarity"] >= ranked[0]["semantic_similarity"]
                and covering["topic_coverage"] >= ranked[0]["topic_coverage"] + CLEARLY_BETTER_COVERAGE):
            ranked = [covering] + [item for item in ranked if item is not covering]
        else:
            # A topic whose own name covers most of the question beats one that only shares words in passing
            # ("is it safe to drive with an overheating engine": the overheating entry, not engine braking,
            # whose overview happens to mention safe driving and overheating brakes).
            named = [item for item in ranked if item["score"] >= ranked[0]["score"] * NAMED_TOPIC_RATIO
                     and item["topic_coverage"] >= NAMED_TOPIC_COVERAGE]
            best = max(
                named, key=lambda item: (item["semantic_similarity"], item["topic_coverage"], item["score"]),
                default=None,
            )
            if (best is not None
                    and best["semantic_similarity"] >= ranked[0]["semantic_similarity"]
                    and best["topic_coverage"] >= ranked[0]["topic_coverage"] + NAMED_TOPIC_GAIN):
                ranked = [best] + [item for item in ranked if item is not best]
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

    def meaning_supports(item, minimum):
        # Answering on meaning with few or no words in common needs two signals: the entry as a whole and one of its
        # own short descriptions are both close. A long, general entry can be near many questions on its own
        # ("which car has the lowest mileage" vs. an engine-power entry about passengers and load).
        return item["entry_similarity"] >= minimum and item["phrase_similarity"] >= SEMANTIC_ONLY_MIN_PHRASE_SIMILARITY

    def may_answer(item):
        """Whether this topic has enough in common with the question to answer it at all."""
        matched = matching_terms(item["document"])
        if not matched and not meaning_supports(item, SEMANTIC_ONLY_MIN_SIMILARITY):
            return False
        if item["phrase_length"] == 0:
            # Without a keyword phrase, one stray word in common ("who won the NBA finals" / "won't start")
            # is not a topic match, and at least one matched word must be specific to cars: generic words
            # ("not working", "won't go", "smells like", "no symptoms", "open safety recall") don't identify a topic.
            specific_matches = matched - SYMPTOM_FILLER - COVERAGE_FILLER - GENERIC_ENGLISH
            if not specific_matches and not meaning_supports(item, SEMANTIC_ONLY_MIN_SIMILARITY):
                return False
            if len(specific_matches) == 1 and not meaning_supports(item, SEMANTIC_SINGLE_TERM_MIN_SIMILARITY):
                return False
        return True

    # Verification: the words a keyword shares with the question can agree by accident ("steering became HARD TO turn
    # after STARTing" holds every word of "hard to start"). When another topic is clearly closer in meaning to the
    # whole question and the question also names something specific to it, the meaning wins.
    # Without a keyword phrase on top, nothing but shared single words put it there, so the closest meaning wins outright;
    # a real keyword phrase is only overruled by a clear margin.
    closest = max(ranked, key=lambda item: (item["semantic_similarity"], item["score"]))
    margin = SEMANTIC_OVERRIDE_MARGIN if ranked[0]["phrase_length"] else 0.0
    if (closest["document"]["id"] != ranked[0]["document"]["id"]
            and closest["semantic_similarity"] > ranked[0]["semantic_similarity"] + margin
            # ... and it must name at least as much of what was asked: "my car is overheating" says nothing that makes a
            # transmission-temperature topic a better fit than the engine-overheating one it ties with.
            and closest["topic_coverage"] >= ranked[0]["topic_coverage"]
            and (closest["phrase_length"] or matching_terms(closest["document"]) - SYMPTOM_FILLER - COVERAGE_FILLER - GENERIC_ENGLISH)
            # ... and it must be able to answer: replacing a keyword match ("red thermometer light") with a topic that is
            # closer in meaning but shares only "red" with the question would end in no answer at all.
            and may_answer(closest)):
        best_passage = max((item for item in ranked if item["document"]["id"] == closest["document"]["id"]), key=lambda item: item["score"])
        ranked = [best_passage] + [item for item in ranked if item is not best_passage]

    primary = ranked[0]["document"]
    if not may_answer(ranked[0]):
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


def _sentence(text):
    text = text.strip()
    text = text[:1].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


# What kind of question it is. This only decides which kind of knowledge answers it, never the wording.
# "Can I still drive it?", "is it safe to drive?", "pwede pa ba i-drive?"
ASKS_DRIVE = re.compile(
    r"\b(?:can i (?:still |keep )?drive|could i (?:still )?drive|can i (?:still |keep )?use (?:it|the car|my car|the vehicle)\b(?! (?:instead|for))|safe to drive|ok(?:ay)? to drive|is it safe|still drive|keep driving"
    r"|drive (?:it )?with|drive on it|pwede (?:pa )?(?:ba )?(?:i-?|ma)?(?:drive|gamitin|byahe)|delikado|dangerous|serious|urgent)\b", re.IGNORECASE)
# "How much does it cost to fix?", "magkano ipagawa?"
ASKS_PRICE = re.compile(r"\b(?:how much (?:does|do|will|would|is|to|for)|cost|costs|price|magkano|expensive|mahal ba|cheap)\b", re.IGNORECASE)
ASKS_ADVICE = re.compile(
    # The message reaches this normalised, sometimes without the "I" ("when should change the timing belt").
    r"^\s*(?:what|which|how much|how often|how many|when)\b.*\b(?:should|do (?:i|we) need|is (?:the )?(?:correct|right|recommended|best|ideal))\b"
    r"|^\s*how often\b|^\s*what(?:'s| is) the (?:correct|right|recommended|ideal|normal)\b", re.IGNORECASE)
ASKS_INTERVAL = re.compile(r"\b(?:how often|how long|how many (?:km|kilometers|kilometres|miles)|interval|lifespan|life span|every how"
                           r"|when (?:should|do|to) (?:i )?(?:change|replace))\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Chat answers built from knowledge. Every statement a topic entry holds (what it is, what each part does,
# causes, clues, checks, safety, cost, intervals) is scored against the question, and the answer is the
# statements that answer it best. No answer text, layout or follow-up question is written in advance.
# ---------------------------------------------------------------------------

# Where statements come from, and what kind of knowledge each field holds.
STATEMENT_FIELDS = (
    ("answer", "explain"), ("overview", "explain"), ("component_functions", "function"), ("possible_causes", "cause"),
    ("symptoms", "clue"), ("triage", "clue"), ("condition_hints", "clue"), ("actions", "check"), ("checks", "check"),
    ("professional", "workshop"), ("diagnostic_steps", "workshop"), ("drive_safety", "drive"), ("stop", "safety"),
    ("urgency", "safety"), ("cost_note", "cost"), ("interval_items", "interval"),
)
# How well each kind of knowledge suits each kind of question.
KIND_WEIGHTS = {
    "drive": {"drive": 3.0, "safety": 1.8, "explain": 0.9, "clue": 0.8},
    "cost": {"cost": 3.0, "workshop": 1.3, "cause": 1.0},
    "function": {"function": 2.6, "explain": 1.6},
    "interval": {"interval": 3.0, "explain": 0.6, "check": 0.8},
    "advice": {"explain": 1.6, "interval": 1.6, "check": 1.4, "function": 1.2},
    "action": {"check": 2.0, "workshop": 1.3, "safety": 1.3, "explain": 2.0, "clue": 1.0},
    "cause": {"cause": 2.0, "clue": 1.6, "explain": 1.2},
    "problem": {"explain": 1.5, "cause": 1.5, "clue": 1.4, "check": 1.2, "safety": 1.1},
}
OTHER_KIND_WEIGHT = 0.5
# Statements that hold only under a condition ("in traffic", "with the coolant full") answer a question only
# when it says something about that condition; a word the whole topic shares ("overheating") isn't enough.
CONDITIONAL_FIELDS = {"triage", "condition_hints", "symptoms"}
CONDITIONAL_WITHOUT_DETAIL = 0.2
# A word in at most this share of the topic's statements is specific enough to pick out a condition.
SPECIFIC_TERM_SHARE = 0.25
# Fields written for technicians or as background count for less than plain advice to a driver.
FIELD_WEIGHTS = {
    "answer": 1.3, "checks": 0.75, "diagnostic_steps": 0.5, "overview": 0.85,
    "possible_causes": 0.9, "triage": 1.15,
}
# Each further statement of a kind already in the answer counts this much less, so an answer covers what is
# wrong, why and what to do rather than three ways of saying one of them.
REPEATED_KIND_FACTOR = 0.55
# Statements of one kind in one answer; one safety instruction is enough.
KIND_LIMITS = {"safety": 1, "drive": 1, "cost": 1}
# The order statements are read in, so the answer goes from what is happening to what to do about it.
READING_ORDER = ("drive", "explain", "function", "interval", "clue", "cause", "cost", "check", "workshop", "safety")
MAX_STATEMENTS = 4
# A follow-up adds to what was said, so it is shorter and only keeps what clearly answers it.
FOLLOW_UP_STATEMENTS = 2
FOLLOW_UP_SCORE_FLOOR = 0.45
MAX_ANSWER_WORDS = 120
# A statement much weaker than the best one doesn't answer the question.
RELATIVE_SCORE_FLOOR = 0.3
STATEMENT_STOP = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to", "in", "on", "at", "by", "for", "with", "from", "as", "is", "are",
    "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those", "there", "here", "i", "me", "my", "we", "our", "you",
    "your", "he", "she", "they", "them", "their", "do", "does", "did", "done", "can", "could", "should", "would", "will", "may", "might",
    "must", "have", "has", "had", "what", "which", "who", "when", "where", "why", "how", "about", "into", "up", "out", "just", "also", "very",
    "really", "still", "any", "some", "all", "more", "most", "other", "such", "only", "than", "too", "s", "t", "ll", "re", "ve", "d", "m",
    "please", "po", "ba", "ang", "ng", "na", "sa", "mga", "lang", "naman", "pa", "not", "no", "yes", "car", "vehicle", "tell", "know",
    "much", "many", "often", "long", "much", "get", "got", "like", "one", "anything", "else", "something", "thing", "things", "way",
    "ok", "okay", "hmm", "now", "even", "suppose", "happen", "happens", "mean", "means", "meaning", "explain", "details", "info", "continue",
    "anymore", "though", "already", "go", "kasi", "din", "rin", "yung", "ito", "iyan", "lng", "pls",
}
NEGATION = re.compile(r"\b(?:not|no|never|cannot|without|hindi|ayaw|wala)\b|n'?t\b", re.IGNORECASE)
QUESTION_KIND_DESCRIPTIONS = {
    "drive": "The user asks whether it is safe to keep driving or operate the vehicle in this condition.",
    "cost": "The user asks for an estimated price or cost to repair or maintain the vehicle.",
    "function": "The user asks what a component does, how it works, or its purpose.",
    "interval": "The user asks how often, when, or at what distance maintenance or replacement is due.",
    "advice": "The user asks what is recommended, normal, correct, or best for the vehicle.",
    "action": "The user asks what to do, how to fix, repair, inspect, or resolve the described vehicle problem.",
    "cause": "The user asks why the problem happened or what may have caused the symptom.",
    "problem": "The user describes or asks about a vehicle symptom without a more specific request.",
}


def has_chat_fields(document):
    return isinstance(document.get("answer"), str) and document["answer"].strip() != ""


def _words(text):
    return set(_normalize(text).split())


def mentions(text, terms):
    """True when any of ``terms`` (or a simple plural/verb variant) appears in ``text``."""
    words = _words(text)
    return any(_terms_match(term, word) for term in terms for word in words)


def _content_terms(text):
    return [word for word in _normalize(text).split() if len(word) > 1 and word not in STATEMENT_STOP]


def _split_sentences(text):
    return [part for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip()) if part]


def knowledge_statements(document):
    """Every statement the topic entry holds, each tagged with the kind of knowledge it is."""
    statements, seen = [], set()
    for field, kind in STATEMENT_FIELDS:
        value = document.get(field)
        if field == "triage":
            items = [option.get("narrow") for question in value or [] for option in question.get("answers", [])]
        elif isinstance(value, dict):
            items = list(value.values())
        elif isinstance(value, list):
            items = value
        else:
            items = [value]
        for item in items:
            if not isinstance(item, str) or not item.strip():
                continue
            # Long passages are split so a question gets the sentence that answers it, not the paragraph around it.
            for text in (_split_sentences(item) if field in {"overview", "urgency"} else [item]):
                text = _sentence(text)
                if text.casefold() in seen:
                    continue
                seen.add(text.casefold())
                statements.append({"key": "s:" + hashlib.md5(text.encode("utf-8")).hexdigest()[:10], "kind": kind,
                                   "field": field, "text": text, "terms": frozenset(_content_terms(text))})
    causes = [cause.strip().rstrip(".") for cause in document.get("key_causes") or [] if isinstance(cause, str) and cause.strip()][:4]
    if causes:
        # The entry's short list of likely causes, read as one statement ("Common causes include worn spark plugs, ...").
        listed = causes[0] if len(causes) == 1 else ", ".join(causes[:-1]) + " or " + causes[-1]
        text = _sentence(f"Common causes include {listed}")
        statements.append({"key": "s:" + hashlib.md5(text.encode("utf-8")).hexdigest()[:10], "kind": "cause",
                           "field": "key_causes", "text": text, "terms": frozenset(_content_terms(text))})
    return statements


def question_kind(query):
    """What the question asks for: whether to drive, cost, what a part does, an interval, advice, what to do, why, or a problem."""
    if ASKS_DRIVE.search(query):
        return "drive"
    if not isinstance(query, str) or not query.strip():
        return "problem"
    if ASKS_PRICE.search(query) and not ASKS_INTERVAL.search(query):
        return "cost"
    if FUNCTION_QUESTION.search(query):
        return "function"
    if ASKS_INTERVAL.search(query):
        return "interval"
    if ASKS_ACTION.search(query):
        return "action"
    if ASKS_CAUSE.search(query):
        return "cause"
    if ASKS_ADVICE.search(query) or ASKS_DEFINITION.search(query):
        return "advice"
    from .rag import KnowledgeIndexUnavailable, embed_texts
    try:
        labels, kind_vectors = _question_kind_vectors()
        query_vector = embed_texts([query], "query")[0]
        similarities = kind_vectors @ query_vector
        best = int(similarities.argmax())
        second = int(similarities.argsort()[-2])
        if similarities[best] >= 0.48 and similarities[best] - similarities[second] >= 0.08:
            return labels[best]
    except KnowledgeIndexUnavailable as error:
        logger.warning("semantic question analysis unavailable; using lexical question-kind checks: %s", error)
    return "problem"


@lru_cache(maxsize=1)
def _question_kind_vectors():
    from .rag import embed_texts
    labels = tuple(QUESTION_KIND_DESCRIPTIONS)
    return labels, embed_texts([QUESTION_KIND_DESCRIPTIONS[label] for label in labels], "passage")


@lru_cache(maxsize=256)
def _statement_vectors(document_id):
    document = knowledge_document(document_id)
    if document is None:
        return (), None
    statements = knowledge_statements(document)
    from .rag import embed_texts
    return tuple(statement["key"] for statement in statements), embed_texts(
        [statement["text"] for statement in statements], "passage",
    )


def _matches(term, statement):
    return any(_terms_match(term, word) for word in statement["terms"])


def rank_statements(statements, query, kind=None, context="", exclude=(), topic_words=(), semantic_scores=None):
    """(score, statement) pairs that share words with the question, best first.

    A word counts more the fewer statements share it ("traffic" says more than "engine" in the overheating
    entry), words from the resolved topic context count less than the new message's, and a question that says
    something is not happening prefers statements about things not happening ("the fan doesn't spin")."""
    kind = kind or question_kind(query)
    terms = set(_content_terms(query))
    context_terms = set(_content_terms(context)) - terms
    count = len(statements) or 1
    weight, specific = {}, set()
    for term in terms | context_terms:
        shared = sum(1 for statement in statements if _matches(term, statement))
        weight[term] = math.log(1 + count / (1 + shared))
        # The topic's own words ("engine", "overheating") say which problem it is, not which condition.
        if shared <= SPECIFIC_TERM_SHARE * count and term not in GENERIC_QUERY_TERMS and term not in QUESTION_ONLY_TERMS \
                and not any(_terms_match(term, word) for word in topic_words):
            specific.add(term)
    negated = bool(NEGATION.search(query))
    suits = KIND_WEIGHTS[kind]
    ranked = []
    for statement in statements:
        if statement["key"] in exclude or (statement["kind"] == "drive" and kind != "drive"):
            # A verdict on driving ("No. Driving on a flat tire...") only answers "can I drive it?".
            continue
        score = sum(weight[term] for term in terms if _matches(term, statement))
        score += 0.35 * sum(weight[term] for term in context_terms if _matches(term, statement))
        semantic_score = (semantic_scores or {}).get(statement["key"], 0.0)
        semantic_strength = max(0.0, (semantic_score - 0.25) / 0.45)
        score += semantic_strength * suits.get(statement["kind"], OTHER_KIND_WEIGHT)
        if score <= 0:
            continue
        if negated and NEGATION.search(statement["text"]):
            score *= 1.25
        if statement["field"] in CONDITIONAL_FIELDS and not any(_matches(term, statement) for term in specific):
            score *= CONDITIONAL_WITHOUT_DETAIL
        # Long statements match more words by length alone.
        score *= suits.get(statement["kind"], OTHER_KIND_WEIGHT) * FIELD_WEIGHTS.get(statement["field"], 1.0)             / max(1, len(statement["terms"])) ** 0.25
        ranked.append((round(score, 4), statement))
    ranked.sort(key=lambda pair: -pair[0])
    return ranked


def _overlap(left, right):
    if not left["terms"] or not right["terms"]:
        return 0.0
    return len(left["terms"] & right["terms"]) / min(len(left["terms"]), len(right["terms"]))


def _select(ranked, limit=MAX_STATEMENTS, floor=RELATIVE_SCORE_FLOOR, overlap_limit=0.6):
    """The best statements that say different things, within a readable length: each pick is the strongest
    remaining statement after discounting kinds the answer already has."""
    chosen, words, per_kind = [], 0, {}
    top = ranked[0][0] if ranked else 0
    remaining = [(score, statement) for score, statement in ranked if score >= top * floor]
    while remaining and len(chosen) < limit:
        candidates = [(score * REPEATED_KIND_FACTOR ** per_kind.get(statement["kind"], 0), statement) for score, statement in remaining
                      if per_kind.get(statement["kind"], 0) < KIND_LIMITS.get(statement["kind"], 2)
                      and not any(_overlap(statement, other) > overlap_limit for other in chosen)
                      and (not chosen or words + len(statement["text"].split()) <= MAX_ANSWER_WORDS)]
        if not candidates:
            break
        _, statement = max(candidates, key=lambda pair: pair[0])
        chosen.append(statement)
        words += len(statement["text"].split())
        per_kind[statement["kind"]] = per_kind.get(statement["kind"], 0) + 1
        remaining = [pair for pair in remaining if pair[1] is not statement]
    return chosen


def _read(chosen, lead=None, safety_first=False):
    """The statements in reading order: the kind the question asks for first ("No." for "can I drive it?", the
    cost note for "how much?"), the safety instruction first when it is happening right now."""
    order = tuple(dict.fromkeys(((lead,) if lead else ()) + (("safety",) if safety_first else ()) + READING_ORDER))
    chosen = sorted(chosen, key=lambda statement: order.index(statement["kind"]))
    paragraphs = []
    for statement in chosen:
        if paragraphs and paragraphs[-1][0] == statement["kind"]:
            paragraphs[-1][1].append(statement["text"])
        else:
            paragraphs.append((statement["kind"], [statement["text"]]))
    return "\n\n".join(" ".join(texts) for _, texts in paragraphs)


def compose_chat_answer(document, query="", shown=(), context="", prior_kind=None):
    """(answer, keys of the statements used) for a question about this topic, built from its knowledge.

    ``shown`` holds statements already given in this conversation, which are not repeated; ``context`` contains
    the resolved topic's knowledge terms, not stored user text."""
    query = query or ""
    statements = knowledge_statements(document)
    kind = question_kind(query)
    if kind == "problem" and prior_kind in KIND_WEIGHTS:
        kind = prior_kind
    exclude = set(shown or ())
    topic_words = frozenset(_content_terms(document.get("title", "")))
    from .rag import KnowledgeIndexUnavailable, embed_texts
    try:
        contextual_query = " ".join(part for part in (query, context) if part)
        query_vector = embed_texts([contextual_query], "query")[0]
        statement_keys, statement_vectors = _statement_vectors(document["id"])
        semantic_scores = dict(zip(statement_keys, statement_vectors @ query_vector))
    except KnowledgeIndexUnavailable as error:
        logger.warning("semantic statement ranking unavailable; using lexical statement ranking: %s", error)
        semantic_scores = {}
    ranked = rank_statements(statements, query, kind, context, exclude, topic_words, semantic_scores)
    if not ranked and context:
        # Nothing in the message itself to go on ("anything else?", "yes"): carry on with the earlier question.
        ranked = rank_statements(
            statements, context, question_kind(context) if kind == "problem" else kind, "",
            exclude, topic_words, semantic_scores,
        )
    if not ranked and not exclude:
        ranked = [(1.0, statement) for statement in statements if statement["kind"] == "explain"][:1]
    chosen = _select(
        ranked, FOLLOW_UP_STATEMENTS, FOLLOW_UP_SCORE_FLOOR, overlap_limit=0.85,
    ) if exclude else _select(ranked)
    if not chosen:
        return "", []
    causes = [statement for statement in statements if statement["field"] == "key_causes" and statement["key"] not in exclude]
    if causes and kind in {"problem", "cause"} and not any(statement["kind"] == "cause" for statement in chosen):
        # Describing a problem or asking why gets its likely causes, not only what the symptom is: the cause list
        # takes the place of the weakest statement when the answer is full.
        if len(chosen) >= (FOLLOW_UP_STATEMENTS if exclude else MAX_STATEMENTS):
            chosen = chosen[:-1]
        chosen = chosen + causes[:1]
    stop = [statement for statement in statements if statement["field"] == "stop" and statement["key"] not in exclude]
    if stop and kind in {"problem", "action", "cause"} and not any(statement["field"] == "stop" for statement in chosen):
        # A problem that can be dangerous always carries its stop instruction, once per conversation, in place of
        # any milder safety note.
        chosen = [statement for statement in chosen if statement["kind"] != "safety"] + stop[:1]
    # The added cause list and stop instruction count toward the length too: the weakest other statements go first.
    kept = {"key_causes", "stop"}
    while len(chosen) > 2 and sum(len(statement["text"].split()) for statement in chosen) > MAX_ANSWER_WORDS:
        droppable = [statement for statement in chosen[1:] if statement["field"] not in kept]
        if not droppable:
            break
        chosen = [statement for statement in chosen if statement is not droppable[-1]]
    # A direct question gets its direct answer first: the verdict on driving, the cost, what the part does, the interval.
    lead = max(KIND_WEIGHTS[kind], key=KIND_WEIGHTS[kind].get) if kind in {"drive", "cost", "function", "interval"} else None
    return _read(chosen, lead, safety_first=bool(EMERGENCY.search(query))), [statement["key"] for statement in chosen]


def compose_follow_up(document, message, shown=(), context="", prior_kind=None):
    """(answer, keys used) for a follow-up about the topic being discussed: what the knowledge says about the
    new message that hasn't been said yet, or None when the knowledge has nothing more on it."""
    answer, keys = compose_chat_answer(document, message, shown, context, prior_kind)
    if answer:
        return answer, keys
    return None
