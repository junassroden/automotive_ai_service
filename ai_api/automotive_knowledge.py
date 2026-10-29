import json
import re
from functools import lru_cache
from pathlib import Path

from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer


MINIMUM_RETRIEVAL_SCORE = 0.12
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


def _normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


def _term_variants(term):
    variants = {term}
    if len(term) > 5 and term.endswith("ing"):
        stem = term[:-3]
        variants.update((stem, stem + "e"))
    if len(term) > 4 and term.endswith("ed"):
        variants.add(term[:-2])
    if len(term) > 4 and term.endswith("ies"):
        variants.add(term[:-3] + "y")
    if len(term) > 4 and term.endswith("es"):
        variants.add(term[:-2])
    if len(term) > 3 and term.endswith("s"):
        variants.add(term[:-1])
    return variants


def _terms_match(left, right):
    return bool(_term_variants(left).intersection(_term_variants(right)))


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
    ranked = []

    for index, (document_index, passage) in enumerate(passages):
        document = documents[document_index]
        keywords = document.get("keywords", [])
        phrase_score = 0.0
        for keyword in keywords:
            normalized_keyword = _normalize(keyword)
            if normalized_keyword and normalized_keyword in normalized_query:
                keyword_length = len(normalized_keyword.split())
                phrase_score = max(
                    phrase_score,
                    min(1.0, keyword_length / max(1, len(query_terms)) + 0.35),
                )

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
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def retrieve_knowledge(
    query,
    minimum_score=MINIMUM_RETRIEVAL_SCORE,
    max_passages=MAX_RETRIEVED_PASSAGES,
):
    normalized_query = _normalize(query)
    if "service bulletin" in normalized_query and "vin" in normalized_query:
        return None
    ranked = rank_knowledge(query)
    if not ranked or ranked[0]["score"] < minimum_score:
        return None
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


def format_knowledge_answer(document, diagnostic_context=None, query=None):
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
    return "\n\n".join(sections)
