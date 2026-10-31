"""Hybrid retrieval over the prebuilt knowledge index (data_pipeline/knowledge_index.py).

Ranking: BM25 on word tokens and TF-IDF cosine on character n-grams are fused
with reciprocal-rank fusion, then reranked by how many of the query's content
terms each passage covers. A DTC code in the question is an exact metadata
match and bypasses fuzzy ranking. Chunks marked audience="admin" are only
returned to administrators.

The returned ``support`` value is a heuristic evidence-strength score in
[0, 1] built from term coverage and score margin. It is not a probability that
an answer is correct; thresholds for it are validated by
scripts/evaluate_chatbot.py.
"""
import json
import logging
import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from django.conf import settings


logger = logging.getLogger(__name__)
DTC_PATTERN = re.compile(r"\b([PBCU][0-9][0-9A-F]{3})\b", re.IGNORECASE)
PART_NUMBER_PATTERN = re.compile(r"\b([0-9]{5}[A-Z0-9]{3,7})\b", re.IGNORECASE)
RRF_K = 60
STOP_TERMS = {
    "what", "does", "mean", "means", "is", "the", "a", "an", "my", "car", "vehicle", "how", "do", "i", "to", "of",
    "and", "or", "in", "on", "for", "it", "can", "you", "me", "about", "tell", "should", "when", "why", "with",
    "this", "that", "are", "be", "there", "please", "code", "codes", "any", "get", "have", "has",
}


class KnowledgeIndexUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_index():
    root = Path(settings.KNOWLEDGE_INDEX_PATH)
    try:
        chunks = [json.loads(line) for line in (root / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
        index = joblib.load(root / "index.joblib")
    except (OSError, ValueError) as error:
        raise KnowledgeIndexUnavailable(
            f"Knowledge index is missing or unreadable at {root}; run python -m data_pipeline.run_all --only knowledge. ({error})"
        ) from error
    if [chunk["id"] for chunk in chunks] != index["chunk_ids"]:
        raise KnowledgeIndexUnavailable("Knowledge index chunks and vectors are out of sync; rebuild the index.")
    by_dtc = {chunk["metadata"]["dtc_code"]: position for position, chunk in enumerate(chunks)
              if chunk["source_type"] == "dtc_reference"}
    logger.info("Loaded knowledge index: %d chunks", len(chunks))
    return chunks, index, by_dtc


@lru_cache(maxsize=1)
def load_parts_lookup():
    path = Path(settings.KNOWLEDGE_INDEX_PATH) / "parts_lookup.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def extract_entities(message):
    text = message.upper()
    return {
        "dtc_codes": sorted(set(match.upper() for match in DTC_PATTERN.findall(text))),
        "part_numbers": sorted(set(match.upper() for match in PART_NUMBER_PATTERN.findall(text))),
        "vehicle_ids": sorted({int(value) for value in re.findall(r"\b(?:vehicle|car)\s*(?:id)?\s*#?\s*(\d{1,9})\b", message, re.IGNORECASE)}),
        "dates": re.findall(r"\b\d{4}-\d{2}-\d{2}\b", message),
        "odometer_km": [int(value.replace(",", "")) for value in re.findall(r"\b(\d{1,3}(?:,\d{3})+|\d{4,7})\s*(?:km|kilomet)", message, re.IGNORECASE)],
    }


def _content_terms(query, analyzer):
    return [term for term in analyzer(query.casefold()) if term not in STOP_TERMS and len(term) > 2]


def _bm25_scores(index, query):
    vocabulary = index["counter"].vocabulary_
    terms = [vocabulary[term] for term in index["counter"].build_analyzer()(query.casefold()) if term in vocabulary]
    scores = np.zeros(index["bm25_tf"].shape[0], dtype=np.float32)
    if not terms:
        return scores
    columns = index["bm25_tf"][:, sorted(set(terms))].toarray()
    idf = index["bm25_idf"][sorted(set(terms))]
    length = (index["bm25_doc_length"] / index["bm25_avg_length"])[:, None]
    scores += (idf * columns * (1.4 + 1) / (columns + 1.4 * (1 - 0.75 + 0.75 * length))).sum(axis=1)
    return scores


def retrieve(query, role="customer", top_k=5, source_types=None):
    """Return ranked passages with citations; empty list when nothing is relevant."""
    chunks, index, by_dtc = load_index()
    entities = extract_entities(query)
    allowed = np.array([
        (chunk["audience"] == "all" or role == "admin")
        and (source_types is None or chunk["source_type"] in source_types)
        for chunk in chunks
    ])
    results = []
    for code in entities["dtc_codes"]:
        position = by_dtc.get(code)
        if position is not None and allowed[position]:
            results.append(_result(chunks[position], 1.0, "exact_dtc_match", 1.0))
    if entities["dtc_codes"] and not results and (source_types is None or "dtc_reference" in source_types):
        # A code was asked about but is not in the generic reference: say so rather than guess.
        return []

    bm25 = _bm25_scores(index, query)
    char = (index["char_matrix"] @ index["char_vectorizer"].transform([query.casefold()]).T).toarray().ravel()
    bm25[~allowed], char[~allowed] = -1, -1
    candidates = np.union1d(np.argsort(-bm25)[:50], np.argsort(-char)[:50])
    rank_bm25 = {position: rank for rank, position in enumerate(sorted(candidates, key=lambda p: -bm25[p]))}
    rank_char = {position: rank for rank, position in enumerate(sorted(candidates, key=lambda p: -char[p]))}
    analyzer = index["counter"].build_analyzer()
    terms = set(_content_terms(query, analyzer))
    fused = []
    for position in candidates:
        if not allowed[position] or (bm25[position] <= 0 and char[position] < 0.15):
            continue
        chunk_terms = set(analyzer(" ".join([chunks[position]["title"], chunks[position]["text"], " ".join(chunks[position].get("keywords", []))]).casefold()))
        coverage = len(terms & chunk_terms) / len(terms) if terms else 0.0
        rrf = 1 / (RRF_K + rank_bm25[position]) + 1 / (RRF_K + rank_char[position])
        fused.append((rrf * (0.5 + coverage), position, coverage))
    fused.sort(reverse=True)
    top_score = fused[0][0] if fused else 0
    for score, position, coverage in fused[: max(0, top_k - len(results))]:
        if chunks[position]["id"] in {item["id"] for item in results}:
            continue
        margin = score / top_score if top_score else 0
        support = round(float(min(1.0, 0.7 * coverage + 0.3 * margin * min(1.0, char[position] / 0.35))), 4)
        results.append(_result(chunks[position], float(score), "hybrid_bm25_char_rrf", support))
    return results


def _result(chunk, score, method, support):
    return {
        "id": chunk["id"], "title": chunk["title"], "text": chunk["text"], "source_type": chunk["source_type"],
        "trust": chunk["trust"], "metadata": chunk.get("metadata", {}), "citation": chunk["citation"],
        "score": round(score, 6), "method": method, "support": support,
    }


def lookup_parts(part_numbers):
    table = load_parts_lookup()
    return {number: table[number] for number in part_numbers if number in table}
