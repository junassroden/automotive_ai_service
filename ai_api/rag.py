"""Hybrid retrieval over the prebuilt knowledge index (data_pipeline/knowledge_index.py).

Ranking: curated automotive references use multilingual E5 semantic similarity
as the primary relevance signal, with BM25 and character n-gram ranks providing
lexical corroboration and candidate recall. Other sources retain reciprocal-rank
fusion. A DTC code in the question is an exact metadata match and bypasses fuzzy
ranking; without a code in the question, code definitions get half weight.
Chunks marked audience="admin" are only returned to administrators.

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
DTC_WITHOUT_CODE_WEIGHT = 0.5
SEMANTIC_CANDIDATES = 50
CURATED_SEMANTIC_RRF_WEIGHT = 10
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
        # Read into memory (about 15 MB), not memory-mapped: on Windows a mapped file cannot be replaced, so a
        # running service would block every index rebuild.
        semantic_vectors = np.load(root / "semantic_vectors.npy")
    except (OSError, ValueError) as error:
        raise KnowledgeIndexUnavailable(
            f"Knowledge index is missing or unreadable at {root}; run python -m data_pipeline.run_all --only knowledge "
            "after installing the pinned local embedding model. "
            f"({error})"
        ) from error
    if [chunk["id"] for chunk in chunks] != index["chunk_ids"]:
        raise KnowledgeIndexUnavailable("Knowledge index chunks and vectors are out of sync; rebuild the index.")
    if [chunk["id"] for chunk in chunks] != index.get("semantic_chunk_ids"):
        raise KnowledgeIndexUnavailable("Knowledge index chunks and semantic vectors are out of sync; rebuild the index.")
    if semantic_vectors.ndim != 2 or semantic_vectors.shape[0] != len(chunks):
        raise KnowledgeIndexUnavailable("Knowledge index semantic vectors are out of sync; rebuild the index.")
    index["semantic_vectors"] = semantic_vectors
    by_dtc = {chunk["metadata"]["dtc_code"]: position for position, chunk in enumerate(chunks)
              if chunk["source_type"] == "dtc_reference"}
    logger.info("Loaded knowledge index: %d chunks", len(chunks))
    return chunks, index, by_dtc


@lru_cache(maxsize=1)
def _load_embedding_model():
    model_path = Path(settings.SEMANTIC_EMBEDDING_MODEL_PATH)
    if not model_path.is_dir():
        raise KnowledgeIndexUnavailable(f"Local semantic embedding model is missing at {model_path}.")
    try:
        import onnxruntime as ort
        from sentence_transformers import SentenceTransformer
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = settings.SEMANTIC_EMBEDDING_THREADS
        session_options.inter_op_num_threads = 1
        model = SentenceTransformer(
            str(model_path), device="cpu", local_files_only=True, trust_remote_code=False,
            backend="onnx",
            model_kwargs={
                "file_name": settings.SEMANTIC_EMBEDDING_ONNX_FILE,
                "session_options": session_options,
            },
        )
    except ImportError as error:
        raise KnowledgeIndexUnavailable(
            "Install the pinned sentence-transformers and optimum-onnx dependencies for semantic retrieval."
        ) from error
    model.max_seq_length = settings.SEMANTIC_EMBEDDING_MAX_SEQUENCE_LENGTH
    return model


def embed_texts(texts, kind):
    """Encode reference passages or one transient query with the local, pinned model."""
    if kind not in {"query", "passage"}:
        raise ValueError("kind must be 'query' or 'passage'.")
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    model = _load_embedding_model()
    prefix = f"{kind}: "
    return model.encode(
        [prefix + text for text in texts],
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)


@lru_cache(maxsize=512)
def query_vector(text):
    """The embedding of one question, cached: a chat turn ranks topics, then statements, with the same text."""
    return embed_texts([text], "query")[0]


@lru_cache(maxsize=1)
def load_phrase_index():
    """Short descriptions of each curated topic and their vectors (data_pipeline/knowledge_index.py), or None
    when the index was built before they existed; topic ranking then uses the whole-entry vectors only."""
    root = Path(settings.KNOWLEDGE_INDEX_PATH)
    try:
        phrases = json.loads((root / "curated_phrases.json").read_text(encoding="utf-8"))
        vectors = np.load(root / "curated_phrase_vectors.npy")
    except (OSError, ValueError):
        logger.warning("curated phrase vectors are missing; rebuild the knowledge index")
        return None
    if vectors.ndim != 2 or vectors.shape[0] != len(phrases):
        logger.warning("curated phrase vectors are out of sync; rebuild the knowledge index")
        return None
    return phrases, vectors


def phrase_similarity(query):
    """{document id: (similarity, best phrase)} between the question and each topic's short descriptions. The similarity
    is the mean of the topic's two closest phrases, so one phrase that happens to share the question's wording
    ("brakes are overheating" for "my car is overheating") can't outrank a topic that several phrases describe."""
    loaded = load_phrase_index()
    if loaded is None:
        return {}
    phrases, vectors = loaded
    # Question and phrases are both short, user-style texts, embedded with the same "query: " prefix.
    similarities = vectors @ query_vector(query)
    found = {}
    for position in np.argsort(-similarities):
        document_id = phrases[position]["document_id"]
        top = found.setdefault(document_id, [])
        if len(top) < 2:
            top.append((float(similarities[position]), phrases[position]["text"]))
    return {document_id: (sum(score for score, _ in top) / len(top), top[0][1]) for document_id, top in found.items()}


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
            results.append(_result(chunks[position], 1.0, "exact_dtc_match", 1.0, 1.0))
    if entities["dtc_codes"] and not results and (source_types is None or "dtc_reference" in source_types):
        # A code was asked about but is not in the generic reference: say so rather than guess.
        return []

    bm25 = _bm25_scores(index, query)
    char = (index["char_matrix"] @ index["char_vectorizer"].transform([query.casefold()]).T).toarray().ravel()
    semantic = np.asarray(index["semantic_vectors"] @ query_vector(query), dtype=np.float32)
    bm25[~allowed], char[~allowed], semantic[~allowed] = -1, -1, -1
    candidates = np.union1d(
        np.union1d(np.argsort(-bm25)[:SEMANTIC_CANDIDATES], np.argsort(-char)[:SEMANTIC_CANDIDATES]),
        np.argsort(-semantic)[:SEMANTIC_CANDIDATES],
    )
    rank_bm25 = {position: rank for rank, position in enumerate(sorted(candidates, key=lambda p: -bm25[p]))}
    rank_char = {position: rank for rank, position in enumerate(sorted(candidates, key=lambda p: -char[p]))}
    rank_semantic = {position: rank for rank, position in enumerate(sorted(candidates, key=lambda p: -semantic[p]))}
    analyzer = index["counter"].build_analyzer()
    terms = set(_content_terms(query, analyzer))
    fused = []
    for position in candidates:
        if not allowed[position] or (bm25[position] <= 0 and char[position] < 0.15 and semantic[position] < 0.35):
            continue
        chunk_terms = set(analyzer(" ".join([chunks[position]["title"], chunks[position]["text"], " ".join(chunks[position].get("keywords", []))]).casefold()))
        coverage = len(terms & chunk_terms) / len(terms) if terms else 0.0
        lexical_rrf = (
            1 / (RRF_K + rank_bm25[position])
            + 1 / (RRF_K + rank_char[position])
        )
        semantic_rrf = 1 / (RRF_K + rank_semantic[position])
        if chunks[position]["source_type"] == "curated_reference":
            rrf = lexical_rrf + CURATED_SEMANTIC_RRF_WEIGHT * semantic_rrf
            score = rrf / ((2 + CURATED_SEMANTIC_RRF_WEIGHT) / RRF_K) + 0.001 * coverage
        else:
            rrf = lexical_rrf + semantic_rrf
            score = (rrf / (3 / RRF_K)) * ((0.5 + coverage) / 1.5)
        if not entities["dtc_codes"] and chunks[position]["source_type"] == "dtc_reference":
            # The ~9,500 code definitions share words with almost any car question ("battery", "light");
            # without a code in the question they rank behind the general references.
            score *= DTC_WITHOUT_CODE_WEIGHT
        fused.append((score, position, coverage, float(semantic[position])))
    fused.sort(reverse=True)
    top_score = fused[0][0] if fused else 0
    for score, position, coverage, semantic_score in fused[: max(0, top_k - len(results))]:
        if chunks[position]["id"] in {item["id"] for item in results}:
            continue
        margin = score / top_score if top_score else 0
        semantic_strength = min(1.0, max(0.0, (semantic_score - 0.35) / 0.35))
        lexical_strength = min(1.0, max(coverage, char[position] / 0.35))
        support = round(float(min(1.0, 0.6 * semantic_strength + 0.25 * coverage
                                  + 0.15 * margin * lexical_strength)), 4)
        results.append(_result(chunks[position], float(score), "hybrid_bm25_char_semantic_rrf", support, semantic_score))
    return results


def _result(chunk, score, method, support, semantic_similarity=None):
    return {
        "id": chunk["id"], "title": chunk["title"], "text": chunk["text"], "source_type": chunk["source_type"],
        "trust": chunk["trust"], "metadata": chunk.get("metadata", {}), "citation": chunk["citation"],
        "score": round(score, 6), "method": method, "support": support,
        "semantic_similarity": round(float(semantic_similarity), 4) if semantic_similarity is not None else None,
    }


def lookup_parts(part_numbers):
    table = load_parts_lookup()
    return {number: table[number] for number in part_numbers if number in table}
