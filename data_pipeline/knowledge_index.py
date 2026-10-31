"""Builds the retrieval index used by the chatbot (ai_api/rag.py).

Only sources whose content can be stated as reference fact are indexed:
  - OBDex generic DTC and PID definitions (CC0)
  - the curated automotive knowledge base (data/automotive_knowledge.json)
  - diagnostic flowcharts (automotive_faults_aktc_obike_et_al.json)
  - the DriveRent system guide written from the application code
  - service-frequency statistics computed from the workshop service records
Synthetic or generated text (CVFFAD, VehicleDiagnosticSample, DTC chat
answers, repair-order sample) is deliberately not indexed. Motorcycle part
numbers from masterparts.xlsx go to an exact-match lookup table, not to
free-text retrieval.

Outputs (data/processed/knowledge_index/): chunks.jsonl, index.joblib
(fitted vectorizers + BM25 statistics), parts_lookup.json, manifest.json.
"""
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


SERVICE_ROOT = Path(__file__).resolve().parents[1]
INDEX_ROOT = SERVICE_ROOT / "data" / "processed" / "knowledge_index"
TOKEN_PATTERN = r"(?u)\b[a-z0-9][a-z0-9\-]+\b"
BM25_K1, BM25_B = 1.4, 0.75


# --- minimal YAML subset reader for OBDex (block maps, block lists, flow lists, scalars) ---

def _scalar(text):
    text = text.strip()
    if text in {"", "~", "null"}:
        return None
    if text in {"true", "false"}:
        return text == "true"
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [_scalar(item) for item in _split_flow(inner)] if inner else []
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        body = text[1:-1]
        return body.replace("\\\"", "\"") if text[0] == "\"" else body.replace("''", "'")
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d*", text):
        return float(text)
    return text


def _split_flow(text):
    items, depth, quote, current = [], 0, None, ""
    for character in text:
        if quote:
            current += character
            if character == quote:
                quote = None
        elif character in "\"'":
            quote = character
            current += character
        elif character == "[":
            depth += 1
            current += character
        elif character == "]":
            depth -= 1
            current += character
        elif character == "," and depth == 0:
            items.append(current)
            current = ""
        else:
            current += character
    if current.strip():
        items.append(current)
    return items


def parse_yaml_subset(text):
    lines = []
    for raw in text.splitlines():
        stripped = raw.split(" #", 1)[0] if " #" in raw and "\"" not in raw and "'" not in raw else raw
        if not stripped.strip() or stripped.lstrip().startswith("#"):
            continue
        lines.append((len(stripped) - len(stripped.lstrip(" ")), stripped.strip()))

    def parse_block(index, indent):
        if index >= len(lines):
            return None, index
        is_list = lines[index][1].startswith("- ") or lines[index][1] == "-"
        result = [] if is_list else {}
        while index < len(lines):
            line_indent, content = lines[index]
            if line_indent < indent:
                break
            if is_list:
                if line_indent != indent or not (content.startswith("- ") or content == "-"):
                    break
                item = content[2:].strip() if content != "-" else ""
                if not item:
                    value, index = parse_block(index + 1, indent + 2)
                    result.append(value)
                    continue
                if re.match(r"^[A-Za-z0-9_\-]+:( |$)", item):
                    # Inline first key of a mapping item: re-read it as a nested map at indent + 2.
                    lines[index] = (indent + 2, item)
                    value, index = parse_block(index, indent + 2)
                    result.append(value)
                    continue
                result.append(_scalar(item))
                index += 1
            else:
                if line_indent != indent:
                    break
                key, _, rest = content.partition(":")
                key = key.strip().strip("\"'")
                if rest.strip():
                    result[key] = _scalar(rest)
                    index += 1
                else:
                    child_indent = lines[index + 1][0] if index + 1 < len(lines) else indent
                    if index + 1 < len(lines) and (child_indent > indent or (child_indent == indent and lines[index + 1][1].startswith("- "))):
                        value, index = parse_block(index + 1, child_indent)
                        result[key] = value
                    else:
                        result[key] = None
                        index += 1
        return result, index

    value, _ = parse_block(0, lines[0][0] if lines else 0)
    return value


def _en(value):
    if isinstance(value, dict):
        return value.get("en") or next((v for v in value.values() if isinstance(v, str)), "")
    return value or ""


# --- chunk builders ---

def obdex_chunks(raw_root):
    base = Path(raw_root) / "OBDex-main" / "data"
    chunks, rejected = [], Counter()
    for path in sorted((base / "generic").glob("*.yaml")):
        entries = parse_yaml_subset(path.read_text(encoding="utf-8")) or []
        for entry in entries:
            code = str(entry.get("code") or "").upper() if isinstance(entry, dict) else ""
            title = _en(entry.get("title")) if isinstance(entry, dict) else ""
            if not re.fullmatch(r"[PBCU][0-9A-F]{4}", code) or not title:
                rejected["entry without a valid code or English title"] += 1
                continue
            causes = [
                f"{_en(cause.get('label'))} ({cause.get('likelihood', 'unrated')} likelihood)"
                for cause in entry.get("common_causes") or [] if isinstance(cause, dict) and _en(cause.get("label"))
            ]
            symptoms = [_en(item) for item in entry.get("symptoms") or [] if _en(item)]
            repair = entry.get("repair") or {}
            flags = entry.get("flags") or {}
            cost = repair.get("estimated_cost_eur")
            parts = [f"{code}: {title}.", _en(entry.get("description"))]
            if causes:
                parts.append("Common causes: " + "; ".join(causes) + ".")
            if symptoms:
                parts.append("Symptoms: " + "; ".join(symptoms) + ".")
            if repair.get("difficulty"):
                parts.append(f"Repair difficulty: {repair['difficulty']}.")
            chunks.append({
                "id": f"dtc:{code}",
                "source_id": "obdex",
                "source_type": "dtc_reference",
                "trust": "verified_reference",
                "audience": "all",
                "title": f"{code} – {title}",
                "text": " ".join(part for part in parts if part),
                "metadata": {
                    "dtc_code": code,
                    "dtc_system": {"P": "powertrain", "B": "body", "C": "chassis", "U": "network"}[code[0]],
                    "affected_components": entry.get("affected_components") or [],
                    "related_codes": entry.get("related_codes") or [],
                    "mil": flags.get("mil"),
                    "limp_mode_possible": flags.get("limp_mode_possible"),
                    "generic_cost_range_eur": cost if isinstance(cost, list) else None,
                },
                "citation": {"name": "OBDex generic OBD-II code reference", "url": "https://github.com/foerbsnavi/obdex",
                             "license": "CC0 1.0", "file": path.relative_to(Path(raw_root)).as_posix(),
                             "note": "Generic SAE J2012 meaning; manufacturer-specific definitions can differ."},
            })
    for path in sorted((base / "pids").glob("*.yaml")):
        for entry in parse_yaml_subset(path.read_text(encoding="utf-8")) or []:
            if not isinstance(entry, dict) or not _en(entry.get("name")):
                rejected["PID entry without a name"] += 1
                continue
            pid = f"{entry.get('mode')}/{entry.get('pid')}"
            text = f"OBD-II PID mode {entry.get('mode')} PID {entry.get('pid')}: {_en(entry.get('name'))}."
            if entry.get("unit"):
                text += f" Unit: {entry['unit']}."
            if entry.get("formula"):
                text += f" Formula: {entry['formula']}."
            if entry.get("range"):
                text += f" Range: {entry['range']}."
            chunks.append({
                "id": f"pid:{pid}", "source_id": "obdex", "source_type": "pid_reference",
                "trust": "verified_reference", "audience": "all", "title": f"PID {pid} – {_en(entry.get('name'))}",
                "text": text, "metadata": {"pid": pid},
                "citation": {"name": "OBDex OBD-II PID reference", "url": "https://github.com/foerbsnavi/obdex", "license": "CC0 1.0",
                             "file": path.relative_to(Path(raw_root)).as_posix()},
            })
    return chunks, dict(rejected)


def curated_chunks(knowledge_path):
    payload = json.loads(Path(knowledge_path).read_text(encoding="utf-8"))
    chunks = []
    for document in payload["documents"]:
        fields = []
        for key in ("overview", "component_functions", "possible_causes", "symptoms", "checks", "urgency", "needed_information"):
            value = document.get(key)
            if isinstance(value, list):
                fields.append(" ".join(str(item) for item in value))
            elif value:
                fields.append(str(value))
        sources = [source for source in document.get("sources", []) if isinstance(source, dict)]
        chunks.append({
            "id": f"kb:{document['id']}", "source_id": "automotive_knowledge", "source_type": "curated_reference",
            "trust": "verified_reference", "audience": "all", "title": document["title"],
            "text": " ".join(fields), "keywords": document.get("keywords", []),
            "metadata": {"category": document["category"], "safety_critical": bool(document.get("safety_critical"))},
            "citation": {"name": sources[0].get("name") if sources else "Curated automotive knowledge base",
                         "url": sources[0].get("url") if sources else None, "file": "data/automotive_knowledge.json"},
        })
    return chunks


def flowchart_chunks(raw_root):
    path = Path(raw_root) / "automotive_faults_aktc_obike_et_al.json"
    entries = json.loads(path.read_text(encoding="utf-8"))
    chunks = []
    for number, entry in enumerate(entries):
        steps = [
            f"{step.get('step')} (possible results: {', '.join(step.get('result', []))})"
            for step in entry.get("diagnosis_steps", []) if isinstance(step, dict) and step.get("step")
        ]
        if not steps:
            continue
        title = f"{entry.get('category')} – {entry.get('subcategory')}"
        chunks.append({
            "id": f"flow:{number}", "source_id": "automotive_fault_flowcharts", "source_type": "diagnostic_flowchart",
            "trust": "unverified_reference", "audience": "all", "title": title,
            "text": f"{title}. Symptoms: {'; '.join(entry.get('symptoms', []))}. Diagnostic steps: {'; '.join(steps)}.",
            "metadata": {"category": entry.get("category")},
            "citation": {"name": "Automotive fault diagnosis flowcharts (AKTC/oBike et al.)", "file": path.name,
                         "note": "Compiled flowchart; source attribution not verified locally."},
        })
    return chunks


def system_chunks(guide_path):
    payload = json.loads(Path(guide_path).read_text(encoding="utf-8"))
    return [{
        "id": f"sys:{document['id']}", "source_id": "system_guide", "source_type": "system_documentation",
        "trust": "system_documentation", "audience": document.get("audience", "all"), "title": document["title"],
        "text": document["text"], "keywords": document.get("keywords", []),
        "metadata": {"category": document["category"]},
        "citation": {"name": "DriveRent system guide", "file": "data/knowledge/system_guide.json",
                     "code_references": document.get("code_references", []), "verified_on": payload.get("verified_on")},
    } for document in payload["documents"]]


def service_schedule_chunks(raw_root):
    from data_pipeline.service_need import TARGETS, load_service_records

    frame, usage, _ = load_service_records(raw_root)
    frame = frame.assign(milestone=(np.round(frame["mileage"] / 10_000) * 10_000).astype(int))
    chunks = []
    for target, definition in TARGETS.items():
        table = frame.groupby("milestone")[target].agg(["mean", "count"])
        rows = [f"{int(m):,} km: {r['mean']:.0%} of {int(r['count'])} visits" for m, r in table.iterrows() if r["count"] >= 10]
        peaks = [f"{int(m):,} km" for m, r in table.iterrows() if r["count"] >= 10 and r["mean"] >= 0.5]
        text = (
            f"Recorded service frequency for {definition['label'].lower()} in {usage['rows_used']} workshop visits "
            f"(Honda Jazz, Amaze and City; Toyota Fortuner; Chennai and Mumbai, model years 2016-2018), by nearest 10,000 km milestone: "
            + "; ".join(rows) + "."
            + (f" Performed at half or more of visits near {', '.join(peaks)}." if peaks else "")
            + " These are observed workshop practices for those models, not a manufacturer schedule for other vehicles."
        )
        chunks.append({
            "id": f"sched:{target}", "source_id": "vehicle_service_records", "source_type": "service_statistics",
            "trust": "derived_from_records", "audience": "all", "title": f"Service frequency: {definition['label']}",
            "text": text, "keywords": [definition["label"].lower(), "maintenance schedule", "service interval", "when to replace"],
            "metadata": {"service_item": target, "component_keys": definition["component_keys"]},
            "citation": {"name": "Vehicle Maintenance - Service Records", "file": "Vehicle Maintenance- Service Records.csv",
                         "note": "Computed by data_pipeline/knowledge_index.py; licence not documented."},
        })
    return chunks


def parts_lookup(raw_root):
    frame = pd.read_excel(Path(raw_root) / "masterparts.xlsx")
    frame.columns = ["part_number", "description", "supply_group"]
    frame["part_number"] = frame["part_number"].astype(str).str.strip().str.upper()
    valid = frame["part_number"].str.fullmatch(r"[0-9A-Z\-]{6,20}")
    frame = frame[valid].drop_duplicates("part_number")
    return {row.part_number: {"description": str(row.description).strip(), "supply_group": str(row.supply_group).strip()}
            for row in frame.itertuples()}, {"rows_read": int(len(valid)), "rows_rejected_invalid_part_number": int((~valid).sum())}


def _search_text(chunk):
    return " ".join([chunk["title"], chunk["text"], " ".join(chunk.get("keywords", []))]).casefold()


def build_index(raw_root, knowledge_path, guide_path, output=INDEX_ROOT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dtc, rejected = obdex_chunks(raw_root)
    sources = {
        "obdex": dtc,
        "automotive_knowledge": curated_chunks(knowledge_path),
        "automotive_fault_flowcharts": flowchart_chunks(raw_root),
        "system_guide": system_chunks(guide_path),
        "vehicle_service_records": service_schedule_chunks(raw_root),
    }
    chunks = [chunk for items in sources.values() for chunk in items]
    ids = [chunk["id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        duplicates = [key for key, count in Counter(ids).items() if count > 1]
        raise ValueError(f"Duplicate chunk ids: {duplicates[:5]}")

    texts = [_search_text(chunk) for chunk in chunks]
    counter = CountVectorizer(token_pattern=TOKEN_PATTERN, stop_words="english")
    term_frequency = counter.fit_transform(texts).tocsc().astype(np.float32)
    document_length = np.asarray(term_frequency.sum(axis=1)).ravel()
    document_frequency = np.diff(term_frequency.indptr)
    idf = np.log1p((len(texts) - document_frequency + 0.5) / (document_frequency + 0.5)).astype(np.float32)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_features=200_000, dtype=np.float32)
    char_matrix = char.fit_transform(texts)

    with (output / "chunks.jsonl").open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    index = {
        "chunk_ids": ids, "counter": counter, "bm25_tf": sparse.csr_matrix(term_frequency), "bm25_idf": idf,
        "bm25_doc_length": document_length, "bm25_avg_length": float(document_length.mean()),
        "char_vectorizer": char, "char_matrix": char_matrix, "built_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(index, output / "index.joblib", compress=3)
    parts, parts_usage = parts_lookup(raw_root)
    (output / "parts_lookup.json").write_text(json.dumps(parts, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "built_at": index["built_at"],
        "chunk_count": len(chunks),
        "chunks_by_source": {name: len(items) for name, items in sources.items()},
        "chunks_by_trust": dict(Counter(chunk["trust"] for chunk in chunks)),
        "rejected_entries": rejected,
        "parts_lookup": {"entries": len(parts), **parts_usage, "source": "masterparts.xlsx"},
        "chunks_sha256": hashlib.sha256((output / "chunks.jsonl").read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256((output / "index.joblib").read_bytes()).hexdigest(),
        "not_indexed": {
            "cvffad": "generated records (source='generated'); Chinese commercial-fleet content",
            "vehicle_diagnostic_sample": "generated diagnostics with random measurements",
            "dtc_chat_sft": "templated assistant answers with truncated unrelated text",
            "auto_repair_orders": "synthetic relational sample (example.org customers)",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
