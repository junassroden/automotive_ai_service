"""Evaluate the chatbot against data/evaluation/chatbot_eval_v2.json.

    python scripts/evaluate_chatbot.py

Scores the current router (ai_api.assistant.answer_chat) and, as a baseline,
the previous intent/knowledge core (chatbot_service._answer_chat_core) on the
same cases.

Every case runs the production flow (scripts/chat_pipeline.py): the AI service
plans which records it needs, the Back-End's ChatRetrieval code (PHP, run through
Back-End/tests/chat_retrieval_cli.php) executes the plan for the case's role and
fixture user on Back-End/tests/fixtures/chat_fixture_db.json, and the AI service
answers from those results. A case therefore cannot pass because data was handed
straight to the AI. "backend_kind" expectations are checked against the kind of
records the plan retrieved (inventory, bookings, reservations, fleet, maintenance,
fleet_risk, or null for none). Cases marked "routing": "direct" inject their
context as written; they test the AI service's own defences (for example an
administrator context sent for a customer) and are reported separately. Cases with
the admin_diagnostic context use the Back-End's explicit diagnostic mode. The
baseline is the previous intent core given each case's old fixture context
directly. Requires the PHP CLI (on PATH or PHP_BINARY). Also measures retrieval recall/precision, a lexical groundedness
check, permission leaks, latency, and overlap with chatbot training data.
Conversation memory writes are rolled back. Results:
data/reports/chatbot_evaluation_v2.json and .md.
"""
import json
import os
import re
import statistics
import sys
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "automotive_ai.settings")

import django  # noqa: E402

django.setup()

import pandas as pd  # noqa: E402
from django.db import transaction  # noqa: E402

from ai_api import chatbot_service  # noqa: E402
from ai_api.assistant import answer_chat  # noqa: E402
from ai_api.rag import extract_entities, load_index, retrieve  # noqa: E402
from chat_pipeline import ask  # noqa: E402


EVAL_PATH = SERVICE_ROOT / "data" / "evaluation" / "chatbot_eval_v2.json"
REPORT_ROOT = SERVICE_ROOT / "data" / "reports"
DTC_VALIDATION = SERVICE_ROOT / "data" / "raw" / "validation-00000-of-00001.parquet"
DIAGNOSTIC_FIXTURES = {"admin_diagnostic"}
# Retrieval tool -> the kind of records it returns, for "backend_kind" expectations.
TOOL_KINDS = {"vehicles.search": "inventory", "vehicles.quote": "inventory", "bookings.mine": "bookings",
              "reservations.summary": "reservations", "fleet.status_counts": "fleet", "maintenance.records": "maintenance",
              "maintenance.inspections": "maintenance", "maintenance.recurring": "maintenance",
              "maintenance.due": "fleet_risk", "maintenance.model_inputs": "fleet_risk", "maintenance.costs": "maintenance",
              "vehicles.status": "fleet", "reservations.revenue": "reservations", "fleet.utilization": "fleet"}
ABSTAIN_INTENTS = {"unknown", "out_of_scope", "clarification_needed", "permission_denied"}
TOKEN = re.compile(r"[a-z0-9]+")


def merge(base, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def routing_mode(case):
    if case.get("context") in DIAGNOSTIC_FIXTURES:
        return "diagnostic_mode"
    return case.get("routing", "backend")


def retrieved_kind(plan):
    """The kind of records a plan retrieved; an admin vehicle search by status is fleet data."""
    if not plan:
        return None
    for request in plan.get("retrievals", []):
        if request["tool"] == "vehicles.search" and request.get("args", {}).get("statuses"):
            return "fleet"
    kinds = [TOOL_KINDS.get(request["tool"]) for request in plan.get("retrievals", [])]
    return kinds[0] if kinds else None


def build_context(case, fixtures):
    """The case's fixture context as written (direct and diagnostic cases, and the baseline)."""
    if not case.get("context"):
        return None
    context = deepcopy(fixtures[case["context"]])
    return merge(context, deepcopy(case.get("context_patch", {})))


def dtc_validation_cases(known_codes):
    """User questions from the held-out DTC chat validation split (answers unused)."""
    if not DTC_VALIDATION.exists():
        return []
    cases = []
    for number, row in enumerate(pd.read_parquet(DTC_VALIDATION).itertuples()):
        codes = extract_entities(row.user)["dtc_codes"]
        if not codes:
            continue
        known = [code for code in codes if code in known_codes]
        expect = {"must_contain": known} if known else {"must_contain": ["not in the generic"], "abstain": True}
        cases.append({"id": f"dtc-val-{number:03d}", "role": "admin", "category": "diagnostics_external",
                      "message": row.user[:4000], "expect": expect, "source": "validation-00000-of-00001.parquet"})
    return cases


def is_abstain(response):
    data = response.get("data") or {}
    evidence = (data.get("evidence") or {}).get("evidence_type")
    return response.get("intent") in ABSTAIN_INTENTS or (evidence == "none" and not data.get("all_categories") and not data.get("prioritized_vehicles"))


def check(case, response, evidence_checks=True, route=None):
    expect = case["expect"]
    answer = response.get("answer", "")
    lowered = answer.casefold()
    data = response.get("data") or {}
    failures = []
    for text in expect.get("must_contain", []):
        if text.casefold() not in lowered:
            failures.append(f"missing '{text}'")
    if expect.get("must_contain_any") and not any(text.casefold() in lowered for text in expect["must_contain_any"]):
        failures.append(f"missing any of {expect['must_contain_any']}")
    leaks = [text for text in expect.get("must_not_contain", []) if text.casefold() in lowered]
    failures += [f"contains forbidden '{text}'" for text in leaks]
    if "abstain" in expect and expect["abstain"] != is_abstain(response):
        failures.append("expected abstention" if expect["abstain"] else "unexpected abstention")
    if "backend_kind" in expect and (route or {}).get("kind") not in expect["backend_kind"]:
        failures.append(f"Back-End routed to {(route or {}).get('kind')!r}, expected one of {expect['backend_kind']}")
    if expect.get("intent") and response.get("intent") not in expect["intent"]:
        failures.append(f"intent {response.get('intent')} not in {expect['intent']}")
    if evidence_checks:
        evidence = (data.get("evidence") or {}).get("evidence_type")
        if expect.get("evidence_type") and evidence not in expect["evidence_type"]:
            failures.append(f"evidence {evidence} not in {expect['evidence_type']}")
        if expect.get("viewer_role") and data.get("viewer_role") != expect["viewer_role"]:
            failures.append(f"viewer_role {data.get('viewer_role')}")
        cited = {source.get("id") for source in (data.get("evidence") or {}).get("sources", []) if isinstance(source, dict)}
        if expect.get("sources") and not set(expect["sources"]) & cited:
            failures.append(f"none of expected sources {expect['sources']} cited (cited {sorted(c for c in cited if c)})")
    return failures, leaks


def groundedness(response, chunks_by_id):
    """Share of answer content tokens that appear in the cited passages (lexical proxy)."""
    data = response.get("data") or {}
    evidence = data.get("evidence") or {}
    if evidence.get("evidence_type") != "reference":
        return None
    cited = [chunks_by_id[s["id"]] for s in evidence.get("sources", []) if isinstance(s, dict) and s.get("id") in chunks_by_id]
    if not cited:
        return None
    source_tokens = set(TOKEN.findall(" ".join(c["title"] + " " + c["text"] for c in cited).casefold()))
    answer_tokens = [t for t in TOKEN.findall(response["answer"].casefold()) if len(t) > 3]
    return round(sum(t in source_tokens for t in answer_tokens) / len(answer_tokens), 4) if answer_tokens else None


def retrieval_metrics(cases):
    rows = []
    for case in cases:
        expected = set(case["expect"].get("sources", []))
        if not expected:
            continue
        results = [item["id"] for item in retrieve(case["message"], case["role"], top_k=3)]
        hits = [rank for rank, item in enumerate(results, start=1) if item in expected]
        rows.append({"id": case["id"], "expected": sorted(expected), "retrieved": results,
                     "recall_at_3": len(set(results) & expected) / len(expected),
                     "precision_at_3": len(set(results) & expected) / max(1, len(results)),
                     "reciprocal_rank": 1 / hits[0] if hits else 0.0})
    mean = lambda key: round(statistics.mean(row[key] for row in rows), 4) if rows else None
    return {"cases": len(rows), "recall_at_3": mean("recall_at_3"), "precision_at_3": mean("precision_at_3"), "mrr": mean("reciprocal_rank"), "rows": rows}


def training_overlap(cases):
    payload = json.loads((SERVICE_ROOT / "data" / "chatbot_intents.json").read_text(encoding="utf-8"))
    normalize = lambda text: " ".join(TOKEN.findall(text.casefold()))
    patterns = {normalize(p) for intent in payload["intents"] for p in intent.get("patterns", [])}
    grams = lambda text: {text[i:i + 3] for i in range(max(1, len(text) - 2))}
    pattern_grams = [(p, grams(p)) for p in patterns]
    exact, near = [], []
    for case in cases:
        message = normalize(case["message"])
        if message in patterns:
            exact.append(case["id"])
            continue
        g = grams(message)
        if any(len(g & pg) / len(g | pg) >= 0.9 for _, pg in pattern_grams if pg):
            near.append(case["id"])
    return {"training_patterns": len(patterns), "exact_matches": exact, "near_duplicates_jaccard_0_9": near}


def run_cases(cases, fixtures, use_router):
    results = []
    conversations = {}
    with transaction.atomic():
        for case in cases:
            route = None
            context = build_context(case, fixtures)
            conversation_id = None
            if case.get("conversation"):
                conversation_id = conversations.setdefault(case["conversation"], str(uuid.uuid5(uuid.NAMESPACE_URL, case["conversation"])))
            started = time.perf_counter()
            try:
                if use_router and routing_mode(case) == "backend":
                    response, plan, _ = ask(case["message"], case["role"], user_id=case.get("user_id"),
                                            conversation_id=conversation_id or str(uuid.uuid4()),
                                            fixture_overrides=case.get("fixture_overrides"))
                    route = {"kind": retrieved_kind(plan)}
                elif use_router:
                    response = answer_chat(case["message"], rental_context=context,
                                           diagnostic_context=(context or {}).get("diagnostic_context"),
                                           viewer_role=case["role"], conversation_id=conversation_id,
                                           conversation_scope=f"{case['role']}:1")
                else:
                    response = chatbot_service._answer_chat_core(case["message"], rental_context=context,
                                                                 diagnostic_context=(context or {}).get("diagnostic_context"))
                error = None
            except Exception as exc:  # an exception is a failed case, not a crashed evaluation
                response, error = {"answer": "", "intent": "error", "data": {}}, f"{type(exc).__name__}: {exc}"
            latency = (time.perf_counter() - started) * 1000
            failures, leaks = check(case, response, evidence_checks=use_router, route=route)
            if error:
                failures.append(error)
            results.append({"id": case["id"], "role": case["role"], "category": case["category"], "critical": case.get("critical", False),
                            "routing": routing_mode(case), "backend_kind": (route or {}).get("kind"),
                            "passed": not failures, "failures": failures, "leaks": leaks, "intent": response.get("intent"),
                            "evidence_type": ((response.get("data") or {}).get("evidence") or {}).get("evidence_type"),
                            "latency_ms": round(latency, 1), "answer_preview": response.get("answer", "")[:300],
                            "groundedness": groundedness(response, CHUNKS) if use_router else None})
        transaction.set_rollback(True)
    return results


def summarize(results):
    def rate(rows):
        return round(sum(r["passed"] for r in rows) / len(rows), 4) if rows else None
    by = lambda key: {value: {"cases": len(rows), "pass_rate": rate(rows)} for value, rows in _group(results, key).items()}
    latencies = sorted(r["latency_ms"] for r in results)
    grounded = [r["groundedness"] for r in results if r["groundedness"] is not None]
    permission = [r for r in results if r["category"] == "permissions"]
    return {
        "cases": len(results), "pass_rate": rate(results),
        "critical_pass_rate": rate([r for r in results if r["critical"]]),
        "by_role": by("role"), "by_category": by("category"), "by_routing": by("routing"),
        "permission_cases": len(permission), "permission_leaks": sum(bool(r["leaks"]) for r in permission),
        "latency_ms": {"p50": latencies[len(latencies) // 2], "p95": latencies[int(len(latencies) * 0.95) - 1], "max": latencies[-1]} if latencies else None,
        "mean_groundedness": round(statistics.mean(grounded), 4) if grounded else None,
        "groundedness_cases": len(grounded),
    }


def _group(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return dict(sorted(groups.items()))


def write_report(report):
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "chatbot_evaluation_v2.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    new, old = report["router"]["summary"], report["baseline_core"]["summary"]
    lines = [
        "# Chatbot evaluation (v2)", "",
        f"Generated {report['generated_at']} with `python scripts/evaluate_chatbot.py`. Evaluation set: `{report['evaluation_set']}` "
        f"({report['hand_written_cases']} author-written cases + {report['external_dtc_cases']} DTC questions from the held-out chat validation split). "
        "Cases are evaluation-only; the training-overlap check is below.", "",
        "| Metric | Current router | Previous core (baseline) |", "|---|---:|---:|",
        f"| Pass rate (all checks) | {new['pass_rate']} | {old['pass_rate']} (content checks only) |",
        f"| Critical-case pass rate | {new['critical_pass_rate']} | {old['critical_pass_rate']} |",
        f"| Permission leaks | {new['permission_leaks']} of {new['permission_cases']} | {old['permission_leaks']} of {old['permission_cases']} |",
        f"| Latency p50 / p95 (ms) | {new['latency_ms']['p50']} / {new['latency_ms']['p95']} | {old['latency_ms']['p50']} / {old['latency_ms']['p95']} |",
        f"| Mean lexical groundedness (reference answers) | {new['mean_groundedness']} over {new['groundedness_cases']} | n/a |", "",
        "Baseline rows apply the same content checks (required/forbidden text, abstention, intent) but not the evidence-type or citation checks, which the old core cannot produce.", "",
        "## Retrieval", "",
        f"Cases with labelled sources: {report['retrieval']['cases']}; recall@3 {report['retrieval']['recall_at_3']}, precision@3 {report['retrieval']['precision_at_3']}, MRR {report['retrieval']['mrr']}.", "",
        "## By role", "", "| Role | Cases | Router pass | Baseline pass |", "|---|---:|---:|---:|",
    ]
    for role, item in new["by_role"].items():
        lines.append(f"| {role} | {item['cases']} | {item['pass_rate']} | {old['by_role'].get(role, {}).get('pass_rate')} |")
    lines += ["", "## By routing", "",
              "`backend`: the production flow, with records retrieved by the Back-End's ChatRetrieval from the fixture database. "
              "`direct`: context injected as written, to test the AI service's own checks. "
              "`diagnostic_mode`: the Back-End's explicit diagnostic mode.", "",
              "| Routing | Cases | Router pass | Baseline pass |", "|---|---:|---:|---:|"]
    for routing, item in new["by_routing"].items():
        lines.append(f"| {routing} | {item['cases']} | {item['pass_rate']} | {old['by_routing'].get(routing, {}).get('pass_rate')} |")
    lines += ["", "## By category", "", "| Category | Cases | Router pass | Baseline pass |", "|---|---:|---:|---:|"]
    for category, item in new["by_category"].items():
        lines.append(f"| {category} | {item['cases']} | {item['pass_rate']} | {old['by_category'].get(category, {}).get('pass_rate')} |")
    failed = [r for r in report["router"]["results"] if not r["passed"]]
    lines += ["", f"## Router failures ({len(failed)})", ""]
    lines += [f"- `{r['id']}` ({r['role']}): {'; '.join(r['failures'])}" for r in failed] or ["- none"]
    overlap = report["training_overlap"]
    lines += ["", "## Training overlap", "",
              f"Compared with {overlap['training_patterns']} intent-training patterns: exact matches {overlap['exact_matches'] or 'none'}; "
              f"near-duplicates (char-trigram Jaccard >= 0.9) {overlap['near_duplicates_jaccard_0_9'] or 'none'}.", "",
              "## Limits of this evaluation", "",
              "- Questions and expected facts are author-written and checked against the code and references; they still need review by a second person.",
              "- This set was also used while developing the router (routing rules and system-guide keywords were adjusted after earlier runs), so it is a regression/development set; add fresh, unseen questions to measure generalisation.",
              "- Groundedness is a lexical overlap proxy, not a human judgement of factual support.",
              "- Database answers are scored on fixture contexts shaped like the Back-End's payloads, not on the live MySQL database; "
              "retrieval runs the Back-End's ChatRetrieval rules on Back-End/tests/fixtures/chat_fixture_db.json, but the HTTP endpoint, "
              "session and SQL queries are not exercised here (see Back-End/tests/chat_integration.ps1 for that).", ""]
    (REPORT_ROOT / "chatbot_evaluation_v2.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    global CHUNKS
    payload = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    chunks, _, by_dtc = load_index()
    CHUNKS = {chunk["id"]: chunk for chunk in chunks}
    cases = payload["cases"] + dtc_validation_cases(set(by_dtc))
    router = run_cases(cases, payload["fixtures"], use_router=True)
    baseline = run_cases(cases, payload["fixtures"], use_router=False)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "evaluation_set": str(EVAL_PATH.relative_to(SERVICE_ROOT)),
        "review_status": payload.get("review_status"),
        "hand_written_cases": len(payload["cases"]), "external_dtc_cases": len(cases) - len(payload["cases"]),
        "router": {"summary": summarize(router), "results": router},
        "baseline_core": {"summary": summarize(baseline), "results": baseline},
        "retrieval": retrieval_metrics(payload["cases"]),
        "training_overlap": training_overlap(cases),
    }
    write_report(report)
    print(json.dumps({"router": report["router"]["summary"], "baseline": report["baseline_core"]["summary"],
                      "retrieval": {k: v for k, v in report["retrieval"].items() if k != "rows"}}, indent=2))


CHUNKS = {}

if __name__ == "__main__":
    main()
