"""Score the sealed evaluation set (data/evaluation/sealed_2026-11-17/questions.json).

    python scripts/evaluate_sealed.py --label NAME [--detail]

Runs every case through ai_api.assistant.answer_chat (the function behind /api/chat/), turn by turn in one
conversation, with conversation-memory writes rolled back. Without --detail only the totals are printed and
saved, so the set can be scored before development without its failures being read. Results go to
data/evaluation/sealed_2026-11-17/result_<label>.json.

Metrics, each over the turns that define it:
  identification  the answer's topic is an accepted topic, or a topic added after sealing whose category is accepted
  relevance       identification passed and no wrong-topic terms appear
  accuracy        every required fact group is present, and no figures/prices where none may be given
  causes          accuracy, over symptom questions (paraphrase, Taglish, misspelled, multi-symptom)
  usefulness      an answer of at least 25 words that tells the user something to do or check
  abstention      out-of-domain questions are declined; unknowable figures are not invented
  safety          safety-critical turns contain an immediate safety instruction
  context         follow-up turns stay on (or correctly switch to) the right topic without repeating the last answer
  retrieval       the answering topic is exactly one of the listed existing topics (when any are listed)
  repetition      no sentence appears twice in one answer
"""
import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))
sys.path.insert(0, str(SERVICE_ROOT / "scripts"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "automotive_ai.settings")

import django  # noqa: E402

django.setup()

from django.db import transaction  # noqa: E402

from ai_api.automotive_knowledge import knowledge_document  # noqa: E402
from evaluate_answer_quality import NUMBER, ask, kind_of, overlap, topic_of, words  # noqa: E402

SET_ROOT = SERVICE_ROOT / "data" / "evaluation" / "sealed_2026-11-17"
MONEY = re.compile(r"(?:₱|\bphp\b|\bpesos?\b|\$|\busd\b)\s*\d|\d[\d,.]*\s*(?:₱|\bphp\b|\bpesos?\b)", re.IGNORECASE)
ACTION = re.compile(r"\b(?:check|have|stop|replace|inspect|test|take|contact|call|pull over|use|turn|top up|add|look|switch|"
                    r"keep|avoid|drive|get|ask|book|bring|let|park|move|note|read|scan|tighten|clean|report|photograph)\b", re.IGNORECASE)
SYMPTOM_TAGS = {"paraphrase", "taglish", "misspelled", "multi_symptom"}


def sentences(text):
    return [part.strip().casefold() for part in re.split(r"(?<=[.!?])\s+", text) if len(part.split()) >= 4]


def topic_ok(turn, topic, baseline_ids):
    if topic and topic in turn.get("topics", []):
        return True
    if topic and topic not in baseline_ids:
        document = knowledge_document(topic)
        return bool(document) and document.get("category") in turn.get("categories", [])
    return False


def score_turn(case, turn, response, previous, config, baseline_ids):
    answer = response.get("answer") or ""
    text = answer.casefold()
    kind, topic = kind_of(response), topic_of(response)
    checks = {}
    expect = turn.get("expect", "answer")
    topical = bool(turn.get("topics") or turn.get("categories"))
    if expect == "abstain":
        checks["abstention"] = kind == "abstain"
    elif expect == "answer":
        checks["answered"] = kind in {"answer", "dtc"}
    elif expect == "clarify_or_answer":
        checks["answered"] = kind in {"answer", "clarify"} and ("?" in answer or kind == "clarify")
    if topical and expect != "any":
        checks["identification"] = topic_ok(turn, topic, baseline_ids)
        checks["relevance"] = checks["identification"] and not any(term.casefold() in text for term in turn.get("exclude", []))
    if turn.get("topics") and expect != "any":
        checks["retrieval"] = topic in turn["topics"]
    accuracy = []
    if turn.get("include"):
        accuracy.append(all(any(option.casefold() in text for option in group) for group in turn["include"]))
    if turn.get("no_numbers"):
        accuracy.append(not NUMBER.search(answer))
        checks["abstention"] = not NUMBER.search(answer)
    if turn.get("no_money"):
        accuracy.append(not MONEY.search(answer))
        if case["tags"][0] == "unknown":
            checks["abstention"] = not MONEY.search(answer)
    if accuracy:
        checks["accuracy"] = all(accuracy)
        if SYMPTOM_TAGS & set(case["tags"]) and turn.get("include"):
            checks["causes"] = checks["accuracy"]
    if expect in {"answer", "clarify_or_answer"} and not turn.get("follow"):
        checks["usefulness"] = len(words(answer)) >= 25 and bool(ACTION.search(answer))
    if turn.get("safety"):
        checks["safety"] = any(phrase in text for phrase in config["safety_phrases"])
    if turn.get("follow") or (previous is not None and topical):
        repeated = overlap(previous, answer) >= 0.5 if previous else False
        checks["context"] = (checks.get("identification", True)) and not repeated and kind in {"answer", "clarify"}
    found = sentences(answer)
    checks["repetition"] = len(found) == len(set(found))
    return {"message": turn["message"], "answer": answer, "kind": kind, "topic": topic, "checks": checks,
            "passed": all(checks.values())}


def run(config, baseline_ids):
    results = []
    with transaction.atomic():
        for case in config["cases"]:
            conversation_id, previous, turns = str(uuid.uuid4()), None, []
            for turn in case["turns"]:
                response = ask(turn["message"], conversation_id)
                turns.append(score_turn(case, turn, response, previous, config, baseline_ids))
                previous = response.get("answer") or ""
            results.append({"id": case["id"], "tags": case["tags"], "turns": turns, "passed": all(t["passed"] for t in turns)})
        transaction.set_rollback(True)
    return results


def summarise(results):
    metrics = {}
    for case in results:
        for turn in case["turns"]:
            for name, passed in turn["checks"].items():
                total = metrics.setdefault(name, [0, 0])
                total[0] += bool(passed)
                total[1] += 1
    by_tag = {}
    for case in results:
        for tag in case["tags"]:
            total = by_tag.setdefault(tag, [0, 0])
            total[0] += case["passed"]
            total[1] += 1
    return {
        "cases_passed": sum(case["passed"] for case in results), "cases": len(results),
        "metrics": {name: {"passed": p, "total": t, "rate": round(p / t, 3)} for name, (p, t) in sorted(metrics.items())},
        "by_tag": {tag: {"passed": p, "total": t} for tag, (p, t) in sorted(by_tag.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--detail", action="store_true", help="Also save and print every turn (reads the set's failures).")
    arguments = parser.parse_args()
    config = json.loads((SET_ROOT / "questions.json").read_text(encoding="utf-8"))
    baseline_ids = set(json.loads((SET_ROOT / "baseline_topic_ids.json").read_text(encoding="utf-8")))
    results = run(config, baseline_ids)
    summary = summarise(results)
    output = {"label": arguments.label, "summary": summary}
    if arguments.detail:
        output["cases"] = results
    (SET_ROOT / f"result_{arguments.label}.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if arguments.detail:
        for case in results:
            for turn in case["turns"]:
                if not turn["passed"]:
                    failed = [name for name, ok in turn["checks"].items() if not ok]
                    print(f"FAIL {case['id']} {turn['message']!r} -> {turn['kind']}/{turn['topic']}: {', '.join(failed)}")


if __name__ == "__main__":
    main()
