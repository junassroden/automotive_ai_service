import json
import os
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "automotive_ai.settings")

import django

django.setup()

from ai_api.automotive_knowledge import format_knowledge_answer, rank_knowledge, retrieve_knowledge


BASE_DIR = Path(__file__).resolve().parent.parent
CASE_PATH = BASE_DIR / "data" / "reports" / "chatbot_knowledge_eval_questions.json"
REPORT_PATH = BASE_DIR / "data" / "reports" / "chatbot_knowledge_evaluation.json"


def evaluate():
    cases = json.loads(CASE_PATH.read_text(encoding="utf-8"))["cases"]
    results = []
    supported_count = 0
    top1_correct = 0
    recall_at_six = 0
    abstention_correct = 0
    abstention_count = 0
    safety_expected = 0
    safety_passed = 0
    uncertainty_expected = 0
    uncertainty_passed = 0

    for case in cases:
        question = case["question"]
        ranked = rank_knowledge(question)
        retrieved = retrieve_knowledge(question)
        expected_abstention = case.get("expected_abstention", False)
        ranked_ids = []
        for item in ranked:
            doc_id = item["document"]["id"]
            if doc_id not in ranked_ids:
                ranked_ids.append(doc_id)
        actual_id = retrieved["document"]["id"] if retrieved else None
        safety_terms = [
            term.casefold() for term in case.get("safety_terms", [])
        ]
        answer = (
            format_knowledge_answer(retrieved, query=question)
            if retrieved
            else ""
        )
        matched_safety_terms = [
            term for term in safety_terms
            if term in answer.casefold()
            or (
                term == "stop safely"
                and any(
                    alternative in answer.casefold()
                    for alternative in ("stopping safely", "pull over where safe")
                )
            )
        ]
        uncertainty_met = None
        if case.get("vehicle_specific_gap_expected"):
            uncertainty_expected += 1
            uncertainty_met = any(
                phrase in answer.casefold()
                for phrase in (
                    "not supplied",
                    "vehicle-specific",
                    "not retrieved",
                    "no exact",
                    "not provided",
                )
            )
            uncertainty_passed += uncertainty_met

        if expected_abstention:
            abstention_count += 1
            abstention_correct += retrieved is None
            passed = retrieved is None
            supported_count += 1
            top1_correct += passed
            recall_at_six += passed
        else:
            expected_ids = case.get(
                "expected_knowledge_ids",
                [case["expected_knowledge_id"]],
            )
            supported_count += 1
            top1_correct += actual_id in expected_ids
            recall_at_six += any(
                expected_id in ranked_ids[:6] for expected_id in expected_ids
            )
            passed = actual_id in expected_ids

        if safety_terms:
            safety_expected += 1
            safety_passed += len(matched_safety_terms) == len(safety_terms)

        results.append({
            "id": case["id"],
            "question": question,
            "expected_knowledge_id": case.get("expected_knowledge_id"),
            "acceptable_knowledge_ids": case.get(
                "expected_knowledge_ids",
                [case["expected_knowledge_id"]]
                if case.get("expected_knowledge_id") else [],
            ),
            "expected_abstention": expected_abstention,
            "retrieved_knowledge_id": actual_id,
            "top_ranked_knowledge_ids": ranked_ids[:6],
            "retrieval_score": retrieved["score"] if retrieved else None,
            "retrieved_passage_count": (
                len(retrieved.get("passages", [])) if retrieved else 0
            ),
            "retrieval_expectation_met": bool(passed),
            "safety_terms_expected": safety_terms,
            "safety_terms_found": matched_safety_terms,
            "safety_expectation_met": (
                len(matched_safety_terms) == len(safety_terms)
                if safety_terms else None
            ),
            "vehicle_specific_uncertainty_expected": case.get(
                "vehicle_specific_gap_expected", False
            ),
            "vehicle_specific_uncertainty_met": uncertainty_met,
        })

    total = len(cases)
    report = {
        "schema_version": 1,
        "evaluation": "automotive_knowledge_retrieval_regression",
        "case_file": str(CASE_PATH.relative_to(BASE_DIR)),
        "case_count": total,
        "metrics": {
            "top1_expected_topic_or_abstention_accuracy": round(
                top1_correct / supported_count, 4
            ) if supported_count else None,
            "expected_topic_recall_at_6_or_abstention_accuracy": round(
                recall_at_six / supported_count, 4
            ) if supported_count else None,
            "unsupported_query_abstention_accuracy": round(
                abstention_correct / abstention_count, 4
            ) if abstention_count else None,
            "safety_term_check_accuracy": round(
                safety_passed / safety_expected, 4
            ) if safety_expected else None,
            "safety_term_cases": safety_expected,
            "vehicle_specific_uncertainty_accuracy": round(
                uncertainty_passed / uncertainty_expected, 4
            ) if uncertainty_expected else None,
            "vehicle_specific_uncertainty_cases": uncertainty_expected,
        },
        "scope_and_limits": {
            "measures": [
                "Deterministic topic retrieval against curated expected IDs.",
                "Abstention on curated unsupported questions.",
                "Presence of specified safety phrases in selected answers.",
            ],
            "not_automatically_verified": [
                "Mechanical factual correctness.",
                "Whether every cited source supports every statement.",
                "Whether diagnostic reasoning matches an exact vehicle's service procedure.",
                "Expert or technician review of answer completeness.",
            ],
            "source_references_fetched_during_answers": 0,
            "human_review_required": True,
            "manual_review_rubric": {
                "score_each_0_to_2": [
                    "Relevance and factual correctness",
                    "Coverage of requested topic",
                    "Diagnostic reasoning and uncertainty",
                    "Source relevance and traceability",
                    "Unsupported or vehicle-specific claims",
                    "Safety of advice",
                    "Handling of unknown questions",
                ],
                "human_scores": None,
            },
        },
        "results": results,
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps({
        "case_count": result["case_count"],
        "metrics": result["metrics"],
        "report": str(REPORT_PATH),
    }, indent=2))
