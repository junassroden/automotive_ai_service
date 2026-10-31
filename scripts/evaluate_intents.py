"""Score the intent classifier on the independent held-out set.

    python scripts/evaluate_intents.py [--label baseline]

Uses the production path (language.normalize_message, then the trained pipeline), reports
accuracy and per-intent recall, and checks that no held-out question is an exact or
near-duplicate of a training pattern. Results: data/reports/intent_holdout_evaluation.json
(each run is appended under its label).
"""
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "automotive_ai.settings")

import django  # noqa: E402

django.setup()

from ai_api.chatbot_service import load_chatbot  # noqa: E402
from ai_api.language import normalize_message  # noqa: E402

HOLDOUT = SERVICE_ROOT / "data" / "evaluation" / "intent_holdout_v1.json"
TRAINING = SERVICE_ROOT / "data" / "chatbot_intents.json"
REPORT = SERVICE_ROOT / "data" / "reports" / "intent_holdout_evaluation.json"
TOKEN = re.compile(r"[a-z0-9]+")


def normalize(text):
    return " ".join(TOKEN.findall(text.casefold()))


def overlap(examples):
    patterns = {normalize(p) for intent in json.loads(TRAINING.read_text(encoding="utf-8"))["intents"] for p in intent.get("patterns", [])}
    grams = lambda text: {text[i:i + 3] for i in range(max(1, len(text) - 2))}
    pattern_grams = [grams(p) for p in patterns]
    exact, near = [], []
    for item in examples:
        text = normalize(item["text"])
        if text in patterns:
            exact.append(item["text"])
        elif any(len(grams(text) & g) / len(grams(text) | g) >= 0.9 for g in pattern_grams if g):
            near.append(item["text"])
    return exact, near


def main():
    label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else time.strftime("%Y-%m-%dT%H:%M:%S")
    examples = json.loads(HOLDOUT.read_text(encoding="utf-8"))["examples"]
    pipeline, _ = load_chatbot()
    predictions = pipeline.predict([normalize_message(item["text"]) for item in examples])
    per_intent = defaultdict(lambda: [0, 0])
    errors = []
    for item, predicted in zip(examples, predictions):
        per_intent[item["intent"]][1] += 1
        if predicted == item["intent"]:
            per_intent[item["intent"]][0] += 1
        else:
            errors.append({"text": item["text"], "expected": item["intent"], "predicted": str(predicted)})
    correct = sum(hits for hits, _ in per_intent.values())
    exact, near = overlap(examples)
    run = {"label": label, "examples": len(examples), "accuracy": round(correct / len(examples), 4),
           "per_intent_recall": {intent: round(hits / total, 3) for intent, (hits, total) in sorted(per_intent.items())},
           "errors": errors, "training_overlap": {"exact": exact, "near_duplicates_jaccard_0_9": near}}
    report = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {"runs": []}
    report["runs"] = [r for r in report["runs"] if r["label"] != label] + [run]
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in run.items() if k != "errors"}, indent=1))
    print(f"{len(errors)} misclassified")


if __name__ == "__main__":
    main()
