
import json
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from data_pipeline.status import record_training_status
from ai_api.language import normalize_message


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "chatbot_intents.json"
MODEL_DIR = BASE_DIR / "trained_models"
MODEL_PATH = MODEL_DIR / "chatbot_nlp.joblib"
REPORT_PATH = MODEL_DIR / "chatbot_evaluation.json"
HOLDOUT_PATH = BASE_DIR / "data" / "evaluation" / "intent_holdout_v1.json"
TRAINING_REPORT_PATH = BASE_DIR / "data" / "reports" / "chatbot_training.json"

RANDOM_STATE = 42
TEST_SIZE = 0.25
MINIMUM_PATTERNS_PER_INTENT = 10


def load_training_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_PATH}"
        )

    try:
        payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Training dataset is not valid JSON at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        ) from error

    if not isinstance(payload, dict) or not isinstance(payload.get("intents"), list):
        raise ValueError("Training dataset must contain an 'intents' list.")

    texts = []
    labels = []
    responses = {}
    seen_patterns = {}

    for index, intent in enumerate(payload["intents"], start=1):
        if not isinstance(intent, dict):
            raise ValueError(f"Intent {index} must be a JSON object.")

        tag = intent.get("tag")
        answer = intent.get("answer")
        patterns = intent.get("patterns")

        if not isinstance(tag, str) or not tag.strip():
            raise ValueError(f"Intent {index} must have a non-empty string tag.")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                f"Intent '{tag}' must have a non-empty string answer."
            )
        if not isinstance(patterns, list):
            raise ValueError(f"Intent '{tag}' must contain a patterns list.")

        if tag in responses:
            raise ValueError(
                f"Duplicate intent tag found: {tag}"
            )

        valid_patterns = []

        for pattern_index, pattern in enumerate(patterns, start=1):
            if not isinstance(pattern, str) or not pattern.strip():
                raise ValueError(
                    f"Pattern {pattern_index} in intent '{tag}' "
                    "must be a non-empty string."
                )

            pattern = " ".join(pattern.split())
            normalized = pattern.casefold()
            previous = seen_patterns.get(normalized)
            if previous is not None:
                previous_tag, previous_pattern = previous
                if previous_tag != tag:
                    raise ValueError(
                        f"Pattern '{pattern}' is duplicated across intents "
                        f"'{previous_tag}' and '{tag}'."
                    )
                print(
                    f"Warning: duplicate pattern skipped: {pattern}"
                )
                continue

            seen_patterns[normalized] = (tag, pattern)
            valid_patterns.append(pattern)

        if len(valid_patterns) < MINIMUM_PATTERNS_PER_INTENT:
            raise ValueError(
                f"Intent '{tag}' has only "
                f"{len(valid_patterns)} unique patterns. "
                f"At least {MINIMUM_PATTERNS_PER_INTENT} are required."
            )

        responses[tag] = {
            "answer": answer.strip()
        }

        # Train on the same normalised text the assistant classifies (typos and Tagalog mapped).
        texts.extend(normalize_message(pattern) for pattern in valid_patterns)
        labels.extend([tag] * len(valid_patterns))

    if not responses:
        raise ValueError(
            "No valid intents were found in the dataset."
        )

    counts = Counter(labels)

    if len(counts) < 2:
        raise ValueError(
            "At least two different intents are required."
        )

    smallest_class = min(counts.values())

    if smallest_class < 2:
        raise ValueError(
            "Every intent needs at least two examples "
            "for a stratified evaluation."
        )

    return texts, labels, responses, counts


def create_pipeline():
    return Pipeline([
        (
            "vectorizer",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ])


def evaluate_model(texts, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    evaluation_pipeline = create_pipeline()
    evaluation_pipeline.fit(X_train, y_train)

    predictions = evaluation_pipeline.predict(X_test)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    labels_order = sorted(set(labels))

    report = classification_report(
        y_test,
        predictions,
        labels=labels_order,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels_order
    )

    evaluation = {
        "total_examples": len(texts),
        "training_examples": len(X_train),
        "test_examples": len(X_test),
        "intent_count": len(labels_order),
        "examples_per_intent": {
            tag: labels.count(tag)
            for tag in labels_order
        },
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "most_frequent_baseline": {
            "accuracy": round(
                float(accuracy_score(y_test, baseline_predictions)),
                4,
            ),
            "macro_f1": round(
                float(f1_score(
                    y_test,
                    baseline_predictions,
                    average="macro",
                    zero_division=0,
                )),
                4,
            ),
        },
        "classification_report": report,
        "confusion_matrix_labels": labels_order,
        "confusion_matrix": matrix.tolist(),
        "note": (
            "Evaluation uses a stratified holdout from manually "
            "authored examples. Similar wording across examples "
            "may make these scores higher than real-world results. "
            "Test with independently collected questions before "
            "production deployment."
        )
    }

    return evaluation


def independent_comparison(texts, labels):
    """Accuracy of the candidate (trained on all examples) and the current model on
    data/evaluation/intent_holdout_v1.json, or None when there is no held-out file or no current model.
    The held-out set is used for this model selection, so its score is not an unbiased final test."""
    if not HOLDOUT_PATH.exists() or not MODEL_PATH.exists():
        return None
    examples = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))["examples"]
    held_out = [normalize_message(item["text"]) for item in examples]
    expected = [item["intent"] for item in examples]
    training = {" ".join(text.casefold().split()) for text in texts}
    leaked = [text for text in held_out if " ".join(text.casefold().split()) in training]
    if leaked:
        return {"error": f"{len(leaked)} held-out question(s) appear in the training data; remove them before comparing."}
    candidate = create_pipeline().fit(texts, labels)
    current = joblib.load(MODEL_PATH)["pipeline"]
    score = lambda pipeline: round(sum(p == e for p, e in zip(pipeline.predict(held_out), expected)) / len(expected), 4)
    return {"held_out_file": str(HOLDOUT_PATH.relative_to(BASE_DIR)), "examples": len(expected),
            "candidate_accuracy": score(candidate), "current_accuracy": score(current)}


def train_final_model(texts, labels, responses, evaluation):
    final_pipeline = create_pipeline()

    final_pipeline.fit(
        texts,
        labels
    )

    artifact = {
        "pipeline": final_pipeline,
        "responses": responses,
        "model_type": (
            "custom_tfidf_logistic_regression_intent_classifier"
        ),
        "external_ai_provider": False,
        "pretrained_language_model": False,
        "training_metadata": evaluation
    }

    temporary_path = MODEL_PATH.with_suffix(".joblib.tmp")
    joblib.dump(artifact, temporary_path)
    temporary_path.replace(MODEL_PATH)


def _train():
    record_training_status(
        "chatbot_intent_classifier",
        {
            "status": "running",
            "task": "custom automotive chatbot intent classification",
            "eligible_for_inference": False,
            "reason": "Chatbot classifier training and holdout evaluation are in progress.",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    try:
        texts, labels, responses, counts = load_training_data()
    except (OSError, ValueError) as error:
        record_training_status(
            "chatbot_intent_classifier",
            {
                "status": "failed",
                "task": "automotive chatbot intent classification",
                "eligible_for_inference": False,
                "reason": str(error),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("CUSTOM AUTOMOTIVE CHATBOT TRAINING")
    print("=" * 40)
    print(f"Dataset: {DATA_PATH}")
    print(f"Total examples: {len(texts)}")
    print(f"Intent count: {len(responses)}")
    print()
    print("Examples per intent:")

    for tag, count in sorted(counts.items()):
        print(f"  {tag}: {count}")

    print()
    print("Evaluating the classifier...")

    evaluation = evaluate_model(
        texts,
        labels
    )
    evaluation.update({
        "model_name": "chatbot_intent_classifier",
        "model_version": "1.0",
        "task": "custom automotive intent classification",
        "model_type": "TF-IDF + LogisticRegression",
        "feature_schema": {
            "vectorizer": "TfidfVectorizer",
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": [1, 2],
            "sublinear_tf": True,
        },
        "model_parameters": {
            "classifier": "LogisticRegression",
            "max_iter": 3000,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },
        "training_data": str(DATA_PATH.relative_to(BASE_DIR)),
        "training_data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "scikit_learn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "acceptance_criteria": {
            "minimum_macro_f1": 0.75,
            "candidate_macro_f1": evaluation["macro_f1"],
            "baseline_macro_f1": evaluation["most_frequent_baseline"]["macro_f1"],
            "acceptance_criteria_passed": (
                evaluation["macro_f1"] >= 0.75
                and evaluation["macro_f1"]
                > evaluation["most_frequent_baseline"]["macro_f1"]
            ),
            "scope": "Internal authored-intent routing only; not maintenance prediction.",
        },
    })

    print()
    print(
        f"Holdout accuracy: "
        f"{evaluation['accuracy']:.2%}"
    )
    print(
        f"Macro F1: "
        f"{evaluation['macro_f1']:.2%}"
    )
    print(
        f"Weighted F1: "
        f"{evaluation['weighted_f1']:.2%}"
    )
    print(
        "Most-frequent baseline macro F1: "
        f"{evaluation['most_frequent_baseline']['macro_f1']:.2%}"
    )
    print("Per-intent results:")

    for tag in sorted(set(labels)):
        metrics = evaluation["classification_report"][tag]

        print(
            f"  {tag}: "
            f"precision={metrics['precision']:.2f}, "
            f"recall={metrics['recall']:.2f}, "
            f"f1={metrics['f1-score']:.2f}, "
            f"support={int(metrics['support'])}"
        )

    print()
    print("Training the final classifier on all examples...")

    current_report = None
    try:
        current_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        pass
    candidate_is_accepted = evaluation["acceptance_criteria"][
        "acceptance_criteria_passed"
    ]
    existing_score = (
        current_report.get("macro_f1")
        if isinstance(current_report, dict)
        else None
    )
    # The internal split changes whenever the training data changes, so the candidate's and the
    # saved model's split scores are only comparable on identical data. With different data, both
    # models are compared on the independent held-out questions instead.
    same_data = isinstance(current_report, dict) and current_report.get("training_data_sha256") == evaluation["training_data_sha256"]
    comparison = None if same_data else independent_comparison(texts, labels)
    if comparison is not None:
        evaluation["acceptance_criteria"]["independent_holdout"] = comparison
    if candidate_is_accepted and comparison is not None:
        if comparison.get("error"):
            candidate_is_accepted = False
            evaluation["acceptance_criteria"]["rejection_reason"] = comparison["error"]
        elif comparison["candidate_accuracy"] < comparison["current_accuracy"]:
            candidate_is_accepted = False
            evaluation["acceptance_criteria"]["rejection_reason"] = (
                "Candidate is less accurate than the current model on the independent held-out questions."
            )
    elif (
        candidate_is_accepted
        and isinstance(existing_score, (int, float))
        and evaluation["macro_f1"] < existing_score
    ):
        candidate_is_accepted = False
        evaluation["acceptance_criteria"]["rejection_reason"] = (
            "Candidate macro F1 is lower than the current saved evaluation."
        )
    evaluation["acceptance_criteria"]["acceptance_criteria_passed"] = candidate_is_accepted

    if candidate_is_accepted:
        train_final_model(texts, labels, responses, evaluation)
        evaluation["artifact_path"] = str(MODEL_PATH.relative_to(BASE_DIR))
        evaluation["artifact_sha256"] = hashlib.sha256(
            MODEL_PATH.read_bytes()
        ).hexdigest()
        temporary_legacy_report = REPORT_PATH.with_suffix(".json.tmp")
        temporary_legacy_report.write_text(
            json.dumps(evaluation, indent=2) + "\n", encoding="utf-8"
        )
        temporary_legacy_report.replace(REPORT_PATH)

    TRAINING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_report_path = TRAINING_REPORT_PATH.with_suffix(".json.tmp")
    temporary_report_path.write_text(
        json.dumps(evaluation, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_report_path.replace(TRAINING_REPORT_PATH)
    record_training_status(
        "chatbot_intent_classifier",
        {
            "status": (
                "evaluated_accepted" if candidate_is_accepted
                else "evaluated_rejected"
            ),
            "task": "custom automotive chatbot intent classification",
            "dataset": str(DATA_PATH.relative_to(BASE_DIR)),
            "source_sha256": evaluation["training_data_sha256"],
            "training_records": evaluation["training_examples"],
            "test_records": evaluation["test_examples"],
            "model_version": evaluation["model_version"],
            "feature_schema": evaluation["feature_schema"],
            "artifact_sha256": evaluation.get("artifact_sha256"),
            "metrics": {
                "accuracy": evaluation["accuracy"],
                "macro_f1": evaluation["macro_f1"],
                "weighted_f1": evaluation["weighted_f1"],
                "baseline_accuracy": evaluation["most_frequent_baseline"]["accuracy"],
                "baseline_macro_f1": evaluation["most_frequent_baseline"]["macro_f1"],
            },
            "artifact_path": evaluation.get("artifact_path"),
            "eligible_for_inference": candidate_is_accepted,
            "reason": (
                "Authored-intent holdout met the internal macro-F1 threshold."
                if candidate_is_accepted
                else evaluation["acceptance_criteria"].get(
                    "rejection_reason",
                    "Authored-intent holdout did not meet the internal threshold.",
                )
            ),
            "updated_at": evaluation["trained_at"],
        },
    )

    print()
    if candidate_is_accepted:
        print(f"Model saved: {MODEL_PATH}")
    else:
        print("Candidate rejected; the existing chatbot model was preserved.")
    print(f"Evaluation report saved: {REPORT_PATH}")
    print()
    print(
        "Training completed. The evaluation score is a baseline, "
        "not a guarantee of real-world accuracy."
    )


def main():
    try:
        _train()
    except Exception as error:
        record_training_status(
            "chatbot_intent_classifier",
            {
                "status": "failed",
                "task": "automotive chatbot intent classification",
                "eligible_for_inference": False,
                "reason": str(error),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    main()
