"""Metric helpers shared by the training pipelines.

Everything here works on plain arrays so the same code scores models and
baselines. Metrics that are undefined for a split (for example ROC-AUC when
the test groups contain only one class) are returned as None, never guessed.
"""
import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.025), 3)


def expected_calibration_error(y_true, probability, bins=10):
    y_true = np.asarray(y_true, dtype=float)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & ((probability < high) if high < 1 else (probability <= high))
        if mask.any():
            error += mask.mean() * abs(probability[mask].mean() - y_true[mask].mean())
    return float(error)


def reliability_table(y_true, probability, bins=10):
    y_true = np.asarray(y_true, dtype=float)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & ((probability < high) if high < 1 else (probability <= high))
        if mask.any():
            rows.append({
                "bin": f"{low:.1f}-{high:.1f}",
                "count": int(mask.sum()),
                "mean_predicted": round(float(probability[mask].mean()), 4),
                "observed_rate": round(float(y_true[mask].mean()), 4),
            })
    return rows


def best_threshold(y_true, probability, beta=1.0):
    """Threshold maximising F-beta on out-of-fold probabilities (ties -> higher threshold)."""
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        return 0.5
    best_score, best = -1.0, 0.5
    weight = beta ** 2
    for threshold in THRESHOLD_GRID:
        predicted = probability >= threshold
        true_positive = float((predicted & (y_true == 1)).sum())
        precision = true_positive / predicted.sum() if predicted.sum() else 0.0
        recall = true_positive / y_true.sum()
        score = 0.0 if precision + recall == 0 else (1 + weight) * precision * recall / (weight * precision + recall)
        if score >= best_score:
            best_score, best = score, float(threshold)
    return best


def _round(value):
    return None if value is None or (isinstance(value, float) and not math.isfinite(value)) else round(float(value), 4)


def binary_metrics(y_true, probability, threshold):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    predicted = (probability >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "records": int(len(y_true)),
        "positives": int(y_true.sum()),
        "threshold": _round(threshold),
        "precision": _round(precision_score(y_true, predicted, zero_division=0)),
        "recall": _round(recall_score(y_true, predicted, zero_division=0)),
        "f1": _round(f1_score(y_true, predicted, zero_division=0)),
        "balanced_accuracy": _round(balanced_accuracy_score(y_true, predicted)) if both_classes else None,
        "pr_auc": _round(average_precision_score(y_true, probability)) if y_true.sum() else None,
        "roc_auc": _round(roc_auc_score(y_true, probability)) if both_classes else None,
        "brier": _round(brier_score_loss(y_true, probability, pos_label=1)) if both_classes else _round(np.mean((probability - y_true) ** 2)),
        "ece": _round(expected_calibration_error(y_true, probability)),
        "prevalence": _round(y_true.mean()),
    }


def macro(per_target, key):
    values = [item[key] for item in per_target.values() if item.get(key) is not None]
    return _round(np.mean(values)) if values else None
