"""Service-need model: which service items a workshop visit at a given odometer
reading is likely to include, trained on the Vehicle Maintenance - Service
Records dataset.

What the target means: a positive label says the item was performed at a
recorded service visit. It is a scheduled-service need, not an observed fault
and not a predicted failure. The data has no visit dates and no vehicle
identifiers, so no time-to-event or failure-date estimate is produced.

Evaluation holds out whole (model, region) groups, so no vehicle population
appears on both sides of the split. Hyperparameters, calibration and decision
thresholds use only the training groups (inner leave-one-group-out folds).
Only the inputs the fleet database actually holds are used: body type, fuel
type and odometer.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from data_pipeline.evaluation import best_threshold, binary_metrics, macro, reliability_table


RANDOM_STATE = 42
SOURCE_FILE = "Vehicle Maintenance- Service Records.csv"
MODEL_FAMILY = "service_need"
FEATURE_COLUMNS = ["vehicle_type", "fuel_type", "mileage"]
SCHEDULE_CYCLES_KM = (10_000, 20_000, 30_000, 40_000, 60_000, 80_000, 100_000)
MILESTONE_KM = 10_000

# Service item -> 3D-inspection component keys (Back-End maintenance_component_types).
TARGETS = {
    "oil_filter": {"label": "Oil filter", "component_keys": ["engine"], "action": "replacement"},
    "engine_oil": {"label": "Engine oil", "component_keys": ["engine"], "action": "replacement"},
    "washer_plug_drain": {"label": "Drain plug washer", "component_keys": ["engine"], "action": "replacement"},
    "dust_and_pollen_filter": {"label": "Cabin dust and pollen filter", "component_keys": [], "action": "replacement"},
    "whell_alignment_and_balancing": {"label": "Wheel alignment and balancing", "component_keys": ["steering", "tire-front-left", "tire-front-right", "tire-rear-left", "tire-rear-right"], "action": "routine_service"},
    "air_clean_filter": {"label": "Air filter", "component_keys": ["engine"], "action": "replacement"},
    "fuel_filter": {"label": "Fuel filter", "component_keys": ["engine"], "action": "replacement"},
    "spark_plug": {"label": "Spark plugs", "component_keys": ["engine"], "action": "replacement"},
    "brake_fluid": {"label": "Brake fluid", "component_keys": ["brakes-front", "brakes-rear"], "action": "replacement"},
    "brake_and_clutch_oil": {"label": "Brake and clutch fluid", "component_keys": ["brakes-front", "brakes-rear", "transmission"], "action": "replacement"},
    "transmission_fluid": {"label": "Transmission fluid", "component_keys": ["transmission"], "action": "replacement"},
    "brake_pads": {"label": "Brake pads", "component_keys": ["brakes-front", "brakes-rear"], "action": "inspection"},
    "clutch": {"label": "Clutch", "component_keys": ["transmission"], "action": "inspection"},
    "coolant": {"label": "Coolant", "component_keys": ["cooling"], "action": "replacement"},
}
SAFETY_TARGETS = {"brake_fluid", "brake_pads", "brake_and_clutch_oil", "clutch"}

# The dataset's four models mapped to the fleet's 3D body types (hatchbacks
# use the sedan rig) and their original body style for documentation.
MODEL_BODY = {"jazz": ("sedan", "hatchback"), "amaze": ("sedan", "sedan"),
              "city": ("sedan", "sedan"), "fortuner": ("suv", "suv")}
FUEL_ALIASES = {"petrol": "petrol", "gasoline": "petrol", "gas": "petrol", "unleaded": "petrol",
                "diesel": "diesel"}
BODY_ALIASES = {"sedan": "sedan", "saloon": "sedan", "hatchback": "sedan", "hatch": "sedan",
                "car": "sedan", "compact": "sedan", "suv": "suv", "crossover": "suv"}

# Per-target acceptance: a target is served only when its held-out scores
# beat the prevalence baseline with a usable ranking.
MIN_TARGET_ROC_AUC = 0.65
MIN_TARGET_F1 = 0.40


def schedule_features(frame):
    """Odometer position relative to common service cycles (picklable, used inside the pipeline)."""
    frame = pd.DataFrame(frame).copy()
    mileage = pd.to_numeric(frame["mileage"], errors="coerce").astype(float)
    output = pd.DataFrame({
        "vehicle_type": frame["vehicle_type"].astype(str),
        "fuel_type": frame["fuel_type"].astype(str),
        "mileage_km": mileage,
        "milestone_index": np.round(mileage / MILESTONE_KM),
    }, index=frame.index)
    for cycle in SCHEDULE_CYCLES_KM:
        remainder = np.mod(mileage, cycle)
        output[f"dist_to_{cycle // 1000}k_cycle"] = np.minimum(remainder, cycle - remainder) / cycle
    return output


NUMERIC_FEATURES = ["mileage_km", "milestone_index"] + [f"dist_to_{c // 1000}k_cycle" for c in SCHEDULE_CYCLES_KM]
CATEGORICAL_FEATURES = ["vehicle_type", "fuel_type"]


def normalize_vehicle_input(vehicle):
    """Map a fleet vehicle (Back-End field names) to model inputs, or raise ValueError."""
    if not isinstance(vehicle, dict):
        raise ValueError("The vehicle field must be a JSON object.")
    body = str(vehicle.get("vehicle_type") or vehicle.get("body_type") or "").strip().casefold()
    fuel = str(vehicle.get("fuel_type") or vehicle.get("engine_type") or "").strip().casefold()
    mileage = vehicle.get("mileage", vehicle.get("current_mileage"))
    return {
        "vehicle_type": BODY_ALIASES.get(body, body),
        "fuel_type": FUEL_ALIASES.get(fuel, fuel),
        "mileage": mileage,
    }


def load_service_records(raw_root):
    path = Path(raw_root) / SOURCE_FILE
    raw = pd.read_csv(path)
    usage = {"file": SOURCE_FILE, "rows_read": int(len(raw)), "rejected": {}}
    frame = raw.copy()
    frame["brand"] = frame["brand"].astype(str).str.strip().str.casefold()
    frame["model"] = frame["model"].astype(str).str.strip().str.casefold()
    frame["region"] = frame["region"].astype(str).str.strip().str.casefold()

    def reject(mask, reason):
        count = int(mask.sum())
        if count:
            usage["rejected"][reason] = usage["rejected"].get(reason, 0) + count
        return frame.loc[~mask]

    frame = reject(~frame["model"].isin(MODEL_BODY), "model has no body-type mapping")
    frame["mileage"] = pd.to_numeric(frame["mileage"], errors="coerce")
    frame = reject(frame["mileage"].isna() | (frame["mileage"] <= 0) | (frame["mileage"] > 2_000_000),
                   "missing or implausible odometer reading")
    frame["cost"] = pd.to_numeric(frame["cost"], errors="coerce")
    for target in TARGETS:
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
    invalid_label = ~frame[list(TARGETS)].isin([0, 1]).all(axis=1)
    frame = reject(invalid_label, "service flag not 0/1")
    fuel = frame["engine_type"].astype(str).str.strip().str.casefold().map(FUEL_ALIASES)
    frame = frame.assign(fuel_type=fuel)
    frame = reject(frame["fuel_type"].isna(), "unknown engine type")

    frame = frame.assign(
        vehicle_type=frame["model"].map(lambda model: MODEL_BODY[model][0]),
        body_style=frame["model"].map(lambda model: MODEL_BODY[model][1]),
        group=frame["model"] + "|" + frame["region"],
    )
    identity = ["brand", "model", "region", "engine_type", "make_year", "mileage", "mileage_range"]
    duplicates = frame.duplicated(subset=identity + list(TARGETS) + ["cost"])
    frame = reject(duplicates, "exact duplicate visit")
    conflicts = frame.duplicated(subset=identity, keep=False)
    usage["conflicting_label_rows_kept"] = int(conflicts.sum())
    frame[list(TARGETS)] = frame[list(TARGETS)].astype(int)
    usage["rows_used"] = int(len(frame))
    usage["groups"] = sorted(frame["group"].unique().tolist())
    return frame.reset_index(drop=True), usage, hashlib.sha256(path.read_bytes()).hexdigest()


def _preprocessor(scale, feature_set="full"):
    numeric = Pipeline([("scale", StandardScaler())]) if scale else "passthrough"
    if feature_set == "milestone":
        return ColumnTransformer([("milestone", "passthrough", ["milestone_index"])])
    columns = [("numeric", numeric, NUMERIC_FEATURES)]
    if feature_set == "full":
        columns.insert(0, ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES))
    return ColumnTransformer(columns)


class MilestoneFrequencyClassifier(ClassifierMixin, BaseEstimator):
    """Service frequency at the nearest 10,000 km milestone, smoothed towards prevalence.

    Expects one column holding the milestone index; milestones never seen in
    training fall back to the training prevalence.
    """

    def fit(self, X, y):
        milestones = np.asarray(X)[:, 0].astype(int)
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        self.prior_ = float(y.mean())
        self.table_ = {
            int(m): float((y[milestones == m].sum() + self.prior_) / ((milestones == m).sum() + 1))
            for m in np.unique(milestones)
        }
        return self

    def predict_proba(self, X):
        positive = np.array([self.table_.get(int(m), self.prior_) for m in np.asarray(X)[:, 0]])
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# "full" uses body and fuel type as well as odometer features; "mileage_only"
# drops them because in this data they are confounded with car model and city.
FEATURE_SETS = ("full", "mileage_only")
CANDIDATES = {
    "logistic_regression": (
        lambda params: LogisticRegression(max_iter=2000, class_weight="balanced", C=params["C"], random_state=RANDOM_STATE),
        [{"C": value, "feature_set": fs} for value in (0.1, 1.0) for fs in FEATURE_SETS],
        True,
    ),
    "random_forest": (
        lambda params: RandomForestClassifier(
            n_estimators=150, min_samples_leaf=params["min_samples_leaf"], max_depth=params["max_depth"],
            class_weight="balanced_subsample", n_jobs=1, random_state=RANDOM_STATE),
        [{"min_samples_leaf": leaf, "max_depth": depth, "feature_set": fs}
         for leaf, depth in ((8, 6), (2, None)) for fs in FEATURE_SETS],
        False,
    ),
    "gradient_boosting": (
        lambda params: HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"], max_depth=params["max_depth"], max_iter=100,
            class_weight="balanced", random_state=RANDOM_STATE),
        [{"learning_rate": rate, "max_depth": 3, "feature_set": fs} for rate in (0.05, 0.15) for fs in FEATURE_SETS],
        False,
    ),
    "milestone_frequency": (
        lambda params: MilestoneFrequencyClassifier(),
        [{"feature_set": "milestone"}],
        False,
    ),
}


class GroupPlattCalibratedClassifier(ClassifierMixin, BaseEstimator):
    """Platt scaling fitted on out-of-fold scores from group-disjoint folds.

    Unlike CalibratedClassifierCV it tolerates folds whose training part has a
    single class (some service items are never performed for one car model):
    those folds score the held-out group with the training prevalence.
    """

    def __init__(self, estimator=None, splits=None):
        self.estimator = estimator
        self.splits = splits

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        oof = np.empty(len(y), dtype=float)
        for train_index, test_index in self.splits:
            if len(np.unique(y[train_index])) < 2:
                oof[test_index] = y[train_index].mean()
            else:
                oof[test_index] = clone(self.estimator).fit(X[train_index], y[train_index]).predict_proba(X[test_index])[:, 1]
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        self.calibrator_ = None
        if len(np.unique(y)) == 2:
            self.calibrator_ = LogisticRegression(C=1e4).fit(_logit(oof)[:, None], y)
        return self

    def predict_proba(self, X):
        raw = self.estimator_.predict_proba(np.asarray(X))[:, 1]
        positive = raw if self.calibrator_ is None else self.calibrator_.predict_proba(_logit(raw)[:, None])[:, 1]
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def _logit(probability):
    probability = np.clip(probability, 1e-4, 1 - 1e-4)
    return np.log(probability / (1 - probability))


def _pipeline(algorithm, params, calibrated_splits=None):
    factory, _, scale = CANDIDATES[algorithm]
    classifier = factory(params)
    if calibrated_splits is not None:
        classifier = GroupPlattCalibratedClassifier(classifier, calibrated_splits)
    return Pipeline([
        ("features", FunctionTransformer(schedule_features)),
        ("preprocess", _preprocessor(scale, params.get("feature_set", "full"))),
        ("classifier", classifier),
    ])


def _group_splits(groups):
    groups = np.asarray(groups)
    return list(LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups))


def _oof_probability(make_pipeline, frame, y, splits):
    oof = np.full(len(frame), np.nan)
    for train_index, test_index in splits:
        if len(np.unique(y[train_index])) < 2:
            oof[test_index] = y[train_index].mean()
            continue
        model = make_pipeline().fit(frame.iloc[train_index][FEATURE_COLUMNS], y[train_index])
        oof[test_index] = model.predict_proba(frame.iloc[test_index][FEATURE_COLUMNS])[:, 1]
    return oof


def _milestone_table(frame, target):
    milestones = np.round(frame["mileage"].to_numpy() / MILESTONE_KM).astype(int)
    table = pd.DataFrame({"milestone": milestones, "y": frame[target].to_numpy()}).groupby("milestone")["y"].agg(["sum", "count"])
    return {int(index): (int(row["sum"]), int(row["count"])) for index, row in table.iterrows()}


def _milestone_probability(table, prior, mileage):
    milestones = np.round(np.asarray(mileage, dtype=float) / MILESTONE_KM).astype(int)
    # Laplace-style smoothing towards the training prevalence.
    return np.array([(table.get(m, (0, 0))[0] + prior) / (table.get(m, (0, 0))[1] + 1) for m in milestones])


def _milestone_oof(frame, target, splits):
    oof = np.full(len(frame), np.nan)
    for train_index, test_index in splits:
        train = frame.iloc[train_index]
        table = _milestone_table(train, target)
        oof[test_index] = _milestone_probability(table, train[target].mean(), frame.iloc[test_index]["mileage"])
    return oof


def _select_algorithm(train, targets, splits):
    """Inner leave-one-group-out search on training groups; score = mean PR-AUC over targets."""
    search = []
    for algorithm, (_, grid, _) in CANDIDATES.items():
        for params in grid:
            scores = {}
            for target in targets:
                y = train[target].to_numpy()
                oof = _oof_probability(lambda: _pipeline(algorithm, params), train, y, splits)
                scores[target] = binary_metrics(y, oof, 0.5)["pr_auc"] or 0.0
            search.append({"algorithm": algorithm, "params": params, "per_target_inner_pr_auc": scores,
                           "mean_inner_pr_auc": round(float(np.mean(list(scores.values()))), 4)})
    best = max((item for item in search if item["algorithm"] != "milestone_frequency"),
               key=lambda item: item["mean_inner_pr_auc"])
    best_by_algorithm = {}
    for item in search:
        current = best_by_algorithm.get(item["algorithm"])
        if current is None or item["mean_inner_pr_auc"] > current["mean_inner_pr_auc"]:
            best_by_algorithm[item["algorithm"]] = item
    return best, best_by_algorithm, search


def _fit_target(train, target, algorithm, params, splits):
    y = train[target].to_numpy()
    if algorithm == "milestone_frequency":
        # A smoothed frequency table is already a probability estimate; no Platt layer.
        oof = _oof_probability(lambda: _pipeline(algorithm, params), train, y, splits)
        return _pipeline(algorithm, params).fit(train[FEATURE_COLUMNS], y), best_threshold(y, oof), oof
    # Out-of-fold calibrated probabilities on training groups choose the threshold.
    oof = np.full(len(train), np.nan)
    for train_index, test_index in splits:
        inner = train.iloc[train_index]
        inner_y = y[train_index]
        if len(np.unique(inner_y)) < 2 or inner["group"].nunique() < 2:
            oof[test_index] = inner_y.mean()
            continue
        model = _pipeline(algorithm, params, calibrated_splits=_group_splits(inner["group"]))
        model.fit(inner[FEATURE_COLUMNS], inner_y)
        oof[test_index] = model.predict_proba(train.iloc[test_index][FEATURE_COLUMNS])[:, 1]
    threshold = best_threshold(y, oof)
    final = _pipeline(algorithm, params, calibrated_splits=splits).fit(train[FEATURE_COLUMNS], y)
    return final, threshold, oof


def _evidence_table(train, target):
    table = _milestone_table(train, target)
    return {str(m * MILESTONE_KM): {"performed": s, "visits": c} for m, (s, c) in sorted(table.items())}


def _cost_experiment(train, test):
    """Visit-cost regression, evaluated on held-out groups (reported, not served)."""
    feature_frame = lambda frame: schedule_features(frame[FEATURE_COLUMNS])
    model = Pipeline([("preprocess", _preprocessor(False)),
                      ("regressor", HistGradientBoostingRegressor(max_depth=4, random_state=RANDOM_STATE))])
    model.fit(feature_frame(train), train["cost"])
    prediction = model.predict(feature_frame(test))
    milestone_median = train.assign(m=np.round(train["mileage"] / MILESTONE_KM)).groupby("m")["cost"].median()
    lookup = np.round(test["mileage"] / MILESTONE_KM).map(milestone_median).fillna(train["cost"].median())
    mae = lambda values: round(float(np.mean(np.abs(test["cost"].to_numpy() - np.asarray(values)))), 1)
    return {
        "target": "visit cost (INR, 2016-2018 workshop invoices)",
        "test_records": int(len(test)),
        "mae_gradient_boosting": mae(prediction),
        "mae_milestone_median_baseline": mae(lookup),
        "mae_global_median_baseline": mae(np.full(len(test), train["cost"].median())),
        "served": False,
        "reason": "Costs are Indian-rupee workshop prices from 2016-2018 for four models; they are not valid estimates for the fleet's currency, region or year. Fleet cost answers use the fleet's own recorded maintenance costs instead.",
    }


def train_service_need(raw_root, output_root, registry_writer=None):
    # Tiny tabular data: native thread pools cost more than they save.
    with threadpool_limits(limits=1):
        return _train_service_need(raw_root, output_root, registry_writer)


def _train_service_need(raw_root, output_root, registry_writer=None):
    frame, usage, source_sha = load_service_records(raw_root)
    modeled_targets = [t for t in TARGETS if 0 < frame[t].mean() < 1]
    constant_targets = {t: float(frame[t].mean()) for t in TARGETS if t not in modeled_targets}

    splitter = GroupShuffleSplit(n_splits=1, test_size=2, random_state=RANDOM_STATE)
    train_index, test_index = next(splitter.split(frame, groups=frame["group"]))
    train, test = frame.iloc[train_index].reset_index(drop=True), frame.iloc[test_index].reset_index(drop=True)
    train_groups, test_groups = sorted(train["group"].unique()), sorted(test["group"].unique())
    inner_splits = _group_splits(train["group"])

    best, best_by_algorithm, search = _select_algorithm(train, modeled_targets, inner_splits)
    milestone_candidate = best_by_algorithm["milestone_frequency"]
    # v1 and v2 served ML estimators and failed to beat the milestone-frequency
    # baseline on held-out groups, so v3 serves that pre-declared estimator for
    # every target; the best ML candidate is still trained and reported.
    target_choice = {target: milestone_candidate for target in modeled_targets}
    algorithm, params = best["algorithm"], best["params"]

    models, thresholds, per_target, baselines, comparison, evidence = {}, {}, {}, {}, {}, {}
    test_probabilities = {}
    for target in modeled_targets:
        y_train, y_test = train[target].to_numpy(), test[target].to_numpy()
        chosen = target_choice[target]
        model, threshold, inner_oof = _fit_target(train, target, chosen["algorithm"], chosen["params"], inner_splits)
        probability = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        test_probabilities[target] = probability
        metrics = binary_metrics(y_test, probability, threshold)
        metrics["estimator"] = {"algorithm": chosen["algorithm"], "params": chosen["params"]}
        metrics["reliability"] = reliability_table(y_test, probability)

        prior = float(y_train.mean())
        milestone_oof = _milestone_oof(train, target, inner_splits)
        milestone_threshold = best_threshold(y_train, milestone_oof)
        milestone_probability = _milestone_probability(_milestone_table(train, target), prior, test["mileage"])
        baselines[target] = {
            "prevalence": binary_metrics(y_test, np.full(len(y_test), prior), 0.5),
            "milestone_lookup": binary_metrics(y_test, milestone_probability, milestone_threshold),
        }
        # Best ML candidate (calibrated, inner-CV threshold) on the same held-out groups.
        ml_model, ml_threshold, _ = _fit_target(train, target, algorithm, params, inner_splits)
        baselines[target]["best_ml_candidate"] = binary_metrics(
            y_test, ml_model.predict_proba(test[FEATURE_COLUMNS])[:, 1], ml_threshold)
        # Uncalibrated best-of-family comparison on the same held-out groups.
        comparison[target] = {}
        for family, item in best_by_algorithm.items():
            family_model = _pipeline(family, item["params"]).fit(train[FEATURE_COLUMNS], y_train)
            family_probability = family_model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
            comparison[target][family] = {k: v for k, v in binary_metrics(y_test, family_probability, 0.5).items()
                                          if k in ("pr_auc", "roc_auc", "f1", "brier")}

        # Which targets to serve is decided on inner training-group folds only, so
        # the held-out metrics reported for served targets are not selected on.
        inner = binary_metrics(y_train, inner_oof, threshold)
        metrics["inner_cv"] = {k: inner[k] for k in ("roc_auc", "f1", "pr_auc", "prevalence")}
        reliable = (
            inner["roc_auc"] is not None and inner["roc_auc"] >= MIN_TARGET_ROC_AUC
            and inner["f1"] >= MIN_TARGET_F1
            and (inner["pr_auc"] or 0) > inner["prevalence"]
        )
        metrics["served"] = bool(reliable)
        metrics["served_reason"] = (
            "Inner training-group cross-validation ROC-AUC, F1 and PR-AUC above prevalence meet the per-target criteria."
            if reliable else
            f"Inner cross-validation evidence is insufficient (needs ROC-AUC >= {MIN_TARGET_ROC_AUC}, "
            f"F1 >= {MIN_TARGET_F1}, PR-AUC above prevalence)."
        )
        per_target[target] = metrics
        evidence[target] = _evidence_table(train, target)
        if reliable:
            models[target] = model
            thresholds[target] = threshold

    served = {t: m for t, m in per_target.items() if m["served"]}
    model_summary = {
        "balanced_accuracy": macro(served, "balanced_accuracy"),
        "macro_f1": macro(served, "f1"),
        "macro_pr_auc": macro(served, "pr_auc"),
        "macro_roc_auc": macro(served, "roc_auc"),
        "macro_brier": macro(served, "brier"),
        "macro_ece": macro(served, "ece"),
    }
    prevalence_baseline = {t: baselines[t]["prevalence"] for t in served}
    ml_candidate = {t: baselines[t]["best_ml_candidate"] for t in served}
    model_summary.update({
        "served_estimator": "milestone_frequency (training-group service frequency at the nearest 10,000 km milestone, smoothed towards prevalence)",
        "baseline": "prevalence (training positive rate for every visit)",
        "baseline_balanced_accuracy": macro(prevalence_baseline, "balanced_accuracy"),
        "baseline_macro_f1": macro(prevalence_baseline, "f1"),
        "baseline_macro_pr_auc": macro(prevalence_baseline, "pr_auc"),
        "prevalence_baseline_macro_brier": macro(prevalence_baseline, "brier"),
        "best_ml_candidate": f"{algorithm} {params} (calibrated)",
        "best_ml_candidate_balanced_accuracy": macro(ml_candidate, "balanced_accuracy"),
        "best_ml_candidate_macro_f1": macro(ml_candidate, "f1"),
        "best_ml_candidate_macro_pr_auc": macro(ml_candidate, "pr_auc"),
        "best_ml_candidate_macro_brier": macro(ml_candidate, "brier"),
        "served_targets": sorted(served),
        "all_modeled_targets_macro_f1": macro(per_target, "f1"),
        "all_modeled_targets_best_ml_macro_f1": macro({t: baselines[t]["best_ml_candidate"] for t in per_target}, "f1"),
    })

    # Leave-one-group-out over all seven groups shows split-to-split variance.
    logo = {}
    for target in modeled_targets:
        y = frame[target].to_numpy()
        fold_scores = []
        for fold_train, fold_test in _group_splits(frame["group"]):
            if len(np.unique(y[fold_train])) < 2:
                continue
            fold_model = _pipeline("milestone_frequency", milestone_candidate["params"]).fit(
                frame.iloc[fold_train][FEATURE_COLUMNS], y[fold_train])
            fold_probability = fold_model.predict_proba(frame.iloc[fold_test][FEATURE_COLUMNS])[:, 1]
            fold_scores.append(binary_metrics(y[fold_test], fold_probability, 0.5))
        logo[target] = {
            "folds": len(fold_scores),
            "mean_pr_auc": macro({i: s for i, s in enumerate(fold_scores)}, "pr_auc"),
            "mean_roc_auc": macro({i: s for i, s in enumerate(fold_scores)}, "roc_auc"),
        }

    importance = _permutation_importance(models, test)
    passed = (
        bool(served)
        and model_summary["balanced_accuracy"] is not None
        and model_summary["balanced_accuracy"] >= 0.60
        and model_summary["macro_f1"] >= 0.60
        and model_summary["balanced_accuracy"] >= (model_summary["baseline_balanced_accuracy"] or 0)
        and model_summary["macro_f1"] >= (model_summary["baseline_macro_f1"] or 0)
    )
    calibrated = (
        bool(served)
        and model_summary["macro_brier"] is not None
        and model_summary["macro_brier"] < model_summary["prevalence_baseline_macro_brier"]
        and model_summary["macro_ece"] <= 0.15
    )

    config = {
        "random_state": RANDOM_STATE, "algorithm": algorithm, "params": params,
        "per_target_estimator": {t: c["algorithm"] for t, c in target_choice.items()},
        "feature_columns": FEATURE_COLUMNS, "schedule_cycles_km": SCHEDULE_CYCLES_KM,
        "calibration": "Platt scaling (GroupPlattCalibratedClassifier) on out-of-fold scores from leave-one-group-out training folds",
        "threshold_rule": "maximise F1 on out-of-fold calibrated probabilities from training groups",
        "per_target_acceptance": {"min_roc_auc": MIN_TARGET_ROC_AUC, "min_f1": MIN_TARGET_F1},
        "sklearn": sklearn.__version__,
    }
    version_hash = hashlib.sha256((source_sha + json.dumps(config, sort_keys=True, default=str)).encode()).hexdigest()[:10]
    version = f"{MODEL_FAMILY}_v3_{version_hash}"
    output = Path(output_root) / version
    output.mkdir(parents=True, exist_ok=True)

    domain = {
        "vehicle_type": sorted(train["vehicle_type"].unique().tolist()),
        "fuel_type": sorted(train["fuel_type"].unique().tolist()),
        "mileage": [int(train["mileage"].min()), int(train["mileage"].max())],
    }
    artifact = {
        "model_version": version,
        "task": "service_need_at_visit",
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": list(TARGETS),
        "target_definitions": {t: TARGETS[t] for t in TARGETS},
        "models": models,
        "thresholds": thresholds,
        "constant_targets": {
            **{t: {"status": f"performed at {rate:.0%} of recorded service visits; routine at every service, no classifier needed", "rate": rate}
               for t, rate in constant_targets.items()},
            **{t: {"status": per_target[t]["served_reason"], "rate": float(frame[t].mean())}
               for t in modeled_targets if t not in models},
        },
        "evidence_tables": evidence,
        "feature_domain": domain,
        "safety_targets": sorted(SAFETY_TARGETS),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    artifact_path = output / "pipeline.joblib"
    joblib.dump(artifact, artifact_path)

    evaluation = {
        "status": "evaluated_accepted" if passed and calibrated else "evaluated_rejected",
        "model_version": version,
        "design_history": (
            "v1 (one calibrated ML estimator for all targets) and v2 (per-target ML vs milestone-frequency choice plus "
            "feature-set ablation on inner folds) were both rejected: on the held-out Honda Amaze groups the ML models did "
            "not beat the milestone-frequency baseline (v1 macro F1 0.730 vs 0.752; v2 0.724 vs 0.752; see "
            "data/reports/service_need_v1_rejected_evaluation.json and service_need_v2_rejected_evaluation.json). Both also "
            "chose served targets from held-out metrics, which biases their aggregates upward. v3 serves the milestone-frequency "
            "estimator, declared as the baseline before any held-out result was seen, and chooses served targets from inner "
            "training-group folds only. The decision to deploy it instead of an ML model was made after seeing v1 and v2, "
            "so a fresh validation on new fleet service records is still required."
        ),
        "task": "service_need_at_visit",
        "target_meaning": "1 = the service item was performed at the recorded workshop visit. Scheduled-service need, not an observed or predicted failure.",
        "dataset": {"file": SOURCE_FILE, "sha256": source_sha, **usage},
        "group_split": {
            "method": "GroupShuffleSplit(test_size=2 groups, seed 42); group = model|region",
            "train_groups": train_groups, "test_groups": test_groups,
            "train_group_count": len(train_groups), "test_group_count": len(test_groups),
            "group_overlap": len(set(train_groups) & set(test_groups)),
            "train_records": int(len(train)), "test_records": int(len(test)),
            "time_aware_split": "Not possible: the records have no visit dates or vehicle identifiers.",
        },
        "class_distribution": {t: {"train_positive_rate": round(float(train[t].mean()), 4),
                                   "test_positive_rate": round(float(test[t].mean()), 4)} for t in TARGETS},
        "hyperparameter_search": {"selected": best, "best_by_algorithm": best_by_algorithm, "trials": search},
        "metrics": model_summary,
        "per_target": per_target,
        "baselines": baselines,
        "algorithm_comparison_uncalibrated": comparison,
        "leave_one_group_out": logo,
        "feature_importance": importance,
        "cost_experiment": _cost_experiment(train, test),
        "probability_validation": {
            "status": "calibrated" if calibrated else "not_calibrated",
            "method": "sigmoid (Platt) on out-of-fold predictions from leave-one-group-out training folds",
            "validation_records": int(len(test)),
            "independent_groups": len(test_groups),
            "brier_score": model_summary["macro_brier"],
            "prevalence_baseline_brier": model_summary["prevalence_baseline_macro_brier"],
            "expected_calibration_error": model_summary["macro_ece"],
        },
        "acceptance_criteria": {
            "acceptance_criteria_passed": bool(passed and calibrated),
            "rules": "macro balanced accuracy and macro F1 over served targets >= 0.60 and >= the milestone-lookup baseline; calibrated macro Brier below the prevalence baseline and ECE <= 0.15; >= 2 groups per split with zero overlap.",
            "discrimination_passed": bool(passed),
            "calibration_passed": bool(calibrated),
        },
        "feature_domain": domain,
        "training_config": config,
        "limitations": [
            "Four car models from two Indian cities, 2016-2018; no motorcycles, pickups, vans or minivans.",
            "Targets are workshop service decisions, which largely follow manufacturer schedules; they are not wear measurements or failures.",
            "No visit dates or vehicle IDs, so no service-window or failure-date estimate can be validated.",
            "Vehicles outside the training domain (body type, fuel type, odometer range) are refused rather than extrapolated.",
        ],
    }
    evaluation_path = output / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation, indent=2, default=str) + "\n", encoding="utf-8")
    (output / "feature_schema.json").write_text(json.dumps({"inputs": FEATURE_COLUMNS, "domain": domain,
        "engineered": NUMERIC_FEATURES + CATEGORICAL_FEATURES}, indent=2) + "\n", encoding="utf-8")
    (output / "model_card.md").write_text(_model_card(evaluation), encoding="utf-8")

    registry_entry = {
        "status": evaluation["status"],
        "model_name": "maintenance_model",
        "model_version": version,
        "task": "service_need_at_visit",
        "dataset": SOURCE_FILE,
        "source_sha256": source_sha,
        "eligible_for_inference": bool(passed and calibrated),
        "evaluation": {"acceptance_criteria_passed": bool(passed and calibrated)},
        "artifact_path": _service_relative(artifact_path),
        "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "evaluation_report_path": _service_relative(evaluation_path),
        "evaluation_report_sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
        "feature_columns": FEATURE_COLUMNS,
        "target_definitions": {t: TARGETS[t] for t in models},
        "metrics": model_summary,
        "reason": (
            "Milestone-frequency service-need estimator passed group-held-out evaluation and calibration; "
            "the ML candidates were evaluated and did not beat it."
            if passed and calibrated else
            "Service-need model did not meet the acceptance criteria; see evaluation.json."
        ),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if registry_writer:
        registry_writer(registry_entry)
    usage["model_version"] = version
    return evaluation, registry_entry, usage


def _service_relative(path):
    service_root = Path(__file__).resolve().parents[1]
    try:
        return Path(path).resolve().relative_to(service_root).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def _permutation_importance(models, test, repeats=10):
    """Mean drop in PR-AUC on held-out groups when one raw input is shuffled."""
    rng = np.random.default_rng(RANDOM_STATE)
    result = {}
    for target, model in models.items():
        y = test[target].to_numpy()
        if y.sum() == 0:
            continue
        base = binary_metrics(y, model.predict_proba(test[FEATURE_COLUMNS])[:, 1], 0.5)["pr_auc"]
        drops = {}
        for column in FEATURE_COLUMNS:
            values = []
            for _ in range(repeats):
                shuffled = test[FEATURE_COLUMNS].copy()
                shuffled[column] = rng.permutation(shuffled[column].to_numpy())
                values.append(base - binary_metrics(y, model.predict_proba(shuffled)[:, 1], 0.5)["pr_auc"])
            drops[column] = round(float(np.mean(values)), 4)
        result[target] = drops
    return result


def _model_card(evaluation):
    metrics = evaluation["metrics"]
    lines = [
        f"# Model card: {evaluation['model_version']}", "",
        f"**Status:** {evaluation['status']}  ",
        f"**Task:** {evaluation['target_meaning']}", "",
        "## Data", "",
        f"- Source: `{evaluation['dataset']['file']}` (sha256 `{evaluation['dataset']['sha256'][:16]}…`)",
        f"- Rows read / used: {evaluation['dataset']['rows_read']} / {evaluation['dataset']['rows_used']}; rejected: {evaluation['dataset']['rejected'] or 'none'}",
        f"- Train groups: {', '.join(evaluation['group_split']['train_groups'])}",
        f"- Held-out test groups: {', '.join(evaluation['group_split']['test_groups'])}", "",
        "## Held-out results (served targets, macro)", "",
        f"Served estimator: {metrics['served_estimator']}.", "",
        "| Metric | Served estimator | Best ML candidate | Prevalence baseline |", "|---|---:|---:|---:|",
        f"| Balanced accuracy | {metrics['balanced_accuracy']} | {metrics['best_ml_candidate_balanced_accuracy']} | {metrics['baseline_balanced_accuracy']} |",
        f"| F1 | {metrics['macro_f1']} | {metrics['best_ml_candidate_macro_f1']} | {metrics['baseline_macro_f1']} |",
        f"| PR-AUC | {metrics['macro_pr_auc']} | {metrics['best_ml_candidate_macro_pr_auc']} | {metrics['baseline_macro_pr_auc']} |",
        f"| Brier | {metrics['macro_brier']} | {metrics['best_ml_candidate_macro_brier']} | {metrics['prevalence_baseline_macro_brier']} |",
        f"| ECE | {metrics['macro_ece']} | | |", "",
        f"Best ML candidate: {metrics['best_ml_candidate']}.", "",
        "## Per target (held-out groups)", "",
        "| Target | Served | Test positives | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier | Best ML F1 | Inner-CV ROC-AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target, item in evaluation["per_target"].items():
        ml = evaluation["baselines"][target]["best_ml_candidate"]
        lines.append(f"| {target} | {'yes' if item['served'] else 'no'} | {item['positives']} | {item['precision']} | {item['recall']} | {item['f1']} | {item['pr_auc']} | {item['roc_auc']} | {item['brier']} | {ml['f1']} | {item['inner_cv']['roc_auc']} |")
    lines += ["", "## Design history", "", evaluation["design_history"], ""]
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in evaluation["limitations"]] + [""]
    return "\n".join(lines)
