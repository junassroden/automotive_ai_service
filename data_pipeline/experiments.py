"""Evaluation-only experiments for the labelled datasets that cannot back a
served fleet prediction. Each experiment trains real models against
baselines and records why it is (not) usable; nothing here is registered for
inference. Results feed data/reports/experiments_report.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, r2_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from data_pipeline.evaluation import best_threshold, binary_metrics


RANDOM_STATE = 42


def _binary_suite(X_train, y_train, X_test, y_test, skip=()):
    models = {
        "prevalence_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(200, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=1, random_state=RANDOM_STATE),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=RANDOM_STATE),
    }
    results = {}
    for name, model in models.items():
        if name in skip:
            continue
        # Threshold from out-of-fold training predictions, never from the test set.
        oof = cross_val_predict(model, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), method="predict_proba")[:, 1]
        threshold = best_threshold(y_train, oof) if name != "prevalence_baseline" else 0.5
        probability = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
        results[name] = binary_metrics(y_test, probability, threshold)
    return results


def engine_condition(raw_root):
    raw_root = Path(raw_root)
    train = pd.read_csv(raw_root / "train.csv")
    test = pd.read_csv(raw_root / "test.csv")
    combined = pd.read_csv(raw_root / "raw_data.csv")
    combined.columns = train.columns
    union = pd.concat([train, test])
    overlap = len(train.merge(test, how="inner"))
    features = [c for c in train.columns if c != "engine_condition"]
    results = _binary_suite(train[features], train["engine_condition"], test[features], test["engine_condition"])
    best = max((k for k in results if k != "prevalence_baseline"), key=lambda k: results[k]["pr_auc"] or 0)
    return {
        "experiment": "engine_condition_experiment",
        "status": "evaluated_not_served",
        "files": ["train.csv", "test.csv", "raw_data.csv"],
        "split": "supplied train.csv / test.csv (row-level; no vehicle or run identifiers exist)",
        "records": {"train": len(train), "test": len(test), "raw_data_equals_union": bool(len(combined.merge(union, how="inner")) == len(combined) == len(union)),
                    "train_test_identical_rows": overlap},
        "results": results,
        "best_model": best,
        "reason_not_served": (
            "The meaning of engine_condition (0/1) is undocumented, there is no source or licence, and rows "
            "cannot be grouped by engine or vehicle, so held-out scores may overstate performance on new vehicles. "
            "The fleet database also holds none of these live sensor readings."
        ),
    }


def synthetic_signal_audit(raw_root):
    raw_root = Path(raw_root)
    ev = pd.read_csv(raw_root / "EV_Predictive_Maintenance_Dataset_15min.csv", parse_dates=["Timestamp"]).sort_values("Timestamp")
    leakage = {"Failure_Probability", "RUL", "TTF", "Maintenance_Type", "Component_Health_Score", "Timestamp"}
    features = [c for c in ev.columns if c not in leakage]
    cut = int(len(ev) * 0.8)
    train, test = ev.iloc[:cut], ev.iloc[cut:]
    # Random forest is skipped here only for run time (it took ~30 min on 140k rows and also scored ROC-AUC 0.50).
    failure = _binary_suite(train[features], train["Failure_Probability"], test[features], test["Failure_Probability"], skip=("random_forest",))
    rul = HistGradientBoostingRegressor(max_iter=200, random_state=RANDOM_STATE).fit(train[features], train["RUL"])
    rul_r2 = r2_score(test["RUL"], rul.predict(test[features]))
    maintenance = HistGradientBoostingClassifier(max_iter=200, random_state=RANDOM_STATE).fit(train[features], train["Maintenance_Type"])
    maintenance_macro_f1 = f1_score(test["Maintenance_Type"], maintenance.predict(test[features]), average="macro")
    majority_f1 = f1_score(test["Maintenance_Type"], np.full(len(test), train["Maintenance_Type"].mode()[0]), average="macro")
    correlation = ev[features + ["Failure_Probability"]].corr()["Failure_Probability"].drop("Failure_Probability").abs().max()

    hyundai = pd.read_csv(raw_root / "cars_hyundai.csv")
    h_features = ["Engine Temperature (°C)", "Brake Pad Thickness (mm)", "Tire Pressure (PSI)"]
    oof = cross_val_predict(
        RandomForestClassifier(200, min_samples_leaf=5, n_jobs=1, random_state=RANDOM_STATE),
        hyundai[h_features], hyundai["Anomaly Indication"],
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), method="predict_proba")[:, 1]
    hyundai_metrics = binary_metrics(hyundai["Anomaly Indication"], oof, 0.5)
    return {
        "experiment": "synthetic_signal_audit",
        "status": "excluded_no_learnable_signal",
        "ev_predictive_maintenance": {
            "split": "time-ordered: first 80% of timestamps train, last 20% test",
            "records": {"train": len(train), "test": len(test)},
            "excluded_leakage_columns": sorted(leakage - {"Timestamp"}),
            "failure_label_results": failure,
            "rul_regression_test_r2": round(float(rul_r2), 4),
            "maintenance_type_macro_f1": round(float(maintenance_macro_f1), 4),
            "maintenance_type_majority_macro_f1": round(float(majority_f1), 4),
            "max_abs_feature_label_correlation": round(float(correlation), 4),
        },
        "hyundai_component_readings": {"split": "stratified 5-fold out-of-fold", "anomaly_results": hyundai_metrics},
        "conclusion": (
            "Models do not beat the prevalence baseline (ROC-AUC near 0.5, R-squared near 0): the labels are "
            "independent of every input, consistent with randomly generated data. Training on them would produce "
            "a model that outputs noise, so both datasets are excluded from every served model."
        ),
    }


def cvffad_fault_system(raw_root):
    events = pd.read_csv(Path(raw_root) / "fault_events.csv", parse_dates=["occurrence_date"])
    events["month"] = events["occurrence_date"].dt.month
    numeric = ["mileage_at_fault", "vehicle_age_at_fault", "temperature_celsius", "pressure_kpa", "engine_rpm",
               "temp_trend_1h", "temp_trend_24h", "pressure_trend_1h", "rpm_variance", "month"]
    categorical = ["vehicle_type", "fuel_type", "owner_type", "primary_route"]
    leakage = ["fault_code", "fault_category", "fault_name", "fault_description_raw", "severity", "difficulty",
               "common_mileage", "typical_season", "required_tool", "related_faults", "mileage_ratio",
               "age_mileage_interaction", "seasonal_risk_score", "system_degradation_index"]
    split = GroupShuffleSplit(1, test_size=0.25, random_state=RANDOM_STATE)
    train_index, test_index = next(split.split(events, groups=events["vehicle_id"]))
    train, test = events.iloc[train_index], events.iloc[test_index]
    model = make_pipeline(
        make_column_transformer((OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical), ("passthrough", numeric)),
        HistGradientBoostingClassifier(max_iter=200, random_state=RANDOM_STATE, class_weight="balanced"),
    ).fit(train[categorical + numeric], train["fault_system"])
    predicted = model.predict(test[categorical + numeric])
    majority = np.full(len(test), train["fault_system"].mode()[0])
    return {
        "experiment": "cvffad_fault_system_experiment",
        "status": "evaluated_not_served",
        "target": "fault_system (7 systems) from operating readings and vehicle context",
        "split": "GroupShuffleSplit by vehicle_id (25% of vehicles held out)",
        "records": {"train": len(train), "test": len(test), "vehicle_overlap": len(set(train["vehicle_id"]) & set(test["vehicle_id"]))},
        "excluded_leakage_columns": leakage,
        "macro_f1": round(float(f1_score(test["fault_system"], predicted, average="macro")), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(test["fault_system"], predicted)), 4),
        "majority_baseline_macro_f1": round(float(f1_score(test["fault_system"], majority, average="macro")), 4),
        "majority_baseline_balanced_accuracy": round(float(balanced_accuracy_score(test["fault_system"], majority)), 4),
        "classes": events["fault_system"].value_counts().to_dict(),
        "reason_not_served": (
            "Every CVFFAD record is marked source='generated' and the dataset covers Chinese commercial trucks/vans "
            "with a placeholder DOI. Patterns learned from generated events must not be presented as real fleet risk."
        ),
    }


def diesel_dtc_presence(raw_root):
    base = Path(raw_root) / "OBD-Dataset-main" / "OBD-Dataset-main"
    train = pd.read_excel(base / "MASTER_TRAIN_X.xlsx")
    test = pd.read_excel(base / "MASTER_TEST_X.xlsx")
    dtc = [c for c in train.columns if c.startswith("P") and c != "P0000"]
    pids = ["LOAD_PCT", "ECT", "MAP", "RPM", "VSS", "IAT", "MAF", "FRP", "BARO", "VPWR", "AAT"]
    frame = pd.concat([train.assign(source="train"), test.assign(source="test")], ignore_index=True)
    frame["signature"] = frame[["P0000"] + dtc].astype(int).astype(str).agg("".join, axis=1)
    ambiguous = frame["signature"] == "0" * (len(dtc) + 1)
    frame = frame.loc[~ambiguous].copy()
    frame["active_dtc"] = (frame[dtc].sum(axis=1) > 0).astype(int)
    signatures = frame.groupby("signature")["active_dtc"].first()

    supplied_train, supplied_test = frame[frame.source == "train"], frame[frame.source == "test"]
    with threadpool_limits(limits=1):
        supplied = HistGradientBoostingClassifier(max_iter=100, random_state=RANDOM_STATE).fit(supplied_train[pids], supplied_train["active_dtc"])
        supplied_metrics = binary_metrics(supplied_test["active_dtc"], supplied.predict_proba(supplied_test[pids])[:, 1], 0.5)

    folds = []
    for signature in signatures.index:
        held = frame["signature"] == signature
        y_train = frame.loc[~held, "active_dtc"]
        model = HistGradientBoostingClassifier(max_iter=100, random_state=RANDOM_STATE).fit(frame.loc[~held, pids], y_train)
        predicted = model.predict(frame.loc[held, pids])
        folds.append({"held_out_signature": signature, "true_label": int(signatures[signature]),
                      "rows": int(held.sum()), "fraction_predicted_active": round(float(predicted.mean()), 4)})
    vehicle_level_correct = sum((f["fraction_predicted_active"] >= 0.5) == bool(f["true_label"]) for f in folds)
    return {
        "experiment": "diesel_dtc_presence_experiment",
        "status": "evaluated_not_served",
        "target": "any active DTC (excluding P0000) from 11 OBD-II PIDs",
        "rows_excluded_ambiguous_signature": int(ambiguous.sum()),
        "dtc_signatures": {k: int(v) for k, v in signatures.items()},
        "supplied_split_results": supplied_metrics,
        "supplied_split_warning": "All 7 DTC signatures (vehicles) appear in both supplied files, so this score measures recognising known vehicles, not detecting faults.",
        "leave_one_signature_out": folds,
        "vehicle_level_correct": f"{vehicle_level_correct}/{len(folds)}",
        "reason_not_served": (
            "DTCs are constant per vehicle, so labels identify the vehicle. With only one fault-free signature the "
            "vehicle-disjoint evaluation cannot support a fault detector; the fleet also has no live PID stream."
        ),
    }


def run_experiments(raw_root):
    with threadpool_limits(limits=1):
        return {
            "engine_condition_experiment": engine_condition(raw_root),
            "synthetic_signal_audit": synthetic_signal_audit(raw_root),
            "cvffad_fault_system_experiment": cvffad_fault_system(raw_root),
            "diesel_dtc_presence_experiment": diesel_dtc_presence(raw_root),
        }
