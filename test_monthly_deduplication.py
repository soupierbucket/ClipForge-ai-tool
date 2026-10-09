"""Compare current aggregation with one row per student-month aggregation."""

from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


HERE = Path(__file__).resolve().parent
FINAL_DIR = HERE.parent / "final"
DATA_DIR = FINAL_DIR / "uts-mdsi-competition-2026"
TARGET = "successful_outcome"
PARAMS = dict(
    n_estimators=1000, learning_rate=0.03, max_depth=3,
    min_child_weight=1, subsample=0.70, colsample_bytree=0.85,
    reg_lambda=1.0,
)


def make_model(X: pd.DataFrame) -> Pipeline:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    prep = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    model = XGBClassifier(
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        random_state=42, n_jobs=-1, **PARAMS,
    )
    return Pipeline([("preprocess", prep), ("model", model)])


def build_deduplicated_engineered():
    sys.path.insert(0, str(FINAL_DIR))
    import Code as merge_code
    import feature_engineering as feature_code

    students = pd.read_csv(DATA_DIR / "students.csv")
    profiles = pd.read_csv(DATA_DIR / "student_profile_metrics.csv")
    suburbs = pd.read_csv(DATA_DIR / "suburbs.csv")
    attractions = pd.read_csv(DATA_DIR / "student_attractions.csv")
    activity = pd.read_csv(DATA_DIR / "student_monthly_activity.csv")
    labels = pd.read_csv(DATA_DIR / "train.csv")
    activity["month_date"] = pd.to_datetime(activity["month_date"])
    duplicate_counts = activity.groupby("student_id").agg(
        row_count=("month_date", "size"), month_count=("month_date", "nunique")
    )
    duplicate_counts["duplicate_month_count"] = (
        duplicate_counts["row_count"] - duplicate_counts["month_count"]
    )
    activity_columns = [c for c in activity.columns if c not in ("student_id", "month_date")]
    activity = (activity.groupby(["student_id", "month_date"], as_index=False)[activity_columns]
                .mean())

    attraction_summary = merge_code.aggregate_attractions(attractions)
    activity_summary = merge_code.aggregate_monthly_activity(activity)
    combined = students.merge(profiles, on="student_id", how="left", validate="one_to_one")
    combined = combined.merge(suburbs, on="suburb_id", how="left", validate="many_to_one")
    combined = combined.merge(attraction_summary, on="student_id", how="left", validate="one_to_one")
    combined = combined.merge(activity_summary, on="student_id", how="left", validate="one_to_one")
    combined = combined.merge(labels, on="student_id", how="left", validate="one_to_one")

    with tempfile.TemporaryDirectory(prefix="dedup_months_") as tmp:
        tmp_path = Path(tmp)
        activity.to_csv(tmp_path / "student_monthly_activity.csv", index=False)
        shutil.copy2(DATA_DIR / "student_attractions.csv", tmp_path / "student_attractions.csv")
        feature_code.DATA_DIR = tmp_path
        engineered = feature_code.add_features(combined)
    return combined, engineered, duplicate_counts["duplicate_month_count"]


def main() -> None:
    current_merged = pd.read_csv(DATA_DIR / "combined_students.csv")
    current_engineered = pd.read_csv(DATA_DIR / "model_ready_students.csv")
    dedup_merged, dedup_engineered, duplicate_counts = build_deduplicated_engineered()
    features = list(dict.fromkeys(
        [c for c in current_merged.columns if c not in ("student_id", TARGET)]
        + [c for c in current_engineered.columns if c.startswith("monthly_change_")]
        + ["work_to_study_ratio"]
    ))
    current = current_engineered.loc[current_engineered[TARGET].notna()].reset_index(drop=True)
    deduped = dedup_engineered.loc[dedup_engineered[TARGET].notna()].reset_index(drop=True)
    if not current["student_id"].equals(deduped["student_id"]):
        raise ValueError("Current and deduplicated data rows are not aligned by student_id.")
    if not current[TARGET].equals(deduped[TARGET]):
        raise ValueError("Current and deduplicated target labels do not match.")
    deduped["duplicate_month_count"] = deduped["student_id"].map(duplicate_counts).fillna(0)
    y = current[TARGET].astype(int)
    rows = []
    for cv_seed in (42, 123):
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed).split(current, y))
        variants = [
            ("current", current, features),
            ("deduplicated_months", deduped, features),
            ("deduplicated_plus_duplicate_month_count", deduped,
             features + ["duplicate_month_count"]),
        ]
        for name, frame, selected_features in variants:
            X = frame[selected_features]
            scores = []
            for train_idx, valid_idx in folds:
                model = make_model(X.iloc[train_idx])
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                pred = model.predict_proba(X.iloc[valid_idx])[:, 1]
                scores.append(roc_auc_score(y.iloc[valid_idx], pred))
            rows.append({
                "cv_seed": cv_seed, "aggregation": name, "feature_count": len(selected_features),
                "mean_roc_auc": float(np.mean(scores)),
                "std_roc_auc": float(np.std(scores, ddof=1)),
                "fold_scores": ",".join(f"{score:.6f}" for score in scores),
            })
            print(f"seed={cv_seed} {name}: mean={np.mean(scores):.6f}, folds={scores}")

    output = HERE / "monthly_deduplication_results.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
