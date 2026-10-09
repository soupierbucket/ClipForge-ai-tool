"""Compare attraction-name visit pivots with the existing aggregates."""

from pathlib import Path

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
DATA_DIR = HERE.parent / "final" / "uts-mdsi-competition-2026"
TARGET = "successful_outcome"
DROP_ATTRACTION_DETAILS = {
    "attraction_types_visited", "attraction_mean_visits", "attraction_max_visits",
    "attraction_top_visit_share", "attraction_diversity_ratio",
    "attraction_visits_per_type", "log1p_attraction_total_visits",
}
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


def main() -> None:
    merged = pd.read_csv(DATA_DIR / "combined_students.csv")
    engineered = pd.read_csv(DATA_DIR / "model_ready_students.csv")
    visits = pd.read_csv(DATA_DIR / "student_attractions.csv")
    visits["attraction_name_clean"] = visits["attraction_name"].astype(str).str.strip().str.casefold()
    pivot = visits.pivot_table(
        index="student_id", columns="attraction_name_clean", values="visit_count",
        aggfunc="sum", fill_value=0,
    )
    pivot.columns = [f"attraction_visit_{str(name).replace(' ', '_')}"
                     for name in pivot.columns]
    pivot = pivot.reset_index()
    data = engineered.merge(pivot, on="student_id", how="left", validate="one_to_one")
    pivot_features = [c for c in pivot.columns if c != "student_id"]
    data[pivot_features] = data[pivot_features].fillna(0)

    labelled = data.loc[data[TARGET].notna()].copy().reset_index(drop=True)
    y = labelled[TARGET].astype(int)
    base = list(dict.fromkeys(
        [c for c in merged.columns if c not in ("student_id", TARGET)]
        + [c for c in engineered.columns if c.startswith("monthly_change_")]
        + ["work_to_study_ratio"]
    ))
    total_only = [c for c in base if c not in DROP_ATTRACTION_DETAILS]
    with_pivot = base + pivot_features
    total_only_with_pivot = total_only + pivot_features
    variants = {
        "reference_53": base,
        "reference_plus_name_pivot": with_pivot,
        "total_visits_plus_name_pivot": total_only_with_pivot,
    }
    rows = []
    for cv_seed in (42, 123):
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed).split(labelled, y))
        for name, columns in variants.items():
            X = labelled[columns]
            scores = []
            for train_idx, valid_idx in folds:
                model = make_model(X.iloc[train_idx])
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                pred = model.predict_proba(X.iloc[valid_idx])[:, 1]
                scores.append(roc_auc_score(y.iloc[valid_idx], pred))
            rows.append({
                "cv_seed": cv_seed, "experiment": name, "feature_count": len(columns),
                "attraction_name_count": len(pivot_features),
                "mean_roc_auc": float(np.mean(scores)),
                "std_roc_auc": float(np.std(scores, ddof=1)),
                "fold_scores": ",".join(f"{score:.6f}" for score in scores),
                "pivot_features": ",".join(pivot_features),
            })
            print(f"seed={cv_seed} {name}: mean={np.mean(scores):.6f}, folds={scores}")
    output = HERE / "attraction_pivot_results.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Attraction names represented: {len(pivot_features)}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
