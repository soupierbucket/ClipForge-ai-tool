"""Compare XGBoost and feature subsets with fixed stratified folds."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


DATA_DIR = Path(__file__).parent / "uts-mdsi-competition-2026"
FEATURE_PATH = DATA_DIR / "model_ready_students.csv"
BASE_PATH = DATA_DIR / "combined_students.csv"
OUTPUT_PATH = DATA_DIR / "xgboost_best_candidate.csv"
ID_COLUMN = "student_id"
TARGET_COLUMN = "successful_outcome"
RANDOM_STATE = 42


def make_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=700,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        reg_alpha=0.0,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def main() -> None:
    data = pd.read_csv(FEATURE_PATH)
    base_columns = pd.read_csv(BASE_PATH, nrows=0).columns.tolist()
    labelled = data[data[TARGET_COLUMN].notna()]
    unlabelled = data[data[TARGET_COLUMN].isna()]
    y = labelled[TARGET_COLUMN].astype(int)
    all_features = [c for c in data.columns if c not in (ID_COLUMN, TARGET_COLUMN)]
    engineered = [c for c in all_features if c not in base_columns]
    attraction = [c for c in all_features if "attraction_" in c]
    weak_profile = [c for c in ("mobile_provider", "favourite_coffee") if c in all_features]
    normalized_trends = [c for c in all_features if c.startswith("monthly_change_")]
    corrected_ratio = ["work_to_study_ratio"] if "work_to_study_ratio" in all_features else []
    absolute_trends = [c for c in all_features if c.startswith("first_to_last_change_")]

    experiments = {
        "all_features": all_features,
        "without_attraction_features": [c for c in all_features if c not in attraction],
        "raw_merged_features_only": [c for c in all_features if c not in engineered],
        "without_mobile_provider_and_coffee": [c for c in all_features if c not in weak_profile],
        "base_plus_normalized_trends_and_work_study_ratio": list(
            dict.fromkeys([c for c in all_features if c not in engineered] + normalized_trends + corrected_ratio)
        ),
        "revised_features_without_absolute_trends": [
            c for c in all_features if c not in absolute_trends
        ],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores_by_name = {}
    for name, columns in experiments.items():
        X = labelled[columns]
        pipeline = make_pipeline(X)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        scores_by_name[name] = scores
        print(
            f"{name} ({len(columns)} features): "
            f"folds={[round(score, 5) for score in scores]}, "
            f"mean={scores.mean():.5f}, std={scores.std(ddof=1):.5f}"
        )

    best_name = max(scores_by_name, key=lambda name: scores_by_name[name].mean())
    best_columns = experiments[best_name]
    print(f"\nBest feature set: {best_name}")
    final_model = make_pipeline(labelled[best_columns])
    final_model.fit(labelled[best_columns], y)
    probabilities = final_model.predict_proba(unlabelled[best_columns])[:, 1]
    pd.DataFrame(
        {ID_COLUMN: unlabelled[ID_COLUMN], TARGET_COLUMN: probabilities}
    ).to_csv(OUTPUT_PATH, index=False)
    print(f"Saved candidate: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
