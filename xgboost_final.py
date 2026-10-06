"""Standalone XGBoost run using the selected 53-feature recipe."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "uts-mdsi-competition-2026"
ID = "student_id"
TARGET = "successful_outcome"
SEED = 42


def make_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    preprocessing = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def main() -> None:
    merged = pd.read_csv(DATA_DIR / "combined_students.csv")
    engineered = pd.read_csv(DATA_DIR / "model_ready_students.csv")
    labelled = engineered[engineered[TARGET].notna()].copy()
    unlabelled = engineered[engineered[TARGET].isna()].copy()

    base_features = [c for c in merged.columns if c not in (ID, TARGET)]
    monthly_trends = [c for c in engineered.columns if c.startswith("monthly_change_")]
    feature_columns = list(
        dict.fromkeys(base_features + monthly_trends + ["work_to_study_ratio"])
    )

    X = labelled[feature_columns]
    y = labelled[TARGET].astype(int)
    X_test = unlabelled[feature_columns]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(
        make_pipeline(X), X, y, cv=cv, scoring="roc_auc", n_jobs=1
    )
    print(f"Features used: {len(feature_columns)}")
    print(f"Fold ROC-AUC: {[round(score, 5) for score in scores]}")
    print(f"Mean ROC-AUC: {scores.mean():.5f}")
    print(f"Standard deviation: {scores.std(ddof=1):.5f}")

    final_model = make_pipeline(X)
    final_model.fit(X, y)
    probabilities = final_model.predict_proba(X_test)[:, 1]
    output_path = HERE / "xgboost_submission.csv"
    pd.DataFrame({ID: unlabelled[ID], TARGET: probabilities}).to_csv(
        output_path, index=False
    )
    print(f"Saved {len(unlabelled)} test predictions to: {output_path}")


if __name__ == "__main__":
    main()
