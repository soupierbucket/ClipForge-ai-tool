"""Evaluate a histogram gradient-boosting classifier with five-fold CV."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_DIR = Path(__file__).parent / "uts-mdsi-competition-2026"
FEATURE_PATH = DATA_DIR / "model_ready_students.csv"
ID_COLUMN = "student_id"
TARGET_COLUMN = "successful_outcome"
RANDOM_STATE = 42


def build_pipeline(features: pd.DataFrame) -> Pipeline:
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

    preprocessing = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric_columns),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            ),
        ],
        sparse_threshold=0.0,
    )
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def main() -> None:
    data = pd.read_csv(FEATURE_PATH)
    labelled = data[data[TARGET_COLUMN].notna()]
    unlabelled = data[data[TARGET_COLUMN].isna()]
    feature_columns = [c for c in data.columns if c not in (ID_COLUMN, TARGET_COLUMN)]
    X = labelled[feature_columns]
    y = labelled[TARGET_COLUMN].astype(int)
    X_test = unlabelled[feature_columns]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipeline = build_pipeline(X)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    print(f"Fold ROC-AUC: {[round(score, 5) for score in scores]}")
    print(f"Mean ROC-AUC: {scores.mean():.5f}")
    print(f"Standard deviation: {scores.std(ddof=1):.5f}")

    pipeline.fit(X, y)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame(
        {ID_COLUMN: unlabelled[ID_COLUMN], TARGET_COLUMN: probabilities}
    )
    output_path = DATA_DIR / "hist_gradient_boosting_submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
