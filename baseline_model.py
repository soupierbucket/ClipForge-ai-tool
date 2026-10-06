"""Stage 3: a simple, interpretable baseline model.

The baseline is logistic regression with median imputation for numeric columns
and one-hot encoding for categorical columns. It reports validation ROC-AUC and
writes probability predictions for the unlabelled students.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path(__file__).parent / "uts-mdsi-competition-2026"
FEATURE_PATH = DATA_DIR / "model_ready_students.csv"
FALLBACK_FEATURE_PATH = DATA_DIR / "combined_students.csv"
OUTPUT_PATH = DATA_DIR / "baseline_submission.csv"
ID_COLUMN = "student_id"
TARGET_COLUMN = "successful_outcome"
RANDOM_STATE = 42


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def main() -> None:
    input_path = FEATURE_PATH if FEATURE_PATH.exists() else FALLBACK_FEATURE_PATH
    if not input_path.exists():
        raise FileNotFoundError(
            "Run Code.py first, then feature_engineering.py, before running this script."
        )

    data = pd.read_csv(input_path)
    labelled = data[data[TARGET_COLUMN].notna()].copy()
    unlabelled = data[data[TARGET_COLUMN].isna()].copy()
    if labelled.empty or unlabelled.empty:
        raise ValueError("Expected both labelled training rows and unlabelled test rows.")

    feature_columns = [
        column for column in data.columns if column not in (ID_COLUMN, TARGET_COLUMN)
    ]
    X = labelled[feature_columns]
    y = labelled[TARGET_COLUMN].astype(int)
    X_test = unlabelled[feature_columns]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    def make_model() -> Pipeline:
        return Pipeline(
            [
                ("preprocess", build_preprocessor(X_train)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    validation_model = make_model()
    validation_model.fit(X_train, y_train)
    valid_probabilities = validation_model.predict_proba(X_valid)[:, 1]
    validation_auc = roc_auc_score(y_valid, valid_probabilities)
    print(f"Input: {input_path.name}")
    print(f"Labelled rows: {len(labelled):,}")
    print(f"Unlabelled rows: {len(unlabelled):,}")
    print(f"Validation ROC-AUC: {validation_auc:.5f}")

    final_model = make_model()
    final_model.fit(X, y)
    test_probabilities = final_model.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame(
        {
            ID_COLUMN: unlabelled[ID_COLUMN],
            TARGET_COLUMN: test_probabilities,
        }
    )
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
