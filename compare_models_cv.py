"""Compare baseline and tree models with stratified cross-validation.

All preprocessing is inside each model pipeline, so medians, category levels,
and scaling are learned from each fold's training rows only.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path(__file__).parent / "uts-mdsi-competition-2026"
FEATURE_PATH = DATA_DIR / "model_ready_students.csv"
FALLBACK_FEATURE_PATH = DATA_DIR / "combined_students.csv"
OUTPUT_PATH = DATA_DIR / "cv_best_model_submission.csv"
ID_COLUMN = "student_id"
TARGET_COLUMN = "successful_outcome"
RANDOM_STATE = 42
FOLDS = 5


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


def build_pipeline(estimator, features: pd.DataFrame) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(features)),
            ("model", estimator),
        ]
    )


def main() -> None:
    input_path = FEATURE_PATH if FEATURE_PATH.exists() else FALLBACK_FEATURE_PATH
    if not input_path.exists():
        raise FileNotFoundError("Run Code.py before model comparison.")

    data = pd.read_csv(input_path)
    labelled = data[data[TARGET_COLUMN].notna()].copy()
    unlabelled = data[data[TARGET_COLUMN].isna()].copy()
    feature_columns = [
        column for column in data.columns if column not in (ID_COLUMN, TARGET_COLUMN)
    ]
    X = labelled[feature_columns]
    y = labelled[TARGET_COLUMN].astype(int)
    X_test = unlabelled[feature_columns]

    estimators = {
        "logistic_regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, estimator in estimators.items():
        pipeline = build_pipeline(estimator, X)
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
            return_train_score=False,
        )["test_score"]
        results.append(
            {
                "model": name,
                "mean_roc_auc": scores.mean(),
                "std_roc_auc": scores.std(ddof=1),
                "fold_scores": scores,
            }
        )
        fold_text = ", ".join(f"{score:.5f}" for score in scores)
        print(f"{name}: folds [{fold_text}]")
        print(f"  mean ROC-AUC={scores.mean():.5f}, std={scores.std(ddof=1):.5f}")

    results.sort(key=lambda result: result["mean_roc_auc"], reverse=True)
    best_name = results[0]["model"]
    print(f"\nBest mean cross-validation ROC-AUC: {best_name}")

    # Fit the selected estimator on every labelled row for a separate output.
    final_model = build_pipeline(estimators[best_name], X)
    final_model.fit(X, y)
    probabilities = final_model.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame(
        {ID_COLUMN: unlabelled[ID_COLUMN], TARGET_COLUMN: probabilities}
    )
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
