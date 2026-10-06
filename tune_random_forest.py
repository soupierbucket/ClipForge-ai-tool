"""Small five-fold random-forest tuning experiment."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path(__file__).parent / "uts-mdsi-competition-2026"
FEATURE_PATH = DATA_DIR / "model_ready_students.csv"
ID_COLUMN = "student_id"
TARGET_COLUMN = "successful_outcome"
RANDOM_STATE = 42


def make_pipeline(features: pd.DataFrame, max_features, min_samples_leaf: int):
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
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
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def main() -> None:
    data = pd.read_csv(FEATURE_PATH)
    labelled = data[data[TARGET_COLUMN].notna()]
    unlabelled = data[data[TARGET_COLUMN].isna()]
    columns = [c for c in data.columns if c not in (ID_COLUMN, TARGET_COLUMN)]
    X = labelled[columns]
    y = labelled[TARGET_COLUMN].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    configurations = [
        ("sqrt", 1),
        ("sqrt", 2),
        ("sqrt", 4),
        (0.5, 1),
        (0.5, 2),
        (0.5, 4),
    ]
    results = []
    for max_features, leaf_size in configurations:
        pipeline = make_pipeline(X, max_features, leaf_size)
        scores = cross_val_score(
            pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=1
        )
        mean_score = scores.mean()
        std_score = scores.std(ddof=1)
        results.append((mean_score, std_score, max_features, leaf_size, scores))
        print(
            f"max_features={max_features}, min_samples_leaf={leaf_size}: "
            f"mean={mean_score:.5f}, std={std_score:.5f}, "
            f"folds={[round(score, 5) for score in scores]}"
        )

    results.sort(key=lambda item: item[0], reverse=True)
    _, _, best_features, best_leaf, _ = results[0]
    print(
        f"\nBest configuration: max_features={best_features}, "
        f"min_samples_leaf={best_leaf}"
    )

    final_model = make_pipeline(X, best_features, best_leaf)
    final_model.fit(X, y)
    probabilities = final_model.predict_proba(unlabelled[columns])[:, 1]
    output = pd.DataFrame(
        {ID_COLUMN: unlabelled[ID_COLUMN], TARGET_COLUMN: probabilities}
    )
    output_path = DATA_DIR / "tuned_random_forest_submission.csv"
    output.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
