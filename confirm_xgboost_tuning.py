"""Check the XGBoost tuning gain on a second stratified fold assignment."""

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
DATA_DIR = HERE.parent / "uts-mdsi-competition-2026"
ID = "student_id"
TARGET = "successful_outcome"
SEED = 123


def make_model(X: pd.DataFrame, params: dict) -> Pipeline:
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
        random_state=42, n_jobs=-1, **params,
    )
    return Pipeline([("preprocess", prep), ("model", model)])


def main() -> None:
    merged = pd.read_csv(DATA_DIR / "combined_students.csv")
    data = pd.read_csv(DATA_DIR / "model_ready_students.csv")
    labelled = data[data[TARGET].notna()]
    base = [c for c in merged.columns if c not in (ID, TARGET)]
    trends = [c for c in data.columns if c.startswith("monthly_change_")]
    columns = list(dict.fromkeys(base + trends + ["work_to_study_ratio"]))
    X = labelled[columns]
    y = labelled[TARGET].astype(int)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))

    settings = {
        "reference": dict(n_estimators=700, learning_rate=0.03, max_depth=4,
                          min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
                          reg_lambda=3.0),
        "tuned_shallow": dict(n_estimators=1000, learning_rate=0.03, max_depth=3,
                              min_child_weight=1, subsample=0.85, colsample_bytree=0.85,
                              reg_lambda=1.0),
    }
    for name, params in settings.items():
        scores = []
        for train_idx, valid_idx in folds:
            model = make_model(X.iloc[train_idx], params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict_proba(X.iloc[valid_idx])[:, 1]
            scores.append(roc_auc_score(y.iloc[valid_idx], pred))
        print(f"{name}, seed={SEED}: folds={[round(s, 5) for s in scores]}, mean={np.mean(scores):.5f}, std={np.std(scores, ddof=1):.5f}")


if __name__ == "__main__":
    main()
