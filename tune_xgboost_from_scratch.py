"""Tune XGBoost settings on the existing 53-feature recipe and fixed folds."""

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
SEED = 42


def build_pipeline(X: pd.DataFrame, params: dict) -> Pipeline:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    prep = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ]
    )
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        **params,
    )
    return Pipeline([("preprocess", prep), ("model", model)])


def main() -> None:
    merged = pd.read_csv(DATA_DIR / "combined_students.csv")
    data = pd.read_csv(DATA_DIR / "model_ready_students.csv")
    labelled = data[data[TARGET].notna()].copy()
    unlabelled = data[data[TARGET].isna()].copy()
    base = [c for c in merged.columns if c not in (ID, TARGET)]
    trends = [c for c in data.columns if c.startswith("monthly_change_")]
    columns = list(dict.fromkeys(base + trends + ["work_to_study_ratio"]))
    X = labelled[columns]
    X_test = unlabelled[columns]
    y = labelled[TARGET].astype(int)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))

    configurations = {
        "reference": dict(n_estimators=700, learning_rate=0.03, max_depth=4,
                          min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
                          reg_lambda=3.0),
        "shallow_regularized": dict(n_estimators=1000, learning_rate=0.03, max_depth=3,
                                    min_child_weight=3, subsample=0.9, colsample_bytree=0.9,
                                    reg_lambda=5.0),
        "shallow_less_regularized": dict(n_estimators=1000, learning_rate=0.03, max_depth=3,
                                        min_child_weight=1, subsample=0.85, colsample_bytree=0.85,
                                        reg_lambda=1.0),
        "depth4_more_trees": dict(n_estimators=1100, learning_rate=0.025, max_depth=4,
                                  min_child_weight=2, subsample=0.9, colsample_bytree=0.9,
                                  reg_lambda=3.0),
        "depth4_stronger_regularization": dict(n_estimators=900, learning_rate=0.03, max_depth=4,
                                               min_child_weight=5, subsample=0.8,
                                               colsample_bytree=0.8, reg_lambda=5.0),
        "depth5_regularized": dict(n_estimators=1100, learning_rate=0.025, max_depth=5,
                                   min_child_weight=5, subsample=0.85, colsample_bytree=0.85,
                                   reg_lambda=5.0),
        "depth5_less_regularized": dict(n_estimators=900, learning_rate=0.03, max_depth=5,
                                        min_child_weight=2, subsample=0.9, colsample_bytree=0.8,
                                        reg_lambda=2.0),
        "depth6_regularized": dict(n_estimators=1300, learning_rate=0.02, max_depth=6,
                                   min_child_weight=5, subsample=0.85, colsample_bytree=0.8,
                                   reg_lambda=7.0),
    }

    results = []
    oof_by_name = {}
    for name, params in configurations.items():
        oof = np.zeros(len(y), dtype=float)
        scores = []
        for train_idx, valid_idx in folds:
            model = build_pipeline(X.iloc[train_idx], params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict_proba(X.iloc[valid_idx])[:, 1]
            oof[valid_idx] = pred
            scores.append(roc_auc_score(y.iloc[valid_idx], pred))
        oof_by_name[name] = oof
        result = {
            "configuration": name,
            "mean_roc_auc": np.mean(scores),
            "std_roc_auc": np.std(scores, ddof=1),
            "fold_scores": ",".join(f"{score:.6f}" for score in scores),
            "parameters": repr(params),
        }
        results.append(result)
        print(f"{name}: folds={[round(s, 5) for s in scores]}, mean={np.mean(scores):.5f}, std={np.std(scores, ddof=1):.5f}")

    result_frame = pd.DataFrame(results).sort_values("mean_roc_auc", ascending=False)
    result_frame.to_csv(HERE / "xgboost_tuning_results.csv", index=False)
    best_name = result_frame.iloc[0]["configuration"]
    print(f"\nBest setting: {best_name}; mean CV ROC-AUC={result_frame.iloc[0]['mean_roc_auc']:.5f}")

    final_model = build_pipeline(X, configurations[best_name])
    final_model.fit(X, y)
    probabilities = final_model.predict_proba(X_test)[:, 1]
    pd.DataFrame({ID: unlabelled[ID], TARGET: probabilities}).to_csv(
        HERE / "xgboost_tuned_candidate.csv", index=False
    )
    print(f"Saved separate candidate: {HERE / 'xgboost_tuned_candidate.csv'}")


if __name__ == "__main__":
    main()
