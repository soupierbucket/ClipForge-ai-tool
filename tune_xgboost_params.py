"""Run and record a focused XGBoost hyperparameter search."""

from datetime import datetime
from pathlib import Path
import csv
import json

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


def make_pipeline(X: pd.DataFrame, params: dict) -> Pipeline:
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
    estimator = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        **params,
    )
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])


def evaluate(X, y, params, seed):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for train_idx, valid_idx in folds.split(X, y):
        model = make_pipeline(X.iloc[train_idx], params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probabilities = model.predict_proba(X.iloc[valid_idx])[:, 1]
        scores.append(roc_auc_score(y.iloc[valid_idx], probabilities))
    return scores


def main() -> None:
    merged = pd.read_csv(DATA_DIR / "combined_students.csv")
    engineered = pd.read_csv(DATA_DIR / "model_ready_students.csv")
    labelled = engineered[engineered[TARGET].notna()].copy()
    unlabelled = engineered[engineered[TARGET].isna()].copy()

    base_features = [c for c in merged.columns if c not in (ID, TARGET)]
    trend_features = [c for c in engineered.columns if c.startswith("monthly_change_")]
    feature_columns = list(
        dict.fromkeys(base_features + trend_features + ["work_to_study_ratio"])
    )
    X = labelled[feature_columns]
    y = labelled[TARGET].astype(int)
    X_test = unlabelled[feature_columns]

    common = dict(learning_rate=0.03, subsample=0.85, colsample_bytree=0.85)
    configurations = {
        "current_tuned": dict(n_estimators=1000, max_depth=3, min_child_weight=1, reg_lambda=1.0, **common),
        "depth_2": dict(n_estimators=1000, max_depth=2, min_child_weight=1, reg_lambda=1.0, **common),
        "depth_4": dict(n_estimators=900, max_depth=4, min_child_weight=3, reg_lambda=3.0, **common),
        "depth_5_regularized": dict(n_estimators=1100, learning_rate=0.025, max_depth=5,
                                     min_child_weight=5, subsample=0.85,
                                     colsample_bytree=0.85, reg_lambda=5.0),
        "depth_3_child_3": dict(n_estimators=1000, max_depth=3, min_child_weight=3, reg_lambda=1.0, **common),
        "depth_3_child_5": dict(n_estimators=1000, max_depth=3, min_child_weight=5, reg_lambda=1.0, **common),
        "lower_rate_more_trees": dict(n_estimators=1400, learning_rate=0.02, max_depth=3,
                                      min_child_weight=1, subsample=0.85,
                                      colsample_bytree=0.85, reg_lambda=1.0),
        "higher_rate_fewer_trees": dict(n_estimators=650, learning_rate=0.05, max_depth=3,
                                        min_child_weight=1, subsample=0.85,
                                        colsample_bytree=0.85, reg_lambda=1.0),
        "row_sample_70pct": dict(n_estimators=1000, max_depth=3, min_child_weight=1,
                                 learning_rate=0.03, subsample=0.70,
                                 colsample_bytree=0.85, reg_lambda=1.0),
        "row_sample_100pct": dict(n_estimators=1000, max_depth=3, min_child_weight=1,
                                  learning_rate=0.03, subsample=1.0,
                                  colsample_bytree=0.85, reg_lambda=1.0),
        "column_sample_70pct": dict(n_estimators=1000, max_depth=3, min_child_weight=1,
                                    learning_rate=0.03, subsample=0.85,
                                    colsample_bytree=0.70, reg_lambda=1.0),
        "column_sample_100pct": dict(n_estimators=1000, max_depth=3, min_child_weight=1,
                                     learning_rate=0.03, subsample=0.85,
                                     colsample_bytree=1.0, reg_lambda=1.0),
        "stronger_l2": dict(n_estimators=1000, max_depth=3, min_child_weight=1,
                            learning_rate=0.03, subsample=0.85,
                            colsample_bytree=0.85, reg_lambda=5.0),
        "weaker_l2": dict(n_estimators=1000, max_depth=3, min_child_weight=1,
                          learning_rate=0.03, subsample=0.85,
                          colsample_bytree=0.85, reg_lambda=0.3),
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    results_path = HERE / f"xgboost_tuning_results_{stamp}.csv"
    candidate_path = HERE / f"xgboost_tuned_submission_{stamp}.csv"
    columns = ["phase", "configuration", "seed", "mean_roc_auc", "std_roc_auc", "fold_scores", "parameters"]

    completed = []
    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        file.flush()

        for name, params in configurations.items():
            scores = evaluate(X, y, params, SEED)
            row = {
                "phase": "search",
                "configuration": name,
                "seed": SEED,
                "mean_roc_auc": float(np.mean(scores)),
                "std_roc_auc": float(np.std(scores, ddof=1)),
                "fold_scores": ",".join(f"{score:.6f}" for score in scores),
                "parameters": json.dumps(params, sort_keys=True),
            }
            writer.writerow(row)
            file.flush()
            completed.append((name, params, scores))
            print(
                f"{name}: folds={[round(score, 5) for score in scores]}, "
                f"mean={np.mean(scores):.5f}, std={np.std(scores, ddof=1):.5f}",
                flush=True,
            )

        best_name, best_params, _ = max(completed, key=lambda item: np.mean(item[2]))
        print(f"\nConfirming {best_name} and current_tuned with seed 123...", flush=True)
        confirmation = []
        for name in dict.fromkeys(["current_tuned", best_name]):
            scores = evaluate(X, y, configurations[name], 123)
            row = {
                "phase": "confirmation",
                "configuration": name,
                "seed": 123,
                "mean_roc_auc": float(np.mean(scores)),
                "std_roc_auc": float(np.std(scores, ddof=1)),
                "fold_scores": ",".join(f"{score:.6f}" for score in scores),
                "parameters": json.dumps(configurations[name], sort_keys=True),
            }
            writer.writerow(row)
            file.flush()
            confirmation.append((name, scores))
            print(
                f"confirmation {name}: folds={[round(score, 5) for score in scores]}, "
                f"mean={np.mean(scores):.5f}",
                flush=True,
            )

    final_model = make_pipeline(X, best_params)
    final_model.fit(X, y)
    probabilities = final_model.predict_proba(X_test)[:, 1]
    pd.DataFrame({ID: unlabelled[ID], TARGET: probabilities}).to_csv(
        candidate_path, index=False
    )
    print(f"\nBest search setting: {best_name}")
    print(f"Iteration results recorded at: {results_path}")
    print(f"Separate candidate saved at: {candidate_path}")


if __name__ == "__main__":
    main()
