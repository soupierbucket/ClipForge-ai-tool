"""Fresh, isolated CV study: baseline XGBoost vs CatBoost variants."""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "uts-mdsi-competition-2026"
OUTPUT_DIR = HERE
ID = "student_id"
TARGET = "successful_outcome"
SEED = 42


def make_xgb_pipeline(X: pd.DataFrame, estimator: XGBClassifier) -> Pipeline:
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
    return Pipeline([("preprocess", prep), ("model", estimator)])


def xgb_estimator() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=700,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )


def catboost_estimator(depth: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=1400,
        learning_rate=0.035,
        depth=depth,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=5.0,
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def run_xgb_cv(X: pd.DataFrame, y: pd.Series, splits, categorical_suburb: bool):
    working = X.copy()
    if categorical_suburb:
        working["suburb_id"] = working["suburb_id"].astype("string")
    oof = np.zeros(len(y), dtype=float)
    fold_scores = []
    for train_idx, valid_idx in splits:
        model = make_xgb_pipeline(working.iloc[train_idx], xgb_estimator())
        model.fit(working.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict_proba(working.iloc[valid_idx])[:, 1]
        oof[valid_idx] = pred
        fold_scores.append(roc_auc_score(y.iloc[valid_idx], pred))
    return oof, fold_scores


def run_catboost_cv(X: pd.DataFrame, y: pd.Series, splits, depth: int):
    working = X.copy()
    cat_columns = working.select_dtypes(exclude="number").columns.tolist()
    if "suburb_id" in working.columns:
        cat_columns.append("suburb_id")
    for column in cat_columns:
        working[column] = working[column].fillna("__MISSING__").astype(str)
    oof = np.zeros(len(y), dtype=float)
    fold_scores = []
    for train_idx, valid_idx in splits:
        model = catboost_estimator(depth)
        model.fit(
            working.iloc[train_idx],
            y.iloc[train_idx],
            cat_features=cat_columns,
        )
        pred = model.predict_proba(working.iloc[valid_idx])[:, 1]
        oof[valid_idx] = pred
        fold_scores.append(roc_auc_score(y.iloc[valid_idx], pred))
    return oof, fold_scores, cat_columns


def fit_full_candidate(name: str, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
    if name.startswith("xgboost"):
        train_frame = X.copy()
        test_frame = X_test.copy()
        if name == "xgboost_suburb_as_category":
            train_frame["suburb_id"] = train_frame["suburb_id"].astype("string")
            test_frame["suburb_id"] = test_frame["suburb_id"].astype("string")
        model = make_xgb_pipeline(train_frame, xgb_estimator())
        model.fit(train_frame, y)
        return model.predict_proba(test_frame)[:, 1]

    train_frame = X.copy()
    test_frame = X_test.copy()
    cat_columns = train_frame.select_dtypes(exclude="number").columns.tolist()
    cat_columns.append("suburb_id")
    for column in cat_columns:
        train_frame[column] = train_frame[column].fillna("__MISSING__").astype(str)
        test_frame[column] = test_frame[column].fillna("__MISSING__").astype(str)
    depth = 6 if name == "catboost_depth_6" else 8
    model = catboost_estimator(depth)
    model.fit(train_frame, y, cat_features=cat_columns)
    return model.predict_proba(test_frame)[:, 1]


def main() -> None:
    merged = pd.read_csv(DATA_DIR / "combined_students.csv")
    engineered = pd.read_csv(DATA_DIR / "model_ready_students.csv")
    labelled = engineered[engineered[TARGET].notna()].copy()
    unlabelled = engineered[engineered[TARGET].isna()].copy()
    y = labelled[TARGET].astype(int)

    # Recreate the winning reference input: merged features, five normalized
    # monthly trends, and the corrected monthly work/study ratio.
    base_features = [c for c in merged.columns if c not in (ID, TARGET)]
    trend_features = [c for c in engineered.columns if c.startswith("monthly_change_")]
    selected_features = list(dict.fromkeys(base_features + trend_features + ["work_to_study_ratio"]))
    X = labelled[selected_features].copy()
    X_test = unlabelled[selected_features].copy()

    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))
    predictions = {}
    results = []

    experiments = [
        ("xgboost_53_reference", "xgboost", None),
        ("xgboost_suburb_as_category", "xgboost", "suburb_category"),
        ("catboost_depth_6", "catboost", 6),
        ("catboost_depth_8", "catboost", 8),
    ]
    for name, model_type, option in experiments:
        if model_type == "xgboost":
            oof, scores = run_xgb_cv(X, y, splits, option == "suburb_category")
        else:
            oof, scores, _ = run_catboost_cv(X, y, splits, option)
        predictions[name] = oof
        results.append({"model": name, "mean_roc_auc": np.mean(scores), "std_roc_auc": np.std(scores, ddof=1),
                        "fold_scores": ",".join(f"{score:.6f}" for score in scores)})
        print(f"{name}: folds={[round(score, 5) for score in scores]}, mean={np.mean(scores):.5f}, std={np.std(scores, ddof=1):.5f}")

    blend_results = []
    reference = predictions["xgboost_53_reference"]
    for candidate in ["catboost_depth_6", "catboost_depth_8", "xgboost_suburb_as_category"]:
        for weight in (0.25, 0.5, 0.75):
            blended = (1 - weight) * reference + weight * predictions[candidate]
            fold_scores = [
                roc_auc_score(y.iloc[valid_idx], blended[valid_idx])
                for _, valid_idx in splits
            ]
            name = f"blend_xgb_cat_candidate_{weight:.2f}_{candidate}"
            blend_results.append((np.mean(fold_scores), name, candidate, weight, blended, fold_scores))
            print(f"{name}: folds={[round(score, 5) for score in fold_scores]}, mean={np.mean(fold_scores):.5f}")

    single_results = list(results)
    for mean_score, blend_name, _, _, _, scores in blend_results:
        results.append({"model": blend_name, "mean_roc_auc": mean_score,
                        "std_roc_auc": np.std(scores, ddof=1),
                        "fold_scores": ",".join(f"{score:.6f}" for score in scores)})
    pd.DataFrame(results).sort_values("mean_roc_auc", ascending=False).to_csv(
        OUTPUT_DIR / "model_results.csv", index=False
    )
    best_single = max(single_results, key=lambda row: row["mean_roc_auc"])
    best_blend = max(blend_results, key=lambda row: row[0])
    use_blend = best_blend[0] > best_single["mean_roc_auc"]

    if use_blend:
        _, best_name, candidate, weight, _, _ = best_blend
        print(f"\nSelected CV candidate: {best_name}, mean={best_blend[0]:.5f}")
        xgb_pred = fit_full_candidate("xgboost_53_reference", X, y, X_test)
        candidate_pred = fit_full_candidate(candidate, X, y, X_test)
        final_probabilities = (1 - weight) * xgb_pred + weight * candidate_pred
    else:
        best_name = best_single["model"]
        print(f"\nSelected CV candidate: {best_name}, mean={best_single['mean_roc_auc']:.5f}")
        final_probabilities = fit_full_candidate(best_name, X, y, X_test)

    pd.DataFrame({ID: unlabelled[ID], TARGET: final_probabilities}).to_csv(
        OUTPUT_DIR / "best_candidate_submission.csv", index=False
    )
    print(f"Saved isolated candidate to {OUTPUT_DIR / 'best_candidate_submission.csv'}")


if __name__ == "__main__":
    main()
