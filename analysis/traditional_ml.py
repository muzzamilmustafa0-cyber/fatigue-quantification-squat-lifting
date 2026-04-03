import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.linear_model    import LogisticRegression, Ridge
from sklearn.svm             import SVC, SVR
from sklearn.tree            import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble        import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, cohen_kappa_score,
                                     mean_absolute_error, mean_squared_error, r2_score)
from sklearn.impute          import SimpleImputer

try:
    from xgboost  import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed — skipping XGBoost models")

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[WARN] lightgbm not installed — skipping LightGBM models")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_JOBS       = -1   # Use all CPU cores
CV_FOLDS     = 10


def build_classifiers() -> dict:
    clfs = {
        "Logistic Regression": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0,
                                       solver="lbfgs",
                                       random_state=RANDOM_STATE)),
        ]),
        "SVM (RBF)": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", RobustScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale",
                        probability=True, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(max_depth=8,
                                           random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                           random_state=RANDOM_STATE, n_jobs=N_JOBS)),
        ]),
    }
    if HAS_XGB:
        clfs["XGBoost"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(n_estimators=200, max_depth=6,
                                  learning_rate=0.05, subsample=0.8,
                                  eval_metric="logloss",
                                  random_state=RANDOM_STATE, n_jobs=N_JOBS,
                                  verbosity=0)),
        ])
    if HAS_LGB:
        clfs["LightGBM"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", LGBMClassifier(n_estimators=200, num_leaves=31,
                                   learning_rate=0.05, random_state=RANDOM_STATE,
                                   n_jobs=N_JOBS, verbose=-1)),
        ])
    return clfs


def build_regressors() -> dict:
    regs = {
        "Logistic Regression (Ridge)": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ]),
        "SVM (RBF)": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", RobustScaler()),
            ("reg", SVR(kernel="rbf", C=10.0, gamma="scale")),
        ]),
        "Decision Tree": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("reg", DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("reg", RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                          random_state=RANDOM_STATE, n_jobs=N_JOBS)),
        ]),
    }
    if HAS_XGB:
        regs["XGBoost"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("reg", XGBRegressor(n_estimators=200, max_depth=6,
                                 learning_rate=0.05, subsample=0.8,
                                 random_state=RANDOM_STATE, n_jobs=N_JOBS,
                                 verbosity=0)),
        ])
    if HAS_LGB:
        regs["LightGBM"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("reg", LGBMRegressor(n_estimators=200, num_leaves=31,
                                  learning_rate=0.05, random_state=RANDOM_STATE,
                                  n_jobs=N_JOBS, verbose=-1)),
        ])
    return regs


def clf_metrics(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Kappa":     cohen_kappa_score(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            metrics["AUC"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["AUC"] = np.nan
    return metrics


def reg_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mse),
        "R2":   r2_score(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100,
    }


def run_kfold_classification(X: np.ndarray, y: np.ndarray,
                              classifiers: dict, n_splits: int = CV_FOLDS) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    records = []
    for name, pipe in tqdm(classifiers.items(), desc="k-Fold classification"):
        fold_metrics = []
        for train_idx, test_idx in skf.split(X, y):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            pipe.fit(Xtr, ytr)
            ypred = pipe.predict(Xte)
            yprob = pipe.predict_proba(Xte)[:, 1] if hasattr(pipe, "predict_proba") else None
            fold_metrics.append(clf_metrics(yte, ypred, yprob))

        agg = pd.DataFrame(fold_metrics).agg(["mean", "std"])
        rec = {"Model": name}
        for col in agg.columns:
            rec[f"{col}_mean"] = round(agg.loc["mean", col], 4)
            rec[f"{col}_std"]  = round(agg.loc["std",  col], 4)
        records.append(rec)

    return pd.DataFrame(records)


def run_kfold_regression(X: np.ndarray, y: np.ndarray,
                          regressors: dict, n_splits: int = CV_FOLDS) -> pd.DataFrame:
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    # For regression use KFold
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    records = []
    for name, pipe in tqdm(regressors.items(), desc="k-Fold regression"):
        fold_metrics = []
        for train_idx, test_idx in kf.split(X, y):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            pipe.fit(Xtr, ytr)
            ypred = pipe.predict(Xte)
            fold_metrics.append(reg_metrics(yte, ypred))

        agg = pd.DataFrame(fold_metrics).agg(["mean", "std"])
        rec = {"Model": name}
        for col in agg.columns:
            rec[f"{col}_mean"] = round(agg.loc["mean", col], 4)
            rec[f"{col}_std"]  = round(agg.loc["std",  col], 4)
        records.append(rec)

    return pd.DataFrame(records)


def run_loso_classification(X: np.ndarray, y: np.ndarray,
                             subjects: np.ndarray, classifiers: dict) -> pd.DataFrame:
    unique_subjects = np.unique(subjects)
    records = []
    for name, pipe in tqdm(classifiers.items(), desc="LOSO classification"):
        fold_metrics = []
        for subj in unique_subjects:
            test_mask  = subjects == subj
            train_mask = ~test_mask
            if train_mask.sum() < 10 or test_mask.sum() < 2:
                continue
            Xtr, Xte = X[train_mask], X[test_mask]
            ytr, yte = y[train_mask], y[test_mask]
            if len(np.unique(ytr)) < 2:
                continue
            try:
                pipe.fit(Xtr, ytr)
                ypred = pipe.predict(Xte)
                yprob = pipe.predict_proba(Xte)[:, 1] if hasattr(pipe, "predict_proba") else None
                fold_metrics.append(clf_metrics(yte, ypred, yprob))
            except Exception as e:
                print(f"  [WARN] {name} LOSO fold {subj} failed: {e}")
                continue

        if not fold_metrics:
            continue
        agg = pd.DataFrame(fold_metrics).agg(["mean", "std"])
        rec = {"Model": name, "CV": "LOSO", "N_Subjects": len(unique_subjects)}
        for col in agg.columns:
            rec[f"{col}_mean"] = round(agg.loc["mean", col], 4)
            rec[f"{col}_std"]  = round(agg.loc["std",  col], 4)
        records.append(rec)

    return pd.DataFrame(records)


def run_loso_regression(X: np.ndarray, y: np.ndarray,
                         subjects: np.ndarray, regressors: dict) -> pd.DataFrame:
    unique_subjects = np.unique(subjects)
    records = []
    for name, pipe in tqdm(regressors.items(), desc="LOSO regression"):
        fold_metrics = []
        for subj in unique_subjects:
            test_mask  = subjects == subj
            train_mask = ~test_mask
            if train_mask.sum() < 10 or test_mask.sum() < 2:
                continue
            Xtr, Xte = X[train_mask], X[test_mask]
            ytr, yte = y[train_mask], y[test_mask]
            try:
                pipe.fit(Xtr, ytr)
                ypred = pipe.predict(Xte)
                fold_metrics.append(reg_metrics(yte, ypred))
            except Exception as e:
                print(f"  [WARN] {name} LOSO fold {subj} failed: {e}")
                continue

        if not fold_metrics:
            continue
        agg = pd.DataFrame(fold_metrics).agg(["mean", "std"])
        rec = {"Model": name, "CV": "LOSO", "N_Subjects": len(unique_subjects)}
        for col in agg.columns:
            rec[f"{col}_mean"] = round(agg.loc["mean", col], 4)
            rec[f"{col}_std"]  = round(agg.loc["std",  col], 4)
        records.append(rec)

    return pd.DataFrame(records)


def wilcoxon_vs_baseline(loso_df: pd.DataFrame,
                          X: np.ndarray, y: np.ndarray,
                          subjects: np.ndarray,
                          classifiers: dict,
                          metric: str = "F1") -> pd.DataFrame:
    from scipy.stats import wilcoxon
    unique_subjects = np.unique(subjects)

    model_fold_scores = {}
    for name, pipe in classifiers.items():
        scores = []
        for subj in unique_subjects:
            test_mask  = subjects == subj
            train_mask = ~test_mask
            if train_mask.sum() < 10 or test_mask.sum() < 2:
                continue
            Xtr, Xte = X[train_mask], X[test_mask]
            ytr, yte = y[train_mask], y[test_mask]
            if len(np.unique(ytr)) < 2:
                continue
            try:
                pipe.fit(Xtr, ytr)
                ypred = pipe.predict(Xte)
                scores.append(f1_score(yte, ypred, average="macro", zero_division=0))
            except Exception:
                scores.append(np.nan)
        model_fold_scores[name] = np.array(scores)

    baseline_name = "Logistic Regression"
    if baseline_name not in model_fold_scores:
        return pd.DataFrame()

    baseline_scores = model_fold_scores[baseline_name]
    records = []
    for name, scores in model_fold_scores.items():
        if name == baseline_name:
            continue
        valid = ~(np.isnan(scores) | np.isnan(baseline_scores))
        if valid.sum() < 5:
            p = np.nan
        else:
            try:
                _, p = wilcoxon(scores[valid], baseline_scores[valid])
            except Exception:
                p = np.nan
        records.append({
            "Model":      name,
            "Baseline":   baseline_name,
            "Metric":     metric,
            "p_value":    round(p, 4) if not np.isnan(p) else np.nan,
            "Significant": "Yes" if (not np.isnan(p) and p < 0.05) else "No",
        })

    return pd.DataFrame(records)


def compute_shap_importance(pipe, X_train: np.ndarray, X_test: np.ndarray,
                             feature_names: list) -> pd.DataFrame:
    try:
        import shap
        clf = pipe.named_steps.get("clf") or pipe.named_steps.get("reg")
        # Transform X through pipeline steps up to classifier
        X_tr_scaled = X_train.copy()
        for step_name, step in pipe.steps[:-1]:
            X_tr_scaled = step.transform(X_tr_scaled)
        X_te_scaled = X_test.copy()
        for step_name, step in pipe.steps[:-1]:
            X_te_scaled = step.transform(X_te_scaled)

        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_te_scaled)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]   # class 1 (Fatigued)
        importance = np.abs(shap_vals).mean(axis=0)
        df = pd.DataFrame({"Feature": feature_names, "SHAP_importance": importance})
        return df.sort_values("SHAP_importance", ascending=False).reset_index(drop=True)
    except Exception as e:
        print(f"  [WARN] SHAP computation failed: {e}")
        # Fallback: feature importance from tree model
        clf = pipe.named_steps.get("clf") or pipe.named_steps.get("reg")
        if hasattr(clf, "feature_importances_"):
            df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP_importance": clf.feature_importances_
            })
            return df.sort_values("SHAP_importance", ascending=False).reset_index(drop=True)
        return pd.DataFrame({"Feature": feature_names, "SHAP_importance": np.zeros(len(feature_names))})


def run_traditional_ml(verbose: bool = True):
    clf_path = RESULTS_DIR / "dataset_classification.csv"
    reg_path = RESULTS_DIR / "dataset_regression.csv"

    if not clf_path.exists():
        print("[ERROR] dataset_classification.csv not found. Run 04_dataset_builder.py first.")
        return None, None

    clf_df = pd.read_csv(clf_path)
    reg_df = pd.read_csv(reg_path)

    # Identify feature columns (exclude meta columns)
    meta_cols = ["Subject", "Subject_idx", "Load_kg", "Squat_No",
                 "RPE", "Fatigued", "HR", "MPF_form", "Weight_kg", "Height_cm", "BMI"]
    feat_cols = [c for c in clf_df.columns if c not in meta_cols]

    X_clf  = clf_df[feat_cols].values.astype(np.float32)
    y_clf  = clf_df["Fatigued"].values.astype(int)
    y_rpe  = reg_df["RPE"].values.astype(np.float32)
    subj   = clf_df["Subject_idx"].values.astype(int)

    if verbose:
        print(f"\n  Feature matrix : {X_clf.shape}")
        print(f"  Class balance  : {y_clf.mean()*100:.1f}% fatigued")
        print(f"  RPE range      : {y_rpe.min():.1f} – {y_rpe.max():.1f}")

    classifiers = build_classifiers()
    regressors  = build_regressors()

    if verbose: print("\n[A] 10-Fold CV — Classification ...")
    kfold_clf_df = run_kfold_classification(X_clf, y_clf, classifiers)
    kfold_clf_df.to_csv(RESULTS_DIR / "cv10_results_classification.csv", index=False)

    if verbose: print("\n[B] 10-Fold CV — Regression ...")
    kfold_reg_df = run_kfold_regression(X_clf, y_rpe, regressors)
    kfold_reg_df.to_csv(RESULTS_DIR / "cv10_results_regression.csv", index=False)

    if verbose: print("\n[C] LOSO CV — Classification ...")
    loso_clf_df = run_loso_classification(X_clf, y_clf, subj, classifiers)
    loso_clf_df.to_csv(RESULTS_DIR / "loso_results_classification.csv", index=False)

    if verbose: print("\n[D] LOSO CV — Regression ...")
    loso_reg_df = run_loso_regression(X_clf, y_rpe, subj, regressors)
    loso_reg_df.to_csv(RESULTS_DIR / "loso_results_regression.csv", index=False)

    if verbose: print("\n[E] Wilcoxon significance tests ...")
    stat_df = wilcoxon_vs_baseline(loso_clf_df, X_clf, y_clf, subj, classifiers)
    stat_df.to_csv(RESULTS_DIR / "statistical_tests.csv", index=False)

    if not loso_clf_df.empty and "F1_mean" in loso_clf_df.columns:
        best_clf_name = loso_clf_df.loc[loso_clf_df["F1_mean"].idxmax(), "Model"]
    else:
        best_clf_name = "Random Forest"

    if verbose: print(f"\n  Best classifier (LOSO F1): {best_clf_name}")

    best_pipe = classifiers.get(best_clf_name, classifiers["Random Forest"])
    best_pipe.fit(X_clf, y_clf)
    joblib.dump(best_pipe, RESULTS_DIR / "best_model_classification.joblib")

    best_reg_name = "Random Forest"
    if not loso_reg_df.empty and "R2_mean" in loso_reg_df.columns:
        best_reg_name = loso_reg_df.loc[loso_reg_df["R2_mean"].idxmax(), "Model"]
    best_reg_pipe = regressors.get(best_reg_name, regressors["Random Forest"])
    best_reg_pipe.fit(X_clf, y_rpe)
    joblib.dump(best_reg_pipe, RESULTS_DIR / "best_model_regression.joblib")

    if verbose: print("\n[F] Computing SHAP feature importance ...")
    shap_df = compute_shap_importance(best_pipe, X_clf, X_clf, feat_cols)
    shap_df.to_csv(RESULTS_DIR / "shap_importance.csv", index=False)

    if verbose:
        print(f"\n  Results saved to: {RESULTS_DIR}")
        if not kfold_clf_df.empty:
            print("\n  10-Fold Classification Results:")
            print(kfold_clf_df[["Model", "Accuracy_mean", "F1_mean", "AUC_mean"]].to_string(index=False))

    return {
        "kfold_clf":  kfold_clf_df,
        "kfold_reg":  kfold_reg_df,
        "loso_clf":   loso_clf_df,
        "loso_reg":   loso_reg_df,
        "stat_tests": stat_df,
        "shap":       shap_df,
        "feat_cols":  feat_cols,
        "X":          X_clf,
        "y_clf":      y_clf,
        "y_rpe":      y_rpe,
        "subjects":   subj,
        "classifiers": classifiers,
        "regressors":  regressors,
    }


if __name__ == "__main__":
    results = run_traditional_ml(verbose=True)
