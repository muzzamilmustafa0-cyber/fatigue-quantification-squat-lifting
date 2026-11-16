import sys, warnings, json
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
ANALYSIS = Path(sys.argv[1])
OUT = Path(sys.argv[2])
sys.path.insert(0, str(ANALYSIS))
from traditional_ml import build_classifiers, build_regressors, clf_metrics, reg_metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, confusion_matrix, r2_score, mean_absolute_error
from scipy.stats import wilcoxon

d = pd.read_csv(ANALYSIS / "results" / "dataset_classification.csv")
meta = ["Subject", "Subject_idx", "Load_kg", "Squat_No", "RPE", "Fatigued", "HR", "MPF_form", "Weight_kg", "Height_cm", "BMI"]
rpe_sub = ["RPE_Upper", "RPE_Middle", "RPE_Lower"]
emg = [c for c in d.columns if c not in meta + rpe_sub + ["Obs_No"]]
sets = {
    "A_published (EMG+RPE sub-scales+block)": emg + rpe_sub + ["Obs_No"],
    "B_EMG+HR (physiological only)": emg + ["HR"],
    "C_EMG only": emg,
    "D_EMG+HR+elapsed block": emg + ["HR", "Obs_No"],
}
y = d.Fatigued.values.astype(int); rpe = d.RPE.values.astype(float)
subj = d.Subject_idx.values
keep = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]

rows, regrows, folds = [], [], {}
for sname, cols in sets.items():
    X = d[cols].values.astype(np.float32)
    clfs = build_classifiers()
    for m in keep:
        pipe = clfs[m]
        prob = np.full(len(y), np.nan); f1s = []
        for s in np.unique(subj):
            te = subj == s; tr = ~te
            pipe.fit(X[tr], y[tr])
            p = pipe.predict_proba(X[te])[:, 1]; prob[te] = p
            f1s.append(f1_score(y[te], (p >= .5).astype(int), average="macro", zero_division=0))
        pred = (prob >= .5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        folds[(sname, m)] = np.array(f1s)
        rows.append(dict(FeatureSet=sname, Model=m, nFeat=len(cols),
                         Acc=accuracy_score(y, pred), F1=f1_score(y, pred, average="macro"),
                         AUC=roc_auc_score(y, prob), Sens=tp / (tp + fn), Spec=tn / (tn + fp),
                         FoldF1_mean=np.mean(f1s), FoldF1_sd=np.std(f1s, ddof=1)))
    regs = build_regressors()
    for m in ["Random Forest", "XGBoost", "LightGBM"]:
        pipe = regs[m]; pr = np.full(len(y), np.nan)
        for s in np.unique(subj):
            te = subj == s; tr = ~te
            pipe.fit(X[tr], rpe[tr]); pr[te] = pipe.predict(X[te])
        regrows.append(dict(FeatureSet=sname, Model=m, R2=r2_score(rpe, pr), MAE=mean_absolute_error(rpe, pr)))
    print("done", sname, flush=True)

res = pd.DataFrame(rows).round(3); reg = pd.DataFrame(regrows).round(3)
res.to_csv(OUT / "sens_classification.csv", index=False); reg.to_csv(OUT / "sens_regression.csv", index=False)
print(res.to_string()); print(reg.to_string())

B = "B_EMG+HR (physiological only)"
for m in ["Random Forest", "XGBoost", "LightGBM"]:
    a, b = folds[(B, m)], folds[(B, "Logistic Regression")]
    print("Wilcoxon B", m, "vs LR p=", round(wilcoxon(a, b).pvalue, 4))
a, b = folds[(B, "Random Forest")], folds[("A_published (EMG+RPE sub-scales+block)", "Random Forest")]
print("Wilcoxon RF B vs A p=", round(wilcoxon(a, b).pvalue, 4))

import shap
X = d[sets[B]].values.astype(np.float32)
pipe = build_classifiers()["Random Forest"]; pipe.fit(X, y)
Xi = pipe.named_steps["imp"].transform(X)
sv = shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Xi)
sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
imp = pd.DataFrame({"Feature": sets[B], "SHAP": np.abs(sv).mean(0)}).sort_values("SHAP", ascending=False)
imp.to_csv(OUT / "sens_shap_B.csv", index=False); print(imp.head(15).to_string())
