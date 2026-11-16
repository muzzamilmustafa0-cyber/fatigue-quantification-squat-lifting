import sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
ANALYSIS = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ANALYSIS))
from traditional_ml import build_classifiers, build_regressors
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, r2_score, mean_absolute_error
from scipy.stats import wilcoxon

d = pd.read_csv(ANALYSIS / "results" / "dataset_classification.csv")
meta = ["Subject", "Subject_idx", "Load_kg", "Squat_No", "RPE", "Fatigued", "HR", "MPF_form", "Weight_kg", "Height_cm", "BMI"]
rpe_sub = ["RPE_Upper", "RPE_Middle", "RPE_Lower"]
emg = [c for c in d.columns if c not in meta + rpe_sub + ["Obs_No"]]
ar = [c for c in emg if "_AR" in c]
ratio_cols = [c for c in emg if c not in ar]

# session baseline = block 1 of the same subject and load (available after the first four lifts)
base = d[d.Obs_No == 1].set_index(["Subject", "Load_kg"])
nb = d.copy()
for c in ratio_cols + ["HR"]:
    b = d.set_index(["Subject", "Load_kg"]).index.map(base[c])
    b = np.asarray(b, dtype=float)
    nb[c + "_rel"] = d[c].values / np.where(np.abs(b) > 1e-9, b, np.nan)
for c in ar:
    b = np.asarray(d.set_index(["Subject", "Load_kg"]).index.map(base[c]), dtype=float)
    nb[c + "_rel"] = d[c].values - b
rel = [c + "_rel" for c in ratio_cols + ar]
nb[rel + ["HR_rel"]] = nb[rel + ["HR_rel"]].replace([np.inf, -np.inf], np.nan)

SETS = {
    "B: EMG + HR (absolute)": (d, emg + ["HR"]),
    "E: EMG + HR (baseline-referenced)": (nb, rel + ["HR_rel"]),
    "F: E + load": (nb, rel + ["HR_rel", "Load_kg"]),
    "G: F + elapsed block": (nb, rel + ["HR_rel", "Load_kg", "Obs_No"]),
}
y = d.Fatigued.values.astype(int); subj = d.Subject_idx.values; rpe = d.RPE.values.astype(float)
rows, folds, probs = [], {}, {}
for sname, (df, cols) in SETS.items():
    X = df[cols].values.astype(np.float32)
    for m in ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]:
        pipe = build_classifiers()[m]; p = np.zeros(len(y)); f1s = []
        for u in np.unique(subj):
            te = subj == u; pipe.fit(X[~te], y[~te]); p[te] = pipe.predict_proba(X[te])[:, 1]
            f1s.append(f1_score(y[te], (p[te] >= .5).astype(int), average="macro", zero_division=0))
        pr = (p >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(y, pr).ravel()
        folds[(sname, m)] = np.array(f1s); probs[(sname, m)] = p
        rows.append(dict(Set=sname, Model=m, Acc=accuracy_score(y, pr), F1=f1_score(y, pr, average="macro"),
                         AUC=roc_auc_score(y, p), Sens=tp / (tp + fn), Spec=tn / (tn + fp)))
    reg = build_regressors()["XGBoost"]; pr = np.zeros(len(y))
    for u in np.unique(subj):
        te = subj == u; reg.fit(X[~te], rpe[~te]); pr[te] = reg.predict(X[te])
    rows[-1].update(R2_XGB=r2_score(rpe, pr), MAE_XGB=mean_absolute_error(rpe, pr))
    print("done", sname, flush=True)
res = pd.DataFrame(rows).round(3); pd.set_option("display.width", 200); print(res.to_string())
res.to_csv(OUT / "baseline_norm.csv", index=False)
for s in SETS:
    for m in ["Random Forest", "XGBoost", "LightGBM"]:
        print(s, m, "vs LR p=", round(wilcoxon(folds[(s, m)], folds[(s, "Logistic Regression")]).pvalue, 4))
for m in ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]:
    print(m, "F vs B p=", round(wilcoxon(folds[("F: E + load", m)], folds[("B: EMG + HR (absolute)", m)]).pvalue, 4))
pd.DataFrame({f"{s}|{m}": v for (s, m), v in probs.items()}).assign(y=y, Load=d.Load_kg, Block=d.Obs_No, Subject=d.Subject).to_csv(OUT / "probs_norm.csv", index=False)
