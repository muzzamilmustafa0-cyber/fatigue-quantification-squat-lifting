import sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
warnings.filterwarnings("ignore")
ANALYSIS = Path(sys.argv[1]); OUT = Path(sys.argv[2]); SENS = Path(sys.argv[3])
sys.path.insert(0, str(ANALYSIS))
from traditional_ml import build_classifiers
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "savefig.dpi": 600, "svg.fonttype": "path"})
BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"

d = pd.read_csv(ANALYSIS / "results" / "dataset_classification.csv")
P = pd.read_csv(SENS / "probs_norm.csv"); y = P.y.values
B, F = "B: EMG + HR (absolute)", "F: E + load"

LI = {4: 0.41, 8: 0.82}
pF = (P[f"{F}|Logistic Regression"] >= .5).astype(int).values
rows = []
for L in (4, 8):
    k = d.Load_kg.values == L; yk = d.Fatigued.values
    onset = d[k & (yk == 1)].groupby("Subject").Obs_No.min()
    rows.append(dict(Load=L, LI=LI[L], n_subj=d[k].Subject.nunique(), obs=int(k.sum()), pct_fat_obs=100 * yk[k].mean(),
                     pct_fat_b6=100 * yk[k & (d.Obs_No.values == 6)].mean(), subj_reach=onset.size,
                     onset_median_lifts=4 * onset.median() if onset.size else np.nan,
                     rpe_b1=d[k & (d.Obs_No == 1)].RPE.mean(), rpe_b6=d[k & (d.Obs_No == 6)].RPE.mean(),
                     hr_b1=d[k & (d.Obs_No == 1)].HR.mean(), hr_b6=d[k & (d.Obs_No == 6)].HR.mean(),
                     sens=100 * pF[k & (yk == 1)].mean(), spec=100 * (1 - pF[k & (yk == 0)].mean())))
nt = pd.DataFrame(rows).round(1); nt.to_csv(OUT / "niosh_table.csv", index=False); print(nt.to_string())

fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7)); blk = np.arange(1, 7)
for L, c, mk, ls in ((4, BLUE, "o", "-"), (8, ORANGE, "s", "--")):
    k = d.Load_kg == L
    rp = np.array([d[k & (d.Obs_No == b)].RPE.mean() for b in blk]); se = np.array([d[k & (d.Obs_No == b)].RPE.sem() for b in blk])
    pr = [100 * d[k & (d.Obs_No == b)].Fatigued.mean() for b in blk]
    lab = f"{L} kg (LI = {LI[L]:.2f})"
    axs[0].plot(blk * 4, rp, color=c, marker=mk, ms=5, lw=2, ls=ls, label=lab); axs[0].fill_between(blk * 4, rp - se, rp + se, color=c, alpha=.15, lw=0)
    axs[1].plot(blk * 4, pr, color=c, marker=mk, ms=5, lw=2, ls=ls, label=lab)
    axs[1].annotate(f"{pr[-1]:.0f}%", (24, pr[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
axs[0].axhline(5, color=INK2, lw=.8, ls=":"); axs[0].text(4.2, 5.15, "Fatigue threshold (RPE = 5, 'hard')", fontsize=7.2, color=INK2)
axs[0].set(xlabel="Cumulative lifts", ylabel="Borg CR-10 (mean ± SEM)", xticks=blk * 4, ylim=(0, 7.5), title="(a) Perceived exertion")
axs[1].set(xlabel="Cumulative lifts", ylabel="Blocks rated fatigued (%)", xticks=blk * 4, ylim=(0, 100), title="(b) Prevalence of fatigued state")
for a in axs:
    a.grid(axis="y", color=GRID, lw=.6); a.set_axisbelow(True); a.title.set_fontsize(9); a.legend(frameon=False, fontsize=7.6, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig_niosh.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
fig.tight_layout(); fig.savefig(OUT / "fig_niosh.svg", bbox_inches="tight", facecolor="white"); plt.close(fig)

fig = plt.figure(figsize=(7.4, 3.0)); gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.25, .8, .8], wspace=.5)
cols = {"Logistic Regression": (INK2, ":"), "Random Forest": (BLUE, "-"), "XGBoost": (ORANGE, "--"), "LightGBM": (AQUA, "-.")}
for i, (s, ttl) in enumerate(((B, "(a) Absolute features"), (F, "(b) Baseline-referenced"))):
    a = fig.add_subplot(gs[i])
    for m, (c, ls) in cols.items():
        p = P[f"{s}|{m}"].values; f_, t_, _ = roc_curve(y, p)
        a.plot(f_, t_, color=c, ls=ls, lw=1.6, label=f"{m.replace('Logistic Regression', 'LR')} ({roc_auc_score(y, p):.2f})")
    a.plot([0, 1], [0, 1], color=GRID, lw=.8); a.set(xlabel="False-positive rate", ylabel="True-positive rate" if i == 0 else "", xlim=(0, 1), ylim=(0, 1.01))
    a.set_title(ttl, fontsize=8.4); a.legend(frameon=False, fontsize=6.6, loc="lower right", title="AUC", title_fontsize=6.6)
for i, (s, m, ttl) in enumerate(((B, "Random Forest", "(c) Random Forest,\nabsolute"), (F, "Logistic Regression", "(d) LR,\nbaseline-referenced"))):
    a = fig.add_subplot(gs[2 + i]); cm = confusion_matrix(y, (P[f"{s}|{m}"] >= .5).astype(int))
    a.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
    for (r, c_), v in np.ndenumerate(cm):
        a.text(c_, r, f"{v}\n({100 * v / cm[r].sum():.0f}%)", ha="center", va="center", fontsize=7.4, color="white" if v > cm.max() * .6 else INK)
    a.set_xticks([0, 1], ["NF", "F"]); a.set_yticks([0, 1], ["NF", "F"]); a.set_xlabel("Predicted"); a.set_ylabel("Actual" if i == 0 else "")
    a.set_title(ttl, fontsize=8.4); [sp.set_visible(False) for sp in a.spines.values()]
fig.savefig(OUT / "fig_loso.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
fig.savefig(OUT / "fig_loso.svg", bbox_inches="tight", facecolor="white"); plt.close(fig)

import shap
meta = ["Subject", "Subject_idx", "Load_kg", "Squat_No", "RPE", "Fatigued", "HR", "MPF_form", "Weight_kg", "Height_cm", "BMI"]
rpe_sub = ["RPE_Upper", "RPE_Middle", "RPE_Lower"]
emg = [c for c in d.columns if c not in meta + rpe_sub + ["Obs_No"]]
ar = [c for c in emg if "_AR" in c]; ratio_cols = [c for c in emg if c not in ar]
base = d[d.Obs_No == 1].set_index(["Subject", "Load_kg"]); idx = d.set_index(["Subject", "Load_kg"]).index
X = pd.DataFrame(index=d.index)
for c in ratio_cols + ["HR"]:
    b = np.asarray(idx.map(base[c]), dtype=float); X[c] = d[c].values / np.where(np.abs(b) > 1e-9, b, np.nan)
for c in ar:
    X[c] = d[c].values - np.asarray(idx.map(base[c]), dtype=float)
X["Load"] = d.Load_kg.values; X = X.replace([np.inf, -np.inf], np.nan)
pipe = build_classifiers()["XGBoost"]; pipe.fit(X.values.astype(np.float32), d.Fatigued.values)
Xi = pipe.named_steps["imp"].transform(X.values.astype(np.float32))
sv = shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Xi)
sv = sv[1] if isinstance(sv, list) else sv
imp = pd.DataFrame({"Feature": X.columns, "SHAP": np.abs(sv).mean(0)}).sort_values("SHAP", ascending=False)
imp.to_csv(OUT / "shap_F.csv", index=False); print(imp.head(12).to_string())
sh = imp.head(15).iloc[::-1]
ren = {"DeltMed": "MD", "DeltAnt": "AD", "Supraspinatus": "SP", "VastLat": "VL", "BicepsFem": "BF", "RectusFem": "RF", "Trapezius": "UT", "Trap": "UT", "Upper": "Upper-limb", "Lower": "Lower-limb", "Middle": "Mid-chain",
       "Gastrocnemius": "GS", "SpectralEntropy": "SpEn", "PeakFreq": "PF"}
def lab(f):
    for k, v in ren.items(): f = f.replace(k, v)
    return f.replace("_", " ")
def cat(f):
    if f in ("HR", "Load"): return ORANGE
    if any(k in f for k in ("MPF", "MDF", "Peak", "Spectral", "_AR")): return BLUE
    return AQUA
fig, a = plt.subplots(figsize=(4.2, 3.4))
a.barh(range(len(sh)), sh.SHAP, color=[cat(f) for f in sh.Feature], height=.7)
a.set_yticks(range(len(sh)), [lab(f) for f in sh.Feature], fontsize=7.6); a.set_xlabel("Mean |SHAP value| (log-odds)")
a.grid(axis="x", color=GRID, lw=.6); a.set_axisbelow(True)
a.legend(handles=[Patch(color=BLUE, label="Spectral / AR"), Patch(color=AQUA, label="Amplitude / time"), Patch(color=ORANGE, label="Heart rate / load")],
         frameon=False, fontsize=7.4, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig_shapF.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
fig.tight_layout(); fig.savefig(OUT / "fig_shapF.svg", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("figures2 done")
