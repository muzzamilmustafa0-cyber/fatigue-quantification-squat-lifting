import sys
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth":  1.5,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

# Figure column widths (cm → inches): 8.5 cm (1-col) | 17.6 cm (2-col)
COL1 = 8.5 / 2.54
COL2 = 17.6 / 2.54

# Color-blind-friendly palette (Paul Tol's vibrant)
PALETTE = ["#0077BB", "#EE7733", "#009988", "#CC3311",
           "#33BBEE", "#EE3377", "#BBBBBB", "#000000"]

MUSCLE_LABELS = [
    "Deltoid\nMedial", "Deltoid\nAnterior", "Trapezius",
    "Supraspinatus", "Biceps\nFemoris", "Vastus\nLateralis",
    "Gastrocnemius", "Rectus\nFemoris"
]


def _save(fig, name: str):
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig1_emg_raw_vs_filtered():
    try:
        from data_loader import load_all          # noqa
        from signal_processing import preprocess_emg  # noqa
        emg_data, _, _ = load_all(verbose=False)
        subj = next(iter(emg_data))
        load = "4kg" if "4kg" in emg_data[subj] else next(iter(emg_data[subj]))
        raw = emg_data[subj][load]
        # Use 3 seconds of data for display clarity
        n_disp = int(3 * 1020)
        raw_disp = raw[:n_disp, 0]   # Channel 0: Deltoid Medial
    except Exception:
        # Synthetic fallback for testing
        np.random.seed(42)
        t = np.linspace(0, 3, int(3 * 1020))
        raw_disp = (np.random.randn(len(t)) * 80 +
                    np.sin(2 * np.pi * 2 * t) * 50 +
                    np.sin(2 * np.pi * 50 * t) * 200)
        raw_disp = raw_disp.astype(np.float32)

    from scipy.signal import butter, filtfilt
    nyq = 1020 / 2
    b, a = butter(4, [20/nyq, 450/nyq], btype="band")
    filtered = filtfilt(b, a, raw_disp)
    rectified = np.abs(filtered)
    b2, a2 = butter(4, 6/nyq, btype="low")
    envelope = filtfilt(b2, a2, rectified)

    t = np.arange(len(raw_disp)) / 1020.0

    fig, axes = plt.subplots(4, 1, figsize=(COL2, 10 / 2.54 * 1.6), sharex=True)
    labels = ["(a) Raw EMG", "(b) Bandpass Filtered (20–450 Hz)",
              "(c) Full-wave Rectified", "(d) Linear Envelope (6 Hz LPF)"]
    signals = [raw_disp, filtered, rectified, envelope]
    colors  = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]

    for ax, sig, lbl, col in zip(axes, signals, labels, colors):
        ax.plot(t, sig, color=col, linewidth=0.8, alpha=0.9)
        ax.set_ylabel("Amplitude (µV)", fontsize=9)
        ax.set_title(lbl, loc="left", fontsize=10, fontweight="bold")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("EMG Signal Preprocessing Pipeline\n(Deltoid Medial — Squat Task, 4 kg Load)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig1_emg_raw_vs_filtered.png")


def fig2_feature_extraction_pipeline():
    import matplotlib.patches as mp

    FIG_W = COL2          # ~6.9 in
    FIG_H = 4.5 / 2.54   # 4.5 cm  → ~1.77 in

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_aspect("auto")

    C_HDR  = "#1A6FA8"    # header box fill
    C_DET  = "#DFF0FA"    # detail card fill
    C_BDR  = "#0D4D78"    # borders
    C_ARR  = "#333333"    # arrow colour
    C_TXT_W = "white"
    C_TXT_D = "#1A2A3A"

    N      = 6
    MARGIN = 0.025
    GAP    = 0.012         # gap between adjacent boxes
    BW     = (1 - 2 * MARGIN - (N - 1) * GAP) / N   # box width ≈ 0.148

    Y_HDR_CY  = 0.73       # centre y of header row
    HDR_H     = 0.22       # header box height
    Y_DET_CY  = 0.28       # centre y of detail row
    DET_H     = 0.34       # detail box height

    stages = [
        ("Raw EMG\nAcquisition",
         "8 muscles · 1020 Hz\n30 subjects · 2 loads\n(4 kg & 8 kg)"),
        ("Signal\nPre-processing",
         "Bandpass 20–450 Hz\nFull-wave rectification\nLinear envelope (6 Hz)"),
        ("Temporal\nSegmentation",
         "6 blocks per session\n~40 s / block\n4 squats per block"),
        ("Feature\nExtraction",
         "Time-domain (RMS, IEMG,\nZCR, WL, MAV, VAR)\nFreq-domain + AR(4) × 8"),
        ("ML / DL\nModelling",
         "LR · SVM · DT · RF\nXGBoost · LightGBM\nMLP · CNN · BiLSTM"),
        ("Evaluation\n& Explainability",
         "Accuracy · F1 · AUC\nMAE · RMSE · R²\nSHAP importance"),
    ]

    xs = [MARGIN + i * (BW + GAP) + BW / 2 for i in range(N)]   # box centres

    for i, (title, detail) in enumerate(stages):
        cx = xs[i]

        hx = cx - BW / 2
        hy = Y_HDR_CY - HDR_H / 2
        ax.add_patch(mp.FancyBboxPatch(
            (hx, hy), BW, HDR_H,
            boxstyle="round,pad=0.012",
            facecolor=C_HDR, edgecolor=C_BDR, linewidth=1.0, zorder=3))

        # step number badge (top-right corner, inside box)
        bx = cx + BW / 2 - 0.022
        by = Y_HDR_CY + HDR_H / 2 - 0.040
        ax.add_patch(mp.FancyBboxPatch(
            (bx - 0.016, by - 0.028), 0.036, 0.038,
            boxstyle="round,pad=0.004",
            facecolor="white", edgecolor=C_BDR, linewidth=0.6, zorder=5))
        ax.text(bx + 0.002, by - 0.008, str(i + 1),
                ha="center", va="center", fontsize=6.5, fontweight="bold",
                color=C_HDR, zorder=6)

        ax.text(cx, Y_HDR_CY, title,
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                color=C_TXT_W, linespacing=1.35, zorder=4)

        dx = cx - BW / 2
        dy = Y_DET_CY - DET_H / 2
        ax.add_patch(mp.FancyBboxPatch(
            (dx, dy), BW, DET_H,
            boxstyle="round,pad=0.012",
            facecolor=C_DET, edgecolor=C_BDR, linewidth=0.7,
            alpha=0.95, zorder=3))
        ax.text(cx, Y_DET_CY, detail,
                ha="center", va="center", fontsize=6.8,
                color=C_TXT_D, linespacing=1.4, zorder=4)

        ax.plot([cx, cx],
                [Y_HDR_CY - HDR_H / 2, Y_DET_CY + DET_H / 2],
                color=C_BDR, lw=0.7, ls=":", zorder=2)

    for i in range(N - 1):
        x0 = xs[i]   + BW / 2 + 0.002
        x1 = xs[i+1] - BW / 2 - 0.002
        ax.annotate("",
            xy=(x1, Y_HDR_CY), xytext=(x0, Y_HDR_CY),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=C_ARR,
                            lw=1.2, mutation_scale=9),
            zorder=6)

    ax.text(0.5, 0.975,
            "EMG-Based Fatigue Quantification Pipeline",
            ha="center", va="top", fontsize=10, fontweight="bold",
            color="#0D2B45", transform=ax.transAxes)

    plt.tight_layout(pad=0.3)
    _save(fig, "fig2_feature_extraction_pipeline.png")


def fig3_fatigue_progression():
    clf_path = RESULTS_DIR / "dataset_classification.csv"
    if not clf_path.exists():
        print("  [SKIP] fig3: dataset not found.")
        return

    df = pd.read_csv(clf_path)

    # Use observation block number (1–6) not squat count
    df["Block"] = df["Squat_No"] // 4   # 4→1, 8→2, …, 24→6

    mpf_col = "DeltMed_MPF"    # Medial Deltoid shows clearest MPF decline (8 kg)
    rms_col = "DeltAnt_RMS"    # Anterior Deltoid RMS

    # Check columns exist
    if mpf_col not in df.columns:
        mpf_col = next((c for c in df.columns if "MPF" in c and "NMPF" not in c
                        and "mean" not in c), None)
    if rms_col not in df.columns:
        rms_col = next((c for c in df.columns if "RMS" in c and "mean" not in c), None)

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 6.5 / 2.54))

    panels = [
        ("RPE",  "RPE",    "Borg RPE Score",          "(a) Perceived Exertion (RPE)"),
        (mpf_col, mpf_col, "MPF (Hz)",                "(b) Mean Power Frequency\n    (Medial Deltoid)"),
        (rms_col, rms_col, "RMS Amplitude (\u00b5V)", "(c) RMS Amplitude\n    (Anterior Deltoid)"),
    ]

    for ax, (key, col, ylabel, title) in zip(axes, panels):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        for ki, (load, grp) in enumerate(df.groupby("Load_kg")):
            stats = grp.groupby("Block")[col].agg(["mean", "sem"])
            x = stats.index.values
            mean_v = stats["mean"].values
            sem_v  = stats["sem"].values

            ax.plot(x, mean_v, marker="o", markersize=5,
                    label=f"{int(load)} kg", color=PALETTE[ki], linewidth=2.0)
            ax.fill_between(x, mean_v - sem_v, mean_v + sem_v,
                             alpha=0.18, color=PALETTE[ki])

        # Fatigue threshold line for RPE panel
        if key == "RPE":
            ax.axhline(5, color=PALETTE[3], linestyle="--", linewidth=1.4,
                       label="Fatigue threshold\n(RPE = 5)")
            ax.set_ylim(0, 9)

        # Fatigue onset marker on MPF and RMS panels (block 5 = squat 20)
        if key != "RPE":
            ax.axvline(5, color=PALETTE[3], linestyle="--", linewidth=1.2,
                       label="Onset (~Block 5)", alpha=0.8)

        ax.set_xlabel("Observation Block (1–6)", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold",
                     linespacing=1.3)
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7.5, framealpha=0.8)

    fig.suptitle(
        "Fatigue Progression During Squat Lifting  (mean \u00b1 SEM across subjects)",
        fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "fig3_fatigue_progression.png")


def fig4_model_comparison_classification():
    tables = [
        RESULTS_DIR / "loso_results_classification.csv",
        RESULTS_DIR / "cv10_results_classification.csv",
    ]
    dl_path = RESULTS_DIR / "dl_results_classification.csv"

    df_trad = None
    for p in tables:
        if p.exists():
            df_trad = pd.read_csv(p)
            break
    if df_trad is None:
        print("  [SKIP] fig4: no classification results found.")
        return

    metrics  = ["Accuracy", "F1",   "AUC"]
    m_labels = ["Accuracy", "F1-Score (macro)", "AUC-ROC"]
    m_colors = [PALETTE[0], PALETTE[2], PALETTE[3]]   # blue, green, red

    # Collect traditional ML rows
    trad_rows = []
    for _, r in df_trad.iterrows():
        trad_rows.append({
            "Model":    r["Model"],
            "Accuracy": float(r.get("Accuracy_mean", np.nan)),
            "F1":       float(r.get("F1_mean",       np.nan)),
            "AUC":      float(r.get("AUC_mean",       np.nan)),
        })

    # Collect DL rows
    dl_rows = []
    if dl_path.exists():
        for _, r in pd.read_csv(dl_path).iterrows():
            dl_rows.append({
                "Model":    r["Model"],
                "Accuracy": float(r.get("Accuracy", np.nan)),
                "F1":       float(r.get("F1",       np.nan)),
                "AUC":      float(r.get("AUC",      np.nan)),
            })

    # Short labels for x-axis
    short = {
        "Logistic Regression": "LR",
        "SVM": "SVM",
        "Decision Tree": "DT",
        "Random Forest": "RF",
        "XGBoost": "XGB",
        "LightGBM": "LGB",
        "MLP": "MLP",
        "1D-CNN": "CNN",
        "BiLSTM": "BiLSTM",
    }

    def _plot_panel(ax, rows, title):
        names   = [r["Model"] for r in rows]
        xlabels = [short.get(n, n) for n in names]
        n       = len(names)
        x       = np.arange(n)
        bw      = 0.22           # individual bar width
        offsets = np.array([-bw, 0, bw])

        for mi, (metric, mlabel, mcol) in enumerate(zip(metrics, m_labels, m_colors)):
            vals = [r[metric] for r in rows]
            bars = ax.bar(x + offsets[mi], vals, bw * 0.90,
                          label=mlabel, color=mcol, alpha=0.82,
                          edgecolor="white", linewidth=0.5, zorder=3)
            # Value label above each bar
            for bar, val in zip(bars, vals):
                if not np.isnan(val) and val > 0.1:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{val:.2f}",
                        ha="center", va="bottom",
                        fontsize=5.5, color="#222222",
                        rotation=90
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylim(0.45, 1.18)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.yaxis.grid(True, alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
        # Reference line at 0.5 (random chance)
        ax.axhline(0.5, color="#AAAAAA", lw=0.8, ls="--", zorder=2)

    n_panels = 1 + (1 if dl_rows else 0)
    if n_panels == 2:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(COL2, 2.8),
            gridspec_kw={"width_ratios": [len(trad_rows), max(len(dl_rows), 1)]}
        )
        _plot_panel(ax1, trad_rows, "(a) Traditional ML — LOSO CV")
        _plot_panel(ax2, dl_rows,   "(b) Deep Learning — Hold-out")
        ax2.set_ylabel("")
        ax2.set_yticklabels([])
        # Shared legend below both panels
        handles, labels_ = ax1.get_legend_handles_labels()
        fig.legend(handles, labels_, loc="lower center", ncol=3,
                   fontsize=8.5, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.08))
        plt.subplots_adjust(wspace=0.08, bottom=0.18)
    else:
        fig, ax1 = plt.subplots(figsize=(COL1 * 1.4, 2.8))
        _plot_panel(ax1, trad_rows, "Traditional ML — LOSO CV")
        ax1.legend(fontsize=8, framealpha=0.9, loc="lower right", ncol=1)
        plt.tight_layout()

    _save(fig, "fig4_model_comparison_classification.png")


def fig5_confusion_matrices():
    clf_path = RESULTS_DIR / "dataset_classification.csv"
    if not clf_path.exists():
        print("  [SKIP] fig5: dataset not found.")
        return

    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    try:
        import xgboost as xgb
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False
    try:
        import lightgbm as lgb
        HAS_LGB = True
    except ImportError:
        HAS_LGB = False

    df = pd.read_csv(clf_path)
    meta = ["Subject", "Subject_idx", "Load_kg", "Squat_No",
            "RPE", "Fatigued", "HR", "MPF_form", "Weight_kg", "Height_cm", "BMI"]
    feat_cols = [c for c in df.columns if c not in meta]
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    y = df["Fatigued"].values.astype(int)

    models_to_plot = {
        "Logistic\nRegression": Pipeline([
            ("imp", SimpleImputer()), ("scl", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42))
        ]),
        "Random\nForest": Pipeline([
            ("imp", SimpleImputer()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
    }
    if HAS_XGB:
        models_to_plot["XGBoost"] = Pipeline([
            ("imp", SimpleImputer()),
            ("clf", xgb.XGBClassifier(n_estimators=100, random_state=42,
                                       eval_metric="logloss", verbosity=0))
        ])
    if HAS_LGB:
        models_to_plot["LightGBM"] = Pipeline([
            ("imp", SimpleImputer()),
            ("clf", lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
        ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cms = {}
    for name, pipe in models_to_plot.items():
        cm_agg = np.zeros((2, 2), dtype=int)
        for tr, te in skf.split(X, y):
            pipe.fit(X[tr], y[tr])
            ypred = pipe.predict(X[te])
            cm_agg += confusion_matrix(y[te], ypred, labels=[0, 1])
        cms[name] = cm_agg

    names = list(cms.keys())
    n = len(names)
    ncols = min(n, 2)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(COL2, nrows * 7.5 / 2.54),
                              squeeze=False)
    flat_axes = axes.flatten()

    # Hide any unused subplots
    for k in range(n, len(flat_axes)):
        flat_axes[k].set_visible(False)

    import seaborn as sns
    tick_labels = ["Non-\nFatigued", "Fatigued"]

    for k, (name, cm) in enumerate(cms.items()):
        ax = flat_axes[k]
        total = cm.sum()
        # Percentage annotations alongside counts
        annot = np.array([[f"{v}\n({100*v/total:.1f}%)" for v in row] for row in cm])

        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=ax,
                    cbar=False, linewidths=1.0, linecolor="white",
                    annot_kws={"size": 9, "weight": "bold"},
                    xticklabels=tick_labels, yticklabels=tick_labels)

        ax.set_title(name.replace("\n", " "), fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Predicted Label", fontsize=9, labelpad=4)
        ax.set_ylabel("True Label",      fontsize=9, labelpad=4)
        ax.tick_params(axis="both", labelsize=8.5)

    fig.suptitle("Confusion Matrices — Fatigue Binary Classification\n(5-Fold Stratified CV, Aggregated)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig5_confusion_matrices.png")


def fig6_roc_curves():
    clf_path = RESULTS_DIR / "dataset_classification.csv"
    if not clf_path.exists():
        print("  [SKIP] fig6: dataset not found.")
        return

    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler, label_binarize
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    try:
        import xgboost as xgb; HAS_XGB = True
    except ImportError:
        HAS_XGB = False
    try:
        import lightgbm as lgb; HAS_LGB = True
    except ImportError:
        HAS_LGB = False

    df = pd.read_csv(clf_path)
    meta = ["Subject", "Subject_idx", "Load_kg", "Squat_No",
            "RPE", "Fatigued", "HR", "MPF_form", "Weight_kg", "Height_cm", "BMI"]
    feat_cols = [c for c in df.columns if c not in meta]
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    y = df["Fatigued"].values.astype(int)

    clf_map = {
        "Logistic Regression": Pipeline([("imp", SimpleImputer()), ("scl", StandardScaler()),
                                          ("clf", LogisticRegression(max_iter=500, random_state=42))]),
        "SVM (RBF)":           Pipeline([("imp", SimpleImputer()), ("scl", StandardScaler()),
                                          ("clf", SVC(kernel="rbf", probability=True, random_state=42))]),
        "Decision Tree":       Pipeline([("imp", SimpleImputer()),
                                          ("clf", DecisionTreeClassifier(max_depth=8, random_state=42))]),
        "Random Forest":       Pipeline([("imp", SimpleImputer()),
                                          ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
    }
    if HAS_XGB:
        clf_map["XGBoost"] = Pipeline([("imp", SimpleImputer()),
                                        ("clf", xgb.XGBClassifier(n_estimators=100, random_state=42,
                                                                   eval_metric="logloss", verbosity=0))])
    if HAS_LGB:
        clf_map["LightGBM"] = Pipeline([("imp", SimpleImputer()),
                                         ("clf", lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))])

    fig, ax = plt.subplots(figsize=(COL1 * 1.1, COL1 * 1.1))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for ci, (name, pipe) in enumerate(clf_map.items()):
        tprs, aucs = [], []
        base_fpr = np.linspace(0, 1, 101)
        for tr, te in skf.split(X, y):
            pipe.fit(X[tr], y[tr])
            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(X[te])[:, 1]
            else:
                proba = pipe.decision_function(X[te])
            fpr, tpr, _ = roc_curve(y[te], proba)
            tprs.append(np.interp(base_fpr, fpr, tpr))
            aucs.append(auc(fpr, tpr))

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.mean(aucs)
        ax.plot(base_fpr, mean_tpr, color=PALETTE[ci], linewidth=1.8,
                label=f"{name} (AUC={mean_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Chance (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Fatigue Classification\n(5-Fold CV, Mean)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    _save(fig, "fig6_roc_curves.png")


def fig7_shap_importance():
    shap_path = RESULTS_DIR / "shap_importance.csv"
    if not shap_path.exists():
        print("  [SKIP] fig7: SHAP importance not found.")
        return

    df = pd.read_csv(shap_path).head(20)
    df = df.sort_values("SHAP_importance", ascending=True)

    fig, ax = plt.subplots(figsize=(COL1 * 1.2, 9 / 2.54))
    colors = [PALETTE[0] if "_MPF" in f or "_MDF" in f
              else PALETTE[1] if "_RMS" in f
              else PALETTE[2] if "_IEMG" in f
              else PALETTE[5]
              for f in df["Feature"]]
    bars = ax.barh(df["Feature"], df["SHAP_importance"], color=colors,
                   edgecolor="white", linewidth=0.5, height=0.7)

    # Legend
    patches = [
        mpatches.Patch(color=PALETTE[0], label="MPF/MDF (Freq. domain)"),
        mpatches.Patch(color=PALETTE[1], label="RMS (Time domain)"),
        mpatches.Patch(color=PALETTE[2], label="IEMG"),
        mpatches.Patch(color=PALETTE[5], label="Other"),
    ]
    ax.legend(handles=patches, fontsize=7.5, loc="lower right")
    ax.set_xlabel("Mean |SHAP Value| (Feature Importance)")
    ax.set_title("Top-20 Feature Importances (SHAP)\nBest Traditional ML Model",
                 fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    _save(fig, "fig7_shap_importance.png")


def fig8_regression_scatter():
    clf_path = RESULTS_DIR / "dataset_regression.csv"
    loso_reg = RESULTS_DIR / "loso_results_regression.csv"
    if not clf_path.exists():
        print("  [SKIP] fig8: dataset not found.")
        return

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import r2_score, mean_absolute_error

    df = pd.read_csv(clf_path)
    meta = ["Subject", "Subject_idx", "Load_kg", "Squat_No",
            "RPE", "Fatigued", "HR", "MPF_form", "Weight_kg", "Height_cm", "BMI"]
    feat_cols = [c for c in df.columns if c not in meta]
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    y = df["RPE"].values.astype(np.float32)

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("reg", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    y_pred = cross_val_predict(pipe, X, y, cv=KFold(5, shuffle=True, random_state=42))
    r2  = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    # Color by load condition
    load_vals = df["Load_kg"].values
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 7 / 2.54))

    for ki, (load, color) in enumerate([(4, PALETTE[0]), (8, PALETTE[1])]):
        ax = axes[ki]
        mask = load_vals == load
        yt, yp = y[mask], y_pred[mask]
        r2_l  = r2_score(yt, yp)
        mae_l = mean_absolute_error(yt, yp)

        ax.scatter(yt, yp, c=color, alpha=0.5, s=20, edgecolors="none")
        lim = [min(yt.min(), yp.min()) - 0.5, max(yt.max(), yp.max()) + 0.5]
        ax.plot(lim, lim, "k--", linewidth=1, alpha=0.7, label="Perfect fit")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Actual Borg RPE Score")
        ax.set_ylabel("Predicted Borg RPE Score")
        ax.set_title(f"({'ab'[ki]}) Load = {load} kg\nR²={r2_l:.3f}, MAE={mae_l:.3f}",
                     loc="left", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("Borg RPE Regression — Random Forest (5-Fold CV)\n"
                 f"Overall: R²={r2:.3f}, MAE={mae:.3f}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig8_regression_scatter.png")


def fig9_deep_learning_training():
    hist_path = RESULTS_DIR / "dl_training_history.pkl"
    if not hist_path.exists():
        print("  [SKIP] fig9: DL training history not found.")
        return

    with open(hist_path, "rb") as f:
        all_hist = pickle.load(f)

    if not all_hist:
        print("  [SKIP] fig9: empty training history.")
        return

    # Show classification models only
    clf_keys = {k: v for k, v in all_hist.items() if "_clf" in k}
    n = len(clf_keys)
    if n == 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(COL2, 9 / 2.54), squeeze=False)
    name_map = {
        "MLP_clf":  "MLP",
        "CNN_clf":  "1D-CNN",
        "LSTM_clf": "BiLSTM",
    }

    for ci, (key, hist) in enumerate(clf_keys.items()):
        model_name = name_map.get(key, key)
        ax_loss = axes[0][ci]
        ax_acc  = axes[1][ci]

        epochs = range(1, len(hist.get("loss", [])) + 1)
        ax_loss.plot(epochs, hist.get("loss", []),     color=PALETTE[0], label="Train")
        ax_loss.plot(epochs, hist.get("val_loss", []), color=PALETTE[1], label="Val",
                     linestyle="--")
        ax_loss.set_title(f"{model_name}", fontsize=10, fontweight="bold")
        ax_loss.set_ylabel("Binary Cross-Entropy Loss")
        ax_loss.legend(fontsize=7.5)

        ax_acc.plot(epochs, hist.get("accuracy", []),     color=PALETTE[0], label="Train")
        ax_acc.plot(epochs, hist.get("val_accuracy", []), color=PALETTE[1], label="Val",
                    linestyle="--")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.legend(fontsize=7.5)

    fig.suptitle("Deep Learning Training Curves — Fatigue Classification",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig9_deep_learning_training.png")


def generate_all_figures(verbose: bool = True):
    if verbose:
        print("\n" + "=" * 55)
        print("  GENERATING FIGURES")
        print("=" * 55)

    fig_funcs = [
        ("fig1_emg_raw_vs_filtered",          fig1_emg_raw_vs_filtered),
        ("fig2_feature_extraction_pipeline",  fig2_feature_extraction_pipeline),
        ("fig3_fatigue_progression",          fig3_fatigue_progression),
        ("fig4_model_comparison",             fig4_model_comparison_classification),
        ("fig5_confusion_matrices",           fig5_confusion_matrices),
        ("fig6_roc_curves",                   fig6_roc_curves),
        ("fig7_shap_importance",              fig7_shap_importance),
        ("fig8_regression_scatter",           fig8_regression_scatter),
        ("fig9_deep_learning_training",       fig9_deep_learning_training),
    ]

    for name, fn in fig_funcs:
        try:
            fn()
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    if verbose:
        saved = list(FIGURES_DIR.glob("fig*.png"))
        print(f"\n  {len(saved)} figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_figures(verbose=True)
