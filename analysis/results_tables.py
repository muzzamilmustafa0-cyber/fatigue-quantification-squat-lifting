import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_table1_demographics() -> pd.DataFrame:
    demo_path = RESULTS_DIR / "dataset_classification.csv"
    if not demo_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(demo_path)
    demo = df[["Subject", "Weight_kg", "Height_cm", "BMI"]].drop_duplicates(subset="Subject")

    stats = {
        "Parameter":      ["Age (years)*",  "Weight (kg)", "Height (cm)", "BMI (kg/m²)",
                            "N subjects", "N observations (4 kg)", "N observations (8 kg)"],
        "Mean":           ["—",
                            f"{demo['Weight_kg'].mean():.1f}",
                            f"{demo['Height_cm'].mean():.1f}",
                            f"{demo['BMI'].mean():.1f}",
                            str(len(demo)),
                            str(len(df[df['Load_kg'] == 4])),
                            str(len(df[df['Load_kg'] == 8]))],
        "SD":             ["—",
                            f"±{demo['Weight_kg'].std():.1f}",
                            f"±{demo['Height_cm'].std():.1f}",
                            f"±{demo['BMI'].std():.1f}",
                            "—", "—", "—"],
        "Range":          ["—",
                            f"{demo['Weight_kg'].min():.1f}–{demo['Weight_kg'].max():.1f}",
                            f"{demo['Height_cm'].min():.1f}–{demo['Height_cm'].max():.1f}",
                            f"{demo['BMI'].min():.1f}–{demo['BMI'].max():.1f}",
                            "—", "—", "—"],
    }
    t1 = pd.DataFrame(stats)
    t1.to_csv(RESULTS_DIR / "table1_demographics.csv", index=False)
    print("  Table 1 (demographics) saved.")
    return t1


def make_table2_feature_statistics() -> pd.DataFrame:
    clf_path = RESULTS_DIR / "dataset_classification.csv"
    if not clf_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(clf_path)

    # Select representative features
    key_features = [c for c in df.columns if any(
        s in c for s in ["_RMS", "_MPF", "_MDF", "_IEMG"]
    )][:24]   # Top 24 for readability

    if "HR" in df.columns:
        key_features = ["HR"] + key_features

    records = []
    for feat in key_features:
        if feat not in df.columns:
            continue
        vals_all   = df[feat].dropna()
        vals_4kg   = df.loc[df["Load_kg"] == 4, feat].dropna()
        vals_8kg   = df.loc[df["Load_kg"] == 8, feat].dropna()
        vals_fat   = df.loc[df["Fatigued"] == 1, feat].dropna()
        vals_nfat  = df.loc[df["Fatigued"] == 0, feat].dropna()

        # Mann-Whitney U test
        try:
            from scipy.stats import mannwhitneyu
            _, p_fatigue = mannwhitneyu(vals_fat, vals_nfat, alternative="two-sided")
        except Exception:
            p_fatigue = np.nan

        records.append({
            "Feature":          feat,
            "Overall Mean±SD":  f"{vals_all.mean():.3f}±{vals_all.std():.3f}",
            "4 kg Mean±SD":     f"{vals_4kg.mean():.3f}±{vals_4kg.std():.3f}",
            "8 kg Mean±SD":     f"{vals_8kg.mean():.3f}±{vals_8kg.std():.3f}",
            "Fatigued Mean±SD": f"{vals_fat.mean():.3f}±{vals_fat.std():.3f}",
            "Non-Fat. Mean±SD": f"{vals_nfat.mean():.3f}±{vals_nfat.std():.3f}",
            "p (Fat vs Non)":   f"{p_fatigue:.4f}" if not np.isnan(p_fatigue) else "N/A",
        })

    t2 = pd.DataFrame(records)
    t2.to_csv(RESULTS_DIR / "table2_feature_statistics.csv", index=False)
    print("  Table 2 (feature statistics) saved.")
    return t2


def make_table3_classification() -> pd.DataFrame:
    records = []

    # Traditional ML — LOSO results preferred
    for fname in ["loso_results_classification.csv", "cv10_results_classification.csv"]:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            cv_type = "LOSO" if "loso" in fname else "10-Fold CV"
            for _, row in df.iterrows():
                r = {"Model": row["Model"], "CV": cv_type, "Source": "Traditional ML"}
                for metric in ["Accuracy", "Precision", "Recall", "F1", "AUC", "Kappa"]:
                    mean_col = f"{metric}_mean"
                    std_col  = f"{metric}_std"
                    if mean_col in row:
                        r[metric] = (f"{row[mean_col]:.4f}" +
                                     (f"±{row[std_col]:.4f}" if std_col in row else ""))
                records.append(r)
            break   # Use only first found

    # Deep Learning
    dl_path = RESULTS_DIR / "dl_results_classification.csv"
    if dl_path.exists():
        dl_df = pd.read_csv(dl_path)
        for _, row in dl_df.iterrows():
            r = {"Model": row["Model"], "CV": "Hold-out", "Source": "Deep Learning"}
            for metric in ["Accuracy", "Precision", "Recall", "F1", "AUC", "Kappa"]:
                if metric in row:
                    r[metric] = f"{row[metric]:.4f}"
            records.append(r)

    t3 = pd.DataFrame(records)
    if not t3.empty:
        t3.to_csv(RESULTS_DIR / "table3_classification_comparison.csv", index=False)
        print("  Table 3 (classification comparison) saved.")
    return t3


def make_table4_regression() -> pd.DataFrame:
    records = []

    for fname in ["loso_results_regression.csv", "cv10_results_regression.csv"]:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            cv_type = "LOSO" if "loso" in fname else "10-Fold CV"
            for _, row in df.iterrows():
                r = {"Model": row["Model"], "CV": cv_type, "Source": "Traditional ML"}
                for metric in ["MAE", "RMSE", "R2", "MAPE"]:
                    mean_col = f"{metric}_mean"
                    std_col  = f"{metric}_std"
                    if mean_col in row:
                        r[metric] = (f"{row[mean_col]:.4f}" +
                                     (f"±{row[std_col]:.4f}" if std_col in row else ""))
                records.append(r)
            break

    dl_path = RESULTS_DIR / "dl_results_regression.csv"
    if dl_path.exists():
        dl_df = pd.read_csv(dl_path)
        for _, row in dl_df.iterrows():
            r = {"Model": row["Model"], "CV": "Hold-out", "Source": "Deep Learning"}
            for metric in ["MAE", "RMSE", "R2", "MAPE"]:
                if metric in row:
                    r[metric] = f"{row[metric]:.4f}"
            records.append(r)

    t4 = pd.DataFrame(records)
    if not t4.empty:
        t4.to_csv(RESULTS_DIR / "table4_regression_comparison.csv", index=False)
        print("  Table 4 (regression comparison) saved.")
    return t4


def make_table5_statistical_tests() -> pd.DataFrame:
    path = RESULTS_DIR / "statistical_tests.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.to_csv(RESULTS_DIR / "table5_statistical_tests.csv", index=False)
    print("  Table 5 (statistical tests) saved.")
    return df


def df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = df.columns.tolist()
    col_fmt = "l" + "c" * (len(cols) - 1)
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join([f"\\textbf{{{c}}}" for c in cols]) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join([str(v) for v in row.values]) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def save_latex_tables(tables: dict):
    captions = {
        "table1": ("Participant Demographic Characteristics", "tab:demographics"),
        "table2": ("Descriptive Statistics of Extracted EMG Features", "tab:features"),
        "table3": ("Fatigue Classification Performance Comparison", "tab:clf"),
        "table4": ("Borg RPE Regression Performance Comparison", "tab:reg"),
        "table5": ("Wilcoxon Signed-Rank Statistical Significance Tests", "tab:stats"),
    }
    latex_blocks = [r"\usepackage{booktabs}", ""]
    for key, df in tables.items():
        if df is not None and not df.empty and key in captions:
            cap, lbl = captions[key]
            latex_blocks.append(df_to_latex(df, cap, lbl))
            latex_blocks.append("")

    tex_path = RESULTS_DIR / "latex_tables.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_blocks))
    print(f"  LaTeX tables saved to: {tex_path}")


def generate_all_tables(verbose: bool = True):
    if verbose:
        print("\n" + "="*55)
        print("  GENERATING RESULT TABLES")
        print("="*55)

    tables = {
        "table1": make_table1_demographics(),
        "table2": make_table2_feature_statistics(),
        "table3": make_table3_classification(),
        "table4": make_table4_regression(),
        "table5": make_table5_statistical_tests(),
    }

    save_latex_tables(tables)

    if verbose:
        print(f"\n  All tables saved to: {RESULTS_DIR}")

    return tables


if __name__ == "__main__":
    tables = generate_all_tables(verbose=True)
    for name, t in tables.items():
        if t is not None and not t.empty:
            print(f"\n{name}:")
            print(t.to_string(index=False))
