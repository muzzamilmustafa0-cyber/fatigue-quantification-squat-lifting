import sys
import time
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def banner(msg: str):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {msg}")
    print(f"{sep}")


def elapsed(t0: float) -> str:
    sec = int(time.time() - t0)
    return f"{sec // 60}m {sec % 60}s"


def step1_build_dataset():
    banner("STEP 1/5 — Building ML Dataset from Raw EMG")
    t0 = time.time()
    from dataset_builder import build_datasets  # noqa
    clf_df, reg_df, dl_data = build_datasets(verbose=True)
    print(f"\n  Done in {elapsed(t0)}")
    return clf_df


def step2_traditional_ml():
    banner("STEP 2/5 — Traditional ML (LR, SVM, RF, XGBoost, LightGBM)")
    t0 = time.time()
    from traditional_ml import run_traditional_ml  # noqa
    results = run_traditional_ml(verbose=True)
    print(f"\n  Done in {elapsed(t0)}")
    return results


def step3_deep_learning():
    banner("STEP 3/5 — Deep Learning (MLP, 1D-CNN, BiLSTM)")
    t0 = time.time()
    from deep_learning import run_deep_learning  # noqa
    results = run_deep_learning(verbose=True)
    print(f"\n  Done in {elapsed(t0)}")
    return results


def step4_tables():
    banner("STEP 4/5 — Generating Result Tables")
    t0 = time.time()
    from results_tables import generate_all_tables  # noqa
    tables = generate_all_tables(verbose=True)
    print(f"\n  Done in {elapsed(t0)}")
    return tables


def step5_figures():
    banner("STEP 5/5 — Generating Figures")
    t0 = time.time()
    from visualizations import generate_all_figures  # noqa
    generate_all_figures(verbose=True)
    print(f"\n  Done in {elapsed(t0)}")


def print_summary():
    banner("PIPELINE COMPLETE — Output Summary")

    fig_files = sorted(FIGURES_DIR.glob("fig*.png"))
    print(f"\n  Figures ({len(fig_files)} files):")
    for f in fig_files:
        sz = f.stat().st_size / 1024
        print(f"    {f.name:<45} ({sz:.0f} KB)")

    res_files = sorted(RESULTS_DIR.glob("table*.csv")) + sorted(RESULTS_DIR.glob("*results*.csv"))
    print(f"\n  Result tables ({len(res_files)} files):")
    for f in res_files:
        print(f"    {f.name}")

    latex = RESULTS_DIR / "latex_tables.tex"
    if latex.exists():
        print(f"\n  LaTeX tables: {latex}")

    print("\n  [OK] All outputs saved.")
    print(f"    Figures directory : {FIGURES_DIR}")
    print(f"    Results directory : {RESULTS_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Full ML analysis pipeline for EMG fatigue quantification."
    )
    parser.add_argument("--skip-dl",    action="store_true", help="Skip deep learning step")
    parser.add_argument("--skip-build", action="store_true", help="Skip dataset rebuild")
    parser.add_argument("--figures-only", action="store_true", help="Only regenerate figures/tables")
    args = parser.parse_args()

    total_t0 = time.time()

    if args.figures_only:
        step4_tables()
        step5_figures()
        print_summary()
        return

    if not args.skip_build:
        step1_build_dataset()
    else:
        clf_path = RESULTS_DIR / "dataset_classification.csv"
        if not clf_path.exists():
            print("[WARN] Cached dataset not found. Running step 1 anyway ...")
            step1_build_dataset()
        else:
            print(f"\n  [SKIP] Dataset already built: {clf_path}")

    step2_traditional_ml()

    if not args.skip_dl:
        step3_deep_learning()
    else:
        print("\n  [SKIP] Deep learning (--skip-dl flag set)")

    step4_tables()
    step5_figures()
    print_summary()

    total_time = elapsed(total_t0)
    print(f"\n  Total pipeline time: {total_time}\n")


if __name__ == "__main__":
    main()
