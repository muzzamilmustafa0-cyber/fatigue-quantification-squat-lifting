import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Make sure sibling modules are importable
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_all, FS, MUSCLE_NAMES  # noqa
from signal_processing import process_subject        # noqa
from feature_extraction import (                     # noqa
    extract_features_from_squats,
    extract_features_from_windows,
    get_feature_names,
)

BASE_DIR    = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FATIGUE_THRESHOLD = 5   # Borg RPE >= 5 → Fatigued

# EMG file name (lower-case) -> subject name in "Filled form S" (lower-case), where spellings differ.
# Fuzzy matching previously assigned the wrong participant's labels to several files.
EMG_TO_FORM_NAME = {
    "arsalan": "asrsalan",
    "asif hussain": "asif husain",
    "ijaz beher": "ijaz behar",
    "mohsin": "mohsin ali",
    "mohsin javed": "mohsijaved",
    "zohaib": "zuhaib",
}
# "Bizzat 4.xlsx" duplicates the 4 kg session in "Bizzat.xlsx" (identical EMG features).
DUPLICATE_EMG_FILES = {"bizzat 4"}


def form_subject_idx(label_df: pd.DataFrame, emg_name: str) -> int:
    key = emg_name.lower().strip()
    if label_df.empty or key in DUPLICATE_EMG_FILES:
        return -1
    fs_name = EMG_TO_FORM_NAME.get(key, key)
    rows = label_df[label_df["Subject"].str.strip().str.lower() == fs_name]
    return int(rows["Subject_idx"].iloc[0]) if not rows.empty else -1


N_OBS_BLOCKS = 6   # Observations per session (6 blocks of 4 squats = 24 squats)


def _extract_block_features(raw: np.ndarray, block_idx: int, n_blocks: int = N_OBS_BLOCKS,
                             fs: float = FS):
    from signal_processing import bandpass_filter, lowpass_filter
    from feature_extraction import segment_to_features

    n_total = raw.shape[0]
    block_len = n_total // n_blocks
    start = block_idx * block_len
    end   = start + block_len

    block = raw[start:end].astype(np.float64)

    # Bandpass + rectify for feature extraction
    nyq = fs / 2.0
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [20 / nyq, min(450 / nyq, 0.999)], btype="band")
    filtered = np.empty_like(block)
    for ch in range(block.shape[1]):
        filtered[:, ch] = filtfilt(b, a, block[:, ch])

    return segment_to_features(filtered.astype(np.float32), fs=fs)


def build_emg_feature_df(emg_data: dict, label_df: pd.DataFrame,
                          demo_df: pd.DataFrame) -> pd.DataFrame:
    all_rows = []
    emg_subjects = sorted(emg_data.keys())

    for emg_subj_idx, subj_name in enumerate(tqdm(emg_subjects, desc="Extracting features")):
        subj_emg = emg_data[subj_name]

        # Anthropometrics
        demo_row = demo_df[demo_df["Subject"].str.lower().str.strip() == subj_name.lower().strip()]
        weight = float(demo_row["Weight_kg"].iloc[0]) if not demo_row.empty else np.nan
        height = float(demo_row["Height_cm"].iloc[0]) if not demo_row.empty else np.nan
        bmi    = float(demo_row["BMI"].iloc[0])        if not demo_row.empty else np.nan

        # Find matching subject in label_df
        form_subj_idx = form_subject_idx(label_df, subj_name)
        if form_subj_idx < 0:
            print(f"  [SKIP] {subj_name} — no exact label-form entry (or duplicate recording)")
            continue

        for load_str in ["4kg", "8kg"]:
            if load_str not in subj_emg:
                continue
            load_kg = int(load_str.replace("kg", ""))
            raw = subj_emg[load_str]

            min_samples = int(FS * 20 * N_OBS_BLOCKS)  # at least 20 s per block
            if raw.shape[0] < min_samples:
                print(f"  [SKIP] {subj_name}/{load_str} — too short ({raw.shape[0]} samples)")
                continue

            # Get label rows from analysis form for this subject+load
            if form_subj_idx >= 0 and not label_df.empty:
                form_rows = label_df[
                    (label_df["Subject_idx"] == form_subj_idx) &
                    (label_df["Load_kg"] == load_kg)
                ].sort_values("Obs_No").reset_index(drop=True)
            else:
                form_rows = pd.DataFrame()

            for block_idx in range(N_OBS_BLOCKS):
                try:
                    feats = _extract_block_features(raw, block_idx, N_OBS_BLOCKS, fs=FS)
                except Exception as e:
                    print(f"  [WARN] Feature extraction failed {subj_name}/{load_str} block {block_idx}: {e}")
                    continue

                row = {
                    "Subject":     subj_name,
                    "Subject_idx": emg_subj_idx,
                    "Load_kg":     load_kg,
                    "Obs_No":      block_idx + 1,
                    "Squat_No":    (block_idx + 1) * 4,
                    "Weight_kg":   weight,
                    "Height_cm":   height,
                    "BMI":         bmi,
                }
                row.update(feats)

                if form_rows.empty or block_idx >= len(form_rows):
                    continue
                lr = form_rows.iloc[block_idx]
                row["RPE"]        = float(lr["RPE"])
                row["RPE_Upper"]  = float(lr.get("RPE_Upper", np.nan))
                row["RPE_Middle"] = float(lr.get("RPE_Middle", np.nan))
                row["RPE_Lower"]  = float(lr.get("RPE_Lower", np.nan))
                row["HR"]         = float(lr.get("HR", np.nan))

                all_rows.append(row)

    return pd.DataFrame(all_rows)


def assign_labels_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["RPE"]).copy()
    df["Fatigued"] = (df["RPE"] >= FATIGUE_THRESHOLD).astype(int)
    # Drop rows with all-zero feature values (artefact segments)
    feat_cols = [c for c in df.columns if c not in
                 ["Subject", "Subject_idx", "Load_kg", "Squat_No",
                  "RPE", "Fatigued", "HR", "MPF_form", "Weight_kg",
                  "Height_cm", "BMI"]]
    row_sums = df[feat_cols].abs().sum(axis=1)
    df = df[row_sums > 0].reset_index(drop=True)
    return df


def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=random_state)
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res
    except Exception as e:
        print(f"  [WARN] SMOTE failed ({e}), using original data.")
        return X, y


def build_dl_dataset(emg_data: dict, label_df: pd.DataFrame) -> dict:
    from signal_processing import sliding_windows  # noqa
    WIN_SEC = 2.0
    FS_val  = FS

    all_windows = []
    all_labels  = []
    all_rpe     = []
    all_subjects = []

    subjects = sorted(emg_data.keys())

    for subj_idx, subj_name in enumerate(tqdm(subjects, desc="Building DL windows")):
        subj_emg = emg_data[subj_name]
        form_idx = form_subject_idx(label_df, subj_name)
        if form_idx < 0:
            continue

        for load_str in ["4kg", "8kg"]:
            if load_str not in subj_emg:
                continue
            load_kg = int(load_str.replace("kg", ""))
            raw = subj_emg[load_str]
            if raw.shape[0] < int(WIN_SEC * FS_val) * 2:
                continue

            from signal_processing import preprocess_emg  # noqa
            filtered, _, _ = preprocess_emg(raw, fs=FS_val)

            windows = sliding_windows(filtered, fs=FS_val, window_sec=WIN_SEC, overlap=0.5)
            n_win   = windows.shape[0]

            # Get RPE for the full recording from the form
            form_rows = label_df[
                (label_df["Subject_idx"] == form_idx) &
                (label_df["Load_kg"] == load_kg)
            ].sort_values("Obs_No")
            if form_rows.empty:
                continue

            # Assign RPE to each window based on its temporal position
            for wi in range(n_win):
                frac = wi / max(n_win - 1, 1)
                n_obs = len(form_rows)
                obs_idx = min(int(frac * n_obs), n_obs - 1)
                rpe_val = float(form_rows["RPE"].iloc[obs_idx])

                all_windows.append(windows[wi])
                all_rpe.append(rpe_val)
                all_labels.append(int(rpe_val >= FATIGUE_THRESHOLD))
                all_subjects.append(subj_idx)

    if not all_windows:
        return {}

    return {
        "windows":  np.array(all_windows, dtype=np.float32),
        "labels":   np.array(all_labels,  dtype=np.int32),
        "rpe":      np.array(all_rpe,     dtype=np.float32),
        "subjects": np.array(all_subjects, dtype=np.int32),
    }


def build_datasets(verbose: bool = True):
    emg_data, label_df, demo_df = load_all(verbose=verbose)

    if verbose:
        print("\n[1/3] Building EMG feature DataFrame ...")
    df = build_emg_feature_df(emg_data, label_df, demo_df)
    df = assign_labels_and_clean(df)

    feat_cols = [c for c in df.columns if c not in
                 ["Subject", "Subject_idx", "Load_kg", "Squat_No",
                  "RPE", "Fatigued", "HR", "MPF_form", "Weight_kg",
                  "Height_cm", "BMI"]]

    # Impute NaNs with median per feature
    for col in feat_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Replace infinities
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    clf_df = df[["Subject", "Subject_idx", "Load_kg", "Squat_No",
                 "Weight_kg", "Height_cm", "BMI", "HR"] + feat_cols + ["RPE", "Fatigued"]].copy()
    clf_df.to_csv(RESULTS_DIR / "dataset_classification.csv", index=False)

    reg_df = clf_df.copy()
    reg_df.to_csv(RESULTS_DIR / "dataset_regression.csv", index=False)

    if verbose:
        print(f"  Classification dataset: {clf_df.shape[0]} rows × {clf_df.shape[1]} cols")
        print(f"  Fatigued: {clf_df['Fatigued'].sum()} / {len(clf_df)}"
              f" ({clf_df['Fatigued'].mean()*100:.1f}%)")
        print(f"  RPE range: {clf_df['RPE'].min():.1f} – {clf_df['RPE'].max():.1f}")

    if verbose:
        print("\n[2/3] Building DL window dataset ...")
    dl_data = build_dl_dataset(emg_data, label_df)
    if dl_data:
        np.savez_compressed(
            RESULTS_DIR / "dataset_dl_windows.npz",
            **dl_data
        )
        if verbose:
            print(f"  DL windows: {dl_data['windows'].shape}")

    if verbose:
        print(f"\n  Datasets saved to: {RESULTS_DIR}")
        print("[3/3] Done.")

    return clf_df, reg_df, dl_data


if __name__ == "__main__":
    clf_df, reg_df, dl_data = build_datasets(verbose=True)
