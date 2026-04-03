import os
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parent.parent          # project root
DATA_DIR   = BASE_DIR / "Data"
FORM_PATH  = BASE_DIR / "FINAL DATA ANALYSIS FORM.xlsx"

MUSCLE_NAMES = [
    "Deltoid_Medial",
    "Deltoid_Anterior",
    "Trapezius",
    "Supraspinatus",
    "Biceps_Femoris",
    "Vastus_Lateralis",
    "Gastrocnemius_Medial",
    "Rectus_Femoris",
]

FS = 1020.0   # Sampling frequency (Hz)


def _find_data_start(df_raw: pd.DataFrame) -> int:
    n_rows = min(120, len(df_raw))
    # Pass 1: look for [DATA] marker
    for i in range(n_rows):
        try:
            val_str = str(df_raw.iat[i, 0]).strip().lower()
            if val_str == "[data]":
                return i + 1
        except Exception:
            continue
    # Pass 2: find first row where most columns are numeric (actual data)
    for i in range(n_rows):
        try:
            row_vals = pd.to_numeric(df_raw.iloc[i, :8], errors="coerce")
            if row_vals.notna().sum() >= 6:
                return i
        except Exception:
            continue
    return 51   # Hard-coded fallback


def load_subject_emg(filepath: Path) -> dict:
    result = {}
    xl = pd.ExcelFile(filepath, engine="openpyxl")
    for sheet in xl.sheet_names:
        key = None
        if "4" in sheet:
            key = "4kg"
        elif "8" in sheet:
            key = "8kg"
        else:
            continue

        df_raw = pd.read_excel(filepath, sheet_name=sheet,
                               header=None, engine="openpyxl")
        start = _find_data_start(df_raw)
        data_block = df_raw.iloc[start:, :9].copy()   # col 0 might be row index; cols 1-8 = channels
        # Drop the first column if it looks like a row counter / label
        # Check: if col 0 is always sequential integers keep it; otherwise it is a label column
        col0 = pd.to_numeric(data_block.iloc[:, 0], errors="coerce")
        if col0.isna().mean() > 0.5:
            data_block = data_block.iloc[:, 1:9]
        else:
            # Could be a time-stamp column — drop it
            data_block = data_block.iloc[:, 1:9]

        data_block = data_block.apply(pd.to_numeric, errors="coerce").dropna()
        arr = data_block.values.astype(np.float32)

        # Ensure exactly 8 channels
        if arr.shape[1] < 8:
            pad = np.zeros((arr.shape[0], 8 - arr.shape[1]), dtype=np.float32)
            arr = np.hstack([arr, pad])
        elif arr.shape[1] > 8:
            arr = arr[:, :8]

        result[key] = arr
    return result


CACHE_DIR = BASE_DIR / "analysis" / "results" / "_emg_cache"


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.npz"


def _save_emg_cache(name: str, data: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for k, v in data.items():
        arrays[k.replace("kg", "kg")] = v
    np.savez_compressed(_cache_path(name), **arrays)


def _load_emg_cache(name: str) -> dict | None:
    p = _cache_path(name)
    if not p.exists():
        return None
    npz = np.load(p)
    return {k: npz[k] for k in npz.files}


def load_all_emg(verbose: bool = True) -> dict:
    subject_files = sorted(DATA_DIR.glob("*.xlsx"))
    emg_data = {}
    iterator = tqdm(subject_files, desc="Loading EMG files") if verbose else subject_files
    for fp in iterator:
        name = fp.stem
        # Try cache first
        cached = _load_emg_cache(name)
        if cached is not None:
            emg_data[name] = cached
            continue
        try:
            data = load_subject_emg(fp)
            emg_data[name] = data
            _save_emg_cache(name, data)
        except Exception as e:
            print(f"  [WARN] Could not load {name}: {e}")
    return emg_data


def load_demographics() -> pd.DataFrame:
    df = pd.read_excel(FORM_PATH, sheet_name="Bio data ", engine="openpyxl")
    df.columns = df.columns.str.strip()
    # Keep only rows with valid weight/height
    df = df[["Names", "Weight (Kg)", "Height (cm)", "BMI"]].dropna(subset=["Weight (Kg)"])
    df["Names"] = df["Names"].astype(str).str.strip()
    df = df.rename(columns={"Names": "Subject", "Weight (Kg)": "Weight_kg",
                             "Height (cm)": "Height_cm"})
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    df["Weight_kg"] = pd.to_numeric(df["Weight_kg"], errors="coerce")
    df["Height_cm"] = pd.to_numeric(df["Height_cm"], errors="coerce")
    return df.reset_index(drop=True)


def _load_offset_map(sheet_name: str, load_kg: int) -> pd.DataFrame:
    df = pd.read_excel(FORM_PATH, sheet_name=sheet_name,
                       header=None, engine="openpyxl")

    records = []
    # Find the first header row containing "rpe" or "mpf" labels
    header_row = 0
    for i in range(min(5, len(df))):
        row_str_vals = [str(v).lower() for v in df.iloc[i].tolist()]
        if any("rpe" in v or "mpf" in v for v in row_str_vals):
            header_row = i
            break

    data = df.iloc[header_row + 1:].reset_index(drop=True)

    n_cols   = data.shape[1]
    subj_idx = 0
    col      = 0
    while col + 1 < n_cols:
        rpe_col = data.iloc[:, col]
        mpf_col = data.iloc[:, col + 1] if col + 1 < n_cols else None
        mdf_col = data.iloc[:, col + 2] if col + 2 < n_cols else None

        rpe_vals = pd.to_numeric(rpe_col, errors="coerce")
        valid    = rpe_vals.notna()
        if valid.sum() < 2:
            col += 3
            subj_idx += 1
            continue

        mpf_vals = pd.to_numeric(mpf_col, errors="coerce") if mpf_col is not None else None
        mdf_vals = pd.to_numeric(mdf_col, errors="coerce") if mdf_col is not None else None

        # Use positional indexing to avoid iloc/loc mismatch
        rpe_arr = rpe_vals[valid].values
        mpf_arr = mpf_vals[valid].values if mpf_vals is not None else [np.nan] * len(rpe_arr)
        mdf_arr = mdf_vals[valid].values if mdf_vals is not None else [np.nan] * len(rpe_arr)

        for sq_no, (rpe, mpf_v, mdf_v) in enumerate(zip(rpe_arr, mpf_arr, mdf_arr)):
            rec = {
                "Subject_idx": subj_idx,
                "Load_kg":     load_kg,
                "Squat_No":    sq_no + 1,
                "RPE":         float(rpe),
                "MPF":         float(mpf_v) if not np.isnan(float(mpf_v) if mpf_v is not None else float('nan')) else np.nan,
                "MDF":         float(mdf_v) if not np.isnan(float(mdf_v) if mdf_v is not None else float('nan')) else np.nan,
            }
            records.append(rec)

        col += 3
        subj_idx += 1

    out = pd.DataFrame(records)
    return out


def _load_vertical_sheet(sheet_name: str, load_kg: int) -> pd.DataFrame:
    df = pd.read_excel(FORM_PATH, sheet_name=sheet_name,
                       header=0, engine="openpyxl")
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    # Standardize column names
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if "subject" in cl:
            rename[c] = "Subject_idx"
        elif "upper" in cl:
            rename[c] = "EMG_Upper"
        elif "lower" in cl:
            rename[c] = "EMG_Lower"
        elif "middle" in cl or "mid" in cl:
            rename[c] = "EMG_Middle"
        elif "h.r" in cl or "hr" in cl or "heart" in cl:
            rename[c] = "HR"
    df = df.rename(columns=rename)
    df["Load_kg"] = load_kg

    for col in ["Subject_idx", "EMG_Upper", "EMG_Lower", "EMG_Middle", "HR"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Subject_idx", "EMG_Upper"])
    return df.reset_index(drop=True)


def load_label_dataframe() -> pd.DataFrame:
    try:
        df_raw = pd.read_excel(FORM_PATH, sheet_name="Filled form S",
                               header=None, engine="openpyxl")
    except Exception as e:
        print(f"  [WARN] Could not load Filled form S: {e}")
        return pd.DataFrame()

    records = []
    current_subject = "Unknown"
    current_load    = None
    subject_idx     = -1

    for i, row in df_raw.iterrows():
        obs_val   = row.iloc[3]  # Observation number column
        squat_val = row.iloc[4]  # Squat count column

        # Check for subject name in col 1 (only real subject names — not numbers)
        col1 = row.iloc[1]
        col2 = row.iloc[2]

        if isinstance(col1, str) and len(str(col1).strip()) > 1:
            s = str(col1).strip()
            # Only update subject if it looks like a name (not a number/unit string)
            if not s.replace(".", "").replace(",", "").isdigit():
                current_subject = s
                subject_idx += 1

        # Check for load condition in col 2
        if isinstance(col2, str):
            c2 = str(col2).strip().lower()
            if "4" in c2 and "kg" in c2:
                current_load = 4
            elif "8" in c2 and "kg" in c2:
                current_load = 8
        elif pd.notna(col2):
            c2_str = str(col2).strip().lower()
            if "4" in c2_str and "kg" in c2_str:
                current_load = 4
            elif "8" in c2_str and "kg" in c2_str:
                current_load = 8

        # Parse data rows: obs_val must be integer 1-9, squat_val integer 4-28
        try:
            obs_int   = int(obs_val)
            squat_int = int(squat_val)
        except (TypeError, ValueError):
            continue

        if not (1 <= obs_int <= 10 and 1 <= squat_int <= 30):
            continue

        rpe_upper = pd.to_numeric(row.iloc[6], errors="coerce")
        rpe_mid   = pd.to_numeric(row.iloc[7], errors="coerce")
        rpe_lower = pd.to_numeric(row.iloc[8], errors="coerce")
        hr_val    = pd.to_numeric(row.iloc[9], errors="coerce")
        time_val  = pd.to_numeric(row.iloc[5], errors="coerce")

        # Use max RPE across regions as the overall RPE (conservative fatigue indicator)
        rpe_max = float(np.nanmax([rpe_upper, rpe_mid, rpe_lower]
                                   if not all(np.isnan([rpe_upper, rpe_mid, rpe_lower]))
                                   else [np.nan]))

        records.append({
            "Subject":     current_subject,
            "Subject_idx": subject_idx,
            "Load_kg":     current_load if current_load else 0,
            "Obs_No":      obs_int,
            "Squat_No":    squat_int,
            "Time_s":      float(time_val) if not np.isnan(time_val) else np.nan,
            "RPE_Upper":   float(rpe_upper) if not np.isnan(rpe_upper) else np.nan,
            "RPE_Middle":  float(rpe_mid)   if not np.isnan(rpe_mid)   else np.nan,
            "RPE_Lower":   float(rpe_lower) if not np.isnan(rpe_lower) else np.nan,
            "RPE":         rpe_max,
            "HR":          float(hr_val) if not np.isnan(hr_val) else np.nan,
        })

    label_df = pd.DataFrame(records)
    if label_df.empty:
        return label_df

    # Remove rows with no valid load
    label_df = label_df[label_df["Load_kg"] > 0].copy()

    # Clip RPE to valid Borg CR10 scale range (0-10); values > 10 are data entry errors
    for col in ["RPE_Upper", "RPE_Middle", "RPE_Lower", "RPE"]:
        label_df[col] = label_df[col].clip(upper=10.0)

    # Add binary fatigue label
    label_df["Fatigued"] = (label_df["RPE"] >= 5).astype(int)

    return label_df.reset_index(drop=True)


def load_all(verbose: bool = True):
    if verbose:
        print("=" * 60)
        print("  DATA LOADER")
        print("=" * 60)
        print(f"  Data directory  : {DATA_DIR}")
        print(f"  Analysis form   : {FORM_PATH}")

    demo_df  = load_demographics()
    label_df = load_label_dataframe()
    emg_data = load_all_emg(verbose=verbose)

    if verbose:
        n_subjects = len(emg_data)
        n_with_both = sum(1 for v in emg_data.values() if "4kg" in v and "8kg" in v)
        print(f"\n  Subjects loaded : {n_subjects}  ({n_with_both} with both 4kg & 8kg)")
        print(f"  Label rows      : {len(label_df)}")
        print(f"  Demo rows       : {len(demo_df)}")
        if not label_df.empty and "Fatigued" in label_df.columns:
            fatigue_pct = label_df["Fatigued"].mean() * 100
            print(f"  Fatigued samples: {fatigue_pct:.1f}%")

    return emg_data, label_df, demo_df


if __name__ == "__main__":
    emg_data, label_df, demo_df = load_all(verbose=True)
    print("\nDemographics sample:")
    print(demo_df.head(5).to_string())
    print("\nLabel DataFrame sample:")
    print(label_df.head(10).to_string())
    if emg_data:
        first_subj = next(iter(emg_data))
        for load in ["4kg", "8kg"]:
            if load in emg_data[first_subj]:
                arr = emg_data[first_subj][load]
                print(f"\n  {first_subj} / {load} — shape: {arr.shape}")
