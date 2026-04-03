import numpy as np
from scipy.signal import welch
from scipy.stats import entropy
from typing import List, Optional, Dict, Tuple

FS = 1020.0

# Channel groupings: upper/middle/lower regions
# Channels 0-3: Upper extremity (Deltoid Med, Deltoid Ant, Trapezius, Supraspinatus)
# Channels 4-7: Lower extremity (Biceps Femoris, Vastus Lat, Gastrocnemius, Rectus Femoris)
UPPER_CH  = [0, 1, 2, 3]
LOWER_CH  = [4, 5, 6, 7]
MIDDLE_CH = [2, 3, 4, 5]   # Trunk-spanning muscles

MUSCLE_NAMES = [
    "DeltMed", "DeltAnt", "Trap", "Supraspinatus",
    "BicepsFem", "VastLat", "Gastrocnemius", "RectusFem"
]

FEATURE_SUFFIXES = ["RMS", "IEMG", "ZCR", "WL", "VAR", "MAV",
                    "MPF", "MDF", "PeakFreq", "SpectralEntropy"]


def rms(sig: np.ndarray) -> float:
    return float(np.sqrt(np.mean(sig ** 2)))

def iemg(sig: np.ndarray, fs: float = FS) -> float:
    return float(np.sum(np.abs(sig)) / fs)

def zcr(sig: np.ndarray, fs: float = FS) -> float:
    crossings = np.sum(np.diff(np.sign(sig)) != 0)
    return float(crossings / (len(sig) / fs))

def wl(sig: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(sig))))

def mav(sig: np.ndarray) -> float:
    return float(np.mean(np.abs(sig)))


def power_spectrum(sig: np.ndarray, fs: float = FS,
                   nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if nperseg is None:
        nperseg = min(len(sig), 256)
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg, scaling="density")
    return freqs, psd


def mpf(sig: np.ndarray, fs: float = FS) -> float:
    freqs, psd = power_spectrum(sig, fs=fs)
    total = np.sum(psd)
    if total == 0:
        return 0.0
    return float(np.sum(freqs * psd) / total)


def mdf(sig: np.ndarray, fs: float = FS) -> float:
    freqs, psd = power_spectrum(sig, fs=fs)
    cumpower = np.cumsum(psd)
    half = cumpower[-1] / 2.0
    idx = np.searchsorted(cumpower, half)
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def peak_freq(sig: np.ndarray, fs: float = FS) -> float:
    freqs, psd = power_spectrum(sig, fs=fs)
    return float(freqs[np.argmax(psd)])


def spectral_entropy(sig: np.ndarray, fs: float = FS) -> float:
    _, psd = power_spectrum(sig, fs=fs)
    psd_norm = psd / (psd.sum() + 1e-12)
    return float(entropy(psd_norm + 1e-12))


def ar_coefficients(sig: np.ndarray, order: int = 4) -> np.ndarray:
    sig = sig - sig.mean()
    n = len(sig)
    R = np.array([np.dot(sig[:n - k], sig[k:]) / n for k in range(order + 1)])
    # Yule-Walker using Levinson-Durbin
    a = np.zeros(order)
    e = R[0]
    for i in range(order):
        if e == 0:
            break
        lam = -R[i + 1] / e
        a_new = a[:i + 1] + lam * a[:i + 1][::-1]
        a[:i + 1] = a_new
        a[i] = lam
        e *= (1 - lam ** 2)
    return a


def channel_features(sig: np.ndarray, fs: float = FS, ar_order: int = 4) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    feats["RMS"]             = rms(sig)
    feats["IEMG"]            = iemg(sig, fs)
    feats["ZCR"]             = zcr(sig, fs)
    feats["WL"]              = wl(sig)
    feats["VAR"]             = float(np.var(sig))
    feats["MAV"]             = mav(sig)
    feats["MPF"]             = mpf(sig, fs)
    feats["MDF"]             = mdf(sig, fs)
    feats["PeakFreq"]        = peak_freq(sig, fs)
    feats["SpectralEntropy"] = spectral_entropy(sig, fs)
    ar = ar_coefficients(sig, order=ar_order)
    for i, v in enumerate(ar):
        feats[f"AR{i+1}"] = float(v)
    return feats


def segment_to_features(segment: np.ndarray, fs: float = FS,
                         ar_order: int = 4) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    for ch in range(segment.shape[1]):
        name = MUSCLE_NAMES[ch] if ch < len(MUSCLE_NAMES) else f"Ch{ch}"
        ch_feats = channel_features(segment[:, ch], fs=fs, ar_order=ar_order)
        for k, v in ch_feats.items():
            feats[f"{name}_{k}"] = v

    # Region-level aggregates (mean RMS and mean MPF)
    for region, chs in [("Upper", UPPER_CH), ("Lower", LOWER_CH), ("Middle", MIDDLE_CH)]:
        region_rms = np.mean([feats[f"{MUSCLE_NAMES[c]}_RMS"] for c in chs if c < len(MUSCLE_NAMES)])
        region_mpf = np.mean([feats[f"{MUSCLE_NAMES[c]}_MPF"] for c in chs if c < len(MUSCLE_NAMES)])
        region_mdf = np.mean([feats[f"{MUSCLE_NAMES[c]}_MDF"] for c in chs if c < len(MUSCLE_NAMES)])
        feats[f"{region}_RMS_mean"] = float(region_rms)
        feats[f"{region}_MPF_mean"] = float(region_mpf)
        feats[f"{region}_MDF_mean"] = float(region_mdf)

    return feats


def extract_features_from_squats(squat_segments: List[np.ndarray],
                                  fs: float = FS) -> np.ndarray:
    if not squat_segments:
        return np.empty((0, 0)), []

    rows = []
    for seg in squat_segments:
        row = segment_to_features(seg, fs=fs)
        rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)
    return df.values, list(df.columns)


def add_nmpf(feature_matrix: np.ndarray, feature_names: List[str]) -> np.ndarray:
    from signal_processing import compute_nmpf_series  # noqa: keep local import
    mpf_indices = [i for i, n in enumerate(feature_names) if "_MPF" in n and "mean" not in n]

    if len(mpf_indices) == 0:
        return feature_matrix, feature_names

    mpf_matrix = feature_matrix[:, mpf_indices]   # [n_squats, n_muscles]
    nmpf_matrix = mpf_matrix / (mpf_matrix[:3].mean(axis=0, keepdims=True) + 1e-8)

    nmpf_names = [feature_names[i].replace("_MPF", "_NMPF") for i in mpf_indices]
    combined = np.hstack([feature_matrix, nmpf_matrix])
    combined_names = feature_names + nmpf_names
    return combined, combined_names


def extract_features_from_windows(windows: np.ndarray,
                                   fs: float = FS) -> np.ndarray:
    rows = []
    for w in windows:
        row = segment_to_features(w, fs=fs)
        rows.append(list(row.values()))
    return np.array(rows, dtype=np.float32)


def get_feature_names(ar_order: int = 4) -> List[str]:
    names = []
    base_feats = ["RMS", "IEMG", "ZCR", "WL", "VAR", "MAV",
                  "MPF", "MDF", "PeakFreq", "SpectralEntropy"]
    ar_feats   = [f"AR{i+1}" for i in range(ar_order)]
    all_feats  = base_feats + ar_feats

    for ch in range(8):
        name = MUSCLE_NAMES[ch]
        for f in all_feats:
            names.append(f"{name}_{f}")

    for region in ["Upper", "Lower", "Middle"]:
        for f in ["RMS_mean", "MPF_mean", "MDF_mean"]:
            names.append(f"{region}_{f}")

    return names


if __name__ == "__main__":
    # Smoke test
    np.random.seed(0)
    fake_seg = np.random.randn(2040, 8).astype(np.float32) * 50
    feats = segment_to_features(fake_seg)
    print(f"Features per segment : {len(feats)}")
    for k, v in list(feats.items())[:10]:
        print(f"  {k:40s} = {v:.4f}")
