import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, sosfiltfilt, butter
from typing import List, Tuple, Optional

FS = 1020.0          # Sampling frequency (Hz)
WINDOW_SEC   = 2.0   # Sliding window duration (s)
OVERLAP      = 0.5   # Overlap fraction for sliding windows
MIN_SQUAT_SAMPLES = int(0.5 * FS)   # Minimum squat duration: 0.5 s


def bandpass_filter(signal: np.ndarray, lowcut: float = 20.0,
                    highcut: float = 450.0, fs: float = FS,
                    order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    low  = lowcut  / nyq
    high = min(highcut / nyq, 0.999)
    b, a = butter(order, [low, high], btype="band")
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    # Multi-channel: filter each column
    out = np.empty_like(signal)
    for ch in range(signal.shape[1]):
        out[:, ch] = filtfilt(b, a, signal[:, ch])
    return out


def lowpass_filter(signal: np.ndarray, cutoff: float = 6.0,
                   fs: float = FS, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    norm_cut = cutoff / nyq
    b, a = butter(order, norm_cut, btype="low")
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    out = np.empty_like(signal)
    for ch in range(signal.shape[1]):
        out[:, ch] = filtfilt(b, a, signal[:, ch])
    return out


def preprocess_emg(raw: np.ndarray, fs: float = FS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    filtered  = bandpass_filter(raw.astype(np.float64), fs=fs)
    rectified = np.abs(filtered)
    envelope  = lowpass_filter(rectified, fs=fs)
    return filtered.astype(np.float32), rectified.astype(np.float32), envelope.astype(np.float32)


def segment_squats_by_peaks(envelope: np.ndarray,
                             fs: float = FS,
                             min_squat_sec: float = 0.5,
                             max_squat_sec: float = 8.0) -> List[Tuple[int, int]]:
    # Combined activation
    combined = envelope.sum(axis=1)

    # Moving average smoothing (0.5 s window)
    win = int(0.5 * fs)
    kernel = np.ones(win) / win
    smoothed = np.convolve(combined, kernel, mode="same")

    # Normalize
    smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)

    # Find peaks with minimum distance of min_squat_sec
    min_dist = int(min_squat_sec * fs)
    max_dist = int(max_squat_sec * fs)
    peaks, _ = find_peaks(smoothed,
                          distance=min_dist,
                          height=0.2,      # at least 20% of max amplitude
                          prominence=0.1)

    if len(peaks) < 2:
        # Fallback: equal-duration segmentation
        return segment_squats_equal(len(envelope), fs=fs, n_squats=25)

    # Find valleys between peaks
    segments = []
    boundaries = [0] + list(peaks) + [len(envelope) - 1]
    for i in range(len(peaks)):
        start = boundaries[i]
        end   = boundaries[i + 2]
        dur   = end - start
        if min_squat_sec * fs <= dur <= max_squat_sec * fs:
            segments.append((start, end))

    if len(segments) < 3:
        return segment_squats_equal(len(envelope), fs=fs, n_squats=25)

    return segments


def segment_squats_equal(n_samples: int, fs: float = FS,
                          n_squats: int = 25) -> List[Tuple[int, int]]:
    seg_len = n_samples // n_squats
    return [(i * seg_len, (i + 1) * seg_len) for i in range(n_squats)]


def extract_squat_segments(filtered: np.ndarray, envelope: np.ndarray,
                            fs: float = FS) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    segs = segment_squats_by_peaks(envelope, fs=fs)
    filt_segs = [filtered[s:e] for s, e in segs]
    env_segs  = [envelope[s:e] for s, e in segs]
    return filt_segs, env_segs


def sliding_windows(signal: np.ndarray,
                    fs: float = FS,
                    window_sec: float = WINDOW_SEC,
                    overlap: float = OVERLAP) -> np.ndarray:
    win_len  = int(window_sec * fs)
    step     = int(win_len * (1 - overlap))
    n_samples = signal.shape[0]
    indices  = range(0, n_samples - win_len + 1, step)
    windows  = np.stack([signal[i:i + win_len] for i in indices], axis=0)
    return windows   # [W, win_len, 8]


def compute_nmpf_series(mpf_series: np.ndarray) -> np.ndarray:
    baseline = mpf_series[0:3].mean(axis=0, keepdims=True)   # Average first 3 squats as baseline
    baseline[baseline == 0] = 1.0
    nmpf = mpf_series / baseline
    return nmpf


def process_subject(raw: np.ndarray, fs: float = FS):
    filtered, rectified, envelope = preprocess_emg(raw, fs=fs)
    squat_filtered, squat_envelope = extract_squat_segments(filtered, envelope, fs=fs)
    dl_windows = sliding_windows(filtered, fs=fs)

    return {
        "filtered":       filtered,
        "rectified":      rectified,
        "envelope":       envelope,
        "squat_filtered": squat_filtered,
        "squat_envelope": squat_envelope,
        "dl_windows":     dl_windows,
    }


if __name__ == "__main__":
    # Quick smoke-test with synthetic data
    np.random.seed(42)
    fake_raw = (np.random.randn(50000, 8) * 100).astype(np.float32)
    result = process_subject(fake_raw)
    print(f"Filtered shape  : {result['filtered'].shape}")
    print(f"Envelope shape  : {result['envelope'].shape}")
    print(f"Squats detected : {len(result['squat_filtered'])}")
    print(f"DL windows      : {result['dl_windows'].shape}")
