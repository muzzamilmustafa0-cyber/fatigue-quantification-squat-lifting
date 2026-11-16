import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ANALYSIS = Path(sys.argv[1]); OUT = Path(sys.argv[2])
sys.path.insert(0, str(ANALYSIS))
from signal_processing import bandpass_filter, lowpass_filter
FS = 1020.0
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "savefig.dpi": 600, "svg.fonttype": "path"})
INK2, GRID = "#52514e", "#e4e3df"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]

raw = np.load(ANALYSIS / "results" / "_emg_cache" / "Imran.npz")["4kg"][:, 0].astype(float)[10:]
filt = bandpass_filter(raw[:, None], fs=FS)[:, 0]
rect = np.abs(filt)
env = lowpass_filter(rect[:, None], fs=FS)[:, 0]
# 3 s window centred on a lift burst in the middle third of the session
n, w = len(env), int(3 * FS)
mid = env[n // 3: 2 * n // 3]
c = n // 3 + int(np.argmax(np.convolve(mid, np.ones(int(0.5 * FS)), "same")))
s0 = max(0, c - w // 2); sl = slice(s0, s0 + w); t = np.arange(w) / FS

panels = [(raw[sl] - np.median(raw[sl]), "(a) Raw signal"), (filt[sl], "(b) Band-pass filtered, 20–450 Hz"),
          (rect[sl], "(c) Full-wave rectified"), (env[sl], "(d) Linear envelope, 6 Hz low-pass")]
fig, axs = plt.subplots(4, 1, figsize=(5.4, 4.6), sharex=True)
for a, (y, ttl), col in zip(axs, panels, COL):
    a.plot(t, y, color=col, lw=0.6 if "envelope" not in ttl else 1.4)
    a.set_title(ttl, fontsize=8.6, loc="left", pad=2)
    a.set_ylabel("µV", fontsize=8.4)
    a.grid(axis="y", color=GRID, lw=.5); a.set_axisbelow(True)
    a.tick_params(labelsize=7.6)
axs[-1].set_xlabel("Time (s)")
axs[-1].set_xlim(0, 3)
fig.tight_layout(h_pad=0.6)
fig.savefig(OUT / "fig_prep.png", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "fig_prep.svg", bbox_inches="tight", facecolor="white")
print("prep done", s0 / FS)
