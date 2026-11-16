import sys
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
ANALYSIS = Path(sys.argv[1]); OUT = Path(sys.argv[2])
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e", "savefig.dpi": 600, "svg.fonttype": "path"})
BLUE, ORANGE, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"
d = pd.read_csv(ANALYSIS / "results" / "dataset_classification.csv")
b1 = d[d.Obs_No == 1].set_index(["Subject", "Load_kg"])
idx = d.set_index(["Subject", "Load_kg"]).index
d["UMDF_rel"] = 100 * d.Upper_MDF_mean.values / np.asarray(idx.map(b1.Upper_MDF_mean), dtype=float)
d["UMPF_rel"] = 100 * d.Upper_MPF_mean.values / np.asarray(idx.map(b1.Upper_MPF_mean), dtype=float)
stats = []
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5)); blk = np.arange(1, 7)
for j, (col, ttl, yl) in enumerate((("UMDF_rel", "(a) Upper-limb median frequency", "MDF (% of block 1)"),
                                    ("UMPF_rel", "(b) Upper-limb mean power frequency", "MPF (% of block 1)"),
                                    ("HR", "(c) Heart rate", "Heart rate (bpm)"))):
    a = axs[j]
    for L, c, mk, ls in ((4, BLUE, "o", "-"), (8, ORANGE, "s", "--")):
        w = d[d.Load_kg == L].pivot_table(index="Subject", columns="Obs_No", values=col).dropna()
        m, se = w.mean(), w.sem()
        a.plot(blk * 4, m.values, color=c, marker=mk, ms=4.5, lw=1.8, ls=ls, label=f"{L} kg")
        a.fill_between(blk * 4, (m - se).values, (m + se).values, color=c, alpha=.15, lw=0)
        fr = friedmanchisquare(*[w[k].values for k in blk])
        wx = wilcoxon(w[1].values, w[6].values) if col == "HR" else wilcoxon(w[6].values - 100)
        stats.append(dict(var=col, load=L, n=len(w), mean_b1=m[1], mean_b6=m[6], friedman_chi2=fr.statistic, friedman_p=fr.pvalue, wilcoxon_b6_p=wx.pvalue))
    a.set(xlabel="Cumulative lifts", ylabel=yl, xticks=blk * 4); a.set_title(ttl, fontsize=8.6)
    a.grid(axis="y", color=GRID, lw=.6); a.set_axisbelow(True); a.legend(frameon=False, fontsize=7.4)
fig.tight_layout(); fig.savefig(OUT / "fig_physio.png", bbox_inches="tight", facecolor="white")
fig.tight_layout(); fig.savefig(OUT / "fig_physio.svg", bbox_inches="tight", facecolor="white")
s = pd.DataFrame(stats); s.to_csv(OUT / "physio_stats.csv", index=False); pd.set_option("display.width", 200); print(s.round(4).to_string())
