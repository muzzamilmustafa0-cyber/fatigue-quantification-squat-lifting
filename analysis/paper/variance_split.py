import sys
import numpy as np, pandas as pd
from pathlib import Path
A = Path(sys.argv[1]); OUT = Path(sys.argv[2])
d = pd.read_csv(A / "results" / "dataset_classification.csv")
feats = {"Upper-limb MDF": "Upper_MDF_mean", "Upper-limb MPF": "Upper_MPF_mean", "Upper-limb RMS": "Upper_RMS_mean",
         "Lower-limb RMS": "Lower_RMS_mean", "Heart rate": "HR"}
b1 = d[d.Obs_No == 1].set_index(["Subject", "Load_kg"]); idx = d.set_index(["Subject", "Load_kg"]).index
rows = []
for name, c in feats.items():
    for kind in ("absolute", "baseline-referenced"):
        x = d[c].astype(float).values
        if kind != "absolute":
            x = x / np.asarray(idx.map(b1[c]), dtype=float)
        df = d[["Subject", "Obs_No", "RPE"]].assign(x=x)
        if kind != "absolute":
            df = df[df.Obs_No > 1]
        tot = df.x.var(ddof=0)
        between = df.groupby("Subject").x.transform("mean").var(ddof=0)
        rows.append(dict(feature=name, scale=kind, between_subject_share=100 * between / tot,
                         spearman_with_RPE=df.x.corr(df.RPE, method="spearman")))
r = pd.DataFrame(rows).round(2); r.to_csv(OUT / "variance_split.csv", index=False)
pd.set_option("display.width", 200); print(r.to_string())
