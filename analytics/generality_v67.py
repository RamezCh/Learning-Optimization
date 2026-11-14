"""
analytics/generality_v67.py
---------------------------
Collects the generality checks of v6.7 into analytics/v67_generality.json (read by write_evaluation_md.py):
  1. per rival bot and leave-one-rival-out (analytics/per_rival_v67.py -> analytics/v67_per_rival.csv)
  2. imperfect copiers: test bed with --copier-mode lag2 / noise / partial on all copier matches (analytics/copier_modes/cop*_<mode>.csv)
  3. data/v6.4, the 8 files not used for any decision (208 matches x 10 seasons, analytics/v67_testbed_v64rest.csv)
In these files duopoly.py is v6.7 and duopoly_v66.py is v6.6.
"""
import json
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
W = os.path.join(ROOT, "work")


def cmp(t, new="duopoly.py", old="duopoly_v66.py"):
    ch = t[new] / t[old] - 1
    cnt = t[old] >= 1000
    return dict(matches=int(len(t)), change=float(100 * (t[new].sum() / t[old].sum() - 1)),
                better=int((ch[cnt] > 0.005).sum()), worse=int((ch[cnt] < -0.005).sum()),
                worst_match=float(100 * ch[cnt].min()) if cnt.any() else 0.0, median_match=float(100 * ch[cnt].median()))


out = {}
g = pd.read_csv(os.path.join(ROOT, "analytics", "v67_per_rival.csv"), index_col=0)
counted = g[g.v66 > 50]
out["per_rival"] = dict(rivals=int(len(g)), matches=int(g.matches.sum()), better=int((counted.change > 0.5).sum()),
                        worse=int((counted.change < -0.5).sum()), unchanged=int((counted.change.abs() <= 0.5).sum()),
                        loo_min=float(g.total_without_this_rival.min()), loo_min_rival=str(g.total_without_this_rival.idxmin()),
                        loo_max=float(g.total_without_this_rival.max()), top3_share=float(g.share_of_gain.head(3).sum()),
                        top3=[dict(rival=r, type=row.type, change=float(row.change)) for r, row in g.head(3).iterrows()],
                        copier_bots=int((g.type == "copier").sum()))
out["copier_modes"] = {}
for mode in ("lag2", "noise", "partial"):
    out["copier_modes"][mode] = {d: cmp(pd.read_csv(os.path.join(ROOT, "analytics", "copier_modes", f"cop{d}_{mode}.csv"))) for d in ("66", "65")}
out["copier_modes"]["perfect"] = {"66": None, "65": None}
p = os.path.join(ROOT, "analytics", "v67_testbed_v64rest.csv")
if os.path.exists(p):
    t = pd.read_csv(p)
    out["v64_rest"] = cmp(t)
    out["v64_rest"]["files"] = int(t.key.str.split("/").str[1].nunique())
    rest = t.assign(gain=t["duopoly.py"] - t["duopoly_v66.py"]).sort_values("gain")
    out["v64_rest"]["without_top5"] = float(100 * (rest.iloc[:-5]["duopoly.py"].sum() / rest.iloc[:-5]["duopoly_v66.py"].sum() - 1))
json.dump(out, open(os.path.join(ROOT, "analytics", "v67_generality.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
