"""
analytics/per_rival_v67.py
--------------------------
Is v6.7's gain general, or carried by a few rival bots? Pools the test-bed results on data/v6.6 (100 seasons)
and data/v6.5 (60 seasons) and reports, per distinct rival bot, v6.7 vs v6.6; plus leave-one-rival-out totals.
Writes analytics/v67_per_rival.csv and prints the summary.
"""
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sb = pd.read_csv(os.path.join(ROOT, "work", "scoreboard_both.csv")).set_index("key")
frames = []
for f, seasons in (("v67_testbed_v66data.csv", 100), ("v67_testbed_v65data.csv", 60)):
    t = pd.read_csv(os.path.join(ROOT, "analytics", f))[["key", "v6.6", "v6.7"]]
    t[["v6.6", "v6.7"]] = t[["v6.6", "v6.7"]] / seasons          # per season, so both data sets weigh the same per match
    frames.append(t)
t = pd.concat(frames)
t["rival"] = t.key.map(sb.rival)
t["type"] = t.key.map(sb.type)
g = t.groupby("rival").agg(matches=("key", "size"), type=("type", lambda x: x.mode()[0]), v66=("v6.6", "sum"), v67=("v6.7", "sum"))
g["change"] = 100 * (g.v67 / g.v66 - 1)
g["share_of_gain"] = (g.v67 - g.v66) / (g.v67 - g.v66).sum()
tot = 100 * (t["v6.7"].sum() / t["v6.6"].sum() - 1)
loo = {r: 100 * (t[t.rival != r]["v6.7"].sum() / t[t.rival != r]["v6.6"].sum() - 1) for r in g.index}
g["total_without_this_rival"] = pd.Series(loo)
g = g.sort_values("share_of_gain", ascending=False)
g.round(3).to_csv(os.path.join(ROOT, "analytics", "v67_per_rival.csv"))
counted = g[g.v66 > 50]
print(f"{len(g)} distinct rivals, {len(t)} matches; total {tot:+.2f}%")
print(f"rivals better (>0.5%) {int((counted.change > 0.5).sum())}, worse (<-0.5%) {int((counted.change < -0.5).sum())}, "
      f"unchanged {int(((counted.change.abs() <= 0.5)).sum())} (of {len(counted)} with >$50/season)")
print(f"leave-one-rival-out total: min {min(loo.values()):+.2f}% (without {min(loo, key=loo.get)}), max {max(loo.values()):+.2f}%")
top = g.head(3)
print(f"top 3 rivals carry {100 * top.share_of_gain.sum():.0f}% of the gain: " + ", ".join(f"{r} ({row.type}, {row.change:+.1f}%)" for r, row in top.iterrows()))
non_cop = t[t.type != "copier"]
print(f"without copier matches: {100 * (non_cop['v6.7'].sum() / non_cop['v6.6'].sum() - 1):+.2f}% over {len(non_cop)} matches")
print(g[["matches", "type", "change", "share_of_gain", "total_without_this_rival"]].round(2).to_string())
