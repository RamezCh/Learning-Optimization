"""EVALUATION.md section 12: revenue against the oracles by rival type, market size and season; figure imgs/v68_situations.png.
Needs analytics/oracle_v6.7.csv (analytics/oracle.py --data v6.7) and analytics/review_v67/reaction_rule.csv."""
import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
A = lambda *p: os.path.join(ROOT, *p)
OWN = {"AirborneTody", "DeftDalmatian", "ProficientCivet"}
o = pd.read_csv(A("analytics", "oracle_v6.7.csv"))
rr = pd.read_csv(A("analytics", "review_v67", "reaction_rule.csv"))
rr["copy_ex"] = rr["copy_0.02"] - rr["copyP_0.02"]; rr["dir_ex"] = rr.same_dir - rr.same_dirP

def rival_type(r):
    if r.rival in OWN: return "our own account"
    if r.copy_ex >= 0.3: return "copier"
    if r.dir_ex >= 0.15 or r.copy_ex >= 0.1: return "follows our direction"
    if pd.isna(r.copy_ex): return "too little price movement"
    return "ignores us"

rr["type"] = rr.apply(rival_type, axis=1)
o = o.merge(rr[["key", "rival", "type"]], on="key", how="left"); o["type"] = o.type.fillna("too little price movement")
mo = o.groupby("key").o1.mean(); q = mo.quantile([1 / 3, 2 / 3]).values
o["market"] = o.key.map(lambda k: "poor" if mo[k] < q[0] else ("rich" if mo[k] > q[1] else "medium"))
o["seasons"] = pd.cut(o.season, [0, 1, 2, 5, 20, 100], labels=["1", "2", "3-5", "6-20", "21-100"])
gap = (o.o2 - o.actual).sum()
for col in ["type", "market", "seasons"]:
    t = o.groupby(col, observed=True).apply(lambda g: pd.Series(dict(
        matches=g.key.nunique(), share_rev=100 * g.actual.sum() / o.actual.sum(), ours=g.actual.mean(), o1=g.o1.mean(),
        eff_o1=100 * g.actual.sum() / g.o1.sum(), eff_o2=100 * g.actual.sum() / g.o2.sum(), gap_share=100 * (g.o2 - g.actual).sum() / gap)))
    print(t.round(1).to_string(), "\n")
print("market thresholds (perfect-info revenue per season):", q.round(0))
print(o.groupby(["market", "type"]).apply(lambda g: round(100 * g.actual.sum() / g.o1.sum(), 0)).unstack().to_string())
order = ["ignores us", "follows our direction", "copier", "our own account", "too little price movement"]
n = o.groupby("type").key.nunique()
lab = {"ignores us": "Ignores us", "follows our direction": "Follows our direction", "copier": "Copies us",
       "our own account": "Our own account", "too little price movement": "Dead market, no price moves"}
g = o.groupby("type").agg(ours=("actual", "mean"), o1=("o1", "mean")).reindex(order)
fig, ax = plt.subplots(figsize=(9, 3.8)); y = np.arange(len(g))[::-1]
ax.barh(y, g.o1, color="#d1d5db", height=0.6, label="best possible (perfect-information DP)")
ax.barh(y, g.ours, color="#2563eb", height=0.6, label="v6.7 earned")
for yi, a, b in zip(y, g.ours, g.o1): ax.text(b + 40, yi, f"{100 * a / b:.0f}%", va="center", fontsize=10, color="#1f2937")
ax.set_yticks(y); ax.set_yticklabels([f"{lab[i]} ({n[i]})" for i in g.index], fontsize=9, color="#6b7280")
ax.set_xlabel("revenue per season ($), all 100 seasons", color="#6b7280")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.set_title("Where the money is: v6.7 by rival type (data/v6.7, number of match-files)", loc="left", fontsize=11, color="#1f2937")
fig.tight_layout(); fig.savefig(A("imgs", "v68_situations.png"), dpi=150)
