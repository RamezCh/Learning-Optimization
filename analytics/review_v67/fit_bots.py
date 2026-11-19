"""Fit a simulated rival (analytics/rival_bot.py) to every v6.7 match; also store v6.7's logged revenue per season."""
import sys, os, json, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "analytics"))
import testbed as tb, rival_bot
tb.set_data("v6.7"); D = tb.load()
out = os.path.join(ROOT, "analytics", "review_v67", "bots"); os.makedirs(out, exist_ok=True)
rows = []
for key, g in D.groupby("key"):
    P = g.price.values.reshape(-1, 100); R = g.price_competitor.values.reshape(-1, 100); C = g.cap.values.reshape(-1, 100).astype(bool)
    ok = C & (P < 120) & (R < 120) & (R > 0)
    if ok.sum() < 500:
        continue
    q = rival_bot.fit(P, R, ok)
    name = key.replace("/", "__").replace(" ", "_") + ".json"
    json.dump(q, open(os.path.join(out, name), "w"))
    rev = (np.where(P < 120, P, 0) * g.demand.values.reshape(-1, 100)).sum(axis=1)
    rows.append(dict(key=key, bot=name, a=q["coef"][0], c=q["coef"][1], b=q["coef"][2], d=q["coef"][3], g=q["coef"][4],
                     rev_logs_6_30=float(rev[5:30].mean())))
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "analytics", "review_v67", "bots.csv"), index=False)
print(len(rows))
