"""Play algorithms against the simulated rivals of chosen v6.7 matches. Usage: bot_eval.py OUT.csv algo1.py [algo2.py ...] --keys file.txt --seasons 30 --seeds 2"""
import sys, os, json, argparse, numpy as np, pandas as pd
from multiprocessing import Pool
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "analytics"))
import selfplay as sp
BOTS = os.path.join(ROOT, "analytics", "review_v67", "bots")

def run(a):
    algo, key, bot, theta, seasons, seed = a
    out = sp.play((algo, "bot:" + os.path.join(BOTS, bot), np.array(theta), seasons, seed))
    late = [r for r in out if r["season"] > 5]
    return dict(algo=os.path.basename(algo), key=key, seed=seed, rev=np.mean([r["rev_a"] for r in late]),
                rival_rev=np.mean([r["rev_b"] for r in late]), p=np.nanmean([r["p_a"] for r in late]),
                r=np.nanmean([r["p_b"] for r in late]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("algos", nargs="+")
    ap.add_argument("--keys", required=True); ap.add_argument("--seasons", type=int, default=30); ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    F = json.load(open(os.path.join(ROOT, "analytics", "demand_fits_v6.7.json")))
    B = pd.read_csv(os.path.join(ROOT, "analytics", "review_v67", "bots.csv")).set_index("key")
    keys = [k.strip() for k in open(a.keys) if k.strip()]
    jobs = [(al, k, B.loc[k, "bot"], F[k]["theta"], a.seasons, s) for al in a.algos for k in keys for s in range(1, a.seeds + 1)]
    with Pool(2) as pool:
        df = pd.DataFrame(pool.map(run, jobs, chunksize=1))
    df.to_csv(a.out, index=False)
    print(df.groupby("algo")[["rev", "rival_rev"]].mean().round(0).to_string())
