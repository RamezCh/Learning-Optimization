"""Self-play over several fitted markets: pairs of algorithms, mean revenue per season (seasons 6+) for each side.
Usage: python analytics/selfplay_markets.py OUT.csv A.py:B.py [A.py:B.py ...] --markets k1,k2,... --seasons 30 --seeds 2"""
import sys, os, json, argparse, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
import selfplay as sp
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run(args):
    a, b, key, theta, seasons, seed = args
    out = sp.play((a, b, np.array(theta), seasons, seed))
    late = [r for r in out if r["season"] > 5]
    return dict(pair=f"{os.path.basename(a)} vs {os.path.basename(b)}", market=key, seed=seed,
                rev_a=np.mean([r["rev_a"] for r in late]), rev_b=np.mean([r["rev_b"] for r in late]),
                p_a=np.nanmean([r["p_a"] for r in late]), p_b=np.nanmean([r["p_b"] for r in late]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("pairs", nargs="+")
    ap.add_argument("--markets", required=True); ap.add_argument("--data", default="v6.7")
    ap.add_argument("--seasons", type=int, default=30); ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    F = json.load(open(os.path.join(ROOT, "analytics", f"demand_fits_{a.data}.json")))
    keys = [k for m in a.markets.split(",") for k in sorted(F) if m in k][: len(a.markets.split(",")) * 1]
    keys = list(dict.fromkeys(next(k for k in sorted(F) if m in k) for m in a.markets.split(",")))
    jobs = [(p.split(":")[0], p.split(":")[1], k, F[k]["theta"], a.seasons, s)
            for p in a.pairs for k in keys for s in range(1, a.seeds + 1)]
    with Pool(2) as pool:
        df = pd.DataFrame(pool.map(run, jobs, chunksize=1))
    df.to_csv(a.out, index=False)
    print(df.groupby(["pair", "market"])[["rev_a", "rev_b", "p_a", "p_b"]].mean().round(0).to_string())
    print(df.groupby("pair")[["rev_a", "rev_b"]].mean().round(0).to_string())
