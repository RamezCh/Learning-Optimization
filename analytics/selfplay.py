"""
analytics/selfplay.py
---------------------
Two pricing algorithms play each other in a simulated market (the test bed replays logged rival prices
and cannot show how two adaptive bots react to each other).

Market: the match's fitted choice model, symmetric for both airlines. Buyers arrive Poisson(n e^{k tau});
each picks us with  e^{b(m-p)} / (1 + e^{b(m-p)} + e^{b(m-r)}),  the rival symmetrically, or no one.
By Poisson thinning our sales and the rival's are independent Poisson draws. 80 seats each; a sold-out
airline posts its price but sells nothing and its rival faces no competitor.

Usage: python analytics/selfplay.py A.py B.py --fit <match key substring> --seasons 30 --seeds 3
"""
import argparse, json, os, sys, zlib, importlib.util
import numpy as np
from scipy.stats import poisson
from multiprocessing import Pool
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "analytics"))
import testbed as tb


def _module(path, tag):
    spec = importlib.util.spec_from_file_location("sp_%s_%d" % (tag, zlib.crc32(path.encode())), path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def play(args):
    pa, pb, theta, n_seasons, seed = args
    A, B = _module(pa, "A"), _module(pb, "B")
    ia = ib = None
    out = []
    for s in range(1, n_seasons + 1):
        rng = np.random.default_rng(seed * 100003 + s * 7919)
        da, db = rng.random(100), rng.random(100)
        pa_h, pb_h, sa, sb = [], [], [], []
        la = lb = 80
        for t in range(1, 101):
            ha = np.array([pa_h, pb_h]) if t > 1 else None
            hb = np.array([pb_h, pa_h]) if t > 1 else None
            qa_hist = np.array(sa) if t > 1 else None
            qb_hist = np.array(sb) if t > 1 else None
            p1, ia = A.p(s, t, ha, qa_hist, lb > 0, ia)
            p2, ib = B.p(s, t, hb, qb_hist, la > 0, ib)
            p1, p2 = float(p1), float(p2)
            r1 = p2 if lb > 0 else 1000.0
            r2 = p1 if la > 0 else 1000.0
            tau = (t - 1) / 99.0
            q1 = int(min(la, poisson.ppf(da[t - 1], tb.rate(theta, p1, r1, tau)))) if la > 0 else 0
            q2 = int(min(lb, poisson.ppf(db[t - 1], tb.rate(theta, p2, r2, tau)))) if lb > 0 else 0
            la -= q1; lb -= q2
            pa_h.append(p1); pb_h.append(p2); sa.append(q1); sb.append(q2)
        A.p(s, 101, np.array([pa_h, pb_h]), np.array(sa), lb > 0, ia)
        B.p(s, 101, np.array([pb_h, pa_h]), np.array(sb), la > 0, ib)
        live = [i for i in range(100) if pa_h[i] < 120 and pb_h[i] < 120]
        out.append(dict(season=s, rev_a=float(np.dot([x if x < 120 else 0 for x in pa_h], sa)),
                        rev_b=float(np.dot([x if x < 120 else 0 for x in pb_h], sb)),
                        sold_a=sum(sa), sold_b=sum(sb),
                        p_a=float(np.median([pa_h[i] for i in live[:50]])) if live else np.nan,
                        p_b=float(np.median([pb_h[i] for i in live[:50]])) if live else np.nan))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--fit", default="3DkTAv")
    ap.add_argument("--data", default="v6.7")
    ap.add_argument("--seasons", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    fits = json.load(open(os.path.join(ROOT, "analytics", f"demand_fits_{a.data}.json")))
    keys = [k for k in sorted(fits) if a.fit in k]
    theta = np.array(fits[keys[0]]["theta"])
    with Pool(min(2, a.seeds)) as pool:
        res = pool.map(play, [(a.a, a.b, theta, a.seasons, s) for s in range(1, a.seeds + 1)])
    import pandas as pd
    df = pd.concat([pd.DataFrame(r).assign(seed=i + 1) for i, r in enumerate(res)])
    g = df.groupby("season").mean(numeric_only=True)
    print(f"market {keys[0]}  m={np.exp(theta[1]):.0f} b={np.exp(theta[2]):.3f}")
    print(g[["rev_a", "rev_b", "sold_a", "sold_b", "p_a", "p_b"]].iloc[[0, 1, 2, 4, 9, 19, -1]].round(1).to_string())
    late = df[df.season > 5]
    print(f"seasons 6+: rev A {late.rev_a.mean():.0f}  rev B {late.rev_b.mean():.0f}  price A {late.p_a.mean():.1f}  B {late.p_b.mean():.1f}")
    return df


if __name__ == "__main__":
    main()
