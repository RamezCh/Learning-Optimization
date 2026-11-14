"""
analytics/testbed.py
--------------------
Offline test bed built from any folder of DPC logs (default data/v6.6). Same code as
analytics/v65_testbed.py, with the data folder as an argument:
    --data v6.6   ->  data/v6.6/**.csv, fits cached in analytics/demand_fits_v6.6.json

Same method as analytics/v6_testbed.py:
1. Per match, a choice model is fitted to our own daily sales (Poisson maximum likelihood):
       buyers per day ~ Poisson(n * exp(k * tau)),  P(buy from us) = e^{b(m-p)} / (1 + e^{b(m-p)} + e^{b(m-r)})
2. The rival replays its logged price and its logged sold-out days. Rivals that set their price to
   our previous price on >= 80% of days are simulated as copiers.
3. Every pricing file plays every match x 100 seasons with the same random draws per
   (seed, match, season), so differences between files are not luck.

Usage:
    python analytics/testbed.py --data v6.6 --compare path/to/candidate.py [--sample 3] [--seasons 40] [--out file.csv]
"""

import argparse
import glob
import importlib.util
import json
import os
import warnings
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.environ.get("TESTBED_DATA", "v6.6")
DATA_GLOB = os.path.join(ROOT, "data", DATA, "**", "*.csv")
FITS = os.path.join(ROOT, "analytics", f"demand_fits_{DATA}.json")
RESULTS = os.path.join(ROOT, "analytics", f"testbed_results_{DATA}.csv")


COPIER_MODE = "perfect"   # how simulated copiers copy: perfect | lag2 | noise | partial


def copier_price(mode, own, logged, t, rng):
    """Rival price of a copier on day t. perfect: our price yesterday (v6.4-v6.7 test beds). lag2: our price two
    days ago. noise: yesterday's price +- N(0, $2). partial: halfway between our price yesterday and its own logged price."""
    if mode == "lag2":
        return own[-2] if len(own) >= 2 else logged
    if mode == "noise":
        return max(0.5, own[-1] + rng.normal(0.0, 2.0))
    if mode == "partial":
        return 0.5 * own[-1] + 0.5 * logged
    return own[-1]


def set_data(version):
    """Point the test bed at data/<version> (call before load / fit_all; workers inherit it)."""
    global DATA, DATA_GLOB, FITS, RESULTS
    DATA = version
    DATA_GLOB = os.path.join(ROOT, "data", version, "**", "*.csv")
    FITS = os.path.join(ROOT, "analytics", f"demand_fits_{version}.json")
    RESULTS = os.path.join(ROOT, "analytics", f"testbed_results_{version}.csv")
PRIORS = [(np.log(0.34), 4.0), (np.log(35.0), 4.0), (np.log(0.25), 20.0), (0.62, 20.0)]


def rate(theta, price, rival, tau):
    ln_n, ln_m, ln_b, k = theta
    b, m = np.exp(ln_b), np.exp(ln_m)
    ours = np.exp(np.clip(b * (m - price), -50, 50))
    theirs = np.exp(np.clip(b * (m - rival), -50, 50))
    return np.exp(ln_n + k * tau) * ours / (1 + ours + theirs)


def load():
    frames = []
    for path in sorted(glob.glob(DATA_GLOB, recursive=True)):
        f = pd.read_csv(path)
        f["key"] = os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path) + "/" + f.competition_id
        frames.append(f)
    d = pd.concat(frames).sort_values(["key", "selling_season", "selling_period"])
    d["tau"] = (d.selling_period - 1) / 99.0
    d["cap"] = d.competitor_has_capacity.astype(bool)
    d["rival_eff"] = np.where(d.cap & (d.price_competitor < 200), d.price_competitor, 1000.0)
    return d


def _fit_one(args):
    key, price, rival, tau, sold, prev_own = args
    def nll(t):
        lam = rate(t, price, rival, tau) + 1e-12
        v = -(sold * np.log(lam) - lam).sum()
        return v + sum(w * (t[i] - m) ** 2 for i, (m, w) in enumerate(PRIORS))
    best = None
    # Several starting points: from a single start Nelder-Mead sometimes ends in a flat region
    # (m > $1000, i.e. "price does not matter") with a much worse likelihood.
    for m0 in (20, 40, 60, 80):
        for b0 in (0.1, 0.25):
            r = minimize(nll, [np.log(0.34), np.log(m0), np.log(b0), 0.62], method="Nelder-Mead",
                         options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-4})
            if best is None or r.fun < best.fun:
                best = r
    ok = ~np.isnan(prev_own) & (rival < 200)
    copier = bool(np.mean(np.abs(rival[ok] - prev_own[ok]) < 0.6) >= 0.8) if ok.any() else False
    return key, {"theta": [float(v) for v in best.x], "copier": copier}


def fit_all(d, procs):
    if os.path.exists(FITS):
        return json.load(open(FITS))
    jobs = []
    for key, g in d.groupby("key"):
        prev = g.groupby("selling_season").price.shift(1).values
        jobs.append((key, g.price.values, g.rival_eff.values, g.tau.values, g.demand.values, prev))
    with Pool(procs) as pool:
        fits = dict(pool.map(_fit_one, jobs, chunksize=4))
    json.dump(fits, open(FITS, "w"), indent=1)
    return fits


_D = None


def _module(path):
    spec = importlib.util.spec_from_file_location("algo_%d" % zlib.crc32(path.encode()), path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _play(args):
    global _D
    path, key, seed, fit, n_seasons, mode = args
    global COPIER_MODE
    COPIER_MODE = mode
    if _D is None:
        _D = load()
    g = _D[_D.key == key]
    rivals = g.price_competitor.values.reshape(-1, 100); caps = g.cap.values.reshape(-1, 100)
    mod = _module(path); theta, copier = fit["theta"], fit["copier"]
    info, rev, seats = None, [], []
    for s in range(1, (n_seasons or rivals.shape[0]) + 1):
        rng = np.random.default_rng(seed * 100003 + s * 7919 + zlib.crc32(key.encode()) % 9973)
        draws = rng.random(100); own, riv, sold, left = [], [], [], 80
        crng = np.random.default_rng(seed * 7 + s)
        for t in range(1, 101):
            cap = True if copier else bool(caps[s - 1, t - 1])
            r = copier_price(COPIER_MODE, own, float(rivals[s - 1, t - 1]), t, crng) if (copier and t > 1) \
                else float(rivals[s - 1, t - 1])
            hist = np.array([own, riv]) if t > 1 else None
            price, info = mod.p(s, t, hist, np.array(sold) if t > 1 else None, cap, info)
            price = float(price)
            r_eff = r if (cap and r < 200) else 1000.0
            q = int(min(left, poisson.ppf(draws[t - 1], rate(theta, price, r_eff, (t - 1) / 99.0)))) if left > 0 else 0
            left -= q; own.append(price); riv.append(r); sold.append(q)
        mod.p(s, 101, np.array([own, riv]), np.array(sold), caps[s - 1, -1], info)
        rev.append(float(np.dot(own, sold))); seats.append(int(sum(sold)))
    return key, rev, seats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copier-mode", default="perfect", choices=["perfect", "lag2", "noise", "partial"])
    ap.add_argument("--data", default=DATA, help="data folder under data/, e.g. v6.6 or v6.5")
    ap.add_argument("--compare", action="append", default=[])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--sample", type=int, default=0, help="use every k-th match (0 = all)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seasons", type=int, default=0, help="play only the first k seasons of each match (0 = all)")
    ap.add_argument("--holdout", action="store_true", help="with --sample: use the matches NOT selected by it")
    ap.add_argument("--out", default="__default__", help="csv file for the per-match results")
    ap.add_argument("--match", default="", help="only matches whose key contains one of these comma-separated ids")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    set_data(a.data)
    if a.out == "__default__":
        a.out = RESULTS
    d = load(); d["rev"] = d.price * d.demand
    fits = fit_all(d, a.procs)
    keys = sorted(fits)
    if a.sample:
        chosen = keys[a.offset::a.sample]
        keys = [k for k in keys if k not in chosen] if a.holdout else chosen
    if a.match:
        keys = [k for k in keys if any(m in k for m in a.match.split(","))]
    played = d[d.key.isin(keys)]
    if a.seasons:
        played = played[played.selling_season <= a.seasons]
    table = played.groupby("key").agg(actual=("rev", "sum"))
    files = {"duopoly.py": os.path.join(ROOT, "duopoly.py")}
    for p in a.compare:
        files[os.path.basename(p)] = os.path.abspath(p)
    for name, path in files.items():
        tot = pd.Series(0.0, index=table.index); st = pd.Series(0.0, index=table.index)
        for seed in range(1, a.seeds + 1):
            with Pool(a.procs) as pool:
                for key, rev, seats in pool.map(_play, [(path, k, seed, fits[k], a.seasons, a.copier_mode) for k in keys], chunksize=1):
                    tot[key] += sum(rev) / a.seeds; st[key] += np.mean(np.array(seats) >= 80) / a.seeds
        table[name] = tot; table[name + ":sellout"] = st
    table.round(3).to_csv(a.out)
    print(f"{len(keys)} matches, actual ${table.actual.sum():,.0f}")
    base = table["duopoly.py"].sum()
    for name in files:
        print(f"  {name:28s} ${table[name].sum():>12,.0f}  {100 * (table[name].sum() / base - 1):+.2f}%  sell-out {100 * table[name + ':sellout'].mean():.1f}%"
              f"  worst match {100 * (table[name] / table['duopoly.py'] - 1)[table['duopoly.py'] > 1000].min():+.1f}%"
              f"  better/worse {int((table[name] > 1.005 * table['duopoly.py']).sum())}/{int((table[name] < 0.995 * table['duopoly.py']).sum())}")


if __name__ == "__main__":
    main()
