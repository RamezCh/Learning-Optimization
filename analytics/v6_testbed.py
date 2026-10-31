"""
analytics/v6_testbed.py
-----------------------
Test bed built from the data/v6 tournament logs (3 tournaments, 34 matches, 100 seasons each).

1. Demand model per match, fitted to our own daily sales (Poisson maximum likelihood):
       buyers per day      ~ Poisson(n * exp(k * tau)),   tau = (day - 1) / 99
       P(buy from us)      = e^{b(m - p)} / (1 + e^{b(m - p)} + e^{b(m - r)})
   m = willingness-to-pay midpoint ($), b = price sensitivity. Weak priors keep matches where
   we never moved our price identified. The fit reproduces our seats per season in every match
   and our sales rate in every $10 price band (printed with --fit-report).
2. Rivals: the logged rival price of the same season and day is replayed. FlashyInchworm,
   which set its price to our previous price on 99% of days, is modelled as a copier.
3. Each pricing file plays all 34 matches x 100 seasons with the same random draws per
   (seed, match, season), so differences between files are not luck.
4. "Best possible" (oracle): the best price for every day, knowing the demand model and the
   rival's price that day (against the copier: the best price to hold all season).

Validation: replaying duopoly.py v6.3 through this test bed gives $1,589,865 vs $1,587,858
actually earned in the tournaments (+0.1%; every match within +-4% except two matches that
earned $100-$200 in total).

Usage:
    python analytics/v6_testbed.py                                   # current duopoly.py, 3 seeds
    python analytics/v6_testbed.py --compare path/to/old_duopoly.py  # add more pricing files
    python analytics/v6_testbed.py --seeds 1 --fit-report

Writes analytics/v6_demand_fits.json and analytics/v6_testbed_results.csv.
"""

import argparse
import glob
import importlib.util
import json
import os
import warnings
import zlib
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "v6")
FITS_PATH = os.path.join(ROOT_DIR, "analytics", "v6_demand_fits.json")
RESULTS_PATH = os.path.join(ROOT_DIR, "analytics", "v6_testbed_results.csv")
COPIERS = {"FlashyInchworm"}

# Weak priors (mean, penalty weight) on [ln n, ln m, ln b, k]
PRIORS = [(np.log(0.34), 4.0), (np.log(35.0), 4.0), (np.log(0.25), 20.0), (0.62, 20.0)]


def load_v6() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
        frame = pd.read_csv(path)
        frame["tournament"] = os.path.basename(path).split("_rank_")[0]
        frame["rank"] = int(os.path.basename(path).split("_rank_")[1].split(".")[0])
        frames.append(frame)
    data = pd.concat(frames).sort_values(["competition_id", "selling_season", "selling_period"])
    data["tau"] = (data.selling_period - 1) / 99.0
    return data


def demand_rate(theta, price, rival, tau):
    """Expected seats per day for us."""
    ln_n, ln_m, ln_b, k = theta
    b, m = np.exp(ln_b), np.exp(ln_m)
    ours = np.exp(np.clip(b * (m - price), -50, 50))
    theirs = np.exp(np.clip(b * (m - rival), -50, 50))
    return np.exp(ln_n + k * tau) * ours / (1 + ours + theirs)


def _neg_log_lik(theta, price, rival, tau, sold):
    lam = demand_rate(theta, price, rival, tau) + 1e-12
    value = -(sold * np.log(lam) - lam).sum()
    for i, (mean, weight) in enumerate(PRIORS):
        value += weight * (theta[i] - mean) ** 2
    return value


def fit_demand(data: pd.DataFrame, report: bool = False) -> Dict[str, Dict]:
    fits = {}
    for cid, match in data.groupby("competition_id"):
        args = (match.price.values, match.price_competitor.values, match.tau.values, match.demand.values)
        best = None
        for m0 in (12, 30, 55):
            res = minimize(_neg_log_lik, [np.log(0.34), np.log(m0), np.log(0.25), 0.62], args=args,
                           method="Nelder-Mead", options={"maxiter": 8000, "xatol": 1e-6, "fatol": 1e-5})
            if best is None or res.fun < best.fun:
                best = res
        fits[cid] = {"theta": [float(v) for v in best.x], "rival": match.competitor_id.iloc[0],
                     "tournament": match.tournament.iloc[0]}
        if report:
            lam = demand_rate(best.x, *args[:3])
            band = (match.price // 10 * 10).astype(int)
            table = pd.DataFrame({"band": band, "act": match.demand.values, "pred": lam}).groupby("band")
            table = table.agg(n=("act", "size"), act=("act", "mean"), pred=("pred", "mean"))
            table = table[table.n >= 200]
            ln_n, ln_m, ln_b, k = best.x
            print(f"{cid} {fits[cid]['rival'][:14]:14s} n={np.exp(ln_n):.2f} m=${np.exp(ln_m):.0f} "
                  f"b={np.exp(ln_b):.3f} k={k:.2f} | seats/season actual {args[3].sum() / 100:.1f} "
                  f"model {lam.sum() / 100:.1f} | per $10 band actual/model: "
                  + " ".join(f"{i}:{r.act:.2f}/{r.pred:.2f}" for i, r in table.iterrows()))
    with open(FITS_PATH, "w") as fh:
        json.dump(fits, fh, indent=1)
    return fits


def oracle_revenue(data: pd.DataFrame, fits: Dict[str, Dict]) -> Dict[str, float]:
    grid = np.arange(1, 100.01, 0.5)
    tau = np.arange(100) / 99.0
    best = {}
    for cid, match in data.groupby("competition_id"):
        theta = fits[cid]["theta"]
        if fits[cid]["rival"] in COPIERS:
            per_season = max(float(np.sum(p * demand_rate(theta, p, p, tau))) for p in grid)
            best[cid] = per_season * 100
        else:
            rivals = match.price_competitor.values.reshape(-1, 100)
            revenue = grid[None, None, :] * demand_rate(theta, grid[None, None, :], rivals[:, :, None],
                                                        tau[None, :, None])
            best[cid] = float(revenue.max(axis=2).sum())
    return best


_DATA = None


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("algo_%d" % zlib.crc32(path.encode()), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _play_match(args: Tuple[str, str, int, Dict]) -> Tuple[str, List[float], List[int]]:
    global _DATA
    path, cid, seed, fit = args
    if _DATA is None:
        _DATA = load_v6()
    module = _load_module(path)
    rivals = _DATA[_DATA.competition_id == cid].price_competitor.values.reshape(-1, 100)
    theta, copier = fit["theta"], fit["rival"] in COPIERS
    info, revenues, seats = None, [], []
    for season in range(1, rivals.shape[0] + 1):
        rng = np.random.default_rng(seed * 100003 + season * 7919 + zlib.crc32(cid.encode()) % 9973)
        draws = rng.random(100)
        own, riv, sold, left = [], [], [], 80
        for t in range(1, 101):
            rival_price = own[-1] if (copier and t > 1) else rivals[season - 1, t - 1]
            hist = np.array([own, riv]) if t > 1 else None
            dem = np.array(sold) if t > 1 else None
            price, info = module.p(season, t, hist, dem, True, info)
            price = float(price)
            q = int(min(left, poisson.ppf(draws[t - 1], demand_rate(theta, price, rival_price, (t - 1) / 99.0))))
            left -= q
            own.append(price)
            riv.append(rival_price)
            sold.append(q)
        module.p(season, 101, np.array([own, riv]), np.array(sold), True, info)
        revenues.append(float(np.dot(own, sold)))
        seats.append(int(sum(sold)))
    return cid, revenues, seats


def simulate(path: str, fits: Dict[str, Dict], seed: int, procs: int) -> Dict[str, Tuple[List[float], List[int]]]:
    jobs = [(path, cid, seed, fits[cid]) for cid in sorted(fits)]
    with Pool(procs) as pool:
        results = pool.map(_play_match, jobs)
    return {cid: (rev, seats) for cid, rev, seats in results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", action="append", default=[], help="extra pricing file (repeatable)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2)))
    parser.add_argument("--fit-report", action="store_true")
    args = parser.parse_args()

    data = load_v6()
    fits = fit_demand(data, report=args.fit_report)
    data["revenue"] = data.price * data.demand
    table = data.groupby("competition_id").agg(
        tournament=("tournament", "first"), rank=("rank", "first"), rival=("competitor_id", "first"),
        rival_median=("price_competitor", "median"), actual=("revenue", "sum"))
    best = oracle_revenue(data, fits)
    table["best_possible"] = [best[c] for c in table.index]

    files = {"duopoly.py": os.path.join(ROOT_DIR, "duopoly.py")}
    for path in args.compare:
        files[os.path.basename(path)] = os.path.abspath(path)
    for name, path in files.items():
        totals = np.zeros(len(table))
        for seed in range(1, args.seeds + 1):
            result = simulate(path, fits, seed, args.procs)
            totals += np.array([sum(result[c][0]) for c in table.index])
        table[name] = totals / args.seeds

    table.round(1).to_csv(RESULTS_PATH)
    print(table.round(0).to_string())
    print(f"\nTotals (mean over {args.seeds} seeds), actual tournament revenue ${table.actual.sum():,.0f}, "
          f"best possible ${table.best_possible.sum():,.0f}:")
    for name in files:
        share = table[name].sum() / table.best_possible.sum() * 100
        print(f"  {name:30s} ${table[name].sum():>12,.0f}   {share:.1f}% of best possible")


if __name__ == "__main__":
    main()
