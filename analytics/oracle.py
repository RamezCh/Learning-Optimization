"""
analytics/oracle.py
-------------------
Market-normalised quality measure: how much of the revenue that was available did we earn?

For every match and season, with the match's fitted demand model (the test bed's choice model):

  O1 "perfect information": knows the demand model AND this season's actual rival price path and sold-out
     days in advance. Dynamic programme over (day, seats left), prices $18-$100. An upper bound nobody can
     reach, because nobody knows the rival's future prices.
  O2 "fair oracle": knows the demand model and the rival's TYPICAL path (median price per day over all
     seasons of the match) but not this season's deviations. Its prices are fixed by the DP on the typical
     path and then evaluated on the actual path. This is what a perfect learner with v6.6's design could get.
  O1_floor1: O1 with prices from $1 (what the $18 floor costs, inside the model).

Copiers (rival copies our previous price) are evaluated with the rival's price = our price.

Efficiency = revenue / O1 revenue. Works for the logs (actual revenue) and for the test bed (same model).
The per-day price comparison (our price minus O2's price at the same day and seats left) shows where we
price too high or too low.

Usage:  python analytics/oracle.py --data v6.6      -> analytics/oracle_v6.6.csv, analytics/oracle_days_v6.6.csv
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import testbed as tb  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
T, S, K = 100, 80, 9
FACT = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880], dtype=float)[:K + 1]
GRID = np.arange(18.0, 100.01, 1.0)
GRID_FLOOR1 = np.arange(1.0, 100.01, 1.0)
_D = None


def pmf_table(lam):
    k = np.arange(K + 1)
    pmf = np.exp(-lam)[:, None] * lam[:, None] ** k[None, :] / FACT[None, :]
    pmf[:, -1] = np.maximum(0.0, 1.0 - pmf[:, :-1].sum(axis=1))
    return pmf


def rates(theta, grid, rival_day, tau, copier):
    return tb.rate(theta, grid, grid if copier else rival_day, tau)


def dp_solve(theta, rival, grid, copier=False):
    """rival: array of 100 effective rival prices (1000 = not selling). Returns V (T+2, S+1), POL."""
    seats = np.arange(S + 1)
    k = np.arange(K + 1)
    sold = np.minimum(k[None, :], seats[:, None])
    idx = seats[:, None] - sold
    V = np.zeros((T + 2, S + 1))
    POL = np.zeros((T + 2, S + 1))
    for t in range(T, 0, -1):
        pmf = pmf_table(rates(theta, grid, rival[t - 1], (t - 1) / 99.0, copier))
        tot = grid[:, None] * (pmf @ sold.T) + pmf @ V[t + 1][idx].T
        V[t] = tot.max(axis=0)
        POL[t] = grid[tot.argmax(axis=0)]
    return V, POL


def dp_evaluate(theta, rival, POL, copier=False):
    """Expected revenue from day 1 with 80 seats when prices follow POL[t][s] and the rival plays `rival`."""
    seats = np.arange(S + 1)
    k = np.arange(K + 1)
    sold = np.minimum(k[None, :], seats[:, None])
    idx = seats[:, None] - sold
    W = np.zeros((T + 2, S + 1))
    for t in range(T, 0, -1):
        price = POL[t]
        r = price if copier else rival[t - 1]
        pmf = pmf_table(tb.rate(theta, price, r, (t - 1) / 99.0))           # one row per seats-left value
        W[t] = (pmf * (price[:, None] * sold + W[t + 1][idx])).sum(axis=1)
    return float(W[1][S])


def _one(args):
    global _D
    key, fit, version = args
    if _D is None:
        tb.set_data(version)
        _D = tb.load()
    g = _D[_D.key == key]
    theta, copier = np.array(fit["theta"]), bool(fit["copier"])
    P = g.price.values.reshape(-1, T)
    D = g.demand.values.reshape(-1, T)
    R = g.rival_eff.values.reshape(-1, T)
    typical = np.nanmedian(np.where(R < 200, R, np.nan), axis=0)
    typical = pd.Series(typical).ffill().bfill().fillna(1000.0).values
    V_typ, POL_typ = dp_solve(theta, typical, GRID, copier)
    rows, days = [], []
    for s in range(P.shape[0]):
        V1, POL1 = dp_solve(theta, R[s], GRID, copier)
        V1f, _ = dp_solve(theta, R[s], GRID_FLOOR1, copier)
        o2 = dp_evaluate(theta, R[s], POL_typ, copier)
        actual = float(np.dot(np.where(P[s] < 120, P[s], 0.0), D[s]))
        rows.append(dict(key=key, season=s + 1, actual=actual, o1=float(V1[1][S]), o2=o2, o1_floor1=float(V1f[1][S]),
                         sold=int(D[s].sum())))
        left = S - np.r_[0, np.cumsum(D[s])[:-1]]
        for t in range(T):
            if left[t] <= 0 or P[s, t] >= 120:
                continue
            days.append((key, s + 1, t + 1, int(left[t]), float(P[s, t]), float(POL1[t + 1][left[t]]),
                         float(POL_typ[t + 1][left[t]]), bool(R[s, t] < 200)))
    return pd.DataFrame(rows), pd.DataFrame(days, columns=["key", "season", "day", "left", "ours", "o1_price", "o2_price",
                                                            "rival_selling"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="v6.6")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    tb.set_data(a.data)
    fits = json.load(open(tb.FITS))
    with Pool(a.procs) as pool:
        out = pool.map(_one, [(k, fits[k], a.data) for k in sorted(fits)], chunksize=1)
    seasons = pd.concat([o[0] for o in out])
    days = pd.concat([o[1] for o in out])
    seasons.to_csv(os.path.join(ROOT, "analytics", f"oracle_{a.data}.csv"), index=False)
    days.to_csv(os.path.join(ROOT, "analytics", f"oracle_days_{a.data}.csv.gz"), index=False, compression="gzip")
    tot = seasons[["actual", "o1", "o2", "o1_floor1"]].sum()
    print(f"{a.data}: actual / O1 = {tot.actual / tot.o1:.3f}, actual / O2 = {tot.actual / tot.o2:.3f}, "
          f"O2 / O1 = {tot.o2 / tot.o1:.3f}, O1(floor $1) / O1 = {tot.o1_floor1 / tot.o1:.3f}")


if __name__ == "__main__":
    main()
