"""
analytics/rival_bot.py
----------------------
A simulated rival fitted to one logged match:
    r_t = a + c*r_{t-1} + b*p_{t-1} + d*tau_t + g*(p_{t-1} - p_{t-2}) + e_t
(OLS on days on which both airlines sold; the g term lets it follow the direction of our last move, which many rivals
do without copying our level; e_t drawn from the match's residuals), day 1 drawn from the rival's logged day-1 prices, clipped to
the range of prices it posted. b > 0 means it moves with our price. Used by analytics/selfplay.py as "bot:<file.json>".
"""
import json, os, sys
import numpy as np


def fit(P, R, ok):
    """P, R: (seasons, 100) our / rival prices; ok: rival selling and both real. Returns the bot parameters."""
    rn, rp, pp, pp2 = R[:, 2:], R[:, 1:-1], P[:, 1:-1], P[:, :-2]
    tau = np.broadcast_to((np.arange(2, 100) / 99.0)[None, :], rn.shape)
    m = ok[:, 2:] & ok[:, 1:-1] & ok[:, :-2]
    X = np.c_[np.ones(m.sum()), rp[m], pp[m], tau[m], pp[m] - pp2[m]]
    coef, *_ = np.linalg.lstsq(X, rn[m], rcond=None)
    res = rn[m] - X @ coef
    day1 = R[:, 0][ok[:, 0]]
    lo, hi = float(np.percentile(R[ok], 0.5)), float(np.percentile(R[ok], 99.5))
    return dict(coef=[float(v) for v in coef], res=[float(v) for v in res[:4000]], day1=[float(v) for v in day1],
                lo=lo, hi=hi)


class Bot:
    def __init__(self, params, seed=0):
        self.q = params
        self.rng = np.random.default_rng(seed)

    def p(self, season, t, hist, demand, cap, info):
        q = self.q
        if t > 100:
            return 0.0, info
        if t == 1 or hist is None:
            return float(self.rng.choice(q["day1"])), info
        a, c, b, d, g = q["coef"]
        own_prev, our_prev = float(hist[0][-1]), float(hist[1][-1])
        our_prev2 = float(hist[1][-2]) if len(hist[1]) >= 2 else our_prev
        if our_prev >= 120:                                   # we are sold out: keep its own drift
            our_prev = our_prev2 = own_prev
        if our_prev2 >= 120:
            our_prev2 = our_prev
        r = a + c * own_prev + b * our_prev + d * (t - 1) / 99.0 + g * (our_prev - our_prev2) + float(self.rng.choice(q["res"]))
        return float(np.clip(r, q["lo"], q["hi"])), info


def load(path, seed=0):
    return Bot(json.load(open(path)), seed)
