"""
analytics/evaluate.py
---------------------
Every number and figure in EVALUATION.md comes from this script. EVALUATION.md always describes the
CURRENT duopoly.py, evaluated on the LATEST tournament logs. For a new version, edit CFG below, re-run the
test beds (section "Reproduce" of EVALUATION.md), then this script and analytics/write_evaluation_md.py.

Inputs (CFG)
    data/<data>/*.csv                   tournaments played by <played>
    analytics/demand_fits_<data>.json   test-bed demand model per match (analytics/testbed.py)
    <testbed_csv>                       test bed on all matches: revenue of each version (columns = versions)
    analytics/oracle_<data>.csv         oracle revenue per season (analytics/oracle.py), also for <prev_data>
    analytics/oracle_days_<data>.csv.gz our price vs the oracle's, per day

Outputs
    analytics/evaluation_<data>.json    all numbers quoted in EVALUATION.md
    imgs/<prefix>_*.png                 figures
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from plot_style import COLORS, apply_style, plt  # noqa: E402
import testbed as tb  # noqa: E402
from testbed import rate  # noqa: E402

CFG = dict(
    data="v6.6", prev_data="v6.5",            # latest logs, and the logs before them
    played="v6.6", prev_played="v6.5",        # versions that played those logs
    current="v6.7", previous="v6.6",          # the version described, and the one it replaces
    prefix="v67",
    testbed_csv="analytics/v67_testbed_v66data.csv",
    prev_testbed_csv="analytics/v67_testbed_v65data.csv",
    ablation=["v6.7 without WTP bound", "v6.7 without monopoly DP", "v6.7 without copier DP", "v6.7 without rival uncertainty",
              "v6.7 alt: fit revert instead of bound"],
    v64_testbed_csv="analytics/v67_testbed_v64file.csv",
)

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMGS = os.path.join(ROOT, "imgs")
OUT = os.path.join(ROOT, "analytics", f"evaluation_{CFG['data']}.json")
ALL_CSV = os.path.join(ROOT, CFG["testbed_csv"])
NEW, OLD = CFG["current"], CFG["previous"]


def img(name):
    return os.path.join(IMGS, f"{CFG['prefix']}_{name}.png")
FLOOR = 18.0
TYPE_ORDER = ["dumper", "copier", "seasonal", "steady", "erratic"]
TYPE_LABEL = {
    "dumper": "Dumper\n(median price < $18)",
    "copier": "Copier\n(their price = ours yesterday)",
    "seasonal": "Seasonal path\n(same price curve every season)",
    "steady": "Steady\n(moves < $1 a day)",
    "erratic": "Erratic\n(moves >= $1 a day, no pattern)",
}


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def r2(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    return float(1.0 - ((actual - predicted) ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan


def adj_r2(value, n, k):
    return float(1.0 - (1.0 - value) * (n - 1) / max(1, n - k - 1))


def lad_fit(X, y, bounds):
    """Least absolute deviations as a linear programme (lecture step 1 / 3)."""
    n, m = X.shape
    cost = np.r_[np.zeros(m), np.ones(n)]
    A = np.block([[-X, -np.eye(n)], [X, -np.eye(n)]])
    res = linprog(cost, A_ub=A, b_ub=np.r_[-y, y], bounds=list(bounds) + [(0, None)] * n, method="highs")
    return res.x[:m]


def bootstrap_ci(new, old, draws=4000, seed=0):
    """95% interval of total(new)/total(old) - 1, resampling matches."""
    rng = np.random.default_rng(seed)
    new, old = np.asarray(new, float), np.asarray(old, float)
    idx = rng.integers(0, len(new), size=(draws, len(new)))
    ratio = new[idx].sum(axis=1) / old[idx].sum(axis=1) - 1.0
    return [float(np.percentile(ratio, 2.5) * 100), float(np.percentile(ratio, 97.5) * 100)]


def match_arrays(g):
    g = g.sort_values(["selling_season", "selling_period"])
    P = g.price.values.reshape(-1, 100)
    R = g.price_competitor.values.reshape(-1, 100)
    C = g.cap.values.reshape(-1, 100)
    D = g.demand.values.reshape(-1, 100)
    selling = C & (R < 200)
    return P, R, D, selling


# ---------------------------------------------------------------------------------------------
# 1. online scoreboard and rival behaviour
# ---------------------------------------------------------------------------------------------
def classify_rival(P, R, selling):
    """One label per match, from the logged prices only. Rules are checked in this order."""
    Rn = np.where(selling, R, np.nan)
    median_price = float(np.nanmedian(Rn))
    prev_own = P[:, :-1]
    ok = selling[:, 1:] & (prev_own < 120)
    copy_share = float(np.mean(np.abs(R[:, 1:][ok] - prev_own[ok]) <= 0.6)) if ok.any() else 0.0
    step = np.abs(np.diff(Rn, axis=1))
    daily_move = float(np.nanmean(step))
    # share of the within-match price variance explained by the day-of-season median of the
    # other seasons' first half (seasons 1-50 predict seasons 51-100)
    profile = np.nanmedian(Rn[:50], axis=0)
    test = Rn[50:]
    resid = test - profile[None, :]
    total = test - np.nanmean(test)
    profile_r2 = float(1.0 - np.nansum(resid ** 2) / max(1e-9, np.nansum(total ** 2)))
    if copy_share >= 0.8:
        label = "copier"
    elif median_price < FLOOR:
        label = "dumper"
    elif profile_r2 >= 0.5 and np.nanstd(profile) >= 2.0:
        label = "seasonal"
    elif daily_move < 1.0:
        label = "steady"
    else:
        label = "erratic"
    return label, dict(rival_median=median_price, copy_share=copy_share, daily_move=daily_move, profile_r2=profile_r2)


def online_scoreboard(d):
    d = d.assign(rev=d.price * d.demand)
    rows = []
    for key, g in d.groupby("key"):
        P, R, D, selling = match_arrays(g)
        label, info = classify_rival(P, R, selling)
        ours_selling = P < 120
        rows.append(dict(
            key=key, tournament=key.split("/")[1].replace(".csv", ""), rival=g.competitor_id.iloc[0], type=label,
            rev=float((P * D).sum() / P.shape[0]), seats=float(D.sum() / P.shape[0]),
            sellout=float((D.sum(axis=1) >= 80).mean()), rival_sellout=float((~g.cap.values.reshape(-1, 100)).any(axis=1).mean()),
            our_median=float(np.median(P[ours_selling])), at_floor=float(np.mean(P[ours_selling] <= FLOOR + 0.01)), **info))
    m = pd.DataFrame(rows)
    by_tournament = {}
    for t, g in m.groupby("tournament"):
        by_tournament[t] = dict(rank=int(t.split("_")[-1]), matches=int(len(g)), rev_mean=float(g.rev.mean()),
                                rev_median=float(g.rev.median()), seats=float(g.seats.mean()), sellout=float(g.sellout.mean()),
                                rival_sellout=float(g.rival_sellout.mean()), our_median=float(g.our_median.median()),
                                rival_median=float(g.rival_median.median()))
    by_type = {}
    total = m.rev.sum()
    for t in TYPE_ORDER:
        g = m[m.type == t]
        by_type[t] = dict(matches=int(len(g)), share_matches=float(len(g) / len(m)), rev_mean=float(g.rev.mean()),
                          rev_median=float(g.rev.median()), share_revenue=float(g.rev.sum() / total),
                          seats=float(g.seats.mean()), sellout=float(g.sellout.mean()), our_median=float(g.our_median.median()),
                          rival_median=float(g.rival_median.median()), at_floor=float(g.at_floor.mean()),
                          rival_sellout=float(g.rival_sellout.mean()))
    overall = dict(matches=int(len(m)), rev_mean=float(m.rev.mean()), rev_median=float(m.rev.median()),
                   seats=float(m.seats.mean()), sellout=float(m.sellout.mean()), rival_sellout=float(m.rival_sellout.mean()),
                   matches_below_500=int((m.rev < 500).sum()), revenue_share_below_500=float(m[m.rev < 500].rev.sum() / total),
                   matches_at_floor=int((m.at_floor > 0.5).sum()),
                   corr_rev_rival_price=float(np.corrcoef(m.rev, m.rival_median)[0, 1]),
                   corr_our_rival_price=float(np.corrcoef(m.our_median, m.rival_median)[0, 1]))
    return m, dict(overall=overall, by_tournament=by_tournament, by_type=by_type)


def plot_scoreboard(m, res):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
    palette = {"dumper": COLORS["warn"], "copier": COLORS["amber"], "seasonal": COLORS["accent"],
               "steady": COLORS["ours"], "erratic": COLORS["rival"]}
    ax = axes[0]
    for t in TYPE_ORDER:
        g = m[m.type == t]
        ax.scatter(g.rival_median, g.rev, s=46, color=palette[t], edgecolor="white", linewidth=0.8,
                   label=f"{t} ({len(g)})", zorder=3)
    ax.axvline(FLOOR, color=COLORS["dark"], linestyle=":", linewidth=1.2)
    ax.text(FLOOR - 1, m.rev.max() * 0.36, "our price floor: $18", fontsize=9, color=COLORS["dark"], rotation=90, ha="right", va="center")
    ax.set_xlabel("Rival's median price in the match ($)")
    ax.set_ylabel("Our revenue per season ($)")
    ax.set_title(f"A. Each dot = one of the {len(m)} {CFG['played']} matches (r = {res['overall']['corr_rev_rival_price']:.2f})")
    ax.legend(title="Rival behaviour (matches)", loc="upper left")

    ax = axes[1]
    types = TYPE_ORDER
    means = [res["by_type"][t]["rev_mean"] for t in types]
    y = np.arange(len(types))[::-1]
    ax.barh(y, means, color=[palette[t] for t in types], height=0.62, zorder=3)
    for yi, t, v in zip(y, types, means):
        info = res["by_type"][t]
        ax.text(v + 60, yi, f"${v:,.0f}  ({info['matches']} matches, {100 * info['share_revenue']:.0f}% of our revenue)",
                va="center", fontsize=9, color=COLORS["dark"])
    ax.set_yticks(y)
    ax.set_yticklabels([TYPE_LABEL[t] for t in types], fontsize=9)
    ax.set_xlim(0, max(means) * 2.15)
    ax.set_xlabel("Our mean revenue per season ($)")
    ax.set_title("B. Revenue by rival behaviour")
    ax.grid(axis="y", visible=False)
    fig.suptitle(f"{CFG['played']} online: {m.tournament.nunique()} tournaments, {len(m)} matches x 100 seasons "
                 f"(data/{CFG['data']})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(img("online_scoreboard"))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# 2. demand models, out of sample (fit on seasons 1-50, scored on seasons 51-100)
# ---------------------------------------------------------------------------------------------
PRIORS = [(np.log(0.34), 4.0), (np.log(35.0), 4.0), (np.log(0.25), 20.0), (0.62, 20.0)]


def fit_choice(price, rival, tau, sold):
    def nll(t):
        lam = rate(t, price, rival, tau) + 1e-12
        return -(sold * np.log(lam) - lam).sum() + sum(w * (t[i] - m) ** 2 for i, (m, w) in enumerate(PRIORS))
    best = None
    for m0, b0 in ((25, 0.25), (45, 0.2), (70, 0.1)):
        res = minimize(nll, [np.log(0.34), np.log(m0), np.log(b0), 0.62], method="Nelder-Mead",
                       options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-3})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def demand_model_quality(d):
    rows, calib = [], []
    for key, g in d.groupby("key"):
        g = g[(g.price < 120)].copy()                       # days on which we were selling
        g["rival_model"] = g.rival_eff                      # 1000 = rival not selling
        g["duo"] = g.rival_eff < 200
        train, test = g[g.selling_season <= 50], g[g.selling_season > 50]
        tr, te = train[train.duo], test[test.duo]
        if len(tr) < 500 or len(te) < 500 or te.demand.var() == 0 or tr.price.std() < 0.5:
            continue
        # (a) linear regression, least squares, one row per day (the v6.0-v6.5 "MLR")
        Xtr = np.c_[np.ones(len(tr)), tr.price, tr.price_competitor, tr.tau]
        Xte = np.c_[np.ones(len(te)), te.price, te.price_competitor, te.tau]
        ols = np.linalg.lstsq(Xtr, tr.demand.values, rcond=None)[0]
        pred_ols = Xte @ ols
        # (b) lecture step 1: least absolute deviations by linear programming, own slope <= 0 and
        #     rival slope >= 0. One row per day ...
        sub = tr.iloc[:: max(1, len(tr) // 1500)]
        Xs = np.c_[np.ones(len(sub)), sub.price, sub.price_competitor, sub.tau]
        bounds = [(None, None), (None, 0.0), (0.0, None), (None, None)]
        lad_day = lad_fit(Xs, sub.demand.values.astype(float), bounds)
        pred_lad_day = Xte @ lad_day
        #     ... and one row per 5-day block (sums), where sales are no longer mostly zero
        def blocks(frame):
            f = frame.assign(block=(frame.selling_period - 1) // 5, one=1.0)
            return f.groupby(["selling_season", "block"]).agg(n=("one", "sum"), p=("price", "sum"),
                                                              r=("price_competitor", "sum"), t=("tau", "sum"), d=("demand", "sum"))
        btr, bte = blocks(tr), blocks(te)
        lad_block = lad_fit(np.c_[btr.n, btr.p, btr.r, btr.t], btr.d.values.astype(float), bounds)
        pred_lad_block = Xte @ lad_block
        # (c) choice model, Poisson maximum likelihood (v6.4+); also learns from "rival sold out" days
        theta = fit_choice(train.price.values, train.rival_model.values, train.tau.values, train.demand.values)
        pred_choice = rate(theta, te.price.values, te.rival_model.values, te.tau.values)
        actual = te.demand.values.astype(float)
        ceiling = max(0.0, 1.0 - actual.mean() / actual.var())     # best possible daily R2 for Poisson sales
        bkey = [te.selling_season.values, (te.selling_period.values - 1) // 10]
        def block_r2(pred):
            f = pd.DataFrame({"a": actual, "p": pred, "s": bkey[0], "b": bkey[1]}).groupby(["s", "b"]).sum()
            return r2(f.a, f.p)
        row = dict(key=key, n=len(te), mean_sales=float(actual.mean()), ceiling=ceiling,
                   zero_days=float((actual == 0).mean()), ols_slope=float(ols[1]), lad_day_slope=float(lad_day[1]),
                   lad_day_all_zero=bool(np.allclose(lad_day, 0.0, atol=1e-9)))
        for name, pred, k in (("ols", pred_ols, 3), ("lad_day", pred_lad_day, 3), ("lad_block", pred_lad_block, 3),
                              ("choice", pred_choice, 4)):
            row[name + "_r2"] = r2(actual, pred)
            row[name + "_adj_r2"] = adj_r2(row[name + "_r2"], len(actual), k)
            row[name + "_block_r2"] = block_r2(pred)
            row[name + "_bias"] = float(pred.sum() / actual.sum() - 1.0)
        rows.append(row)
        calib.append(pd.DataFrame({"actual": actual, "ols": pred_ols, "lad_block": pred_lad_block, "choice": pred_choice,
                                   "price": te.price.values}))
    q = pd.DataFrame(rows)
    c = pd.concat(calib)
    res = dict(matches=int(len(q)), test_days=int(q.n.sum()), mean_sales_per_day=float(c.actual.mean()),
               zero_days=float((c.actual == 0).mean()), ceiling_median=float(q.ceiling.median()),
               lad_day_all_zero_matches=int(q.lad_day_all_zero.sum()))
    for name in ("ols", "lad_day", "lad_block", "choice"):
        res[name] = dict(r2_median=float(q[name + "_r2"].median()), adj_r2_median=float(q[name + "_adj_r2"].median()),
                         r2_q25=float(q[name + "_r2"].quantile(0.25)), r2_q75=float(q[name + "_r2"].quantile(0.75)),
                         block_r2_median=float(q[name + "_block_r2"].median()),
                         share_of_ceiling=float((q[name + "_r2"] / q.ceiling.replace(0, np.nan)).median()),
                         abs_bias_median=float(q[name + "_bias"].abs().median()),
                         best_in_matches=int((q[[n + "_r2" for n in ("ols", "lad_day", "lad_block", "choice")]].idxmax(axis=1)
                                              == name + "_r2").sum()))
    return q, c, res


def plot_demand_quality(q, c, res):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    names = [("ols", "Linear,\nleast squares\n(v6.0 MLR)", COLORS["rival"]),
             ("lad_day", "Linear, LP,\none row per day\n(lecture as is)", COLORS["warn"]),
             ("lad_block", "Linear, LP,\n5-day blocks\n(lecture, adapted)", COLORS["amber"]),
             ("choice", "Choice model,\nPoisson\n(used by the DP)", COLORS["ours"])]
    ax = axes[0]
    x = np.arange(len(names))
    vals = [res[n]["r2_median"] for n, _, _ in names]
    ax.bar(x, vals, color=[c_ for _, _, c_ in names], width=0.62, zorder=3)
    for xi, (n, _, _) in zip(x, names):
        v = res[n]["r2_median"]
        ax.text(xi, max(v, 0) + 0.004, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold", color=COLORS["dark"])
    ax.axhline(res["ceiling_median"], color=COLORS["dark"], linestyle="--", linewidth=1.3)
    ax.text(0.5, res["ceiling_median"] - 0.003, f"best possible for daily counts: {res['ceiling_median']:.3f}",
            ha="center", va="top", fontsize=9, color=COLORS["dark"])
    ax.axhline(0, color=COLORS["dark"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l, _ in names], fontsize=8.5)
    ax.set_ylabel("R2 on seasons 51-100, median match")
    ax.set_title("A. Daily sales explained, out of sample")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    vals = [res[n]["block_r2_median"] for n, _, _ in names]
    ax.bar(x, vals, color=[c_ for _, _, c_ in names], width=0.62, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, max(v, 0) + 0.01, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold", color=COLORS["dark"])
    ax.axhline(0, color=COLORS["dark"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l, _ in names], fontsize=8.5)
    ax.set_ylabel("R2 of 10-day totals, median match")
    ax.set_title("B. Same forecasts, added up over 10 days")
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    for name, label, color in (("ols", "Linear, least squares", COLORS["rival"]),
                               ("lad_block", "Linear, LP per block", COLORS["amber"]),
                               ("choice", "Choice model", COLORS["ours"])):
        bins = pd.qcut(c[name], 12, duplicates="drop")
        g = c.groupby(bins, observed=True).agg(pred=(name, "mean"), act=("actual", "mean"))
        ax.plot(g.pred, g.act, marker="o", markersize=5, linewidth=2, color=color, label=label, zorder=3)
    top = float(c.actual.groupby(pd.qcut(c.choice, 12, duplicates="drop"), observed=True).mean().max()) * 1.1
    ax.plot([0, top], [0, top], color=COLORS["dark"], linestyle=":", linewidth=1.2, label="perfect")
    ax.set_xlim(-0.05, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Forecast seats per day (12 equal groups of days)")
    ax.set_ylabel("Actual seats per day in the group")
    ax.set_title("C. Is the forecast level right? (all matches pooled)")
    ax.legend(loc="upper left")
    fig.suptitle(f"Demand models fitted on seasons 1-50, scored on seasons 51-100 "
                 f"({res['matches']} matches, {res['test_days']:,} days; {100 * res['zero_days']:.0f}% of days sell 0 seats)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(img("demand_model_quality"))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# 3. rival price forecast (every forecast uses earlier seasons / earlier days only)
# ---------------------------------------------------------------------------------------------
def rival_forecast_quality(d, types):
    rows = []
    for key, g in d.groupby("key"):
        P, R, D, selling = match_arrays(g)
        Rn = np.where(selling, R, np.nan)
        Pn = np.where(P < 120, P, np.nan)
        one = {k: [] for k in ("yesterday", "avg5", "reaction")}
        many = {k: [] for k in ("flat", "path")}
        gate_on = []
        for s in range(3, P.shape[0]):
            past = Rn[max(0, s - 10):s]
            seen = np.isfinite(past)
            has = seen.any(axis=0)
            if has.sum() < 10:
                continue
            days = np.arange(100)
            med = np.array([np.median(past[seen[:, k], k]) if has[k] else np.nan for k in days])
            profile = np.interp(days, days[has], med[has])
            gap = past - profile[None, :]
            x, y = gap[:, :-1].ravel(), gap[:, 1:].ravel()
            ok = np.isfinite(x) & np.isfinite(y)
            phi = float(np.clip((x[ok] * y[ok]).sum() / max(1e-9, (x[ok] ** 2).sum()), 0, 1)) if ok.sum() > 20 else 0.0
            coef = None
            if s % 7 == 3:                                   # the LP is refitted on a sample of seasons (speed)
                rr, r1, p1 = Rn[s - 3:s, 1:].ravel(), Rn[s - 3:s, :-1].ravel(), Pn[s - 3:s, :-1].ravel()
                ok = np.isfinite(rr) & np.isfinite(r1) & np.isfinite(p1)
                if ok.sum() >= 60:
                    X = np.c_[np.ones(ok.sum()), r1[ok], p1[ok]]
                    coef = lad_fit(X, rr[ok], [(None, None), (0, 1.5), (0, 1.5)])
                    gate_on.append(np.abs(rr[ok] - X @ coef).mean() <= 0.7 * np.abs(rr[ok] - r1[ok]).mean())
            for t in range(5, 100):
                if not (np.isfinite(Rn[s, t]) and np.isfinite(Rn[s, t - 1])):
                    continue
                if coef is not None and np.isfinite(Pn[s, t - 1]):
                    one["yesterday"].append(abs(Rn[s, t] - Rn[s, t - 1]))
                    one["avg5"].append(abs(Rn[s, t] - np.nanmean(Rn[s, t - 5:t])))
                    one["reaction"].append(abs(Rn[s, t] - (coef[0] + coef[1] * Rn[s, t - 1] + coef[2] * Pn[s, t - 1])))
            for t in (5, 10, 20, 30, 40, 50, 60, 70, 80, 90):
                future = Rn[s, t:]
                window = Rn[s, max(0, t - 5):t]
                if np.isfinite(future).sum() < 3 or not np.isfinite(Rn[s, t - 1]):
                    continue
                mask = np.isfinite(future)
                h = np.arange(1, 100 - t + 1)
                path = profile[t:] + phi ** h * (Rn[s, t - 1] - profile[t - 1])
                many["flat"].append(np.mean(np.abs(future[mask] - np.nanmean(window))))
                many["path"].append(np.mean(np.abs(future[mask] - path[mask])))
        if not many["flat"]:
            continue
        rows.append(dict(key=key, type=types[key], reaction_gate=float(np.mean(gate_on)) if gate_on else 0.0,
                         **{"one_" + k: float(np.mean(v)) if v else np.nan for k, v in one.items()},
                         **{"many_" + k: float(np.mean(v)) for k, v in many.items()}))
    f = pd.DataFrame(rows)
    res = dict(matches=int(len(f)))
    for col in ("one_yesterday", "one_avg5", "one_reaction", "many_flat", "many_path"):
        res[col] = dict(mean=float(f[col].mean()), median=float(f[col].median()))
    res["path_better"] = int((f.many_path < f.many_flat - 0.25).sum())
    res["path_worse"] = int((f.many_path > f.many_flat + 0.25).sum())
    res["path_reduction"] = float(1 - f.many_path.mean() / f.many_flat.mean())
    res["reaction_reduction"] = float(1 - f.one_reaction.mean() / f.one_yesterday.mean())
    res["reaction_gate_matches"] = int((f.reaction_gate >= 0.5).sum())
    res["by_type"] = {t: dict(flat=float(g.many_flat.mean()), path=float(g.many_path.mean()), matches=int(len(g)))
                      for t, g in f.groupby("type")}
    return f, res


def plot_rival_forecast(f, res, d):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    peak = d[d.cap & (d.price_competitor < 200)].groupby("key").price_competitor.max()
    cand = f[(f.type == "seasonal") & (f.key.map(peak) < 100)]
    key = cand.assign(gain=cand.many_flat - cand.many_path).sort_values("gain").key.iloc[-1]
    g = d[d.key == key]
    P, R, D, selling = match_arrays(g)
    Rn = np.where(selling, R, np.nan)
    days = np.arange(1, 101)
    for s in range(60, 70):
        ax.plot(days, Rn[s], color=COLORS["rival"], linewidth=0.9, alpha=0.55, label="seasons 61-70" if s == 60 else None)
    s, t = 70, 30
    profile = np.nanmedian(Rn[60:70], axis=0)
    ax.plot(days, Rn[s], color=COLORS["dark"], linewidth=2.2, label="season 71 (actual)")
    ax.plot(days[t:], np.full(100 - t, np.nanmean(Rn[s, t - 5:t])), color=COLORS["warn"], linewidth=2.2, linestyle="--",
            label="flat 5-day average (v6.5's DP)")
    ax.plot(days[t:], profile[t:], color=COLORS["ours"], linewidth=2.2, linestyle="--",
            label="path of earlier seasons (v6.6+)")
    ax.axvline(t, color=COLORS["dark"], linestyle=":", linewidth=1)
    ax.set_xlabel("Day of the season")
    ax.set_ylabel("Rival price ($)")
    ax.set_title(f"A. One rival ({g.competitor_id.iloc[0]}, match {key[-6:]})")
    ax.set_ylim(0, float(np.nanmax(Rn[60:71])) * 1.45)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    top = float(max(f.many_flat.max(), f.many_path.max())) * 1.05
    ax.scatter(f.many_flat, f.many_path, s=42, color=COLORS["accent"], edgecolor="white", linewidth=0.8, zorder=3)
    ax.plot([0, top], [0, top], color=COLORS["dark"], linestyle=":", linewidth=1.2)
    ax.text(top * 0.55, top * 0.12, f"below the line: path forecast is better\n{res['path_better']} matches better, "
            f"{res['path_worse']} worse (by > $0.25)", fontsize=9, color=COLORS["dark"])
    ax.set_xlabel("Error of the flat forecast ($ per day, rest of the season)")
    ax.set_ylabel("Error of the path forecast ($)")
    ax.set_title(f"B. All {res['matches']} matches: mean error ${res['many_flat']['mean']:.2f} -> ${res['many_path']['mean']:.2f}")

    ax = axes[2]
    labels = ["5-day\naverage", "Yesterday's\nprice", "Reaction model\n(LP, lecture step 3)"]
    vals = [res["one_avg5"]["mean"], res["one_yesterday"]["mean"], res["one_reaction"]["mean"]]
    ax.bar(np.arange(3), vals, color=[COLORS["rival"], COLORS["rival"], COLORS["ours"]], width=0.6, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"${v:.2f}", ha="center", fontweight="bold", fontsize=10, color=COLORS["dark"])
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean absolute error, next day ($)")
    ax.set_title("C. Forecast of the rival's price tomorrow")
    ax.grid(axis="x", visible=False)
    fig.suptitle(f"Rival price forecasts on data/{CFG['data']}, each made only from earlier days and seasons", fontweight="bold")
    fig.tight_layout()
    fig.savefig(img("rival_forecast"))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# 4. correlation vs causation
# ---------------------------------------------------------------------------------------------
def correlation_vs_causation(d, types):
    rows = []
    pooled = []
    for key, g in d.groupby("key"):
        duo = g[(g.price < 120) & (g.rival_eff < 200)]
        if len(duo) < 1000 or duo.price.std() < 0.5 or duo.demand.var() == 0:
            continue
        raw = float(np.corrcoef(duo.price, duo.demand)[0, 1])
        X = np.c_[np.ones(len(duo)), duo.price, duo.price_competitor, duo.tau]
        beta = np.linalg.lstsq(X, duo.demand.values, rcond=None)[0]
        resid_p = duo.price - np.c_[np.ones(len(duo)), duo.price_competitor, duo.tau] @ np.linalg.lstsq(
            np.c_[np.ones(len(duo)), duo.price_competitor, duo.tau], duo.price.values, rcond=None)[0]
        resid_d = duo.demand - np.c_[np.ones(len(duo)), duo.price_competitor, duo.tau] @ np.linalg.lstsq(
            np.c_[np.ones(len(duo)), duo.price_competitor, duo.tau], duo.demand.values, rcond=None)[0]
        partial = float(np.corrcoef(resid_p, resid_d)[0, 1])

        def partial_given(z):
            Z = np.c_[np.ones(len(duo)), z]
            rp = duo.price.values - Z @ np.linalg.lstsq(Z, duo.price.values, rcond=None)[0]
            rd = duo.demand.values - Z @ np.linalg.lstsq(Z, duo.demand.values.astype(float), rcond=None)[0]
            return float(np.corrcoef(rp, rd)[0, 1])
        given_day, given_rival = partial_given(duo.tau.values), partial_given(duo.price_competitor.values)
        price_rival = float(np.corrcoef(duo.price, duo.price_competitor)[0, 1])
        sales_rival = float(np.corrcoef(duo.demand, duo.price_competitor)[0, 1])
        P, R, D, selling = match_arrays(g)
        ok = selling[:, 1:] & selling[:, :-1] & (P[:, :-1] < 120) & (P[:, 1:] < 120)
        they_follow = float(np.corrcoef(R[:, 1:][ok], P[:, :-1][ok])[0, 1]) if ok.sum() > 100 and R[:, 1:][ok].std() > 0 else np.nan
        we_follow = float(np.corrcoef(P[:, 1:][ok], R[:, :-1][ok])[0, 1]) if ok.sum() > 100 and R[:, :-1][ok].std() > 0 else np.nan
        rows.append(dict(key=key, type=types[key], raw=raw, partial=partial, given_day=given_day, given_rival=given_rival,
                         price_rival=price_rival, sales_rival=sales_rival, through_rival=price_rival * sales_rival, slope_raw=float(np.polyfit(duo.price, duo.demand, 1)[0]),
                         slope_controlled=float(beta[1]), corr_price_tau=float(np.corrcoef(duo.price, duo.tau)[0, 1]),
                         corr_demand_tau=float(np.corrcoef(duo.demand, duo.tau)[0, 1]), they_follow=they_follow, we_follow=we_follow))
        pooled.append(duo[["price", "demand", "tau"]].assign(key=key, price_rel=duo.price - duo.price.mean(),
                                                             late=duo.tau >= 0.5))
    c = pd.DataFrame(rows)
    pooled = pd.concat(pooled)
    res = dict(matches=int(len(c)), raw_median=float(c.raw.median()), raw_positive=int((c.raw > 0).sum()),
               partial_median=float(c.partial.median()), partial_negative=int((c.partial < 0).sum()),
               slope_controlled_negative=int((c.slope_controlled < 0).sum()),
               corr_price_tau_median=float(c.corr_price_tau.median()), corr_demand_tau_median=float(c.corr_demand_tau.median()),
               they_follow_median=float(c.they_follow.median()), we_follow_median=float(c.we_follow.median()),
               both_follow_above_half=int(((c.they_follow > 0.5) & (c.we_follow > 0.5)).sum()),
               copiers=int((c.type == "copier").sum()),
               positive_given_day_only=int((c.given_day > 0).sum()), positive_given_rival_only=int((c.given_rival > 0).sum()),
               positive_given_both=int((c.partial > 0).sum()),
               price_rival_median=float(c.price_rival.median()), sales_rival_median=float(c.sales_rival.median()),
               sales_rival_positive=int((c.sales_rival > 0).sum()),
               day_hypothesis_r=float(np.corrcoef(c.corr_price_tau, c.raw)[0, 1]),
               rival_hypothesis_r=float(np.corrcoef(c.through_rival, c.raw)[0, 1]))
    return c, pooled, res


def plot_causation(c, pooled, res):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    order = c.sort_values("raw").reset_index(drop=True)
    x = np.arange(len(order))
    ax.vlines(x, order.raw, order.partial, color=COLORS["grid"], linewidth=1.2, zorder=2)
    ax.scatter(x, order.raw, s=22, color=COLORS["warn"], label=f"raw correlation (> 0 in {res['raw_positive']} matches)", zorder=3)
    ax.scatter(x, order.partial, s=22, color=COLORS["ours"],
               label=f"rival price and day held fixed (< 0 in {res['partial_negative']})", zorder=3)
    ax.axhline(0, color=COLORS["dark"], linewidth=1)
    ax.set_xlabel(f"{len(order)} matches, sorted by raw correlation")
    ax.set_ylabel("Correlation of our price with our daily sales")
    ax.set_title("A. Does a higher price sell more seats? No")
    ax.legend(loc="upper left", fontsize=8.5)

    ax = axes[1]
    ax.scatter(c.through_rival, c.raw, s=42, color=COLORS["accent"], edgecolor="white", linewidth=0.8, zorder=3)
    slope, intercept = np.polyfit(c.through_rival, c.raw, 1)
    xs = np.array([c.through_rival.min(), c.through_rival.max()])
    ax.plot(xs, intercept + slope * xs, color=COLORS["dark"], linewidth=1.5, linestyle="--")
    ax.axhline(0, color=COLORS["dark"], linewidth=1)
    ax.axvline(0, color=COLORS["dark"], linewidth=1)
    ax.set_xlabel("corr(our price, rival price) x corr(our sales, rival price)")
    ax.set_ylabel("Raw correlation of our price with our sales")
    ax.set_title(f"B. The rival's price is the third variable (r = {res['rival_hypothesis_r']:.2f})")

    ax = axes[2]
    palette = {"dumper": COLORS["warn"], "copier": COLORS["amber"], "seasonal": COLORS["accent"],
               "steady": COLORS["ours"], "erratic": COLORS["rival"]}
    for t in TYPE_ORDER:
        g = c[c.type == t]
        ax.scatter(g.we_follow, g.they_follow, s=42, color=palette[t], edgecolor="white", linewidth=0.8, label=t, zorder=3)
    ax.plot([-1, 1], [-1, 1], color=COLORS["dark"], linestyle=":", linewidth=1.2)
    ax.set_xlim(-0.6, 1.05)
    ax.set_ylim(-0.6, 1.05)
    ax.set_xlabel("corr(our price today, their price yesterday)")
    ax.set_ylabel("corr(their price today, our price yesterday)")
    ax.set_title("C. Who follows whom? Correlation cannot tell")
    ax.legend(loc="lower right", fontsize=8.5, title="Rival behaviour")
    fig.suptitle(f"Correlation vs causation in the {CFG['played']} logs", fontweight="bold")
    fig.tight_layout()
    fig.savefig(img("correlation_vs_causation"))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# 5. + 6. test bed: validity and results
# ---------------------------------------------------------------------------------------------
def testbed_results(types):
    if not os.path.exists(ALL_CSV):
        return None, None
    t = pd.read_csv(ALL_CSV)
    t["type"] = t.key.map(types)
    t["seasons"] = 100.0
    valid = t[t.actual > 0]
    err = (valid[OLD] / valid.actual - 1.0)
    big = valid[valid.actual >= 50000]
    validity = dict(matches=int(len(t)), actual_total=float(t.actual.sum()), replay_total=float(t[OLD].sum()),
                    total_error=float(t[OLD].sum() / t.actual.sum() - 1.0),
                    r2_levels=r2(t.actual, t[OLD]), pearson=float(np.corrcoef(t.actual, t[OLD])[0, 1]),
                    median_abs_error=float(err.abs().median()),
                    median_abs_error_big=float((big[OLD] / big.actual - 1.0).abs().median()),
                    within_10pct=int(((big[OLD] / big.actual - 1.0).abs() <= 0.10).sum()), big_matches=int(len(big)),
                    worst=float((big[OLD] / big.actual - 1.0).abs().max()))
    change = t[NEW] / t[OLD] - 1.0
    counted = t[OLD] >= 1000
    result = dict(matches=int(len(t)), old_total=float(t[OLD].sum()), new_total=float(t[NEW].sum()),
                  change=float(t[NEW].sum() / t[OLD].sum() - 1.0) * 100, ci95=bootstrap_ci(t[NEW], t[OLD]),
                  better=int((change[counted] > 0.005).sum()), worse=int((change[counted] < -0.005).sum()),
                  unchanged=int(len(t) - (change[counted] > 0.005).sum() - (change[counted] < -0.005).sum()),
                  median_match=float(change[counted].median() * 100), worst_match=float(change[counted].min() * 100),
                  best_match=float(change[counted].max() * 100),
                  old_per_season=float(t[OLD].sum() / len(t) / 100), new_per_season=float(t[NEW].sum() / len(t) / 100),
                  sellout_old=float(t[OLD + ":sellout"].mean()), sellout_new=float(t[NEW + ":sellout"].mean()))
    gain = (t[NEW] - t[OLD]).sort_values()
    rest = t.loc[gain.index[:-5]]
    trusted = t[(t.actual >= 50000) & ((t[OLD] / t.actual - 1.0).abs() <= 0.10)]
    trusted_change = trusted[NEW] / trusted[OLD] - 1.0
    result["robust"] = dict(
        top5_share_of_gain=float(gain.iloc[-5:].sum() / gain.sum()), top1_share_of_gain=float(gain.iloc[-1] / gain.sum()),
        top1_match=str(gain.index[-1] and t.loc[gain.index[-1], "key"][-6:]),
        top1_testbed_error=float(t.loc[gain.index[-1], OLD] / t.loc[gain.index[-1], "actual"] - 1.0),
        change_without_top5=float(rest[NEW].sum() / rest[OLD].sum() - 1.0) * 100,
        trusted_matches=int(len(trusted)), trusted_change=float(trusted[NEW].sum() / trusted[OLD].sum() - 1.0) * 100,
        trusted_ci95=bootstrap_ci(trusted[NEW], trusted[OLD]),
        trusted_better=int((trusted_change > 0.005).sum()), trusted_worse=int((trusted_change < -0.005).sum()),
        trusted_median=float(trusted_change.median() * 100))
    result["by_type"] = {}
    for name in TYPE_ORDER:
        g = t[t.type == name]
        if len(g):
            result["by_type"][name] = dict(matches=int(len(g)), change=float(g[NEW].sum() / g[OLD].sum() - 1.0) * 100,
                                           dollars_per_season=float((g[NEW].sum() - g[OLD].sum()) / len(g) / 100))
    return t, dict(validity=validity, result=result)


def compare(t, new, old, types=None):
    """Revenue of version `new` vs `old` on the same matches (paired)."""
    change = t[new] / t[old] - 1.0
    counted = t[old] >= 1000
    out = dict(matches=int(len(t)), change=float(t[new].sum() / t[old].sum() - 1.0) * 100, ci95=bootstrap_ci(t[new], t[old]),
               better=int((change[counted] > 0.005).sum()), worse=int((change[counted] < -0.005).sum()),
               median_match=float(change[counted].median() * 100), worst_match=float(change[counted].min() * 100),
               best_match=float(change[counted].max() * 100),
               dollars_per_season=float((t[new].sum() - t[old].sum()) / len(t) / 100))
    if types is not None:
        out["by_type"] = {}
        for name in TYPE_ORDER:
            g = t[t.key.map(types) == name]
            if len(g):
                out["by_type"][name] = dict(matches=int(len(g)), change=float(g[new].sum() / g[old].sum() - 1.0) * 100,
                                            dollars_per_season=float((g[new].sum() - g[old].sum()) / len(g) / 100))
    return out


def oracle_section(m_now, m_prev, t):
    """Online efficiency = revenue / perfect-information oracle, for both tournament sets, plus the test-bed
    versions in the latest markets (same oracle) to split the online change into market mix and code."""
    res = {}
    frames = []
    for data, m in ((CFG["prev_data"], m_prev), (CFG["data"], m_now)):
        o = pd.read_csv(os.path.join(ROOT, "analytics", f"oracle_{data}.csv"))
        o["data"] = data
        o["type"] = o.key.map(dict(zip(m.key, m.type)))
        o["rival"] = o.key.map(dict(zip(m.key, m.rival)))
        frames.append(o)
    o = pd.concat(frames)

    def eff(g):
        s = g[["actual", "o1", "o2", "o1_floor1"]].sum()
        return dict(matches=int(g.key.nunique()), vs_o1=float(s.actual / s.o1), vs_o2=float(s.actual / s.o2),
                    o2_vs_o1=float(s.o2 / s.o1), floor_value=float(s.o1_floor1 / s.o1 - 1.0),
                    regret_per_season=float((s.o1 - s.actual) / len(g)), o1_per_season=float(s.o1 / len(g)))
    for data, g in o.groupby("data"):
        res[data] = eff(g)
        res[data]["by_type"] = {t_: eff(gg) for t_, gg in g.groupby("type")}
        res[data]["season1"] = eff(g[g.season == 1])
        res[data]["later"] = eff(g[g.season > 1])
    common = set(o[o.data == CFG["prev_data"]].rival) & set(o[o.data == CFG["data"]].rival)
    c = o[o.rival.isin(common)]
    per_rival = c.groupby(["rival", "data"]).apply(lambda g: g.actual.sum() / g.o1.sum()).unstack()
    res["matched_rivals"] = dict(rivals=len(common), **{d_: eff(g) for d_, g in c.groupby("data")},
                                 rivals_better=int((per_rival[CFG["data"]] > per_rival[CFG["prev_data"]]).sum()))
    now = o[o.data == CFG["data"]]
    o1 = now.o1.sum()
    res["testbed_in_latest_markets"] = {v: float(t[v].sum() / o1) for v in (CFG["prev_played"], OLD, NEW) if v in t}
    regret = now.assign(gap=now.o1 - now.actual).groupby("type").gap.sum()
    res["regret_share_by_type"] = {k: float(v / regret.sum()) for k, v in regret.items()}
    # price bias vs the fair oracle (O2), duopoly vs monopoly days, non-copier matches
    days = pd.read_csv(os.path.join(ROOT, "analytics", f"oracle_days_{CFG['data']}.csv.gz"))
    days["type"] = days.key.map(dict(zip(m_now.key, m_now.type)))
    days = days[~days.type.isin(["copier", "dumper"])]
    days["bias"] = days.ours - days.o2_price
    duo, mono = days[days.rival_selling], days[~days.rival_selling]
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    by_day = duo.groupby(pd.cut(duo.day, bins), observed=True).bias.agg(["median", "mean"])
    res["price_bias"] = dict(duopoly_median=float(duo.bias.median()), duopoly_mean=float(duo.bias.mean()),
                             monopoly_median=float(mono.bias.median()), monopoly_mean=float(mono.bias.mean()),
                             monopoly_days=int(len(mono)), duopoly_days=int(len(duo)),
                             season1_mean=float(days[days.season == 1].bias.mean()),
                             by_day=[dict(day=str(k), median=float(v["median"]), mean=float(v["mean"])) for k, v in by_day.iterrows()])
    return o, res


def plot_oracle(o, res):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw={"width_ratios": [1.1, 1.2, 1]})
    ax = axes[0]
    tb_ = res["testbed_in_latest_markets"]
    labels = [f"{CFG['prev_played']} online\n(its own markets)", f"{CFG['played']} online\n(its own markets)"] + \
             [f"{v}\nin {CFG['data']} markets\n(test bed)" for v in tb_]
    vals = [res[CFG["prev_data"]]["vs_o1"], res[CFG["data"]]["vs_o1"]] + list(tb_.values())
    colors = [COLORS["rival"], COLORS["ours"]] + [COLORS["grid"], COLORS["accent"], COLORS["amber"]][:len(tb_)]
    ax.bar(np.arange(len(vals)), np.array(vals) * 100, color=colors, width=0.65, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v * 100 + 0.3, f"{100 * v:.1f}%", ha="center", fontsize=10, fontweight="bold", color=COLORS["dark"])
    ax.set_ylim(80, 100)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Revenue as % of the perfect-information oracle")
    ax.set_title("A. Share of the available revenue earned")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    types = [t_ for t_ in TYPE_ORDER if t_ in res[CFG["data"]]["by_type"] and t_ in res[CFG["prev_data"]]["by_type"]]
    x = np.arange(len(types))
    for j, (data, played, color) in enumerate(((CFG["prev_data"], CFG["prev_played"], COLORS["rival"]),
                                               (CFG["data"], CFG["played"], COLORS["ours"]))):
        vals = [100 * res[data]["by_type"][t_]["vs_o1"] for t_ in types]
        ax.bar(x + (j - 0.5) * 0.38, vals, width=0.36, color=color, label=f"{played} online", zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi + (j - 0.5) * 0.38, v + 0.4, f"{v:.0f}", ha="center", fontsize=8.5, color=COLORS["dark"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t_}\n({res[CFG['data']]['by_type'][t_]['matches']} / {res[CFG['prev_data']]['by_type'][t_]['matches']})"
                        for t_ in types], fontsize=9)
    ax.set_ylim(60, 103)
    ax.set_ylabel("% of oracle revenue")
    ax.set_title(f"B. By rival behaviour (matches {CFG['data']} / {CFG['prev_data']})")
    ax.legend(loc="lower right")
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    pb = res["price_bias"]
    mids = [int(r["day"].strip("(]").split(",")[1]) - 5 for r in pb["by_day"]]
    ax.plot(mids, [r["mean"] for r in pb["by_day"]], marker="o", color=COLORS["accent"], linewidth=2, label="rival selling: mean", zorder=3)
    ax.plot(mids, [r["median"] for r in pb["by_day"]], marker="s", color=COLORS["ours"], linewidth=2, label="rival selling: median", zorder=3)
    ax.axhline(pb["monopoly_median"], color=COLORS["warn"], linewidth=2, linestyle="--",
               label=f"rival sold out (hand rule): median +${pb['monopoly_median']:.1f}")
    ax.axhline(0, color=COLORS["dark"], linewidth=1)
    ax.set_xlabel("Day of the season")
    ax.set_ylabel("Our price minus the fair oracle's price ($)")
    ax.set_title(f"C. Where {CFG['played']} priced off ({CFG['data']}, non-copier, non-dumper)")
    ax.set_ylim(-1.5, pb["monopoly_median"] + 3)
    ax.legend(loc="center right", fontsize=8.5)
    fig.suptitle(f"Online quality, normalised by what each market made possible (oracle with true demand and the rival's actual prices)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(img("oracle_efficiency"))
    plt.close(fig)


def plot_testbed(t, res, tprev):
    r = res["result"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    shown = t[t[OLD] >= 1000].assign(change=lambda f: (f[NEW] / f[OLD] - 1.0) * 100).sort_values("change").reset_index(drop=True)
    colors = np.where(shown.change > 0.5, COLORS["ours"], np.where(shown.change < -0.5, COLORS["warn"], COLORS["rival"]))
    cap = 30.0
    ax.bar(np.arange(len(shown)), shown.change.clip(upper=cap), color=colors, width=0.8, zorder=3)
    cut = [(i, v) for i, v in enumerate(shown.change) if v > cap]
    for k, (i, v) in enumerate(cut):
        ax.annotate(f"+{v:.0f}%", xy=(i, cap), xytext=(i - 3.5 * (len(cut) - k), cap + 3), fontsize=9, color=COLORS["dark"],
                    ha="right", va="center", arrowprops=dict(arrowstyle="-", color=COLORS["dark"], linewidth=0.8))
    ax.set_ylim(min(-3.0, shown.change.min() - 1), cap + 6)
    ax.axhline(0, color=COLORS["dark"], linewidth=1)
    ax.set_xlabel(f"{len(shown)} matches with >= $10 revenue per season ({CFG['data']} test bed), sorted")
    ax.set_ylabel(f"Revenue change, {NEW} vs {OLD} (%)")
    ax.set_title(f"A. Per match: {r['better']} better, {r['worse']} worse (beyond 0.5%), median {r['median_match']:+.1f}%")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    rows = [(f"{NEW} on {CFG['data']} matches", res["result"]["change"], COLORS["ours"]),
            (f"{NEW} on {CFG['prev_data']} matches", res["prev_result"]["change"], COLORS["ours"]),
            (f"{NEW} on data/v6.4 file\n(18 matches)", res["v64_file"][NEW]["change"], COLORS["ours"])]
    for name in CFG["ablation"]:
        label = name.replace(f"{NEW} alt: ", "alternative: ").replace(f"{NEW} ", "")
        rows.append((label + f"\n({CFG['data']})", res["ablation"][name]["change"], COLORS["accent"]))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [v for _, v, _ in rows], color=[c_ for _, _, c_ in rows], height=0.6, zorder=3)
    for yi, (lab, v, _) in zip(y, rows):
        ax.text(max(v, 0) + 0.05, yi, f"{v:+.2f}%", va="center", fontsize=9.5, color=COLORS["dark"])
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _, _ in rows], fontsize=9)
    ax.axvline(0, color=COLORS["dark"], linewidth=1)
    ax.set_xlim(min(0, min(v for _, v, _ in rows)) - 0.3, max(v for _, v, _ in rows) * 1.6)
    ax.set_xlabel(f"Revenue change vs {OLD} (%)")
    ax.set_title("B. Both data sets, and with each change removed")
    ax.grid(axis="y", visible=False)
    fig.suptitle(f"Test bed: {NEW} earns {r['change']:+.2f}% vs {OLD} on data/{CFG['data']} "
                 f"(95% interval {r['ci95'][0]:+.2f}% to {r['ci95'][1]:+.2f}%)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(img("testbed_results"))
    plt.close(fig)

    v = res["validity"]
    fig, ax = plt.subplots(figsize=(6.6, 6))
    top = float(max(t.actual.max(), t[OLD].max())) / 100 * 1.05
    ax.scatter(t.actual / 100, t[OLD] / 100, s=42, color=COLORS["accent"], edgecolor="white", linewidth=0.8, zorder=3)
    ax.plot([0, top], [0, top], color=COLORS["dark"], linestyle=":", linewidth=1.2)
    ax.set_xlabel(f"Actual revenue per season in the tournament ($)")
    ax.set_ylabel(f"Revenue per season when {OLD} replays the match on the test bed ($)")
    ax.set_title(f"Test bed vs reality, {v['matches']} matches (data/{CFG['data']})\nR2 = {v['r2_levels']:.3f}, total "
                 f"{100 * v['total_error']:+.2f}%, median match error {100 * v['median_abs_error_big']:.1f}%")
    fig.tight_layout()
    fig.savefig(img("testbed_validity"))
    plt.close(fig)


def prediction_vs_reality(out):
    """What EVALUATION.md (v6.6, built on data/v6.5) predicted, against what data/v6.6 showed."""
    old = json.load(open(os.path.join(ROOT, "analytics", "v66_evaluation.json")))
    sim = json.load(open(os.path.join(ROOT, "analytics", "v66_simulator_check.json")))
    R, RB = old["testbed"]["result"], old["testbed"]["result"]["robust"]
    pred = out["prediction_check"]
    rows = [
        ("Test bed reproduces real revenue (total error)", 100 * old["testbed"]["validity"]["total_error"],
         100 * out["testbed"]["validity"]["total_error"], "%"),
        ("v6.6 vs v6.5, all matches (test bed)", R["change"], pred["change"], "%"),
        ("v6.6 vs v6.5, best-reproduced matches (observed: all matches)", RB["trusted_change"], pred["change"], "%"),
        ("v6.6 vs v6.5, median match", R["median_match"], pred["median_match"], "%"),
        ("v6.6 vs v6.5 against copiers", R["by_type"]["copier"]["change"], pred["by_type"]["copier"]["change"], "%"),
        ("v6.6 vs v6.5 against steady rivals", R["by_type"]["steady"]["change"], pred["by_type"]["steady"]["change"], "%"),
        ("v6.6 vs v6.5 against erratic rivals", R["by_type"]["erratic"]["change"], pred["by_type"]["erratic"]["change"], "%"),
        ("v6.6 vs v6.5 against seasonal rivals", R["by_type"]["seasonal"]["change"], pred["by_type"]["seasonal"]["change"], "%"),
        ("v6.6 vs v6.5 against dumpers", R["by_type"]["dumper"]["change"], pred["by_type"]["dumper"]["change"], "%"),
        ("Rival path forecast: error reduction vs flat", 100 * old["rival_forecast"]["path_reduction"],
         100 * out["rival_forecast"]["path_reduction"], "%"),
        ("Reaction model: next-day error reduction", 100 * old["rival_forecast"]["reaction_reduction"],
         100 * out["rival_forecast"]["reaction_reduction"], "%"),
        ("Choice model: share of the achievable daily R2", 100 * old["demand_models"]["choice"]["share_of_ceiling"],
         100 * out["demand_models"]["choice"]["share_of_ceiling"], "%"),
        ("Server call time, mean (ms)", 18.0 * sim["latency_new_ms"] / sim["latency_old_ms"], out["latency"]["mean_ms"], "ms"),
    ]
    return [dict(claim=c, predicted=float(p), observed=float(o), unit=u) for c, p, o, u in rows]


def plot_prediction(rows):
    pct = [r for r in rows if r["unit"] == "%" and "share" not in r["claim"] and "reduction" not in r["claim"]]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    y = np.arange(len(pct))[::-1]
    for yi, r in zip(y, pct):
        ax.plot([r["predicted"], r["observed"]], [yi, yi], color=COLORS["grid"], linewidth=2.5, zorder=2)
    ax.scatter([r["predicted"] for r in pct], y, s=70, color=COLORS["rival"], label="predicted (EVALUATION.md for v6.6, built on data/v6.5)", zorder=3)
    ax.scatter([r["observed"] for r in pct], y, s=70, color=COLORS["ours"], label="observed on data/v6.6", zorder=3)
    for yi, r in zip(y, pct):
        ax.text(max(r["predicted"], r["observed"]) + 1.5, yi, f"{r['predicted']:+.1f} -> {r['observed']:+.1f}", va="center",
                fontsize=8.5, color=COLORS["dark"])
    ax.set_yticks(y)
    ax.set_yticklabels([r["claim"] for r in pct], fontsize=9)
    ax.axvline(0, color=COLORS["dark"], linewidth=1)
    lo = min(min(r["predicted"], r["observed"]) for r in pct)
    hi = max(max(r["predicted"], r["observed"]) for r in pct)
    ax.set_xlim(lo - 2, hi + 6)
    ax.set_xlabel("Revenue change (%)")
    ax.set_title("Did the last evaluation predict the new tournaments?\n(revenue change, re-measured on the new matches; "
                 "model-fit checks are in the table)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9, frameon=False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(img("prediction_vs_reality"))
    plt.close(fig)


def main():
    apply_style()
    os.makedirs(IMGS, exist_ok=True)
    out = {"config": CFG}
    tb.set_data(CFG["prev_data"])
    m_prev, out["prev_online"] = online_scoreboard(tb.load())
    tb.set_data(CFG["data"])
    d = tb.load()
    m, out["online"] = online_scoreboard(d)
    types = dict(zip(m.key, m.type))
    plot_scoreboard(m, out["online"])
    out["latency"] = dict(mean_ms=float(1000 * d.calculation_duration.mean()), max_s=float(d.calculation_duration.max()),
                          errors=int(d.errors.notna().sum()))
    print("online done")
    q, calib, out["demand_models"] = demand_model_quality(d)
    plot_demand_quality(q, calib, out["demand_models"])
    print("demand models done")
    f, out["rival_forecast"] = rival_forecast_quality(d, types)
    plot_rival_forecast(f, out["rival_forecast"], d)
    print("rival forecast done")
    c, pooled, out["causation"] = correlation_vs_causation(d, types)
    plot_causation(c, pooled, out["causation"])
    t, out["testbed"] = testbed_results(types)
    out["testbed"]["prev_result"] = compare(pd.read_csv(os.path.join(ROOT, CFG["prev_testbed_csv"])), NEW, OLD)
    out["testbed"]["ablation"] = {name: compare(t, name, OLD) for name in CFG["ablation"]}
    out["testbed"]["copier_matches"] = dict(
        latest=compare(t[t.key.map(types) == "copier"], NEW, OLD),
        prev=compare(pd.read_csv(os.path.join(ROOT, CFG["prev_testbed_csv"])).pipe(
            lambda f: f[f.key.map(dict(zip(m_prev.key, m_prev.type))) == "copier"]), NEW, OLD))
    t64 = pd.read_csv(os.path.join(ROOT, CFG["v64_testbed_csv"]))
    out["testbed"]["v64_file"] = {v: compare(t64, v, OLD) for v in (NEW, CFG["prev_played"], "v6.7 alt: fit revert instead of bound")}
    out["prediction_check"] = compare(t, CFG["played"], CFG["prev_played"], types)
    nc = t[t.key.map(types) != "copier"]
    out["prediction_check"]["change_no_copier"] = compare(nc, CFG["played"], CFG["prev_played"])["change"]
    plot_testbed(t, out["testbed"], None)
    o, out["oracle"] = oracle_section(m, m_prev, t)
    plot_oracle(o, out["oracle"])
    out["prediction_vs_reality"] = prediction_vs_reality(out)
    plot_prediction(out["prediction_vs_reality"])
    m.round(3).to_csv(os.path.join(ROOT, "analytics", f"match_table_{CFG['data']}.csv"), index=False)
    json.dump(out, open(OUT, "w"), indent=1)
    print("written", OUT)


if __name__ == "__main__":
    main()
