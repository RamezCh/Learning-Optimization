"""
analytics/generate_v6_plots.py
------------------------------
Figures for the data/v6 post-mortem and the v6.4 results.

imgs/v6_tournament_postmortem.png  (from data/v6 only)
  A. Revenue per season in every match vs the rival's median price, per tournament.
  B. Seats per season, seasons 21-100: variance vs mean. Points on the line var = mean mean
     Poisson noise around a fixed average, i.e. the same demand model every season.
  C. The price war against a copying rival (FlashyInchworm, 39jdMo): season 1 vs season 30.
  D. Demand model check: actual vs model sales per day in every $10 price band of every match.

imgs/v64_testbed_results.png  (from analytics/v6_testbed_results.csv)
  A. Revenue per season per match: v6.3, v6.4 and the best possible.
  B. Totals over the 34 matches.

Run analytics/v6_testbed.py first, with the v6.3 file and the $5-floor variant as --compare
files (their names must contain "v63" and "floor5").
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from plot_style import COLORS, apply_style, plt  # noqa: E402
from v6_testbed import FITS_PATH, RESULTS_PATH, demand_rate, load_v6  # noqa: E402

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOURNAMENT_COLORS = {12: COLORS["ours"], 19: COLORS["accent"], 49: COLORS["amber"]}


def postmortem(data: pd.DataFrame, fits: dict) -> None:
    data = data.assign(revenue=data.price * data.demand)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # A. revenue per season vs rival price
    ax = axes[0, 0]
    match = data.groupby("competition_id").agg(rank=("rank", "first"), rival=("price_competitor", "median"),
                                               revenue=("revenue", "sum"))
    for rank, group in match.groupby("rank"):
        mean = group.revenue.sum() / 100 / len(group)
        ax.scatter(group.rival, group.revenue / 100, s=70, color=TOURNAMENT_COLORS.get(rank, COLORS["dark"]),
                   edgecolor="white", label=f"rank {rank}: {len(group)} matches, ${mean:,.0f}/season")
    ax.set_xlabel("Rival's median price ($)")
    ax.set_ylabel("Our revenue per season ($)")
    ax.set_title("A. The rivals we drew decided most of the score")
    ax.legend(loc="upper left", title="tournament")

    # B. variance vs mean of seats per season
    ax = axes[0, 1]
    seasons = data[data.selling_season > 20].groupby(["competition_id", "selling_season"]).demand.sum()
    stats = seasons.groupby("competition_id").agg(["mean", "var"])
    stats = stats[stats["mean"] > 0.5]
    rival_moves = {"ManipulativeJaybird", "StoicCentipede", "GiftedChachalaca", "KickassMoth"}
    names = data.groupby("competition_id").competitor_id.first()
    moving = stats.index.map(lambda c: names[c] in rival_moves)
    ax.scatter(stats["mean"][~moving], stats["var"][~moving], s=60, color=COLORS["accent"],
               label="rival with a steady strategy")
    ax.scatter(stats["mean"][moving], stats["var"][moving], s=60, color=COLORS["warn"],
               label="rival that changes strategy between seasons")
    top = float(stats["mean"].max()) * 1.15
    ax.plot([0, top], [0, top], color=COLORS["dark"], linestyle="--", label="variance = mean (Poisson)")
    ax.set_xlabel("Mean seats per season (seasons 21-100)")
    ax.set_ylabel("Variance of seats per season")
    ax.set_title("B. Same demand every season: only Poisson noise")
    ax.legend(loc="lower right")

    # C. copier price war
    ax = axes[1, 0]
    copier = data[data.competition_id == "39jdMo"]
    for season, style in ((1, "-"), (30, ":")):
        s = copier[copier.selling_season == season]
        revenue = float((s.price * s.demand).sum())
        ax.plot(s.selling_period, s.price, style, color=COLORS["ours"], linewidth=2.2,
                label=f"season {season}: our price (revenue ${revenue:,.0f})")
        ax.plot(s.selling_period, s.price_competitor, style, color=COLORS["rival"], linewidth=1.4,
                label=f"season {season}: FlashyInchworm (copies our price of the day before)")
    ax.set_xlabel("Day of the season")
    ax.set_ylabel("Price ($)")
    ax.set_title("C. v6.3's season-end clearance started a price war with a copier")
    ax.legend(loc="upper right", fontsize=8)

    # D. demand model check
    ax = axes[1, 1]
    act, pred = [], []
    for cid, m in data.groupby("competition_id"):
        lam = demand_rate(fits[cid]["theta"], m.price.values, m.price_competitor.values, m.tau.values)
        band = (m.price // 10 * 10).astype(int)
        table = pd.DataFrame({"band": band, "act": m.demand.values, "pred": lam}).groupby("band")
        table = table.agg(n=("act", "size"), act=("act", "mean"), pred=("pred", "mean"))
        table = table[table.n >= 200]
        act.extend(table.act)
        pred.extend(table.pred)
    ax.scatter(pred, act, s=40, color=COLORS["accent"], alpha=0.8)
    ax.plot([0, 0.65], [0, 0.65], color=COLORS["dark"], linestyle="--")
    err = np.mean(np.abs(np.array(act) - np.array(pred)))
    ax.set_xlabel("Model: seats per day")
    ax.set_ylabel("Actual: seats per day")
    ax.set_title(f"D. Demand model vs data: {len(act)} price bands, mean error {err:.3f} seats/day")

    fig.suptitle("data/v6: 3 tournaments, 34 matches, v6.3 unchanged in all of them", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT_DIR, "imgs", "v6_tournament_postmortem.png"))
    plt.close(fig)


def results() -> None:
    table = pd.read_csv(RESULTS_PATH, index_col=0)
    old = next(c for c in table.columns if "v63" in c)
    floor = next((c for c in table.columns if "floor5" in c), None)
    new = "duopoly.py"
    table = table.assign(gain=table[new] / table[old] - 1).sort_values("gain")

    fig, axes = plt.subplots(1, 2, figsize=(16, 10), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    y = np.arange(len(table))
    for i, (_, r) in enumerate(table.iterrows()):
        ax.plot([r[old] / 100, r[new] / 100], [i, i], color=COLORS["grid"], linewidth=3, zorder=1)
    ax.scatter(table.best_possible / 100, y, marker="|", s=160, color=COLORS["dark"], label="best possible", zorder=2)
    ax.scatter(table[old] / 100, y, s=45, color=COLORS["rival"], label="v6.3", zorder=3)
    ax.scatter(table[new] / 100, y, s=45, color=COLORS["ours"], label="v6.4", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.rival} ({c})  {r.gain * 100:+.0f}%" for c, r in table.iterrows()], fontsize=8)
    ax.set_xlabel("Revenue per season ($), mean of 3 seeds")
    ax.set_title("A. Every match: v6.3 vs v6.4 (sorted by change)")
    ax.legend(loc="lower right")

    ax = axes[1]
    labels = ["Actual\n(tournaments)", "v6.3\n(test bed)", "v6.4"]
    values = [table.actual.sum(), table[old].sum(), table[new].sum()]
    colors = [COLORS["grid"], COLORS["rival"], COLORS["ours"]]
    if floor:
        labels.append("$5 floor\n(extrapolated)")
        values.append(table[floor].sum())
        colors.append(COLORS["amber"])
    labels.append("Best\npossible")
    values.append(table.best_possible.sum())
    colors.append(COLORS["dark"])
    bars = ax.bar(range(len(values)), np.array(values) / 1e6, color=colors, edgecolor=COLORS["dark"])
    for bar, value in zip(bars, values):
        share = value / table.best_possible.sum() * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"${value / 1e6:.2f}M\n{share:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Total revenue, 34 matches x 100 seasons ($M)")
    ax.set_ylim(0, max(values) / 1e6 * 1.18)
    ax.set_title(f"B. Totals: v6.4 {table[new].sum() / table[old].sum() * 100 - 100:+.1f}% vs v6.3")

    fig.suptitle("v6.4 on the data/v6 test bed (fitted demand, replayed rivals, same random draws)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT_DIR, "imgs", "v64_testbed_results.png"))
    plt.close(fig)


if __name__ == "__main__":
    apply_style()
    data = load_v6()
    with open(FITS_PATH) as fh:
        fits = json.load(fh)
    postmortem(data, fits)
    if os.path.exists(RESULTS_PATH):
        results()
    print("wrote imgs/v6_tournament_postmortem.png and imgs/v64_testbed_results.png")
