"""
analytics/fit_regime_priors.py
------------------------------
One place that decides which tournament data counts as low, average or high demand, and
fits the three regime regression models (REGIME_PRESETS_MLR in duopoly.py) on all of it.

Classification: every season of every match is labelled with the same rule duopoly.py
applies at the end of a season (_classify_and_update_regime), so a prior is fitted on the
kind of season in which duopoly.py would select it:
  low      seats sold < 42, or (no sale at >= $55 in 4+ days so far in the match and < 60 seats)
  high     seats sold >= 72 and no sale at >= $55 so far in the match (truncated willingness to pay)
  average  everything else
A match's label is the label of most of its seasons; the season counts per label are kept.

Fit: OLS of seats sold per day on [our price, rival price - our price, tau], tau = (day-1)/99,
on duopoly days (rival in stock, valid rival price) of all seasons with that label, exactly the
features of the online model in duopoly.py. Stored as the identified slopes
B = b_price - b_diff (own price) and C = b_diff (rival price). Holdout check: fit on seasons
1-80, R^2 on seasons 81-100.

Outputs (re-run whenever new data arrives in data/<folder>/):
  DATA_REGISTRY.md            table of every dataset and match with its demand label
  analytics/data_registry.csv the same, one row per match
  analytics/regime_priors.json fitted priors (paste into REGIME_PRESETS_MLR)

Usage:
    python analytics/fit_regime_priors.py
"""

import glob
import json
import os
from typing import Dict, List

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
REGISTRY_MD = os.path.join(ROOT_DIR, "DATA_REGISTRY.md")
REGISTRY_CSV = os.path.join(ROOT_DIR, "analytics", "data_registry.csv")
PRIORS_JSON = os.path.join(ROOT_DIR, "analytics", "regime_priors.json")

LOW_SEATS, HIGH_SEATS, WEAK_SEATS = 42, 72, 60
HIGH_PRICE = 55.0
sys.path.insert(0, ROOT_DIR)
from duopoly import REGIME_PRESETS_MLR  # noqa: E402

REGIMES = ["low_demand", "average_demand", "high_demand"]
SHORT = {"low_demand": "low", "average_demand": "average", "high_demand": "high"}

# Algorithm version that produced each dataset (who "we" are in the log).
VERSION_OF = {"v1": "v1.0", "v3": "v3.0", "v4": "v4.1", "v5": "v5.0", "v5.1": "v5.1",
              "mlr": "MLR variant", "poisson_logit": "Poisson-logit variant", "v6": "v6.3"}


def load_all() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*", "*.csv"))):
        folder = os.path.basename(os.path.dirname(path))
        frame = pd.read_csv(path)
        frame["dataset"] = folder
        frame["file"] = os.path.basename(path)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["competitor_has_capacity"] = data["competitor_has_capacity"].astype(str).str.lower().isin(["true", "1", "1.0"])
    return data.sort_values(["dataset", "file", "competition_id", "selling_season", "selling_period"])


def classify_seasons(data: pd.DataFrame) -> pd.DataFrame:
    """One row per (file, match, season) with the regime label duopoly.py would give it."""
    rows = []
    for (dataset, file, cid), match in data.groupby(["dataset", "file", "competition_id"], sort=False):
        high_days = high_sales = 0
        for season, s in match.groupby("selling_season"):
            seats = int(s.demand.sum())
            at_high = s[s.price >= HIGH_PRICE]
            high_days += len(at_high)
            high_sales += int(at_high.demand.sum())
            truncated = high_days >= 4 and high_sales == 0
            if seats < LOW_SEATS or (truncated and seats < WEAK_SEATS):
                regime = "low_demand"
            elif seats >= HIGH_SEATS and high_sales == 0:
                regime = "high_demand"
            else:
                regime = "average_demand"
            rows.append({"dataset": dataset, "file": file, "competition_id": cid, "season": int(season),
                         "rival": s.competitor_id.iloc[0], "seats": seats,
                         "revenue": float((s.price * s.demand).sum()),
                         "rival_sold_out": bool((~s.competitor_has_capacity).any()),
                         "rival_median": float(s.price_competitor[s.price_competitor < 200].median()),
                         "regime": regime})
    return pd.DataFrame(rows)


def fit_regime(days: pd.DataFrame) -> Dict:
    x = np.column_stack([days.price, days.price_competitor - days.price, (days.selling_period - 1) / 99.0])
    y = days.demand.values.astype(float)
    model = LinearRegression().fit(x, y)
    b_p, b_diff, b_t = (float(c) for c in model.coef_)
    train = days.selling_season <= 80
    holdout = LinearRegression().fit(x[train.values], y[train.values])
    return {"intercept": round(float(model.intercept_), 4), "B": round(b_p - b_diff, 4), "C": round(b_diff, 4),
            "b_t": round(b_t, 4), "avg_demand": round(float(y.mean()), 3), "days": int(len(y)),
            "r2": round(float(model.score(x, y)), 3),
            "holdout_r2": round(float(holdout.score(x[~train.values], y[~train.values])), 3)}


def main() -> None:
    data = load_all()
    seasons = classify_seasons(data)
    data = data.merge(seasons[["file", "competition_id", "season", "regime"]],
                      left_on=["file", "competition_id", "selling_season"],
                      right_on=["file", "competition_id", "season"])
    duopoly_days = data[data.competitor_has_capacity & (data.price_competitor < 200)]

    priors = {}
    for regime in REGIMES:
        days = duopoly_days[duopoly_days.regime == regime]
        priors[regime] = fit_regime(days) if len(days) > 1000 else None
        if priors[regime]:
            priors[regime]["seasons"] = int((seasons.regime == regime).sum())
            priors[regime]["matches"] = int(seasons[seasons.regime == regime].groupby(["file", "competition_id"]).ngroups)
    with open(PRIORS_JSON, "w") as fh:
        json.dump(priors, fh, indent=1)

    # Registry, one row per match
    match = seasons.groupby(["dataset", "file", "competition_id"], sort=False).agg(
        rival=("rival", "first"), seasons=("season", "count"), seats_per_season=("seats", "mean"),
        revenue_per_season=("revenue", "mean"), rival_median_price=("rival_median", "median"),
        rival_sold_out_pct=("rival_sold_out", "mean"),
        low=("regime", lambda r: int((r == "low_demand").sum())),
        average=("regime", lambda r: int((r == "average_demand").sum())),
        high=("regime", lambda r: int((r == "high_demand").sum()))).reset_index()
    match["label"] = match[["low", "average", "high"]].idxmax(axis=1)
    match["algorithm"] = match.dataset.map(VERSION_OF).fillna("?")
    match["rival_sold_out_pct"] = (match.rival_sold_out_pct * 100).round(0)
    match = match.round({"seats_per_season": 1, "revenue_per_season": 0, "rival_median_price": 1})
    match.to_csv(REGISTRY_CSV, index=False)

    write_markdown(match, seasons, priors)
    print(match.groupby(["dataset", "label"]).size().unstack(fill_value=0))
    print(json.dumps(priors, indent=1))


def write_markdown(match: pd.DataFrame, seasons: pd.DataFrame, priors: Dict) -> None:
    lines: List[str] = [
        "# Data Registry: which tournament data counts as which demand regime",
        "",
        "Generated by `python analytics/fit_regime_priors.py`; re-run it whenever a new CSV is added under "
        "`data/<folder>/`. Do not edit by hand. The CSV files themselves are not in git (`data/` is ignored).",
        "",
        "## Rule",
        "",
        "Every season is labelled with the rule `duopoly.py` applies at the end of a season, so each "
        "regime model is fitted on the kind of season in which `duopoly.py` would use it:",
        "",
        f"* **low**: we sold fewer than {LOW_SEATS} seats, or no sale at ≥ \\${HIGH_PRICE:.0f} in 4+ days so far "
        f"in the match and fewer than {WEAK_SEATS} seats",
        f"* **high**: we sold {HIGH_SEATS}+ seats and had no sale at ≥ \\${HIGH_PRICE:.0f} so far in the match",
        "* **average**: everything else",
        "",
        "A match's label is the label of most of its seasons. The label describes the seasons *we* "
        "experienced: our own sales depend on our prices, so a version that priced too high (v1.0) makes "
        "a market look like low demand.",
        "",
        "## Regime models fitted on this data",
        "",
        "OLS of seats sold per day on our price, rival price − our price and τ, on days the rival had seats. "
        "$B$ = own-price slope, $C$ = rival-price slope. Holdout $R^2$: fitted on seasons 1–80, tested on 81–100.",
        "",
        "| Regime | Matches | Seasons | Days | $\\beta_0$ | $B$ | $C$ | $\\beta_t$ | Mean seats/day | $R^2$ | Holdout $R^2$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for regime in REGIMES:
        p = priors.get(regime)
        if p:
            lines.append(f"| {SHORT[regime]} | {p['matches']} | {p['seasons']:,} | {p['days']:,} | {p['intercept']} | "
                         f"{p['B']} | {p['C']} | {p['b_t']} | {p['avg_demand']} | {p['r2']} | {p['holdout_r2']} |")
    lines += ["", "**Models `duopoly.py` uses now** (`REGIME_PRESETS_MLR`, older fits by dataset folder: "
              "low = mlr, v5.1, poisson_logit; average = v3, v4; high = v1):", "",
              "| Regime | $\\beta_0$ | $B$ | $C$ | $\\beta_t$ |", "| :--- | :---: | :---: | :---: | :---: |"]
    for regime in REGIMES:
        q = REGIME_PRESETS_MLR[regime]
        lines.append(f"| {SHORT[regime]} | {q['intercept']} | {q['B']} | {q['C']} | {q['b_t']} |")
    lines += ["",
              "The refitted models were tested in `duopoly.py` (2026-10-31) and **not adopted**: revenue changed by "
              "−0.8% on the `data/v6` test bed and by +0.8% / −0.4% / −0.2% (average / high / low) on the old "
              "simulator, within noise. Blending the pooled model into each match's own fit (weight 500 or 2,000 days) "
              "did not help either. These models only set prices until a match has 80 days of its own data "
              "(most of season 1); after that each match's own fit describes that match better than pooled data "
              "(pooled fits explain 4–13% of daily sales). See ALGORITHM_CHANGES §20.8.",
              "", "## Datasets", "",
              "| Folder | File | Algorithm | Matches | low / average / high seasons | Seats per season |",
              "| :--- | :--- | :--- | :---: | :---: | :---: |"]
    for (dataset, file), g in match.groupby(["dataset", "file"], sort=False):
        s = seasons[(seasons.dataset == dataset) & (seasons.file == file)]
        counts = " / ".join(str(int((s.regime == r).sum())) for r in REGIMES)
        lines.append(f"| `{dataset}` | `{file}` | {g.algorithm.iloc[0]} | {len(g)} | {counts} | {s.seats.mean():.1f} |")
    lines += ["", "## Matches", "",
              "| Folder | Competition ID | Rival | Label | low / avg / high seasons | Seats per season | "
              "Revenue per season | Rival median price | Rival sold out |",
              "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for _, r in match.iterrows():
        lines.append(f"| `{r.dataset}` | `{r.competition_id}` | {r.rival} | **{r.label}** | "
                     f"{r.low} / {r.average} / {r.high} | {r.seats_per_season} | \\${r.revenue_per_season:,.0f} | "
                     f"\\${r.rival_median_price} | {r.rival_sold_out_pct:.0f}% |")
    lines.append("")
    with open(REGISTRY_MD, "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
