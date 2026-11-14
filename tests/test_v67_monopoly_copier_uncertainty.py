"""
Tests for v6.7: monopoly DP capped at the hand rule, copier DP with a willingness-to-pay guard and the
season-revenue copier search, willingness-to-pay bound in the choice-model fit, today's price under rival
uncertainty.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import duopoly  # noqa: E402
from test_v64_market_sizing import _rate  # noqa: E402

PLAUSIBLE = [float(np.log(1.48)), float(np.log(56.0)), float(np.log(0.161)), 0.63]   # data/v6.6 3yKzJN (copier market)
IMPLAUSIBLE = [float(np.log(1.2)), float(np.log(447.0)), float(np.log(0.126)), 0.24]  # data/v6.6 48gs8r online fit


def _monopoly_state(theta, period=60):
    state = duopoly._initialize_or_get_state(None, 1)
    state.update(logit_theta=list(theta), logit_fitted=True, period_now=period, detected_regime="average_demand")
    return state


def _hand_price(burn, state):
    p_low, p_high = duopoly.REGIME_PRESETS_MLR[state["detected_regime"]]["monopoly_band"]
    return p_high - float(np.clip((burn - 0.30) / 0.90, 0.0, 1.0)) * (p_high - p_low)


def test_monopoly_dp_can_lower_the_hand_price():
    state = _monopoly_state(PLAUSIBLE, period=80)
    seats, burn = 50, 50 / 21.0                         # many seats, 21 days left
    price = duopoly._calculate_monopoly_price(burn, seats, state)
    dp = duopoly._dp_monopoly_price(80, seats, state)
    assert dp < _hand_price(burn, state)                  # the DP wants less than the hand band here
    assert price == dp


def test_monopoly_dp_never_raises_the_hand_price():
    # m = $447: once the rival is gone the model thinks buyers barely react to price -> DP says ~$100
    state = _monopoly_state(IMPLAUSIBLE)
    seats, burn = 30, 30 / 41.0
    assert duopoly._dp_monopoly_price(60, seats, state) > 90.0
    assert abs(duopoly._calculate_monopoly_price(burn, seats, state) - _hand_price(burn, state)) < 1e-9


def test_copier_search_uses_whole_season_revenue():
    # sold out on day 75 at $30: 40 days held before sell-out; revenue = 80 * 30 = 2400
    state = {"copier_days": 40, "copier_rev": 1600.0}
    own = np.r_[np.full(75, 30.0), np.full(25, duopoly.TERMINAL_SOLDOUT_PRICE)]
    sold = np.r_[np.full(75, 80 / 75.0), np.zeros(25)]
    duopoly._copier_revenue_from_season(np.array([own, own]), sold, state)
    assert abs(state["copier_rev"] / state["copier_days"] - 2400.0 / 100.0) < 1e-9   # per season day, not per day held


def test_copier_dp_prices_the_copier_market_near_the_oracle():
    state = {"logit_theta": list(PLAUSIBLE), "seats_left_now": 80}
    day1 = duopoly._dp_copier_price(1, state)
    assert 45.0 <= day1 <= 60.0                                        # oracle: ~$53 (v6.6 held ~$30)
    state["seats_left_now"] = 10
    assert duopoly._dp_copier_price(80, state) >= duopoly._dp_copier_price(80, {**state, "seats_left_now": 70})


def test_copier_dp_is_not_used_with_an_implausible_willingness_to_pay():
    state = duopoly._initialize_or_get_state(None, 1)
    state.update(logit_theta=list(IMPLAUSIBLE), logit_fitted=True, copy_checks=[1] * 20, seats_left_now=60)
    assert duopoly._rival_is_copier(state)
    price = duopoly._calculate_revenue_max_price(30, 40.0, state)
    assert state.get("copier_hold") is not None                       # fell back to the search
    assert price == state["copier_hold"]["price"]
    state["logit_theta"] = list(PLAUSIBLE)
    assert duopoly._calculate_revenue_max_price(30, 40.0, state) == float(
        np.clip(duopoly._dp_copier_price(30, state), *duopoly.COPIER_PRICE_RANGE))


def _history_where_price_barely_matters(seed=0, days=3000):
    """Duopoly days with our price within a few $ of the rival's, sales from a model with m = $400."""
    rng = np.random.default_rng(seed)
    theta = [np.log(0.5), np.log(400.0), np.log(0.3), 0.6]
    rival = rng.uniform(25, 60, days)
    price = rival + rng.normal(0, 3, days)
    tau = rng.uniform(0, 1, days)
    sold = rng.poisson(duopoly._logit_rate(theta, price, rival, tau))
    return np.c_[price, rival - price, tau, sold].tolist()


def test_willingness_to_pay_midpoint_is_bounded_in_the_fit(monkeypatch):
    hist = _history_where_price_barely_matters()
    monkeypatch.setattr(duopoly, "LOGIT_M_MAX", 1e9)
    free = {"training_history": hist, "logit_theta": list(duopoly.LOGIT_PRIOR)}
    duopoly._fit_logit_demand(free)
    monkeypatch.setattr(duopoly, "LOGIT_M_MAX", 150.0)
    bounded = {"training_history": hist, "logit_theta": list(duopoly.LOGIT_PRIOR)}
    duopoly._fit_logit_demand(bounded)
    assert np.exp(free["logit_theta"][1]) > 200.0                   # unbounded: drifts to "price hardly matters"
    assert np.exp(bounded["logit_theta"][1]) <= 150.0 * 1.01


DATA_446 = os.path.join(os.path.dirname(__file__), "..", "data", "v6.4", "day 1", "Tody_Rank_33.csv")


@pytest.mark.skipif(not os.path.exists(DATA_446), reason="data/v6.4 not present")
def test_rival_parked_at_100_is_not_priced_at_100():
    # data/v6.4 match 446Pbr: the rival sits at $100. Replayed with the test bed's demand model for that match,
    # v6.6 collapses in 1 of 8 random replays (seed 1 here: $237, $148, $207 in seasons 2-4, fitted m = $265);
    # on the test bed's own draws it priced ~$100 for ~30 seasons. v6.7 earns ~$4,700 per season on seed 1.
    import pandas as pd
    from scipy.stats import poisson
    g = pd.read_csv(DATA_446)
    g = g[g.competition_id == "446Pbr"].sort_values(["selling_season", "selling_period"])
    rivals = g.price_competitor.values.reshape(-1, 100)
    caps = g.competitor_has_capacity.values.reshape(-1, 100)
    n, m, b, k = 1.08, 61.1, 0.202, 0.71
    rng = np.random.default_rng(1)
    info, revenue = None, []
    for s in range(1, 4):
        own, riv, dem = [], [], []
        for t in range(1, 101):
            cap, r = bool(caps[s - 1, t - 1]), float(rivals[s - 1, t - 1])
            hist = np.array([own, riv]) if t > 1 else None
            p, info = duopoly.p(s, t, hist, np.array(dem) if t > 1 else None, cap, info)
            r_eff = r if (cap and r < 200) else 1000.0
            q = int(min(80 - sum(dem), poisson.ppf(rng.random(), _rate(p, r_eff, (t - 1) / 99.0, n=n, m=m, b=b, k=k))))
            own.append(p); riv.append(r); dem.append(q)
        duopoly.p(s, 101, np.array([own, riv]), np.array(dem), caps[s - 1, -1], info)
        revenue.append(float(np.dot(own, dem)))
    assert min(revenue[1:]) > 3000.0                                  # v6.6: ~$150 per season here


def test_rival_scenarios_appear_after_enough_forecast_errors():
    state = {"rival_last_price": 40.0, "my_last_price": 45.0}
    assert duopoly._rival_today(state, 40.0, 10) == 40.0               # no errors yet: one forecast
    state["rival_fc_errors"] = list(np.linspace(-20, 20, 30))
    scen = duopoly._rival_today(state, 40.0, 11)
    assert len(scen) == 30 and abs(np.mean(scen) - 40.0) < 1e-9 and scen.min() >= 20.0 - 1e-9


def test_forecast_errors_are_recorded_only_while_the_rival_sells():
    info, own, riv, dem = None, [], [], []
    rng = np.random.default_rng(1)
    for t in range(1, 41):
        rival = float(rng.uniform(5, 87))
        hist = np.array([own, riv]) if t > 1 else None
        p, info = duopoly.p(1, t, hist, np.array(dem) if t > 1 else None, t < 30, info)
        info["rival_fc"] = [t, 46.0]                                  # pretend a forecast was made every day
        own.append(p); riv.append(rival); dem.append(0)
    # rival sells on days 1-29; the error of day d is recorded on day d+1 -> 29 errors, none for days 30-39
    assert len(info.get("rival_fc_errors", [])) == 29


def test_price_under_uncertainty_is_at_least_as_good_as_the_grid():
    theta = np.array(PLAUSIBLE)
    V, _ = duopoly._dp_values(theta, 45.0, 1)
    rivals = np.clip(45.0 + np.linspace(-35, 35, 40), 0.1, 199)
    state = {"logit_theta": list(theta)}
    price = duopoly._solve_price_today(40, 30, rivals, V[41], state)
    k = np.arange(duopoly.DP_KMAX + 1)
    fact = np.array([1, 1, 2, 6, 24, 120, 720, 5040], dtype=float)
    sold = np.minimum(k, 30)

    def value(p):
        lam = duopoly._logit_rate(theta, p, rivals, 39 / 99.0)
        pmf = duopoly._poisson_pmf(lam, k, fact).mean(axis=0)
        return float(pmf @ (p * sold + V[41][30 - sold]))
    assert value(price) >= max(value(p) for p in duopoly.DP_PRICE_GRID) - 0.02
    point = duopoly._solve_price_today(40, 30, 45.0, V[41], state)
    assert value(price) >= value(point) - 1e-9                         # never worse than pricing on the mean
