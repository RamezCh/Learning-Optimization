"""
Tests for v6.6: rival price forecast (linear programme + time series), the DP on the forecast
path, today's price from a scalar solver, "rival sold out" days left out of the demand data, early fit.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import duopoly  # noqa: E402
from test_v64_market_sizing import _rate  # noqa: E402

THETA = np.array([np.log(0.6), np.log(45.0), np.log(0.25), 0.6])


def _state_with_paths(rival_paths, own_paths):
    state = duopoly._initialize_or_get_state(None, 1)
    state["rival_paths"] = [list(map(float, r)) for r in rival_paths]
    state["own_paths"] = [list(map(float, o)) for o in own_paths]
    duopoly._fit_rival_model(state)
    return state


def test_least_absolute_fit_ignores_outliers():
    rng = np.random.default_rng(0)
    x = rng.uniform(20, 60, 200)
    y = 5.0 + 0.8 * x + rng.normal(0, 0.2, 200)
    y[:20] += 80.0                                       # 10% wild price spikes
    X = np.c_[np.ones(200), x]
    lad = duopoly._least_absolute_fit(X, y, [(None, None), (None, None)])
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    assert abs(lad[1] - 0.8) < 0.02 and abs(lad[0] - 5.0) < 1.0
    assert abs(ols[0] - 5.0) > 3.0                       # least squares is pulled away


def test_seasonal_rival_path_is_learned():
    days = np.arange(100)
    truth = 30.0 + 0.4 * days                            # rival raises its price every day
    rng = np.random.default_rng(1)
    state = _state_with_paths([truth + rng.normal(0, 0.5, 100) for _ in range(4)], [np.full(100, 45.0)] * 4)
    state["rival_recent_prices"] = [float(truth[28])]
    state["rival_last_price"] = float(truth[28])
    path, share = duopoly._forecast_rival_path(30, state)
    assert abs(path[80] - truth[79]) < 2.0               # day 80, forecast on day 30
    assert path[80] - path[30] > 15.0                    # not a flat line
    assert np.all(share[30:101] == 1.0)


def test_before_two_seasons_the_forecast_is_the_flat_5_day_average():
    state = duopoly._initialize_or_get_state(None, 1)
    state["rival_recent_prices"] = [40.0, 42.0, 44.0]
    path, _ = duopoly._forecast_rival_path(10, state)
    assert np.allclose(path, 42.0)


def test_reaction_model_finds_a_rival_that_follows_our_price():
    rng = np.random.default_rng(2)
    own = [rng.uniform(30, 60, 100) for _ in range(3)]
    rival = [np.r_[50.0, 0.9 * o[:-1] + 3.0] for o in own]           # rival_t = 3 + 0.9 * our_{t-1}
    state = _state_with_paths(rival, own)
    a, c, b = state["rival_model"]["reaction"]
    assert abs(b - 0.9) < 0.02 and abs(c) < 0.02 and abs(a - 3.0) < 0.5
    state["rival_last_price"], state["my_last_price"] = 40.0, 50.0
    assert abs(duopoly._forecast_rival_today(state, 40.0) - 48.0) < 0.5


def test_reaction_model_is_dropped_when_it_is_no_better_than_yesterdays_price():
    rng = np.random.default_rng(3)
    rival = [45.0 + np.cumsum(rng.normal(0, 1.0, 100)) for _ in range(3)]   # random walk
    own = [rng.uniform(30, 60, 100) for _ in range(3)]
    state = _state_with_paths(rival, own)
    assert state["rival_model"]["reaction"] is None
    state["rival_last_price"] = 41.0
    assert duopoly._forecast_rival_today(state, 41.0) == 41.0


def test_dp_on_a_flat_path_equals_the_dp_on_one_price():
    V1, P1 = duopoly._dp_values(THETA, 50.0, 1)
    V2, P2 = duopoly._dp_values(THETA, np.full(102, 50.0), 1)
    assert np.allclose(V1, V2) and np.allclose(P1, P2)


def test_seat_is_worth_more_when_the_rival_will_raise_its_price():
    flat = np.full(102, 35.0)
    rising = np.r_[np.full(51, 35.0), np.full(51, 70.0)]
    V_flat, _ = duopoly._dp_values(THETA, flat, 1)
    V_rise, P_rise = duopoly._dp_values(THETA, rising, 1)
    assert V_rise[1][80] > V_flat[1][80]
    seat_flat = V_flat[21][30] - V_flat[21][29]
    seat_rise = V_rise[21][30] - V_rise[21][29]
    assert seat_rise > seat_flat                          # keep seats for the expensive-rival days


def test_solver_price_is_at_least_as_good_as_the_one_dollar_grid():
    V, POL = duopoly._dp_values(THETA, 50.0, 1)
    state = {"logit_theta": list(THETA)}
    for day, seats, rival in ((10, 70, 50.0), (50, 20, 47.3), (90, 5, 61.8)):
        price = duopoly._solve_price_today(day, seats, rival, V[day + 1], state)
        assert 18.0 <= price <= 100.0

        def value(p):
            k = np.arange(duopoly.DP_KMAX + 1)
            fact = np.array([1, 1, 2, 6, 24, 120, 720, 5040], dtype=float)
            pmf = duopoly._poisson_pmf(duopoly._logit_rate(THETA, p, rival, (day - 1) / 99.0), k, fact)[0]
            sold = np.minimum(k, seats)
            return float(pmf @ (p * sold + V[day + 1][seats - sold]))
        best_on_grid = max(value(p) for p in duopoly.DP_PRICE_GRID)
        assert value(price) >= best_on_grid - 0.02        # rounding to 10 cents costs < 2 cents


def test_sold_out_rival_days_are_left_out_of_the_demand_data():
    info, own, riv, dem = None, [], [], []
    for t in range(1, 41):
        hist = np.array([own, riv]) if t > 1 else None
        has_capacity = t < 30                              # rival sells out on day 30, keeps posting $20
        p, info = duopoly.p(1, t, hist, np.array(dem) if t > 1 else None, has_capacity, info)
        own.append(p); riv.append(20.0); dem.append(0)
    rows = np.array(info["training_history"])
    assert len(rows) == 29                                 # days 1-29 only; days 30-39 are not logged
    assert np.allclose(rows[:, 0] + rows[:, 1], 20.0)      # every logged day has the rival at $20


def test_choice_model_is_fitted_within_the_first_season():
    rng = np.random.default_rng(0)
    info, own, riv, dem = None, [], [], []
    for t in range(1, 101):
        hist = np.array([own, riv]) if t > 1 else None
        p, info = duopoly.p(1, t, hist, np.array(dem) if t > 1 else None, True, info)
        if t == 49:
            assert not info.get("logit_fitted", False)
        if t == 50:
            assert info.get("logit_fitted", False)
        q = int(min(80 - sum(dem), rng.poisson(_rate(p, 20.0, (t - 1) / 99.0, m=15.0))))
        own.append(p); riv.append(20.0); dem.append(q)
    # Low willingness to pay ($15): the price comes down in the same season and the seats sell.
    assert np.median(own[60:]) <= 20.0
    assert sum(dem[50:]) > 3 * max(1, sum(dem[:50]))


def test_full_match_is_fast_and_keeps_only_ten_seasons_of_paths():
    rng = np.random.default_rng(0)
    info, calls, t0 = None, 0, time.time()
    for s in range(1, 14):
        own, riv, dem = [], [], []
        for t in range(1, 101):
            rival = 35.0 + 0.2 * t + rng.normal(0, 1.0)
            hist = np.array([own, riv]) if t > 1 else None
            p, info = duopoly.p(s, t, hist, np.array(dem) if t > 1 else None, True, info)
            q = int(min(80 - sum(dem), rng.poisson(_rate(p, rival, (t - 1) / 99.0, n=0.9))))
            own.append(p); riv.append(rival); dem.append(q); calls += 1
        duopoly.p(s, 101, np.array([own, riv]), np.array(dem), True, info)
    assert (time.time() - t0) / calls < 0.05              # competition limit: 0.2 s on average
    assert len(info["rival_paths"]) == 10 and len(info["own_paths"]) == 10
    assert info["rival_model"] is not None and abs(info["rival_model"]["profile"][79] - 51.0) < 2.0
