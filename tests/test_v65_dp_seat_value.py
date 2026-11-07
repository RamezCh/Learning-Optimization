"""
Tests for v6.5: seat value (mu) and price from dynamic programming on the fitted choice model.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import duopoly  # noqa: E402
from test_v64_market_sizing import _play  # noqa: E402

THETA = np.array([np.log(0.6), np.log(45.0), np.log(0.25), 0.6])


def test_seat_value_is_positive_and_falls_with_more_seats():
    V, POL = duopoly._dp_values(THETA, 50.0, 1)
    mu = V[11][1:] - V[11][:-1]                     # value of the s-th seat after day 10
    assert np.all(mu >= -1e-9)
    assert np.all(np.diff(mu) <= 1e-6)             # diminishing value of extra seats
    assert V[1][80] > V[1][40] > 0


def test_policy_prices_lie_on_grid_and_rise_when_seats_are_scarce():
    V, POL = duopoly._dp_values(THETA, 50.0, 1)
    assert POL[50][1:].min() >= 18.0 and POL[50][1:].max() <= 100.0
    assert POL[50][5] >= POL[50][80]               # few seats left -> higher price


def test_last_day_seat_value_is_zero():
    V, _ = duopoly._dp_values(THETA, 50.0, 1)
    assert np.allclose(V[101], 0.0)


def test_dp_is_fast():
    t0 = time.time()
    for _ in range(5):
        duopoly._dp_values(THETA, 50.0, 1)
    assert (time.time() - t0) / 5 < 0.1


def test_dp_takes_over_once_choice_model_is_fitted():
    info, sold, _ = _play(3, lambda t, own: 60.0, n=1.5)
    assert info["logit_fitted"]
    assert info.get("dp_active") or info.get("unconstrained")
    assert "dp_cache" in info or info.get("unconstrained")


def test_no_dp_before_model_is_fitted():
    state = {"logit_fitted": False, "logit_theta": list(duopoly.LOGIT_PRIOR)}
    assert duopoly._dp_seat_value(10, 50, 60.0, state) is None
