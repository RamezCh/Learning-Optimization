"""
Tests for v6.4: market sizing (mu = 0 when capacity cannot bind) and copying-rival detection.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import duopoly  # noqa: E402


def _rate(price, rival, tau, n=0.34, m=45.0, b=0.25, k=0.6):
    ours = np.exp(b * (m - price))
    theirs = np.exp(b * (m - rival))
    return np.exp(np.log(n) + k * tau) * ours / (1.0 + ours + theirs)


def _play(seasons, rival_rule, n=0.34, m=45.0, seed=0):
    """Small logit market. rival_rule(t, our_prices) -> rival price."""
    rng = np.random.default_rng(seed)
    info, sold = None, []
    for s in range(1, seasons + 1):
        own, riv, dem = [], [], []
        for t in range(1, 101):
            r = rival_rule(t, own)
            hist = np.array([own, riv]) if t > 1 else None
            dh = np.array(dem) if t > 1 else None
            p, info = duopoly.p(s, t, hist, dh, True, info)
            q = int(min(80 - sum(dem), rng.poisson(_rate(p, r, (t - 1) / 99.0, n=n, m=m))))
            own.append(p)
            riv.append(r)
            dem.append(q)
        duopoly.p(s, 101, np.array([own, riv]), np.array(dem), True, info)
        sold.append(sum(dem))
    return info, sold, own


def test_small_market_switches_to_revenue_pricing():
    # ~47 buyers a season can never fill 80 seats: capacity is slack and the model is fitted.
    info, _, _ = _play(3, lambda t, own: 40.0)
    assert info["logit_fitted"]
    assert duopoly._is_unconstrained(info)


def test_big_market_keeps_v63_capacity_logic():
    # ~200 buyers a season: demand at revenue-maximising prices exceeds the seats left.
    info, sold, _ = _play(3, lambda t, own: 60.0, n=1.5)
    assert max(sold) >= 75
    assert not duopoly._is_unconstrained(info)


def test_model_learns_from_zero_sales():
    # Low willingness to pay ($15 midpoint): almost no sales at v6.3's season-1 prices. From
    # season 2 the price must come down to where the buyers are, and sales must recover.
    info, sold, own = _play(4, lambda t, own: 20.0, m=15.0)
    assert np.median(own) <= 20.0
    assert np.mean(sold[1:]) > 2 * sold[0]


def test_copier_is_detected_and_held():
    # Rival sets its price to our price yesterday.
    info, _, own = _play(3, lambda t, own: own[-1] if own else 60.0)
    assert duopoly._rival_is_copier(info)
    assert info["copier_hold"] is not None
    # One held price for most of the last season (no season-end clearance against a copier).
    assert np.mean(np.abs(np.array(own[30:]) - own[-1]) < 0.01) > 0.9


def test_flat_rival_is_not_a_copier():
    # A rival that never moves must not count as a copier, even when we sit at its price.
    info, _, _ = _play(2, lambda t, own: 18.0)
    assert not duopoly._rival_is_copier(info)


def test_copier_hold_search_moves_toward_better_revenue():
    state = {"copier_hold": {"price": 40.0, "best_price": None, "best_rev": None,
                             "direction": -1.0, "step": 3.0},
             "copier_rev": 400.0, "copier_days": 80}
    duopoly._update_copier_hold(state)
    assert state["copier_hold"]["best_price"] == 40.0
    assert state["copier_hold"]["price"] == 37.0
    # Worse revenue at 37: turn around with a smaller step from the best price.
    state["copier_rev"], state["copier_days"] = 300.0, 80
    duopoly._update_copier_hold(state)
    assert state["copier_hold"]["direction"] == 1.0
    assert abs(state["copier_hold"]["price"] - 41.8) < 1e-9


def test_calls_stay_fast():
    info, _, _ = _play(2, lambda t, own: 30.0)
    hist = np.array([[30.0] * 99, [30.0] * 99])
    start = time.perf_counter()
    duopoly.p(3, 101, hist, np.zeros(99), True, info)   # season end includes a model fit
    assert time.perf_counter() - start < 0.2
