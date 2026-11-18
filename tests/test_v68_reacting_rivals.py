"""
Tests for v6.8: copier flag that can clear, copier price up to $80, leader pricing on the rival's reaction
model (tested, off), cooperative offer against a rival that follows us (and its price-war evidence gate), self-play.
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analytics"))
import duopoly  # noqa: E402

THETA = [float(np.log(0.80)), float(np.log(57.0)), float(np.log(0.240)), 0.62]   # data/v6.7 3AbmmC (floor-war market)


def _state(**kw):
    state = duopoly._initialize_or_get_state(None, 5)
    state.update(logit_theta=list(THETA), logit_fitted=True, last_season_seen=5, seats_left_now=80, period_now=10)
    state.update(kw)
    return state


def test_copier_flag_clears_when_we_hold_and_the_rival_does_not_copy():
    state = _state(copy_checks=[1] * 20)                 # latched copier flag
    assert duopoly._rival_is_copier(state)
    own = [60.0] * 12
    for day in range(3, 12):                              # we hold $60, the rival posts ~$70
        hist = np.array([own[:day - 1], [70.0] * (day - 1)])
        duopoly._update_real_time_signals(day, hist, np.zeros(day - 1, dtype=int), True, state)
    assert not duopoly._rival_is_copier(state)


def test_copier_flag_stays_when_the_rival_posts_our_held_price():
    state = _state(copy_checks=[1] * 20)
    own = [60.0] * 12
    for day in range(3, 12):                              # a real copier posts our price
        hist = np.array([own[:day - 1], own[:day - 1]])
        duopoly._update_real_time_signals(day, hist, np.zeros(day - 1, dtype=int), True, state)
    assert duopoly._rival_is_copier(state)


def test_copier_price_may_exceed_60():
    assert duopoly.COPIER_PRICE_MAX == 80.0
    state = _state(copy_checks=[1] * 20)
    price = duopoly._calculate_revenue_max_price(10, 50.0, state)
    assert 18.0 <= price <= 80.0


def test_follower_from_reaction_model(monkeypatch):
    monkeypatch.setattr(duopoly, "LEADER_PRICING", True)         # tested, not adopted (off in v6.8)
    state = _state(rival_model={"reaction": [-0.11, 0.11, 0.90], "range": [18.0, 58.0]})
    alpha, beta, lo, hi = duopoly._rival_follower(state)
    assert beta == pytest.approx(0.90 / 0.89, rel=1e-6) and (lo, hi) == (18.0, 58.0)
    state["rival_model"]["reaction"] = [20.0, 0.5, 0.1]   # slope 0.2: not a follower
    assert duopoly._rival_follower(state) is None


def test_leader_price_above_best_reply_against_a_follower(monkeypatch):
    monkeypatch.setattr(duopoly, "LEADER_PRICING", True)
    follower = _state(rival_model={"reaction": [0.0, 0.0, 1.0], "range": [18.0, 80.0]}, rival_last_price=18.0)
    ignorer = _state(rival_model={"reaction": None, "range": [18.0, 80.0]}, rival_last_price=18.0)
    assert duopoly._calculate_revenue_max_price(10, 18.0, follower) > duopoly._calculate_revenue_max_price(10, 18.0, ignorer) + 10


def test_no_cooperation_without_price_war_evidence():
    state = _state(rival_model={"reaction": [30.0, 0.2, 0.05], "range": [18.0, 60.0]},
                   own_paths=[[40.0] * 100], rival_paths=[[45.0] * 100])
    assert not duopoly._price_war_evidence(state)
    assert duopoly._coop_price(10, state, 20.0) is None


def test_cooperation_offered_after_a_floor_war_and_dropped_after_undercuts():
    state = _state(own_paths=[[18.0] * 100], rival_paths=[[18.0] * 100], rival_cap_days=[True] * 100)
    assert duopoly._price_war_evidence(state)
    offer = duopoly._coop_price(1, state, 18.0)
    assert offer is not None and offer > 30.0
    for day in range(2, 2 + duopoly.COOP_PATIENCE):       # the rival stays at $18: below halfway
        state["rival_last_price"] = 18.0
        last = duopoly._coop_price(day, state, 18.0)
    assert last is None and state["coop_broken"]


def test_cooperation_matches_a_rival_between_halfway_and_the_offer():
    state = _state(own_paths=[[18.0] * 100], rival_paths=[[18.0] * 100], rival_cap_days=[True] * 100)
    offer = duopoly._coop_price(1, state, 18.0)
    state["rival_last_price"] = offer - 3.0                # the other copy's (slightly lower) offer
    assert duopoly._coop_price(2, state, 18.0) == pytest.approx(offer - 3.0)


def test_cooperative_seasons_must_earn_at_least_best_reply():
    state = _state(coop_held_last=True, coop_revs=[900.0, 950.0], br_revs=[1200.0, 1250.0], last_season_seen=7)
    assert not duopoly._coop_allowed(state)
    state["last_season_seen"] = 10                         # retry every 10th season
    assert duopoly._coop_allowed(state)


@pytest.mark.skipif(not (os.path.exists(os.path.join(ROOT, "analytics", "demand_fits_v6.7.json"))
                         and os.path.exists(os.path.join(ROOT, "analytics", "experiments", "duopoly_v67.py"))),
                    reason="needs analytics/demand_fits_v6.7.json and analytics/experiments/duopoly_v67.py")
def test_selfplay_escapes_the_floor_war():
    """data/v6.7 3DkTAv: two copies of v6.7 sit at $18; two copies of v6.8 earn clearly more (same buyers)."""
    import selfplay
    fits = json.load(open(os.path.join(ROOT, "analytics", "demand_fits_v6.7.json")))
    theta = np.array(fits["day 2/AirborneTody_rank_7.csv/3DkTAv"]["theta"])

    def late_revenue(path):                               # 3 seeds x 25 seasons; seeds differ a lot in when
        rev = []                                          # cooperation forms
        for seed in (1, 2, 3):
            out = selfplay.play((path, path, theta, 25, seed))
            rev += [(r["rev_a"] + r["rev_b"]) / 2 for r in out if r["season"] > 5]
        return np.mean(rev)

    v68 = late_revenue(os.path.join(ROOT, "duopoly.py"))
    v67 = late_revenue(os.path.join(ROOT, "analytics", "experiments", "duopoly_v67.py"))
    assert v68 > 1.25 * v67
