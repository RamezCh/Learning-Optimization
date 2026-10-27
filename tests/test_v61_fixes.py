"""
test_v61_fixes.py
-----------------
Regression tests for the v6.1 fixes:
1. No phantom 0-sale season is recorded at the start of each new season.
2. The in-season low-demand test does not fire on normal, sparse early-season sales.
3. Monopoly price falls as the required burn rate rises (and is not stuck at the floor).
4. The demand regression uses full-rank features and the identified slopes B and C.
5. The simulator honours avg_arrivals.
"""

import os
import sys

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'analytics'))

import duopoly  # noqa: E402
from simulation_engine import MarketDemandModel  # noqa: E402


def _play_season(season, info, demand_pattern, our_price=45.0, rival_price=50.0):
    """Feeds one season (periods 1..101) with a fixed demand pattern."""
    prices, demands = [], []
    for t in range(1, 102):
        hist_p = None if t == 1 else np.array([prices, [rival_price] * len(prices)])
        hist_d = None if t == 1 else np.array(demands)
        _, info = duopoly.p(season, t, hist_p, hist_d, True, info)
        if t <= 100:
            prices.append(our_price)
            demands.append(demand_pattern(t))
    return info


def test_no_phantom_season_and_no_permanent_low_latch():
    good = lambda t: 1 if t % 5 != 0 else 0   # 80 sales/season  # noqa: E731
    info = None
    for season in (1, 2, 3):
        info = _play_season(season, info, good)
    assert info['season_sales_history'] == [80, 80, 80]
    assert info['consecutive_low_sales'] == 0
    assert info['detected_regime'] != 'low_demand'


def test_one_bad_season_counts_once_and_unlatches():
    info = _play_season(1, None, lambda t: 1 if t % 4 == 0 else 0)   # 25 sales
    assert info['consecutive_low_sales'] == 1
    assert info['detected_regime'] == 'low_demand'
    info = _play_season(2, info, lambda t: 1 if t % 5 != 0 else 0)   # 80 sales
    assert info['season_sales_history'] == [25, 80]
    assert info['consecutive_low_sales'] == 0
    assert info['detected_regime'] != 'low_demand'


def test_sprt_ignores_short_zero_streaks():
    # 6 zero days in a row while cheaper than the rival, then a sale: normal early-season noise.
    state = {'low_demand_llr': 0.0, 'low_demand_obs': 0}
    for t in range(2, 8):
        duopoly._update_low_demand_sprt(t, 0.0, state)
    duopoly._update_low_demand_sprt(8, 1.0, state)
    assert not state.get('low_demand_in_season', False)


def test_sprt_fires_on_long_drought():
    state = {'low_demand_llr': 0.0, 'low_demand_obs': 0}
    for t in range(2, 30):
        duopoly._update_low_demand_sprt(t, 0.0, state)
    assert state['low_demand_in_season']


def test_monopoly_price_decreases_with_burn_rate():
    for regime in ('low_demand', 'average_demand', 'high_demand'):
        st = {'detected_regime': regime}
        prices = [duopoly._calculate_monopoly_price(b, 20, st) for b in (0.2, 0.6, 1.0, 1.5)]
        assert all(a >= b for a, b in zip(prices, prices[1:])), (regime, prices)
        assert prices[0] > prices[-1], (regime, prices)


def test_regression_is_full_rank_and_recovers_slopes():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(600):
        p_our = rng.uniform(30, 70)
        p_riv = rng.uniform(30, 70)
        tau = rng.uniform(0, 1)
        demand = 1.0 - 0.02 * p_our + 0.01 * p_riv + 0.4 * tau + rng.normal(0, 0.05)
        rows.append([p_our, p_riv - p_our, tau, demand])
    features = np.array(rows)[:, :3]
    assert np.linalg.matrix_rank(features) == 3

    state = {'training_history': rows}
    duopoly._fit_online_regression(state)
    assert abs(state['mlr_B'] - (-0.02)) < 0.003     # own-price slope
    assert abs(state['mlr_C'] - 0.01) < 0.003        # rival-price slope


def test_simulator_honours_avg_arrivals():
    for target in (68.0, 92.0, 135.0):
        m = MarketDemandModel(avg_arrivals=target)
        assert abs(sum(m.get_arrival_rate(t) for t in range(1, 101)) - target) < 1e-6


def test_price_formula_uses_yesterdays_rival_price():
    """A2: the closed-form price uses yesterday's rival price; guardrails keep the 5-day average."""
    seen = {}
    original = duopoly._solve_closed_form_optimal_price

    def spy(period, gap, burn, rival_price, state):
        seen['rival_price'] = rival_price
        return original(period, gap, burn, rival_price, state)

    duopoly._solve_closed_form_optimal_price = spy
    try:
        state = duopoly._initialize_or_get_state(None, 1)
        state['rival_recent_prices'] = [40.0, 40.0, 40.0, 40.0, 50.0]
        state['rival_last_price'] = 50.0
        duopoly._calculate_duopoly_price(30, 0.0, 0.8, 50, state)
    finally:
        duopoly._solve_closed_form_optimal_price = original
    assert seen['rival_price'] == 50.0          # yesterday, not the 5-day average of 42
    assert duopoly._get_recent_rival_average(state, 50.0) == 42.0


def _season_with_bend(bend, total=80):
    """Daily sales whose cumulative share follows (1 - bend)*tau + bend*tau^2."""
    tau = np.arange(101) / 99.0
    cum = total * ((1 - bend) * tau + bend * tau ** 2)
    cum = np.clip(np.round(cum), 0, total)
    return np.diff(cum)[:100].astype(int)


def test_season_bend_is_recovered():
    for bend in (0.0, 0.3, 0.6):
        assert abs(duopoly._fit_season_target_bend(_season_with_bend(bend)) - bend) < 0.05


def test_target_bend_defaults_then_learns_from_good_seasons():
    state = duopoly._initialize_or_get_state(None, 1)
    assert duopoly._current_target_bend(state) == duopoly.TARGET_C_DEFAULT      # nothing learned yet
    state['detected_regime'] = 'high_demand'
    state['target_c_obs'] = [0.8]
    expected = (3 * 0.60 + 0.8) / 4                                            # prior counts as 3 seasons
    assert abs(duopoly._current_target_bend(state) - expected) < 1e-9
    state['target_c_obs'] = [2.0] * 50                                          # clipped to the safety limit
    assert duopoly._current_target_bend(state) == duopoly.TARGET_C_MAX


def test_end_of_season_learns_only_from_full_seasons():
    info = duopoly._initialize_or_get_state(None, 1)
    prices = np.array([[50.0] * 100, [55.0] * 100])
    full = _season_with_bend(0.6)                                               # 80 seats
    _, info = duopoly.p(1, 101, prices, full, True, info)
    assert len(info['target_c_obs']) == 1 and info['target_c_now'] != duopoly.TARGET_C_DEFAULT
    info['last_season_seen'] = 2
    weak = np.array([1 if t % 3 == 0 else 0 for t in range(100)])              # 34 seats
    _, info = duopoly.p(2, 101, prices, weak, True, info)
    assert len(info['target_c_obs']) == 1                                        # not learned from
