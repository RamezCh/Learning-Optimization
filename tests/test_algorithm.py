"""
test_algorithm.py
-----------------
Unit tests verifying the pricing function duopoly.p:
- Contract signature & return types
- Handling None inputs on period 1
- Handling period 101 end-of-season summary
- Strict price bounds ($25 to $100)
- Multi-period state persistence in information_dump
"""

import sys
import os
import pytest
import numpy as np

# Ensure root is importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import duopoly


def test_pricing_contract_signature():
    """Verify function signature and return type (float, dict)."""
    price, info = duopoly.p(
        current_selling_season=1,
        selling_period_in_current_season=1,
        prices_historical_in_current_season=None,
        demand_historical_in_current_season=None,
        competitor_has_capacity_current_period_in_current_season=True,
        information_dump=None
    )

    assert isinstance(price, (float, int, np.floating)), f"Returned price must be float, got {type(price)}"
    assert isinstance(info, dict), f"Returned information_dump must be dict, got {type(info)}"
    assert 25.0 <= price <= 100.0, f"Period 1 price {price} out of bounds [25, 100]"


def test_information_dump_initial_keys():
    """Verify standard keys are initialized in information_dump."""
    _, info = duopoly.p(1, 1, None, None, True, None)
    expected_keys = [
        'total_capacity', 'rival_season_medians', 'rival_ema_price',
        'rival_profile', 'season_sales_history', 'price_demand_stats'
    ]
    for k in expected_keys:
        assert k in info, f"Missing key in information_dump: {k}"


def test_multi_period_decision_sequence():
    """Verify stepping through multiple periods within a season."""
    info = None
    our_prices = []
    rival_prices = []
    our_demands = []

    for t in range(1, 15):
        if t == 1:
            prices_hist = None
            demand_hist = None
        else:
            prices_hist = np.array([our_prices, rival_prices])
            demand_hist = np.array(our_demands)

        price, info = duopoly.p(
            current_selling_season=1,
            selling_period_in_current_season=t,
            prices_historical_in_current_season=prices_hist,
            demand_historical_in_current_season=demand_hist,
            competitor_has_capacity_current_period_in_current_season=True,
            information_dump=info
        )

        assert 25.0 <= price <= 100.0, f"Period {t} price {price} violates bounds"
        our_prices.append(price)
        rival_prices.append(55.0)  # Simulated rival at $55
        our_demands.append(1 if t % 2 == 0 else 0)

    assert len(our_prices) == 14
    assert info['rival_ema_price'] is not None


def test_period_101_end_of_season_feedback():
    """Verify period 101 executes without throwing errors and updates medians."""
    our_prices = [60.0] * 100
    rival_prices = [55.0] * 100
    our_demands = [1] * 80 + [0] * 20

    prices_hist = np.array([our_prices, rival_prices])
    demand_hist = np.array(our_demands)

    _, info = duopoly.p(1, 1, None, None, True, None)
    
    price_101, info_101 = duopoly.p(
        current_selling_season=1,
        selling_period_in_current_season=101,
        prices_historical_in_current_season=prices_hist,
        demand_historical_in_current_season=demand_hist,
        competitor_has_capacity_current_period_in_current_season=True,
        information_dump=info
    )

    assert len(info_101['rival_season_medians']) >= 1
    assert info_101['rival_season_medians'][-1] == 55.0
    assert len(info_101['season_sales_history']) >= 1


def test_monopoly_mode_pricing():
    """When competitor_has_capacity is False, monopoly price should stay bounded."""
    our_prices = [60.0] * 50
    rival_prices = [30.0] * 50
    our_demands = [0] * 50

    prices_hist = np.array([our_prices, rival_prices])
    demand_hist = np.array(our_demands)

    price, _ = duopoly.p(
        current_selling_season=1,
        selling_period_in_current_season=51,
        prices_historical_in_current_season=prices_hist,
        demand_historical_in_current_season=demand_hist,
        competitor_has_capacity_current_period_in_current_season=False,  # Rival sold out!
        information_dump=None
    )

    assert 25.0 <= price <= 100.0, f"Monopoly price {price} out of bounds"
