"""
test_simulation_engine.py
-------------------------
Unit tests verifying the simulation engine:
- MarketDemandModel capacity capping & choice logic
- Exact revenue accounting: sum(price * demand) == total_revenue
- Verification of official DPC tournament winning condition
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ANALYTICS_DIR = os.path.join(ROOT_DIR, 'analytics')
sys.path.append(ROOT_DIR)
sys.path.append(ANALYTICS_DIR)

import duopoly
from simulation_engine import (
    MarketDemandModel,
    DuopolySimulator,
    CopycatCompetitor,
    BoundedCompetitor
)


def test_market_demand_capacity_capping():
    """Verify that market demand never exceeds available seats."""
    market = MarketDemandModel(seed=42)

    # When capacity is 0, demand must be 0
    d1, d2 = market.step(t=50, price1=50.0, price2=50.0, cap1=0, cap2=0)
    assert d1 == 0 and d2 == 0

    # When cap1 is 1, d1 can be at most 1 regardless of how low the price is
    for _ in range(50):
        d1, d2 = market.step(t=95, price1=20.0, price2=90.0, cap1=1, cap2=0)
        assert d1 <= 1, f"Demand {d1} exceeded available capacity 1"
        assert d2 == 0


def test_market_demand_price_sensitivity():
    """Verify that significantly lower prices capture market share."""
    market = MarketDemandModel(seed=123)
    sales_cheap = 0
    sales_expensive = 0

    for _ in range(200):
        d_cheap, d_exp = market.step(t=50, price1=35.0, price2=85.0, cap1=80, cap2=80)
        sales_cheap += d_cheap
        sales_expensive += d_exp

    assert sales_cheap > sales_expensive * 3, f"Cheap airline should capture majority: {sales_cheap} vs {sales_expensive}"


def test_simulation_exact_revenue_accounting():
    """
    Verify that summary total revenue exactly matches the mathematical sum
    of period-by-period price * demand across all simulated seasons.
    """
    market = MarketDemandModel(seed=42)
    simulator = DuopolySimulator(algorithm_func=duopoly.p, market_demand=market)
    competitor = CopycatCompetitor(initial_price=60.0)

    res = simulator.run(competitor, n_seasons=10, competitor_name="Copycat")
    summary = res.summary()
    df = res.df

    # Manual period-by-period sum
    computed_our_total = (df['our_price'] * df['our_sales']).sum()
    computed_rival_total = (df['rival_price'] * df['rival_sales']).sum()
    computed_diff = computed_our_total - computed_rival_total

    assert np.isclose(summary['our_total_revenue'], computed_our_total), "Our total revenue does not match period sum."
    assert np.isclose(summary['rival_total_revenue'], computed_rival_total), "Rival total revenue does not match period sum."
    assert np.isclose(summary['revenue_diff'], computed_diff), "Revenue difference does not match subtraction."


def test_official_dPC_winning_rule_ruling():
    """Verify that tournament_result reflects highest total revenue across all seasons."""
    market = MarketDemandModel(seed=42)
    simulator = DuopolySimulator(algorithm_func=duopoly.p, market_demand=market)
    competitor = BoundedCompetitor(mean_price=55.0)

    res = simulator.run(competitor, n_seasons=10, competitor_name="Bounded")
    summary = res.summary()

    if summary['our_total_revenue'] > summary['rival_total_revenue']:
        assert summary['is_winner'] is True
        assert summary['tournament_result'] == "WINNER"
    else:
        assert summary['is_winner'] is False
        assert summary['tournament_result'] == "RUNNER-UP"
