"""
test_regression_variants.py
---------------------------
Unit tests verifying duopoly_mlr.py (Rule-Governed Multiple Linear Regression):
- Contract signature & return types (float, dict)
- Handling None inputs on period 1
- Handling period 101 end-of-season summary
- Strict price bounds ($25 to $175)
- Multi-period state persistence
"""

import sys
import os
import pytest
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import duopoly_mlr

def test_contract_signature():
    price, info = duopoly_mlr.p(
        current_selling_season=1,
        selling_period_in_current_season=1,
        prices_historical_in_current_season=None,
        demand_historical_in_current_season=None,
        competitor_has_capacity_current_period_in_current_season=True,
        information_dump=None
    )
    assert isinstance(price, (float, int, np.floating)), "Returned price must be float"
    assert isinstance(info, dict), "Returned info must be dict"
    assert 25.0 <= price <= 175.0, f"Period 1 price {price} out of bounds"

def test_multi_period_flow():
    info = None
    prices_hist = []
    demand_hist = []
    
    for t in range(1, 102):
        price, info = duopoly_mlr.p(
            current_selling_season=1,
            selling_period_in_current_season=t,
            prices_historical_in_current_season=prices_hist if t > 1 else None,
            demand_historical_in_current_season=demand_hist if t > 1 else None,
            competitor_has_capacity_current_period_in_current_season=True,
            information_dump=info
        )
        assert 25.0 <= price <= 175.0, f"Period {t} price out of bounds: {price}"
        assert isinstance(info, dict)
        prices_hist.append(50.0)
        demand_hist.append(1)
