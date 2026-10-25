"""
test_data_integrity.py
-----------------------
Unit tests verifying the integrity of the historical tournament dataset,
computations of total revenue, seat sales, and data_loader functions.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure root and analytics are importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ANALYTICS_DIR = os.path.join(ROOT_DIR, 'analytics')
sys.path.append(ROOT_DIR)
sys.path.append(ANALYTICS_DIR)

from data_loader import (
    discover_datasets,
    load_competition_file,
    get_grouped_competitors,
    get_historical_summary
)


@pytest.fixture(scope="module")
def dataset_df():
    """Loads the primary v1 competition dataset."""
    datasets = discover_datasets()
    assert len(datasets) > 0, "No competition datasets discovered in data/."
    v1_matches = [d['path'] for d in datasets.values() if d['version'] == 'v1']
    target_path = v1_matches[0] if v1_matches else list(datasets.values())[0]['path']
    df = load_competition_file(target_path)
    return df


def test_csv_structure_and_row_count(dataset_df):
    """Verify standard schema and exact row count."""
    expected_cols = [
        'competition_id', 'selling_season', 'selling_period',
        'competitor_id', 'price_competitor', 'price',
        'demand', 'competitor_has_capacity'
    ]
    for col in expected_cols:
        assert col in dataset_df.columns, f"Missing expected column: {col}"

    assert len(dataset_df) == 60000, f"Expected 60,000 rows, found {len(dataset_df)}"
    assert dataset_df['competition_id'].nunique() == 6, "Expected 6 unique competitions."
    assert dataset_df['selling_season'].nunique() == 100, "Expected 100 seasons."
    assert dataset_df['selling_period'].nunique() == 100, "Expected 100 periods per season."


def test_csv_total_revenue_accounting(dataset_df):
    """Verify total revenue matches exact sum of price * demand across all 60,000 periods."""
    raw_revenue = (dataset_df['price'] * dataset_df['demand']).sum()
    loaded_revenue = dataset_df['revenue'].sum()

    assert raw_revenue == 336492.0, f"Expected exact historical revenue $336,492.00, got {raw_revenue}"
    assert loaded_revenue == raw_revenue, "Computed revenue column does not match raw product."


def test_individual_tournament_revenues(dataset_df):
    """Verify ground-truth revenue and sales per tournament run."""
    expected_tournaments = {
        '3DRpAU': {'competitor': 'FuturisticSalamander', 'revenue': 0.0, 'sales': 0},
        '3PTJwD': {'competitor': 'UncoveredElephant', 'revenue': 132877.0, 'sales': 2041},
        '3XkNYY': {'competitor': 'ZealousMosquito', 'revenue': 0.0, 'sales': 0},
        '3n5PCY': {'competitor': 'ZealousMosquito', 'revenue': 0.0, 'sales': 0},
        'D9Le8X': {'competitor': 'FuturisticSalamander', 'revenue': 203615.0, 'sales': 2818},
        'HhTiQz': {'competitor': 'UncoveredElephant', 'revenue': 0.0, 'sales': 0},
    }

    for comp_id, exp in expected_tournaments.items():
        sub = dataset_df[dataset_df['competition_id'] == comp_id]
        actual_rev = (sub['price'] * sub['demand']).sum()
        actual_sales = sub['demand'].sum()
        actual_comp = sub['competitor_id'].iloc[0]

        assert actual_comp == exp['competitor'], f"Competition {comp_id} rival mismatch: {actual_comp} vs {exp['competitor']}"
        assert actual_rev == exp['revenue'], f"Competition {comp_id} revenue mismatch: {actual_rev} vs {exp['revenue']}"
        assert actual_sales == exp['sales'], f"Competition {comp_id} sales mismatch: {actual_sales} vs {exp['sales']}"


def test_data_loader_historical_summary(dataset_df):
    """Verify get_historical_summary outputs exact values matching raw dataframe."""
    for comp_id in dataset_df['competition_id'].unique():
        summary = get_historical_summary(dataset_df, comp_id)
        sub = dataset_df[dataset_df['competition_id'] == comp_id]
        
        raw_rev = (sub['price'] * sub['demand']).sum()
        raw_sales = int(sub['demand'].sum())

        assert summary['total_revenue'] == raw_rev, f"Summary revenue discrepancy for {comp_id}"
        assert summary['total_sales'] == raw_sales, f"Summary sales discrepancy for {comp_id}"
        assert summary['total_seasons'] == 100, f"Expected 100 seasons in {comp_id}"


def test_grouped_competitors_structure(dataset_df):
    """Verify that each competitor has exactly two simulation runs."""
    grouped = get_grouped_competitors(dataset_df)
    assert set(grouped.keys()) == {'FuturisticSalamander', 'UncoveredElephant', 'ZealousMosquito'}

    for rival, sims in grouped.items():
        assert len(sims) == 2, f"Competitor {rival} expected 2 simulations, got {len(sims)}"
        assert sims[0]['sim_num'] == 1
        assert sims[1]['sim_num'] == 2
        assert sims[0]['seasons'] == 100
        assert sims[1]['seasons'] == 100


def test_historical_capacity_limits(dataset_df):
    """Verify that in no season did our algorithm exceed the 80-seat capacity limit."""
    season_sales = dataset_df.groupby(['competition_id', 'selling_season'])['demand'].sum()
    assert (season_sales <= 80).all(), "Found season where historical sales exceeded 80 seats capacity limit."
