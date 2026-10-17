"""
run_benchmark.py
----------------
Fast automated benchmark suite for testing duopoly.py.
Runs full 100-season simulations against:
1. Copycat Competitor (mirrors our price)
2. Historical FuturisticSalamander (from CSV)
3. Historical UncoveredElephant (from CSV)
4. Historical ZealousMosquito (from CSV)
5. Our Old v1.0 Baseline Algorithm (head-to-head)
6. Aggressive Undercutter ($18-$38)
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Optional

# Ensure parent and analytics directory are on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import duopoly
from simulation_engine import (
    DuopolySimulator,
    CopycatCompetitor,
    AggressiveDumperCompetitor,
    BoundedCompetitor,
    OldV1Competitor,
    HistoricalCSVCompetitor,
    MarketDemandModel
)
from data_loader import discover_datasets, load_competition_file

def run_all_benchmarks(csv_path: Optional[str] = None):
    print("\n" + "="*80)
    print(" DPC ALGORITHM BENCHMARK SUITE: EVALUATING duopoly.py")
    print("="*80)

    # Automatically discover CSV if not explicitly passed
    if csv_path is None or not os.path.exists(csv_path):
        datasets = discover_datasets()
        if datasets:
            csv_path = list(datasets.values())[0]['path']
            print(f"Auto-detected dataset: {csv_path}")

    # 1. Initialize Simulator
    market = MarketDemandModel(seed=42)
    simulator = DuopolySimulator(algorithm_func=duopoly.p, market_demand=market)

    results = []

    # Scenario A: Copycat Competitor
    print("\n[1/6] Simulating against COPYCAT Competitor (100 Seasons)...")
    copycat = CopycatCompetitor(initial_price=65.0, undercut=0.0, noise=0.5)
    res_copycat = simulator.run(copycat, n_seasons=100, competitor_name="Copycat")
    res_copycat.print_summary()
    results.append(res_copycat.summary())

    # Scenario B: Old v1.0 Baseline Algorithm (Head-to-Head)
    print("\n[2/6] Simulating against OLD v1.0 Algorithm (100 Seasons)...")
    old_v1 = OldV1Competitor()
    res_v1 = simulator.run(old_v1, n_seasons=100, competitor_name="Old v1.0 Baseline")
    res_v1.print_summary()
    results.append(res_v1.summary())

    # Scenario C: Aggressive Dumper ($18-$38)
    print("\n[3/6] Simulating against Aggressive Dumper ($18-$38, 100 Seasons)...")
    dumper = AggressiveDumperCompetitor(low=20.0, high=38.0)
    res_dumper = simulator.run(dumper, n_seasons=100, competitor_name="Aggressive Dumper ($18-$38)")
    res_dumper.print_summary()
    results.append(res_dumper.summary())

    # Scenario D: Bounded Mid-Market ($40-$70)
    print("\n[4/6] Simulating against Bounded Competitor ($40-$70, 100 Seasons)...")
    bounded = BoundedCompetitor(mean_price=55.0, spread=12.0)
    res_bounded = simulator.run(bounded, n_seasons=100, competitor_name="Bounded Competitor ($40-$70)")
    res_bounded.print_summary()
    results.append(res_bounded.summary())

    # Scenario E & F: Historical CSV Competitors
    if os.path.exists(csv_path):
        df_csv = pd.read_csv(csv_path)
        print("\nLoaded historical competition data successfully.")

        # Replay ZealousMosquito (Tournament 3XkNYY)
        print("\n[5/6] Simulating against Historical ZEALOUS MOSQUITO (Tournament 3XkNYY)...")
        hist_zm = HistoricalCSVCompetitor(df_csv, competition_id="3XkNYY")
        res_zm = simulator.run(hist_zm, n_seasons=100, competitor_name="Hist: ZealousMosquito (3XkNYY)")
        res_zm.print_summary()
        results.append(res_zm.summary())

        # Replay UncoveredElephant (Tournament 3PTJwD)
        print("\n[6/6] Simulating against Historical UNCOVERED ELEPHANT (Tournament 3PTJwD)...")
        hist_ue = HistoricalCSVCompetitor(df_csv, competition_id="3PTJwD")
        res_ue = simulator.run(hist_ue, n_seasons=100, competitor_name="Hist: UncoveredElephant (3PTJwD)")
        res_ue.print_summary()
        results.append(res_ue.summary())
    else:
        print(f"\nNote: CSV file not found at {csv_path}. Skipping historical replay.")

    # Grand Scorecard Table
    print("\n" + "="*105)
    print(" EXECUTIVE BENCHMARK SCORECARD: duopoly.py PERFORMANCE ACROSS ALL SCENARIOS")
    print(" (Official DPC Tournament Metric: Total Cumulative Revenue across all 100 Seasons)")
    print("="*105)
    summary_df = pd.DataFrame(results)[[
        'competitor', 'tournament_result', 'our_total_revenue', 'rival_total_revenue', 
        'revenue_diff', 'our_load_factor_pct', 'our_avg_price', 'rival_avg_price', 'season_win_rate_pct'
    ]]
    summary_df.columns = [
        'Competitor Scenario', 'Tournament Result', 'Our Total Rev', 'Rival Total Rev', 
        'Revenue Lead', 'Our Load %', 'Our Avg Price', 'Rival Avg Price', 'Season Win%'
    ]
    summary_df['Our Total Rev'] = summary_df['Our Total Rev'].apply(lambda x: f"${x:,.0f}")
    summary_df['Rival Total Rev'] = summary_df['Rival Total Rev'].apply(lambda x: f"${x:,.0f}")
    summary_df['Revenue Lead'] = summary_df['Revenue Lead'].apply(lambda x: f"${x:+,.0f}")
    summary_df['Our Load %'] = summary_df['Our Load %'].apply(lambda x: f"{x}%")
    summary_df['Our Avg Price'] = summary_df['Our Avg Price'].apply(lambda x: f"${x:.2f}")
    summary_df['Rival Avg Price'] = summary_df['Rival Avg Price'].apply(lambda x: f"${x:.2f}")
    summary_df['Season Win%'] = summary_df['Season Win%'].apply(lambda x: f"{x}%")

    print(summary_df.to_string(index=False))
    print("="*105)
    print(" * Note: Winner is determined strictly by Cumulative Total Revenue across all 100 seasons.\n")

    return results

if __name__ == '__main__':
    run_all_benchmarks()
