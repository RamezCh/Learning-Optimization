"""
analytics/benchmark_3regime_models.py
------------------------------------
Head-to-head evaluation across 3 distinct macroeconomic market regimes:
- Regime 1: High WTP (Valuations $45-$115, Premium travelers like v1/v4)
- Regime 2: Medium WTP (Valuations $35-$92, Standard balanced market like v3)
- Regime 3: Low WTP / Volume Surge (Valuations $25-$78, High volume like v5)

Competitors:
- CopycatCompetitor
- AggressiveDumperCompetitor ($18-$38)
- BoundedCompetitor ($40-$70)
"""

import os
import sys
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ANALYTICS_DIR = os.path.join(ROOT_DIR, 'analytics')
sys.path.append(ROOT_DIR)
sys.path.append(ANALYTICS_DIR)

import duopoly
import duopoly_mlr

from simulation_engine import (
    DuopolySimulator,
    CopycatCompetitor,
    AggressiveDumperCompetitor,
    BoundedCompetitor,
    MarketDemandModel
)

class HighWTPMarket(MarketDemandModel):
    def __init__(self, seed=42):
        super().__init__(avg_arrivals=90, price_sensitivity=0.04, seed=seed)
    def sample_customer_wtp(self, n, t):
        return self.rng.uniform(45.0, 115.0, size=n)

class MediumWTPMarket(MarketDemandModel):
    def __init__(self, seed=42):
        super().__init__(avg_arrivals=95, price_sensitivity=0.07, seed=seed)
    def sample_customer_wtp(self, n, t):
        return self.rng.uniform(35.0, 92.0, size=n)

class LowWTPMarket(MarketDemandModel):
    def __init__(self, seed=42):
        super().__init__(avg_arrivals=130, arrival_convexity=2.0, price_sensitivity=0.12, seed=seed)
    def sample_customer_wtp(self, n, t):
        return self.rng.uniform(25.0, 78.0, size=n)

def run_3regime_benchmark():
    print("="*85)
    print(" HEAD-TO-HEAD BENCHMARK ACROSS 3 MACROECONOMIC REGIMES")
    print("="*85)

    regimes = {
        "Regime 1: High WTP ($45-$115)": HighWTPMarket,
        "Regime 2: Medium WTP ($35-$92)": MediumWTPMarket,
        "Regime 3: Low WTP Surge ($25-$78)": LowWTPMarket,
    }

    algorithms = {
        "Master (duopoly.py)": duopoly.p,
        "Rule-Based MLR (duopoly_mlr.py)": duopoly_mlr.p,
    }

    rivals = {
        "Copycat": CopycatCompetitor(initial_price=60.0, undercut=0.0, noise=0.5),
        "Aggressive Dumper": AggressiveDumperCompetitor(low=18.0, high=36.0),
        "Bounded ($40-$70)": BoundedCompetitor(mean_price=55.0, spread=12.0)
    }

    records = []

    for reg_name, market_cls in regimes.items():
        print(f"\n>>> Evaluating {reg_name} <<<")
        for algo_name, algo_func in algorithms.items():
            algo_rev = 0.0
            algo_wins = 0
            total_matches = 0
            
            for riv_name, riv_agent in rivals.items():
                market = market_cls(seed=123)
                sim = DuopolySimulator(algorithm_func=algo_func, market_demand=market)
                res = sim.run(riv_agent, n_seasons=15, competitor_name=riv_name)
                s = res.summary()

                algo_rev += s["our_total_revenue"]
                algo_wins += s["seasons_won"]
                total_matches += 15

            win_rate = (algo_wins / float(total_matches)) * 100.0
            avg_rev = algo_rev / float(total_matches)
            records.append({
                "Regime": reg_name,
                "Algorithm": algo_name,
                "Total Revenue ($k)": round(algo_rev / 1000.0, 2),
                "Rev / Season ($)": round(avg_rev, 2),
                "Win Rate (%)": round(win_rate, 1)
            })
            print(f"  {algo_name:<30}: Total Rev = ${algo_rev/1000.0:6.1f}k | Rev/Season = ${avg_rev:6.1f} | Win Rate = {win_rate:5.1f}%")

    df_summary = pd.DataFrame(records)
    print("\n" + "="*85)
    print(" OVERALL AGGREGATE SCORECARD")
    print("="*85)
    agg = df_summary.groupby("Algorithm").agg({
        "Total Revenue ($k)": "sum",
        "Rev / Season ($)": "mean",
        "Win Rate (%)": "mean"
    }).reset_index().sort_values(by="Total Revenue ($k)", ascending=False)
    print(agg.to_string(index=False))
    return df_summary

if __name__ == "__main__":
    run_3regime_benchmark()
