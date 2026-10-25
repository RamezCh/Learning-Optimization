"""
analytics/benchmark_3regime_models.py
------------------------------------
Head-to-head evaluation across 3 distinct macroeconomic market regimes:
- Regime 1: Low Demand Market (Arrivals ~68, Valuations $20-$48, e.g. v5.1 / MLR week)
- Regime 2: Average Demand Market (Arrivals ~95, Valuations $35-$92, Standard balanced market like v3/v4)
- Regime 3: High Demand Surge (Arrivals ~135, Valuations $25-$78, High volume surge like v5)

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

class LowDemandMarket(MarketDemandModel):
    """Severe low-demand shock: ~68 total arrivals, high price elasticity, low WTP."""
    def __init__(self, seed=42):
        super().__init__(avg_arrivals=68, arrival_convexity=1.3, price_sensitivity=0.15, seed=seed)
    def sample_customer_wtp(self, n, t):
        return self.rng.uniform(20.0, 48.0, size=n)

class AverageDemandMarket(MarketDemandModel):
    """Standard balanced market: ~95 total arrivals, balanced leisure/business mix."""
    def __init__(self, seed=42):
        super().__init__(avg_arrivals=95, arrival_convexity=1.8, price_sensitivity=0.07, seed=seed)
    def sample_customer_wtp(self, n, t):
        return self.rng.uniform(35.0, 92.0, size=n)

class HighDemandMarket(MarketDemandModel):
    """High volume arrival surge: ~135 total arrivals, convex late surge, low WTP cap."""
    def __init__(self, seed=42):
        super().__init__(avg_arrivals=135, arrival_convexity=2.0, price_sensitivity=0.11, seed=seed)
    def sample_customer_wtp(self, n, t):
        return self.rng.uniform(25.0, 78.0, size=n)

def run_3regime_benchmark():
    print("="*85)
    print(" HEAD-TO-HEAD BENCHMARK ACROSS 3 MACROECONOMIC REGIMES")
    print("="*85)

    regimes = {
        "Regime 1: Low Demand (~68 arrivals, $20-$48 WTP)": LowDemandMarket,
        "Regime 2: Average Demand (~95 arrivals, $35-$92 WTP)": AverageDemandMarket,
        "Regime 3: High Demand Surge (~135 arrivals, $25-$78 WTP)": HighDemandMarket,
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
