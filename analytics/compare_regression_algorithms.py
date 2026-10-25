"""
compare_regression_algorithms.py
--------------------------------
Head-to-head simulation and benchmarking of:
1. v6.0 Unified MLR (duopoly.py)
2. Baseline v1.0 Algorithm (OldV1Competitor)

Evaluates on:
- Copycat Competitor
- Aggressive Dumper ($18-$38)
- Bounded Competitor ($40-$70)
- Historical ZealousMosquito (3XkNYY)
- Historical UncoveredElephant (3PTJwD)

Generates comparison figures:
- `imgs/algorithm_variant_comparison.png`
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure paths
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

def run_head_to_head_comparison():
    print("="*80)
    print(" HEAD-TO-HEAD BENCHMARK: v6.0 UNIFIED MLR vs BASELINE v1.0")
    print("="*80)

    old_v1_agent = OldV1Competitor()

    def v1_pricing_func(season, period, p_hist, d_hist, comp_cap, info):
        old_v1_agent.info_dump = info
        return old_v1_agent.p(season, period, p_hist, d_hist, comp_cap)

    algorithms = {
        "v6.0 Unified MLR": duopoly.p,
        "Baseline v1.0": v1_pricing_func
    }

    # Prepare rivals
    scenarios = {
        "Copycat": CopycatCompetitor(initial_price=65.0, undercut=0.0, noise=0.5),
        "Aggressive Dumper ($18-$38)": AggressiveDumperCompetitor(low=20.0, high=38.0),
        "Bounded ($40-$70)": BoundedCompetitor(mean_price=55.0, spread=12.0)
    }

    csv_candidates = [
        'data/v1/2026-10-14_day2_duopoly_competition_details.csv',
        '../data/v1/2026-10-14_day2_duopoly_competition_details.csv'
    ]
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)

    if csv_path:
        df_csv = pd.read_csv(csv_path)
        scenarios["Hist: ZealousMosquito"] = HistoricalCSVCompetitor(df_csv, "3XkNYY")
        scenarios["Hist: UncoveredElephant"] = HistoricalCSVCompetitor(df_csv, "3PTJwD")

    records = []

    for algo_name, algo_func in algorithms.items():
        print(f"\n--- Testing Algorithm: {algo_name} ---")
        for sc_name, rival_agent in scenarios.items():
            market = MarketDemandModel(seed=42)
            sim = DuopolySimulator(algorithm_func=algo_func, market_demand=market)
            res = sim.run(rival_agent, n_seasons=100, competitor_name=sc_name)
            s = res.summary()

            records.append({
                "Algorithm": algo_name,
                "Scenario": sc_name,
                "Our_Revenue_k": s["our_total_revenue"] / 1000.0,
                "Rival_Revenue_k": s["rival_total_revenue"] / 1000.0,
                "Win_Rate": s["season_win_rate_pct"],
                "Load_Factor": s["our_load_factor_pct"],
                "Avg_Price": s["our_avg_price"]
            })
            print(f"  {sc_name:<25}: Rev = ${s['our_total_revenue']/1000.0:5.1f}k vs Rival ${s['rival_total_revenue']/1000.0:5.1f}k | Win Rate = {s['season_win_rate_pct']:.0f}%")

    df_res = pd.DataFrame(records)

    # -------------------------------------------------------------
    # Generate Chart: algorithm_variant_comparison.png
    # -------------------------------------------------------------
    os.makedirs('imgs', exist_ok=True)
    chart_path = 'imgs/algorithm_variant_comparison.png'

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=200)
    fig.patch.set_facecolor('#ffffff')

    scenarios_list = list(scenarios.keys())
    x = np.arange(len(scenarios_list))
    width = 0.35

    # Panel 1: Revenue by Scenario
    ax1 = axes[0]
    ax1.set_facecolor('#f8fafc')
    ax1.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')

    v6_rev = df_res[df_res['Algorithm'] == "v6.0 Unified MLR"]['Our_Revenue_k'].values
    v1_rev = df_res[df_res['Algorithm'] == "Baseline v1.0"]['Our_Revenue_k'].values

    b1 = ax1.bar(x - width/2, v6_rev, width, label='v6.0 Unified MLR (duopoly.py)', color='#10b981', edgecolor='#047857')
    b2 = ax1.bar(x + width/2, v1_rev, width, label='Baseline v1.0 Heuristic', color='#94a3b8', edgecolor='#475569')

    ax1.set_title('Panel 1: Total Revenue ($k) Across 100 Seasons per Scenario', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios_list, rotation=15, ha='right', fontsize=9.5)
    ax1.set_ylabel('Total Revenue ($k)', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True)

    for bar in b1:
        h = bar.get_height()
        ax1.annotate(f'${h:.1f}k', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#047857')
    for bar in b2:
        h = bar.get_height()
        ax1.annotate(f'${h:.1f}k', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#475569')

    # Panel 2: Win Rate by Scenario
    ax2 = axes[1]
    ax2.set_facecolor('#f8fafc')
    ax2.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')

    v6_wins = df_res[df_res['Algorithm'] == "v6.0 Unified MLR"]['Win_Rate'].values
    v1_wins = df_res[df_res['Algorithm'] == "Baseline v1.0"]['Win_Rate'].values

    bw1 = ax2.bar(x - width/2, v6_wins, width, label='v6.0 Unified MLR (duopoly.py)', color='#3b82f6', edgecolor='#1d4ed8')
    bw2 = ax2.bar(x + width/2, v1_wins, width, label='Baseline v1.0 Heuristic', color='#cbd5e1', edgecolor='#64748b')

    ax2.set_title('Panel 2: Season Win Rate (%) Across 100 Seasons per Scenario', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios_list, rotation=15, ha='right', fontsize=9.5)
    ax2.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 115)
    ax2.legend(loc='upper right', frameon=True)

    for bar in bw1:
        h = bar.get_height()
        ax2.annotate(f'{h:.0f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1d4ed8')
    for bar in bw2:
        h = bar.get_height()
        ax2.annotate(f'{h:.0f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#475569')

    plt.tight_layout()
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison chart to: {chart_path}")

    # Aggregate summary
    print("\n" + "="*80)
    print(" AGGREGATE SUMMARY ACROSS ALL 5 SCENARIOS (500 SEASONS EACH)")
    print("="*80)
    agg = df_res.groupby('Algorithm').agg({
        'Our_Revenue_k': 'sum',
        'Win_Rate': 'mean',
        'Load_Factor': 'mean'
    }).rename(columns={'Our_Revenue_k': 'Total_Revenue'}).sort_values(by='Total_Revenue', ascending=False)
    print(agg)
    print("="*80)

    return df_res

if __name__ == '__main__':
    run_head_to_head_comparison()
