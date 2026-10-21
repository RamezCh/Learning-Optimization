"""
compare_regression_algorithms.py
--------------------------------
Head-to-head simulation and benchmarking of:
1. Master v5.0 (duopoly.py)
2. Multiple Linear Regression Variant (duopoly_mlr.py)
3. Poisson / Logistic Variant (duopoly_poisson_logit.py)

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
import duopoly_mlr
import duopoly_poisson_logit

from simulation_engine import (
    DuopolySimulator,
    CopycatCompetitor,
    AggressiveDumperCompetitor,
    BoundedCompetitor,
    HistoricalCSVCompetitor,
    MarketDemandModel
)

def run_head_to_head_comparison():
    print("="*80)
    print(" HEAD-TO-HEAD BENCHMARK: MASTER v5.0 vs MLR vs POISSON/LOGISTIC")
    print("="*80)

    algorithms = {
        "Master (v5.0)": duopoly.p,
        "Multiple Linear Reg (MLR)": duopoly_mlr.p,
        "Poisson/Logistic Reg": duopoly_poisson_logit.p
    }

    # Prepare rivals
    scenarios = {
        "Copycat": CopycatCompetitor(initial_price=65.0, undercut=0.0, noise=0.5),
        "Aggressive Dumper ($18-$38)": AggressiveDumperCompetitor(low=20.0, high=38.0),
        "Bounded ($40-$70)": BoundedCompetitor(mean_price=55.0, spread=12.0)
    }

    csv_path = 'data/v1/2026-10-14_day2_duopoly_competition_details.csv'
    if os.path.exists(csv_path):
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
            
            summ = res.summary()
            records.append({
                "Algorithm": algo_name,
                "Scenario": sc_name,
                "Our Revenue ($k)": summ["our_total_revenue"] / 1000.0,
                "Rival Revenue ($k)": summ["rival_total_revenue"] / 1000.0,
                "Revenue Ratio": summ["our_total_revenue"] / max(1.0, summ["rival_total_revenue"]),
                "Win Rate (%)": summ["season_win_rate_pct"],
                "Our Load Factor (%)": summ["our_load_factor_pct"],
                "Our Mean Price ($)": summ["our_avg_price"]
            })
            print(f"  {sc_name:<25}: Rev = ${summ['our_total_revenue']/1000.0:.1f}k vs Rival ${summ['rival_total_revenue']/1000.0:.1f}k | Win Rate = {summ['season_win_rate_pct']:.0f}%")

    df_res = pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # Generate Visual Comparison Plot
    # -------------------------------------------------------------------------
    os.makedirs('imgs', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), dpi=200)
    fig.patch.set_facecolor('#ffffff')

    sc_list = list(scenarios.keys())
    x = np.arange(len(sc_list))
    width = 0.25

    colors = {
        "Master (v5.0)": '#0f766e',               # Teal
        "Multiple Linear Reg (MLR)": '#2563eb',   # Blue
        "Poisson/Logistic Reg": '#7c3aed'         # Purple
    }

    # Subplot 1: Total Revenue ($k)
    ax1 = axes[0]
    ax1.set_facecolor('#f8fafc')
    ax1.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')

    for i, (algo_name, color) in enumerate(colors.items()):
        sub_df = df_res[df_res['Algorithm'] == algo_name]
        revs = [sub_df[sub_df['Scenario'] == sc]['Our Revenue ($k)'].values[0] for sc in sc_list]
        bars = ax1.bar(x + (i - 1) * width, revs, width, label=algo_name, color=color, alpha=0.9)
        for b in bars:
            h = b.get_height()
            ax1.annotate(f'${h:.0f}k', xy=(b.get_x() + b.get_width()/2, h),
                         xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax1.set_title('100-Season Revenue Comparison ($k)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sc_list, rotation=15, ha='right', fontsize=9)
    ax1.set_ylabel('Total Revenue ($k)', fontsize=10, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, fontsize=9)

    # Subplot 2: Win Rate (%)
    ax2 = axes[1]
    ax2.set_facecolor('#f8fafc')
    ax2.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')

    for i, (algo_name, color) in enumerate(colors.items()):
        sub_df = df_res[df_res['Algorithm'] == algo_name]
        win_rates = [sub_df[sub_df['Scenario'] == sc]['Win Rate (%)'].values[0] for sc in sc_list]
        bars = ax2.bar(x + (i - 1) * width, win_rates, width, label=algo_name, color=color, alpha=0.9)
        for b in bars:
            h = b.get_height()
            ax2.annotate(f'{h:.0f}%', xy=(b.get_x() + b.get_width()/2, h),
                         xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax2.set_title('Win Rate Against Rival (%)', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sc_list, rotation=15, ha='right', fontsize=9)
    ax2.set_ylabel('Win Rate (%)', fontsize=10, fontweight='bold')
    ax2.set_ylim(0, 115)
    ax2.legend(loc='upper right', frameon=True, fontsize=9)

    plt.tight_layout()
    out_fig = 'imgs/algorithm_variant_comparison.png'
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"\nSaved comparison chart to: {out_fig}")

    # Summary table
    print("\n" + "="*80)
    print(" AGGREGATE SUMMARY ACROSS ALL 5 SCENARIOS (500 SEASONS EACH)")
    print("="*80)
    agg_table = df_res.groupby('Algorithm').agg(
        Total_Revenue=('Our Revenue ($k)', 'sum'),
        Mean_Win_Rate=('Win Rate (%)', 'mean'),
        Mean_Load_Factor=('Our Load Factor (%)', 'mean')
    ).reset_index()
    print(agg_table.to_string(index=False))
    print("="*80)

if __name__ == '__main__':
    run_head_to_head_comparison()
