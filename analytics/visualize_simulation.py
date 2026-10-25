"""
visualize_simulation.py
-----------------------
Generates visual comparison plots for duopoly.py (v2.0) simulations against:
- Copycat Competitor
- Historical Competitors (ZealousMosquito, UncoveredElephant)
- Baseline v1.0 Algorithm

Saves publication-quality figures:
1. `analytics/benchmark_comparison.png`: Bar chart of revenue & load factor
2. `analytics/price_trajectory_simulation.png`: Period-by-period price & inventory trajectories
"""

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Ensure paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import duopoly
from simulation_engine import (
    DuopolySimulator,
    CopycatCompetitor,
    BoundedCompetitor,
    OldV1Competitor,
    HistoricalCSVCompetitor,
    MarketDemandModel
)

def generate_visualizations():
    print("Generating simulation visualizations...")
    market = MarketDemandModel(seed=42)
    simulator = DuopolySimulator(algorithm_func=duopoly.p, market_demand=market)

    # 1. Run key scenarios
    scenarios = {
        "Copycat": CopycatCompetitor(initial_price=65.0, undercut=0.0),
        "ZealousMosquito (Hist)": None,
        "UncoveredElephant (Hist)": None,
        "Old v1.0 Baseline": OldV1Competitor(),
        "Bounded ($40-$70)": BoundedCompetitor()
    }

    csv_candidates = [
        'data/day2_duopoly_competition_details.csv',
        'data/v1/2026-10-14_day2_duopoly_competition_details.csv',
        '../data/v1/2026-10-14_day2_duopoly_competition_details.csv',
        '../data/day2_duopoly_competition_details.csv'
    ]
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)

    if csv_path and os.path.exists(csv_path):
        df_csv = pd.read_csv(csv_path)
        scenarios["ZealousMosquito (Hist)"] = HistoricalCSVCompetitor(df_csv, "3XkNYY")
        scenarios["UncoveredElephant (Hist)"] = HistoricalCSVCompetitor(df_csv, "3PTJwD")
    else:
        del scenarios["ZealousMosquito (Hist)"]
        del scenarios["UncoveredElephant (Hist)"]

    sim_results = {}
    summaries = []

    for name, agent in scenarios.items():
        res = simulator.run(agent, n_seasons=100, competitor_name=name)
        sim_results[name] = res
        summaries.append(res.summary())

    df_sum = pd.DataFrame(summaries)

    # -------------------------------------------------------------
    # Plot 1: Benchmark Comparison (Revenue & Load Factor)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#ffffff')

    labels = df_sum['competitor'].tolist()
    x = np.arange(len(labels))
    width = 0.35

    # Subplot A: Total Revenue
    ax1 = axes[0]
    ax1.set_facecolor('#f8f9fa')
    ax1.grid(True, linestyle='--', alpha=0.6, color='#dee2e6')
    r1 = ax1.bar(x - width/2, df_sum['our_total_revenue'] / 1000.0, width, label='Our Algorithm (v2.0)', color='#10b981')
    r2 = ax1.bar(x + width/2, df_sum['rival_total_revenue'] / 1000.0, width, label='Competitor', color='#ef4444')

    ax1.set_title('100-Season Total Revenue ($k)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax1.set_ylabel('Total Revenue ($k)')
    ax1.legend(loc='upper right', frameon=True)

    for bar in r1:
        h = bar.get_height()
        ax1.annotate(f'${h:.0f}k', xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar in r2:
        h = bar.get_height()
        ax1.annotate(f'${h:.0f}k', xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # Subplot B: Win Rate & Load Factor
    ax2 = axes[1]
    ax2.set_facecolor('#f8f9fa')
    ax2.grid(True, linestyle='--', alpha=0.6, color='#dee2e6')
    b1 = ax2.bar(x - width/2, df_sum['win_rate_pct'], width, label='Win Rate (%)', color='#3b82f6')
    b2 = ax2.bar(x + width/2, df_sum['our_load_factor_pct'], width, label='Our Load Factor (%)', color='#8b5cf6')

    ax2.set_title('Win Rate & Inventory Load Factor (%)', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax2.set_ylabel('Percentage (%)')
    ax2.set_ylim(0, 115)
    ax2.legend(loc='upper right', frameon=True)

    for bar in b1:
        h = bar.get_height()
        ax2.annotate(f'{h:.0f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar in b2:
        h = bar.get_height()
        ax2.annotate(f'{h:.0f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_path1 = 'analytics/benchmark_comparison.png'
    plt.savefig(out_path1, dpi=200)
    plt.close()
    print(f"Saved: {out_path1}")

    # -------------------------------------------------------------
    # Plot 2: Detailed Season Trajectories (Price & Remaining Inventory)
    # -------------------------------------------------------------
    # Take ZealousMosquito or Bounded scenario
    target_res = sim_results.get("ZealousMosquito (Hist)") or sim_results["Bounded ($40-$70)"]
    df_detail = target_res.df[target_res.df['season'] == 25].copy()  # Pick season 25

    fig, (ax_p, ax_c) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor('#ffffff')

    periods = df_detail['period'].values
    our_prices = df_detail['our_price'].values
    rival_prices = df_detail['rival_price'].values
    our_cap = df_detail['our_capacity_remaining'].values
    rival_cap = df_detail['rival_capacity_remaining'].values

    # Pacing reference curve
    tau = (periods - 1) / 100.0
    target_cap = 80.0 * (1.0 - (0.85 * tau + 0.15 * (tau ** 2)))

    # Price chart
    ax_p.set_facecolor('#f8f9fa')
    ax_p.grid(True, linestyle='--', alpha=0.6, color='#dee2e6')
    ax_p.plot(periods, our_prices, color='#10b981', linewidth=2.5, label='Our Algorithm Price (v2.0)')
    ax_p.plot(periods, rival_prices, color='#ef4444', linewidth=1.8, linestyle='--', label=f'Rival Price ({target_res.competitor_name})')
    ax_p.axhline(100.0, color='gray', linestyle=':', label='Empirical WTP Ceiling ($100)')
    ax_p.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
    ax_p.set_title(f'Sample Season Price Trajectory & Inventory Depletion (vs {target_res.competitor_name})', fontsize=13, fontweight='bold', pad=10)
    ax_p.legend(loc='upper right', frameon=True)

    # Capacity chart
    ax_c.set_facecolor('#f8f9fa')
    ax_c.grid(True, linestyle='--', alpha=0.6, color='#dee2e6')
    ax_c.plot(periods, our_cap, color='#10b981', linewidth=2.5, label='Our Remaining Seats')
    ax_c.plot(periods, rival_cap, color='#ef4444', linewidth=1.8, linestyle='--', label='Rival Remaining Seats')
    ax_c.plot(periods, target_cap, color='#3b82f6', linewidth=1.5, linestyle='-.', label='Target Pacing Curve')
    ax_c.set_xlabel('Selling Period (1 to 100)', fontsize=11, fontweight='bold')
    ax_c.set_ylabel('Remaining Capacity', fontsize=11, fontweight='bold')
    ax_c.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    out_path2 = 'analytics/price_trajectory_simulation.png'
    plt.savefig(out_path2, dpi=200)
    plt.close()
    print(f"Saved: {out_path2}")

if __name__ == '__main__':
    generate_visualizations()
