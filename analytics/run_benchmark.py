"""
run_benchmark.py
----------------
Automated benchmark and visualization suite for testing duopoly.py (v6.2).
Single seed (42) per scenario; use benchmark_3regime_models.py for multi-seed confidence intervals.
Runs full 100-season simulations against:
1. Copycat Competitor (mirrors our price)
2. Old v1.0 Baseline Algorithm (head-to-head)
3. Aggressive Undercutter ($18-$38)
4. Bounded Mid-Market ($40-$70)
5. Historical ZealousMosquito (from CSV)
6. Historical UncoveredElephant (from CSV)

Outputs:
- CLI Executive Scorecard
- Publication-quality comparison chart: `imgs/algorithm_variant_comparison.png`
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

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
from plot_style import apply_style, plt

def save_comparison_chart(results: List[Dict[str, Any]], chart_path: str = 'imgs/algorithm_variant_comparison.png'):
    """Generates a high-resolution bar chart comparing revenue and win rates across scenarios."""
    apply_style()
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=200)
    fig.patch.set_facecolor('#ffffff')

    scenarios = df['competitor'].tolist()
    x = np.arange(len(scenarios))
    width = 0.38

    # -------------------------------------------------------------
    # Panel 1: Revenue Comparison (Our Revenue vs Rival Revenue)
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.set_facecolor('#f8fafc')
    ax1.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')

    our_rev = (df['our_total_revenue'] / 1000.0).values
    rival_rev = (df['rival_total_revenue'] / 1000.0).values

    b1 = ax1.bar(x - width/2, our_rev, width, label='duopoly.py (v6.2)', color='#10b981', edgecolor='#047857')
    b2 = ax1.bar(x + width/2, rival_rev, width, label='Opponent Total Revenue', color='#94a3b8', edgecolor='#475569')

    ax1.set_title('Panel 1: Total Revenue ($k) over 100 Seasons (simulator, seed 42)', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=20, ha='right', fontsize=9)
    ax1.set_ylabel('Total Revenue ($k)', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, max(our_rev.max(), rival_rev.max()) * 1.15)
    ax1.legend(loc='upper right', frameon=True)

    for bar in b1:
        h = bar.get_height()
        ax1.annotate(f'${h:.1f}k', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#047857')
    for bar in b2:
        h = bar.get_height()
        ax1.annotate(f'${h:.1f}k', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#475569')

    # -------------------------------------------------------------
    # Panel 2: Season Win Rate (%)
    # -------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_facecolor('#f8fafc')
    ax2.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')

    win_rates = df['season_win_rate_pct'].values
    load_factors = df['our_load_factor_pct'].values

    bw1 = ax2.bar(x - width/2, win_rates, width, label='Season Win Rate (%)', color='#3b82f6', edgecolor='#1d4ed8')
    bw2 = ax2.bar(x + width/2, load_factors, width, label='Our Load Factor (%)', color='#818cf8', edgecolor='#4338ca')

    ax2.set_title('Panel 2: Season Win Rate & Load Factor (%) Across 100 Seasons', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=20, ha='right', fontsize=9)
    ax2.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 115)
    ax2.legend(loc='lower right', frameon=True)

    for bar in bw1:
        h = bar.get_height()
        ax2.annotate(f'{h:.0f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1d4ed8')
    for bar in bw2:
        h = bar.get_height()
        ax2.annotate(f'{h:.0f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#3730a3')

    plt.tight_layout()
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    print(f"\n[Visual] Saved benchmark comparison chart to: {chart_path}")

def run_all_benchmarks(csv_path: Optional[str] = None, generate_chart: bool = True):
    print("\n" + "="*80)
    print(" DPC ALGORITHM BENCHMARK SUITE: EVALUATING duopoly.py (v6.2)")
    print("="*80)

    # 1. Target v1 dataset specifically for historical match replay
    v1_csv = 'data/v1/2026-10-14_day2_duopoly_competition_details.csv'
    if not os.path.exists(v1_csv) and csv_path and os.path.exists(csv_path):
        v1_csv = csv_path

    # Initialize Simulator
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
    if os.path.exists(v1_csv):
        df_csv = pd.read_csv(v1_csv)
        competitions = df_csv['competition_id'].astype(str).unique() if 'competition_id' in df_csv.columns else []

        # Replay ZealousMosquito (Tournament 3XkNYY)
        if "3XkNYY" in competitions:
            print("\n[5/6] Simulating against Historical ZEALOUS MOSQUITO (Tournament 3XkNYY)...")
            hist_zm = HistoricalCSVCompetitor(df_csv, competition_id="3XkNYY")
            res_zm = simulator.run(hist_zm, n_seasons=100, competitor_name="Hist: ZealousMosquito (3XkNYY)")
            res_zm.print_summary()
            results.append(res_zm.summary())

        # Replay UncoveredElephant (Tournament 3PTJwD)
        if "3PTJwD" in competitions:
            print("\n[6/6] Simulating against Historical UNCOVERED ELEPHANT (Tournament 3PTJwD)...")
            hist_ue = HistoricalCSVCompetitor(df_csv, competition_id="3PTJwD")
            res_ue = simulator.run(hist_ue, n_seasons=100, competitor_name="Hist: UncoveredElephant (3PTJwD)")
            res_ue.print_summary()
            results.append(res_ue.summary())
    else:
        print(f"\nNote: CSV file not found at {v1_csv}. Skipping historical replay.")

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

    if generate_chart:
        save_comparison_chart(results)

    return results

if __name__ == '__main__':
    run_all_benchmarks()
