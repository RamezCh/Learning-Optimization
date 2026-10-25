"""
analytics/generate_strategy_visualization.py
---------------------------------------------
Generates high-resolution visualization for the Unified v6.0 Dynamic Pricing Strategy:
1. Panel 1: Dynamic Pricing Lifecycle (v6.0 Closed-Form MLR & Fast Low-Demand Detection)
2. Panel 2: Target Curve vs Booking Curve & Capacity Utilization Pacing Corridor
3. Panel 3: Empirical 3 Macroeconomic Demand Regimes (377,570 Periods Analyzed)

Saves output to: `imgs/dpc_strategy_visualization.png`
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def create_v6_visualization():
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'imgs', 'dpc_strategy_visualization.png'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Set high-DPI figure
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), dpi=200)
    fig.patch.set_facecolor('#ffffff')

    # Color Palette
    c_blue = '#2563eb'
    c_emerald = '#059669'
    c_red = '#ef4444'
    c_amber = '#d97706'
    c_purple = '#7c3aed'
    c_dark = '#1e293b'
    c_slate = '#64748b'

    # =========================================================================
    # PANEL 1: Dynamic Pricing Lifecycle (v6.0 Unified Econometric Engine)
    # =========================================================================
    ax1 = axes[0]
    ax1.set_facecolor('#ffffff')
    ax1.grid(True, linestyle='--', alpha=0.5, color='#e2e8f0')

    t_vals = np.arange(1, 101)
    
    # Rival price: competitor pricing high ($68-$72) in a low-WTP market
    comp_price = 70.0 + 2.5 * np.sin(t_vals / 9.0)
    
    # Our price trajectory in v6.0:
    # Days 1-10: Test price at ~$48.00.
    # Day 10: Zero sales detected -> Fast In-Season Low-Demand Latch triggers!
    # Days 11-70: Optimal MLR pricing snaps to empirical low-demand sweet spot ($24-$28)
    # Dynamic ceiling guard ensures we never get dragged into rival's high price ($70)
    # Days 71-100: Rival stockout / monopoly clearance down to terminal floor
    our_price = np.zeros(100)
    for idx, t in enumerate(t_vals):
        if t <= 10:
            our_price[idx] = 48.0 - (t - 1) * 0.2
        elif t <= 65:
            # Low-demand MLR sweet spot with continuous pacing
            base = 26.5
            pacing_adj = 1.5 * np.sin(t / 6.0)
            our_price[idx] = base + pacing_adj
        elif t <= 88:
            # Late season monopoly monetization
            our_price[idx] = 28.0 - (t - 65) * 0.25
        else:
            # Terminal clearance (t >= 89..100)
            our_price[idx] = max(18.0, 22.0 - (t - 88) * 0.35)

    # Background shading for phases
    ax1.axvspan(1, 10, color='#f1f5f9', alpha=0.7, label='Phase 1: Initial Discovery (t=1..10, ~$48)')
    ax1.axvspan(10, 65, color='#ecfdf5', alpha=0.6, label='Phase 2: Low-Demand MLR Regime ($24–$28)')
    ax1.axvspan(65, 88, color='#fef3c7', alpha=0.5, label='Phase 3: Monopoly Harvest ($24–$28)')
    ax1.axvspan(88, 100, color='#fee2e2', alpha=0.5, label='Phase 4: Terminal Clearance ($18–$22)')

    # Plot trajectories
    ax1.plot(t_vals, comp_price, color=c_slate, linestyle='--', linewidth=2.5, label='High Rival Price ($68–$72)')
    ax1.plot(t_vals, our_price, color=c_emerald, linewidth=3.2, label='duopoly.py v6.0 MLR Optimal Price')
    ax1.axvline(10, color=c_red, linestyle=':', linewidth=2.5, label='Day 10 Fast Latch (0 Sales Trigger)')

    # Annotations
    ax1.annotate('Day 10 Fast Latch:\nZero sales at $48 triggers\nimmediate low-demand shift!',
                 xy=(10, 46.2), xytext=(14, 57),
                 arrowprops=dict(facecolor=c_red, edgecolor=c_red, width=1.5, headwidth=7),
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#fee2e2', edgecolor=c_red, alpha=0.95),
                 fontsize=9.5, fontweight='bold', color='#991b1b')

    ax1.annotate('Dynamic Ceiling Guard:\nIgnores high rival ($70+);\nlocks in low-WTP sweet spot ($26)',
                 xy=(38, 27.5), xytext=(32, 42),
                 arrowprops=dict(facecolor=c_emerald, edgecolor=c_emerald, width=1.5, headwidth=7),
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#d1fae5', edgecolor=c_emerald, alpha=0.95),
                 fontsize=9.5, fontweight='bold', color='#065f46')

    ax1.set_title('Panel 1: Dynamic Pricing Lifecycle (v6.0 Unified MLR Engine)', fontsize=13, fontweight='bold', pad=12, color=c_dark)
    ax1.set_xlabel('Selling Period (Day 1..100)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Ticket Price ($)', fontsize=11, fontweight='bold')
    ax1.set_xlim(1, 100)
    ax1.set_ylim(15, 80)
    ax1.legend(loc='upper right', frameon=True, fontsize=8.5)

    # =========================================================================
    # PANEL 2: Target Curve vs Booking Curve & Capacity Utilization
    # =========================================================================
    ax2 = axes[1]
    ax2.set_facecolor('#ffffff')
    ax2.grid(True, linestyle='--', alpha=0.5, color='#e2e8f0')

    days = np.arange(1, 101)
    tau = (days - 1) / 99.0
    
    # 1. Target Curve: Optimal planned inventory depletion trajectory
    target_sold = 80.0 * (0.85 * tau + 0.15 * (tau ** 2))
    target_remaining = 80.0 - target_sold

    # Pacing Corridor (Behind Pace vs Ahead of Pace)
    behind_corridor = np.clip(target_remaining + 6.0, 0, 80)
    ahead_corridor = np.clip(target_remaining - 6.0, 0, 80)

    # Fill zones
    ax2.fill_between(days, behind_corridor, 80, color='#fee2e2', alpha=0.45, label='Behind Pace (Low Utilization -> Lower mu, Stimulate Sales)')
    ax2.fill_between(days, ahead_corridor, behind_corridor, color='#dcfce7', alpha=0.55, label='On-Pace Corridor (Balanced Dynamic Pricing)')
    ax2.fill_between(days, 0, ahead_corridor, color='#dbeafe', alpha=0.45, label='Ahead of Pace (High Utilization -> Raise mu, Protect Yield)')

    # Realized Booking Curve (Simulated actual remaining capacity under v6.0)
    # Starts flat during days 1-10 (zero sales), then catches up smoothly after day 10
    actual_remaining = np.zeros(100)
    for idx, d in enumerate(days):
        if d <= 10:
            actual_remaining[idx] = 80.0
        else:
            # Accelerated catchup to target curve
            catchup_tau = (d - 10) / 90.0
            actual_remaining[idx] = 80.0 - 80.0 * (0.92 * catchup_tau + 0.08 * (catchup_tau**2))
    actual_remaining = np.clip(actual_remaining, 0, 80)

    # Plot Target Curve and Realized Booking Curve
    ax2.plot(days, target_remaining, color=c_dark, linewidth=3.5, label='Target Curve (Optimal Planned Depletion)')
    ax2.plot(days, actual_remaining, color=c_blue, linewidth=3.0, linestyle='-.', label='Booking Curve (Realized Remaining Seats)')

    # Annotate Target Curve vs Booking Curve distinction
    ax2.annotate('Target Curve:\nOptimal depletion benchmark\nTarget(t) = 80*(0.85t + 0.15t²)',
                 xy=(45, target_remaining[44]), xytext=(12, 16),
                 arrowprops=dict(facecolor=c_dark, edgecolor=c_dark, width=1.5, headwidth=6),
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor=c_dark, alpha=0.9),
                 fontsize=9.0, fontweight='bold', color=c_dark)

    ax2.annotate('Realized Booking Curve:\nReflects Capacity Utilization (Sold/80).\nGap = Remaining - Target drives shadow cost mu!',
                 xy=(70, actual_remaining[69]), xytext=(48, 62),
                 arrowprops=dict(facecolor=c_blue, edgecolor=c_blue, width=1.5, headwidth=6),
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#eff6ff', edgecolor=c_blue, alpha=0.9),
                 fontsize=9.0, fontweight='bold', color='#1e40af')

    ax2.set_title('Panel 2: Target Curve vs Booking Curve & Capacity Pacing', fontsize=13, fontweight='bold', pad=12, color=c_dark)
    ax2.set_xlabel('Selling Period (Day 1..100)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Remaining Seats (Capacity)', fontsize=11, fontweight='bold')
    ax2.set_xlim(1, 100)
    ax2.set_ylim(0, 80)
    ax2.legend(loc='lower left', frameon=True, fontsize=8.0)

    # =========================================================================
    # PANEL 3: Empirical 3 Macroeconomic Demand Regimes (377,570 Periods)
    # =========================================================================
    ax3 = axes[2]
    ax3.set_facecolor('#ffffff')
    ax3.grid(True, linestyle='--', alpha=0.5, color='#e2e8f0')

    regimes = ['Low Demand\n(mlr, v5.1, poisson)', 'Average Demand\n(v3, v4 balanced)', 'High Demand\n(v1 holiday surge)']
    baseline_demands = [0.21, 0.97, 2.23]     # beta_0
    sweet_spot_low = [22.0, 32.0, 44.0]
    sweet_spot_high = [30.0, 44.0, 52.0]
    sweet_spot_mid = [26.0, 38.0, 48.0]

    x_indices = np.arange(len(regimes))
    bar_width = 0.35

    bars1 = ax3.bar(x_indices - bar_width/2, baseline_demands, bar_width, label='Baseline Demand Rate (beta_0)', color='#93c5fd', edgecolor='#2563eb', alpha=0.9)
    bars2 = ax3.bar(x_indices + bar_width/2, [p / 25.0 for p in sweet_spot_mid], bar_width, label='Optimal Sweet Spot Price ($ / 25)', color=c_emerald, edgecolor='#065f46', alpha=0.9)

    # Annotate bars
    for idx, bar in enumerate(bars1):
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f'b0={yval:.2f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1e3a8a')

    for idx, bar in enumerate(bars2):
        price_range = f"${sweet_spot_low[idx]:.0f}-${sweet_spot_high[idx]:.0f}"
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, price_range, ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#065f46')

    # Annotation callout
    ax3.annotate('Fast Regime Detection:\n1. Day 10 zero sales at $48-$55\n2. Cheaper than rival with 0 sales for 4d\n3. Season 1 sales < 42 seats',
                 xy=(0 + bar_width/2, bars2[0].get_height()), xytext=(0.15, 1.65),
                 arrowprops=dict(facecolor=c_emerald, edgecolor=c_emerald, width=1.5, headwidth=6),
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecfdf5', edgecolor=c_emerald, alpha=0.95),
                 fontsize=9.0, fontweight='bold', color='#065f46')

    ax3.set_title('Panel 3: The 3 Macroeconomic Regimes (377k Periods Analyzed)', fontsize=13, fontweight='bold', pad=12, color=c_dark)
    ax3.set_ylabel('Demand Arrival Rate / Normalized Price Scale', fontsize=11, fontweight='bold')
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels(regimes, fontsize=9.5, fontweight='bold')
    ax3.set_ylim(0, 2.7)
    ax3.legend(loc='upper right', frameon=True, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Successfully generated v6.0 visualization at: {output_path}")

if __name__ == '__main__':
    create_v6_visualization()
