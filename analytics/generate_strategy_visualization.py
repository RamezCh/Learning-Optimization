import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def create_v5_visualization():
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'imgs', 'dpc_strategy_visualization.png'))
    
    # Set high-DPI figure
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), dpi=200)
    fig.patch.set_facecolor('#ffffff')

    # Color Palette
    c_blue = '#3b82f6'
    c_green = '#10b981'
    c_emerald = '#059669'
    c_red = '#ef4444'
    c_amber = '#f59e0b'
    c_purple = '#8b5cf6'
    c_dark = '#1f2937'

    # =========================================================================
    # PANEL 1: Dynamic Pricing Lifecycle (v5.0 Master Architecture)
    # =========================================================================
    ax1 = axes[0]
    ax1.set_facecolor('#ffffff')
    ax1.grid(True, linestyle='--', alpha=0.5, color='#e5e7eb')

    t_duopoly = np.arange(1, 66)
    t_monopoly = np.arange(65, 101)
    
    # Competitor price trajectory (oscillating around $45, selling out at t=65)
    comp_price = 45.0 + 2.0 * np.sin(t_duopoly / 8.0)
    
    # Our price trajectory in v5.0
    # Duopoly: dynamic pacing undercut at ~$43.50 with probing pulse
    our_price_duo = comp_price - 1.50
    # Add probing pulse at t=30..36
    our_price_duo[29:32] += 3.5
    our_price_duo[32:35] -= 3.0
    
    # Monopoly: calibrated step-down: $78 -> $72 -> $54 (Sweet Spot) -> $48 (Clearance)
    our_price_mono = np.zeros(len(t_monopoly))
    for idx, t in enumerate(t_monopoly):
        if t <= 75:
            our_price_mono[idx] = 78.0
        elif t <= 88:
            our_price_mono[idx] = 72.0
        elif t <= 95:
            our_price_mono[idx] = 54.0  # Empirical sweet spot ($70.28/period yield)
        else:
            our_price_mono[idx] = 48.0  # Final unconditional clearance

    # Background shading for phases
    ax1.axvspan(1, 65, color='#f0fdf4', alpha=0.5, label='Phase 1: Duopoly (Pacing & Circuit Breakers)')
    ax1.axvspan(65, 100, color='#fffbeb', alpha=0.5, label='Phase 2: Monopoly (Calibrated Yield Sweet Spot)')

    # Plot trajectories
    ax1.plot(t_duopoly, comp_price, color='#64748b', linestyle='--', linewidth=2.5, label='Competitor Price (~$45)')
    ax1.plot(t_duopoly, our_price_duo, color=c_emerald, linewidth=3.2, label='duopoly.py v5.0 Price')
    ax1.plot(t_monopoly, our_price_mono, color=c_emerald, linewidth=3.2)
    ax1.axvline(65, color=c_red, linestyle=':', linewidth=2.5, label='Rival Sells Out (Period 65)')

    # Annotations
    ax1.annotate('Duopoly: Pacing Undercut -$1.50\n(Probing & Smooth Anchor)',
                 xy=(35, 43.5), xytext=(8, 55),
                 arrowprops=dict(facecolor=c_emerald, edgecolor=c_emerald, width=1.5, headwidth=8),
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#d1fae5', edgecolor=c_emerald, alpha=0.9),
                 fontsize=10, fontweight='bold', color='#065f46')

    ax1.annotate('v5.0 Calibrated Monopoly:\n$78 → $72 → $54 (Sweet Spot)\n→ $48 (Final Clearance)',
                 xy=(88, 54.0), xytext=(68, 85),
                 arrowprops=dict(facecolor=c_amber, edgecolor=c_amber, width=1.5, headwidth=8),
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef3c7', edgecolor=c_amber, alpha=0.95),
                 fontsize=10, fontweight='bold', color='#92400e')

    ax1.set_title('Panel 1: Dynamic Pricing Lifecycle (v5.0)', fontsize=14, fontweight='bold', pad=12, color=c_dark)
    ax1.set_xlabel('Selling Period (Day 1..100)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Ticket Price ($)', fontsize=11, fontweight='bold')
    ax1.set_xlim(1, 100)
    ax1.set_ylim(15, 100)
    ax1.legend(loc='upper left', frameon=True, fontsize=9)

    # =========================================================================
    # PANEL 2: Inventory Pacing & Control Zones (v5.0)
    # =========================================================================
    ax2 = axes[1]
    ax2.set_facecolor('#ffffff')
    ax2.grid(True, linestyle='--', alpha=0.5, color='#e5e7eb')

    t_vals = np.arange(1, 101)
    progress = (t_vals - 1) / 100.0
    target_sold = 80.0 * (0.85 * progress + 0.15 * (progress ** 2))
    target_remaining = 80.0 - target_sold

    # Zone boundaries
    behind_pace_boundary = np.clip(target_remaining + 6.0, 0, 80)
    ahead_pace_boundary = np.clip(target_remaining - 6.0, 0, 80)

    # Fill zones
    ax2.fill_between(t_vals, behind_pace_boundary, 80, color='#fee2e2', alpha=0.5, label='Behind Pace (Pacing Undercut -$1.50 to -$8.00)')
    ax2.fill_between(t_vals, ahead_pace_boundary, behind_pace_boundary, color='#dcfce7', alpha=0.6, label='On-Pace Corridor (Standard Undercut -$1.00)')
    ax2.fill_between(t_vals, 0, ahead_pace_boundary, color='#dbeafe', alpha=0.5, label='Ahead of Pace (Scarcity Premium +$1.00 to +$14.00)')

    # Unconditional clearance shading at t >= 90
    ax2.axvspan(90, 100, color='#fef08a', alpha=0.35, label='Final 10-Day Unconditional Clearance (t ≥ 90)')

    # Target Curve
    ax2.plot(t_vals, target_remaining, color=c_dark, linewidth=3.5, label='Target Depletion Curve')

    ax2.set_title('Panel 2: Inventory Pacing & Control Zones', fontsize=14, fontweight='bold', pad=12, color=c_dark)
    ax2.set_xlabel('Selling Period (Day 1..100)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Remaining Seats (Capacity)', fontsize=11, fontweight='bold')
    ax2.set_xlim(1, 100)
    ax2.set_ylim(0, 80)
    ax2.legend(loc='lower left', frameon=True, fontsize=9)

    # =========================================================================
    # PANEL 3: The 3 Empirical WTP Segments (190,000 Periods Analyzed)
    # =========================================================================
    ax3 = axes[2]
    ax3.set_facecolor('#ffffff')
    ax3.grid(True, linestyle='--', alpha=0.5, color='#e5e7eb')

    categories = ['Budget\n($25 – $55)', 'Leisure\n($55 – $80)', 'Business\n($80 – $105+)']
    sales_volume_pct = [85.0, 12.0, 3.0]
    expected_rev_period = [46.51, 36.90, 3.36]

    x_indices = np.arange(len(categories))
    bar_width = 0.35

    bars1 = ax3.bar(x_indices - bar_width/2, sales_volume_pct, bar_width, label='% of Total Sales Volume', color=c_blue, alpha=0.9)
    bars2 = ax3.bar(x_indices + bar_width/2, expected_rev_period, bar_width, label='Expected Rev / Period ($)', color=c_emerald, alpha=0.9)

    # Annotate bars
    for bar in bars1:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    for bar in bars2:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'${yval:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Sweet spot highlight callout
    ax3.annotate('Monopoly Sweet Spot ($52–$56):\nRealizes $70.28 / Period Peak Yield!',
                 xy=(0 + bar_width/2, 46.51), xytext=(0.35, 72),
                 arrowprops=dict(facecolor=c_emerald, edgecolor=c_emerald, width=1.5, headwidth=7),
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecfdf5', edgecolor=c_emerald, alpha=0.95),
                 fontsize=9.5, fontweight='bold', color='#065f46')

    ax3.set_title('Panel 3: The 3 Empirical WTP Segments (190k Periods)', fontsize=14, fontweight='bold', pad=12, color=c_dark)
    ax3.set_ylabel('Percentage (%) / Expected Rev ($)', fontsize=11, fontweight='bold')
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper right', frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Successfully generated v5.0 visualization at: {output_path}")

create_v5_visualization()
