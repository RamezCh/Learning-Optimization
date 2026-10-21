import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def create_v51_visualization():
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
    # PANEL 1: Dynamic Pricing Lifecycle (v5.1 Master Architecture)
    # =========================================================================
    ax1 = axes[0]
    ax1.set_facecolor('#ffffff')
    ax1.grid(True, linestyle='--', alpha=0.5, color='#e5e7eb')

    t_duopoly = np.arange(1, 71)
    t_monopoly = np.arange(70, 101)
    
    # Competitor price trajectory (oscillating around $50.50, selling out at t=70)
    comp_price = 50.5 + 2.0 * np.sin(t_duopoly / 8.0)
    
    # Our price trajectory in v5.1
    # Duopoly: smooth pacing undercut at ~$49.50 with probing pulse
    our_price_duo = comp_price - 1.00
    our_price_duo[25:28] += 3.5
    our_price_duo[28:31] -= 2.5
    
    # Monopoly: Branching based on detected WTP
    # Branch A: Standard WTP Market ($84 -> $78 -> $72 -> $54)
    # Branch B: v5.1 Low-WTP Empirical Sweet Spot ($64 -> $60 -> $56 -> $52)
    our_price_mono_low = np.zeros(len(t_monopoly))
    our_price_mono_std = np.zeros(len(t_monopoly))
    for idx, t in enumerate(t_monopoly):
        if t <= 80:
            our_price_mono_std[idx] = 84.0
            our_price_mono_low[idx] = 64.0
        elif t <= 90:
            our_price_mono_std[idx] = 78.0
            our_price_mono_low[idx] = 60.0
        elif t <= 95:
            our_price_mono_std[idx] = 72.0
            our_price_mono_low[idx] = 56.0
        else:
            our_price_mono_std[idx] = 54.0
            our_price_mono_low[idx] = 50.0

    # Background shading for phases
    ax1.axvspan(1, 70, color='#f0fdf4', alpha=0.5, label='Phase 1: Duopoly (Smooth Undercut & Probing)')
    ax1.axvspan(70, 100, color='#fffbeb', alpha=0.5, label='Phase 2: Adaptive Monopoly (WTP-Calibrated)')

    # Plot trajectories
    ax1.plot(t_duopoly, comp_price, color='#64748b', linestyle='--', linewidth=2.5, label='Rival Price (~$50.50)')
    ax1.plot(t_duopoly, our_price_duo, color=c_emerald, linewidth=3.2, label='duopoly.py v5.1 Price')
    ax1.plot(t_monopoly, our_price_mono_low, color=c_emerald, linewidth=3.2, label='v5.1 Low-WTP Sweet Spot ($52–$64)')
    ax1.plot(t_monopoly, our_price_mono_std, color='#6366f1', linestyle=':', linewidth=2.5, label='Standard High-WTP Path ($72–$84)')
    ax1.axvline(70, color=c_red, linestyle=':', linewidth=2.5, label='Rival Sells Out (Period 70)')

    # Annotations
    ax1.annotate('Duopoly: Smooth Undercut -$1.00\n(Holds $49.50 Ticket Yield)',
                 xy=(35, 49.5), xytext=(8, 62),
                 arrowprops=dict(facecolor=c_emerald, edgecolor=c_emerald, width=1.5, headwidth=8),
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#d1fae5', edgecolor=c_emerald, alpha=0.9),
                 fontsize=9.5, fontweight='bold', color='#065f46')

    ax1.annotate('v5.1 Adaptive Monopoly:\n$64 → $60 → $56 → $50\n(Cures 3UmBuN Zero-Demand Trap)',
                 xy=(85, 60.0), xytext=(65, 32),
                 arrowprops=dict(facecolor=c_amber, edgecolor=c_amber, width=1.5, headwidth=8),
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef3c7', edgecolor=c_amber, alpha=0.95),
                 fontsize=9.5, fontweight='bold', color='#92400e')

    ax1.set_title('Panel 1: Dynamic Pricing Lifecycle (v5.1 Master Architecture)', fontsize=13, fontweight='bold', pad=12, color=c_dark)
    ax1.set_xlabel('Selling Period (Day 1..100)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Ticket Price ($)', fontsize=11, fontweight='bold')
    ax1.set_xlim(1, 100)
    ax1.set_ylim(20, 100)
    ax1.legend(loc='upper left', frameon=True, fontsize=8.5)

    # =========================================================================
    # PANEL 2: Inventory Pacing & Control Zones (v5.1)
    # =========================================================================
    ax2 = axes[1]
    ax2.set_facecolor('#ffffff')
    ax2.grid(True, linestyle='--', alpha=0.5, color='#e5e7eb')

    t_vals = np.arange(1, 101)
    progress = (t_vals - 1) / 100.0
    target_sold = 80.0 * (0.85 * progress + 0.15 * (progress ** 2))
    target_remaining = 80.0 - target_sold

    # Zone boundaries
    behind_pace_boundary = np.clip(target_remaining + 5.5, 0, 80)
    ahead_pace_boundary = np.clip(target_remaining - 5.5, 0, 80)

    # Fill zones
    ax2.fill_between(t_vals, behind_pace_boundary, 80, color='#fee2e2', alpha=0.5, label='Behind Pace (Smooth Undercut -$0.80 to -$7.50)')
    ax2.fill_between(t_vals, ahead_pace_boundary, behind_pace_boundary, color='#dcfce7', alpha=0.6, label='On-Pace Corridor (Moderate Undercut -$0.80)')
    ax2.fill_between(t_vals, 0, ahead_pace_boundary, color='#dbeafe', alpha=0.5, label='Ahead of Pace (Scarcity Premium +$1.00 to +$14.00)')

    # Unconditional clearance shading at t >= 90
    ax2.axvspan(90, 100, color='#fef08a', alpha=0.35, label='Final Unconditional Clearance (t ≥ 90, rem_cap ≥ 2)')

    # Target Curve
    ax2.plot(t_vals, target_remaining, color=c_dark, linewidth=3.5, label='Target Depletion Curve')

    ax2.set_title('Panel 2: Inventory Pacing & Control Zones (v5.1)', fontsize=13, fontweight='bold', pad=12, color=c_dark)
    ax2.set_xlabel('Selling Period (Day 1..100)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Remaining Seats (Capacity)', fontsize=11, fontweight='bold')
    ax2.set_xlim(1, 100)
    ax2.set_ylim(0, 80)
    ax2.legend(loc='lower left', frameon=True, fontsize=8.5)

    # =========================================================================
    # PANEL 3: The 3 Empirical WTP Segments (230,000 Periods Analyzed)
    # =========================================================================
    ax3 = axes[2]
    ax3.set_facecolor('#ffffff')
    ax3.grid(True, linestyle='--', alpha=0.5, color='#e5e7eb')

    categories = ['Budget\n($25 – $55)', 'Leisure\n($55 – $80)', 'Business\n($80 – $105+)']
    sales_volume_pct = [87.5, 11.2, 1.3]
    expected_rev_period = [46.51, 36.90, 1.15]

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
    ax3.annotate('Monopoly Sweet Spot ($52–$64):\nRealizes Peak Revenue in Both\nHigh-WTP & Low-WTP (v5) Rounds!',
                 xy=(0 + bar_width/2, 46.51), xytext=(0.28, 72),
                 arrowprops=dict(facecolor=c_emerald, edgecolor=c_emerald, width=1.5, headwidth=7),
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecfdf5', edgecolor=c_emerald, alpha=0.95),
                 fontsize=9.5, fontweight='bold', color='#065f46')

    ax3.set_title('Panel 3: Empirical WTP Segments (230k Periods across v1–v5)', fontsize=13, fontweight='bold', pad=12, color=c_dark)
    ax3.set_ylabel('Percentage (%) / Expected Rev ($)', fontsize=11, fontweight='bold')
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper right', frameon=True, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Successfully generated v5.1 visualization at: {output_path}")

if __name__ == '__main__':
    create_v51_visualization()
