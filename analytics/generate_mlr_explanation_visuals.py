"""
analytics/generate_mlr_explanation_visuals.py
---------------------------------------------
Generates intuitive, publication-quality infographics explaining Rule-Based MLR
for non-technical audiences:
1. Panel 1: The Demand & Revenue "Hill" (Why there is an optimal sweet-spot price).
2. Panel 2: The Ideal Booking Pacing Curve (Why convex acceleration beats linear pace).
3. Panel 3: The Capacity Opportunity Cost (How scarcity shifts the sweet spot).
4. Panel 4: The 3 Macroeconomic Regimes (High WTP vs Medium WTP vs Low WTP).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def generate_visuals():
    os.makedirs('imgs', exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=200)
    fig.patch.set_facecolor('#ffffff')

    # Color Palette
    c_blue = '#2563eb'
    c_teal = '#0f766e'
    c_amber = '#d97706'
    c_rose = '#e11d48'
    c_slate = '#475569'

    # -------------------------------------------------------------------------
    # Panel 1: The Revenue Hill (Why Math Finds the Peak)
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor('#f8fafc')
    ax1.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')

    prices = np.linspace(25, 90, 200)
    # Demand function: d(p) = 2.4 - 0.03 * p
    demands = np.maximum(0, 2.4 - 0.03 * prices)
    revenue = (prices - 20) * demands  # Surplus revenue above baseline cost 20
    p_opt = prices[np.argmax(revenue)]
    rev_max = np.max(revenue)

    ax1.plot(prices, revenue, color=c_blue, linewidth=3.5, label='Expected Revenue ($ per period)')
    ax1.scatter([p_opt], [rev_max], color=c_rose, s=140, zorder=5)
    ax1.axvline(p_opt, color=c_rose, linestyle=':', alpha=0.7)

    ax1.annotate(f'Optimal Peak Price: ${p_opt:.1f}\n(Max Revenue: ${rev_max:.1f})',
                 xy=(p_opt, rev_max), xytext=(p_opt + 6, rev_max - 5),
                 arrowprops=dict(arrowstyle='->', color=c_rose, lw=2),
                 fontsize=10, fontweight='bold', color=c_rose)

    ax1.annotate('Too Cheap:\nLots of passengers,\nno ticket profit!',
                 xy=(32, 10), xytext=(28, 22),
                 arrowprops=dict(arrowstyle='->', color=c_slate), fontsize=9, color=c_slate)

    ax1.annotate('Too Expensive:\nHigh margins,\nbut empty seats!',
                 xy=(75, 10), xytext=(70, 22),
                 arrowprops=dict(arrowstyle='->', color=c_slate), fontsize=9, color=c_slate)

    ax1.set_title('1. The Revenue Hill: Finding the Golden Price', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Our Ticket Price ($)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Expected Revenue ($ / period)', fontsize=10, fontweight='bold')
    ax1.legend(loc='lower center', fontsize=9)

    # -------------------------------------------------------------------------
    # Panel 2: The Ideal Booking Pacing Curve (Depletion)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor('#f8fafc')
    ax2.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')

    days = np.arange(1, 101)
    tau = (days - 1) / 99.0
    convex_curve = 80 * (0.85 * tau + 0.15 * (tau ** 2))
    linear_curve = 80 * tau

    ax2.plot(days, convex_curve, color=c_teal, linewidth=3.5, label='Our Smart Curved Target (Holds seats for late surge)')
    ax2.plot(days, linear_curve, color=c_slate, linestyle='--', linewidth=2, label='Naïve Flat Target (Sells out too early)')

    ax2.fill_between(days, linear_curve, convex_curve, color=c_teal, alpha=0.15, label='High-Yield Seats Saved for Final Weeks')

    ax2.set_title('2. The 100-Day Booking Curve: Pacing the Flight', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Days Until Takeoff (1 to 100)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Target Cumulative Seats Sold', fontsize=10, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)

    # -------------------------------------------------------------------------
    # Panel 3: Capacity Opportunity Cost (Shadow Cost mu)
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor('#f8fafc')
    ax3.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')

    gaps = np.linspace(-10, 15, 100)
    mu_normal = np.clip(28.0 - 1.8 * gaps, 12, 56)
    mu_low_wtp = np.clip(24.0 - 2.4 * gaps, 2, 56)

    ax3.plot(gaps, mu_normal, color=c_blue, linewidth=3, label='Standard Opportunity Cost')
    ax3.plot(gaps, mu_low_wtp, color=c_amber, linewidth=3, linestyle='-.', label='Low-WTP Clearance Cost (Drops to $2 to clear plane)')

    ax3.axvline(0, color=c_slate, linestyle=':', alpha=0.7)
    ax3.annotate('Ahead of Pace (Scarcity):\nRaise price, seats are precious!',
                 xy=(-5, 38), xytext=(-9, 44),
                 arrowprops=dict(arrowstyle='->', color=c_blue), fontsize=9, color=c_blue)

    ax3.annotate('Behind Pace (Surplus):\nDrop price, empty seats = $0!',
                 xy=(8, 12), xytext=(4, 25),
                 arrowprops=dict(arrowstyle='->', color=c_amber), fontsize=9, color=c_amber)

    ax3.set_title('3. Opportunity Cost (Shadow Price): Scarcity vs Surplus', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Inventory Gap (Seats Behind [+] or Ahead [-] of Target)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Opportunity Value per Seat ($)', fontsize=10, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)

    # -------------------------------------------------------------------------
    # Panel 4: The 3 Macroeconomic Passenger Worlds
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor('#f8fafc')
    ax4.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')

    regimes = ['Regime 1: Premium Route\n(v1/v4 Business Route)',
               'Regime 2: Balanced Route\n(v3 Standard Duopoly)',
               'Regime 3: Budget Surge\n(v5 Holiday Crowds)']
    max_wtps = [115.0, 92.0, 78.0]
    sweet_spots = [79.0, 68.0, 57.0]

    x_pos = np.arange(len(regimes))
    w = 0.35

    b1 = ax4.bar(x_pos - w/2, max_wtps, w, label='Max Willingness to Pay ($)', color='#93c5fd', edgecolor='#1d4ed8')
    b2 = ax4.bar(x_pos + w/2, sweet_spots, w, label='Optimal Ticket Price ($)', color='#2563eb', edgecolor='#1e3a8a')

    for bar in b1:
        ax4.annotate(f"${bar.get_height():.0f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 2), textcoords="offset points", ha='center', fontsize=9, fontweight='bold', color='#1d4ed8')
    for bar in b2:
        ax4.annotate(f"${bar.get_height():.0f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()/2),
                     ha='center', fontsize=9, fontweight='bold', color='#ffffff', va='center')

    ax4.set_title('4. The 3 Passenger Worlds: Auto-Detected in Season 1', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(regimes, fontsize=9)
    ax4.set_ylabel('Price Level ($)', fontsize=10, fontweight='bold')
    ax4.set_ylim(0, 135)
    ax4.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    out_path = 'imgs/mlr_strategy_explained.png'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated explanation chart at: {out_path}")

if __name__ == '__main__':
    generate_visuals()
