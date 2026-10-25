"""
analytics/generate_mlr_explanation_visuals.py
---------------------------------------------
Generates intuitive, publication-quality infographics explaining Rule-Based MLR
for non-technical audiences:
1. Panel 1: The Demand & Revenue "Hill" (Why there is an optimal sweet-spot price).
2. Panel 2: Target Curve vs Realized Booking Curve & Pacing (Convex benchmark vs linear).
3. Panel 3: The Capacity Opportunity Cost (How scarcity shifts the sweet spot).
4. Panel 4: The 3 Macroeconomic Regimes (Low Demand vs Average Demand vs High Demand).
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

    prices = np.linspace(15, 80, 200)
    # Empirical demand function: d(p) = 1.05 - 0.015 * p
    demands = np.maximum(0, 1.05 - 0.015 * prices)
    revenue = (prices - 15) * demands  # Surplus revenue above baseline cost 15
    p_opt = prices[np.argmax(revenue)]
    rev_max = np.max(revenue)

    ax1.plot(prices, revenue, color=c_blue, linewidth=3.5, label='Expected Surplus Revenue ($ per period)')
    ax1.scatter([p_opt], [rev_max], color=c_rose, s=140, zorder=5)
    ax1.axvline(p_opt, color=c_rose, linestyle=':', alpha=0.7)

    ax1.annotate(f'Optimal Closed-Form Price: ${p_opt:.1f}\n(Max Revenue: ${rev_max:.1f}/period)',
                 xy=(p_opt, rev_max), xytext=(p_opt + 4, rev_max - 2.5),
                 arrowprops=dict(arrowstyle='->', color=c_rose, lw=2),
                 fontsize=10, fontweight='bold', color=c_rose)

    ax1.annotate('Too Cheap ($18):\nHigh volume, but\ngiving away margin!',
                 xy=(22, 4), xytext=(18, 9),
                 arrowprops=dict(arrowstyle='->', color=c_slate), fontsize=9, color=c_slate)

    ax1.annotate('Too Expensive ($65+):\nHigh margin, but\nseats fly empty!',
                 xy=(62, 3), xytext=(55, 8),
                 arrowprops=dict(arrowstyle='->', color=c_slate), fontsize=9, color=c_slate)

    ax1.set_title('1. The Revenue Hill: Finding the Closed-Form Optimal Price', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Our Ticket Price ($)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Expected Surplus Revenue ($ / period)', fontsize=10, fontweight='bold')
    ax1.legend(loc='lower center', fontsize=9)

    # -------------------------------------------------------------------------
    # Panel 2: Target Curve vs Booking Curve (Depletion & Capacity Utilization)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor('#f8fafc')
    ax2.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')

    days = np.arange(1, 101)
    tau = (days - 1) / 99.0
    target_curve = 80 * (0.85 * tau + 0.15 * (tau ** 2))
    linear_curve = 80 * tau

    ax2.plot(days, target_curve, color=c_teal, linewidth=3.5, label='Target Curve (Optimal Planned Depletion)')
    ax2.plot(days, linear_curve, color=c_slate, linestyle='--', linewidth=2, label='Naïve Linear Pace')

    ax2.fill_between(days, linear_curve, target_curve, color=c_teal, alpha=0.15, label='Late Seats Protected for Inelastic Surge')

    ax2.annotate('Target Curve: TargetSold(t) = 80*(0.85τ + 0.15τ²)\nControls Capacity Utilization Pace',
                 xy=(50, target_curve[49]), xytext=(10, 55),
                 arrowprops=dict(arrowstyle='->', color=c_teal, lw=2),
                 fontsize=9.5, fontweight='bold', color=c_teal)

    ax2.set_title('2. Target Curve vs Realized Booking Pace (Pacing the Flight)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Days of Season (1 to 100)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Cumulative Seats Sold (Booking Pace)', fontsize=10, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)

    # -------------------------------------------------------------------------
    # Panel 3: Capacity Opportunity Cost (Shadow Price mu)
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor('#f8fafc')
    ax3.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')

    gaps = np.linspace(-10, 15, 100)
    mu_normal = np.clip(28.0 - 1.8 * gaps, 12, 56)
    mu_low_wtp = np.clip(22.0 - 2.2 * gaps, 2, 50)

    ax3.plot(gaps, mu_normal, color=c_blue, linewidth=3, label='Standard Opportunity Cost mu(t, c)')
    ax3.plot(gaps, mu_low_wtp, color=c_amber, linewidth=3, linestyle='-.', label='Low-Demand Opportunity Cost (Drops toward $2)')

    ax3.axvline(0, color=c_slate, linestyle=':', alpha=0.7)
    ax3.annotate('Ahead of Pace (Scarcity):\nRaise price, remaining seats are precious!',
                 xy=(-5, 37), xytext=(-9, 44),
                 arrowprops=dict(arrowstyle='->', color=c_blue), fontsize=9, color=c_blue)

    ax3.annotate('Behind Pace (Surplus):\nDrop price, unsold seats spoil at takeoff!',
                 xy=(8, 11), xytext=(3, 23),
                 arrowprops=dict(arrowstyle='->', color=c_amber), fontsize=9, color=c_amber)

    ax3.set_title('3. Opportunity Cost (Shadow Price): Scarcity vs Surplus', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Inventory Gap: RemainingSeats - TargetSeats (Behind [+] or Ahead [-])', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Opportunity Value per Seat ($)', fontsize=10, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)

    # -------------------------------------------------------------------------
    # Panel 4: The 3 Macroeconomic Demand Regimes (377,570 Periods)
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor('#f8fafc')
    ax4.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')

    regimes = ['Low Demand\n(mlr, v5.1, poisson)',
               'Average Demand\n(v3, v4 balanced)',
               'High Demand\n(v1 holiday surge)']
    baseline_arrivals = [0.21, 0.97, 2.23]
    sweet_spots = [26.0, 38.0, 48.0]

    x_pos = np.arange(len(regimes))
    w = 0.35

    b1 = ax4.bar(x_pos - w/2, [b * 30 for b in baseline_arrivals], w, label='Baseline Demand Arrival Rate (scaled)', color='#93c5fd', edgecolor='#1d4ed8')
    b2 = ax4.bar(x_pos + w/2, sweet_spots, w, label='Optimal Sweet Spot Price ($)', color='#2563eb', edgecolor='#1e3a8a')

    for idx, bar in enumerate(b1):
        ax4.annotate(f"b0={baseline_arrivals[idx]:.2f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 2), textcoords="offset points", ha='center', fontsize=9, fontweight='bold', color='#1d4ed8')
    for idx, bar in enumerate(b2):
        ax4.annotate(f"${sweet_spots[idx]:.0f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()/2),
                     ha='center', fontsize=9, fontweight='bold', color='#ffffff', va='center')

    ax4.set_title('4. The 3 Macroeconomic Regimes (377k Periods Empirical Analysis)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(regimes, fontsize=9)
    ax4.set_ylabel('Price Level / Scaled Arrival Rate', fontsize=10, fontweight='bold')
    ax4.set_ylim(0, 95)
    ax4.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    out_path = 'imgs/mlr_strategy_explained.png'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated explanation chart at: {out_path}")

if __name__ == '__main__':
    generate_visuals()
