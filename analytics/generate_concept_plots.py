"""
analytics/generate_concept_plots.py
-----------------------------------
Textbook/concept figures used in README.md section 1-2. These are illustrations of the
theory and of the course's game rules, not results.

- imgs/price_response_plots.png: linear price-response d(p) = 100 - 10p, elastic vs inelastic
  demand, revenue R(p) = p*d(p) peaking at p* = 5 (elasticity 1).
- imgs/dpc_game_rules_visualization.png: competition hierarchy, arrival-intensity shape
  (scaled to the simulator's default ~92 arrivals per season), willingness-to-pay segments
  from the course slides (up to ~$50 / ~$85 / ~$150), and a duopoly -> monopoly transition.
"""

import os
import sys

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(ROOT_DIR, 'analytics'))

from plot_style import COLORS, apply_style, plt  # noqa: E402
from simulation_engine import MarketDemandModel  # noqa: E402


def price_response_figure(path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    p = np.linspace(0, 10, 200)

    ax = axes[0]
    ax.plot(p, 100 - 10 * p, color=COLORS['accent'], linewidth=3, label='d(p) = 100 − 10p')
    ax.scatter([0, 10], [100, 0], color=COLORS['warn'], zorder=5)
    ax.annotate('Max demand D = 100 (at p = $0)', (0, 100), xytext=(1.0, 92), fontsize=9.5)
    ax.annotate('Satiating price P = $10 (demand 0)', (10, 0), xytext=(4.6, 10), fontsize=9.5)
    ax.set_title('1. Downward-sloping price-response curve')
    ax.set_xlabel('Price p ($)')
    ax.set_ylabel('Demand d(p) (units)')
    ax.legend(loc='upper right')

    ax = axes[1]
    pp = np.linspace(1, 9, 200)
    ax.plot(pp, 200 / pp ** 2, color=COLORS['warn'], linewidth=3, label='Elastic: d = 200/p² (ε = 2)')
    ax.plot(pp, 80 / pp ** 0.4, color=COLORS['ours'], linewidth=3, linestyle='--', label='Inelastic: d = 80/p^0.4 (ε = 0.4)')
    ax.set_title('2. Elastic vs inelastic demand')
    ax.set_xlabel('Price p ($)')
    ax.set_ylabel('Demand d(p) (units)')
    ax.legend(loc='upper right')

    ax = axes[2]
    ax.plot(p, p * (100 - 10 * p), color='#7c3aed', linewidth=3, label='R(p) = p · d(p)')
    ax.axvline(5, color=COLORS['amber'], linestyle=':', linewidth=2.5, label='p* = $5 (ε = 1, MR = 0)')
    ax.scatter([5], [250], color=COLORS['warn'], zorder=5)
    ax.set_title('3. Total revenue and the optimal price')
    ax.set_xlabel('Price p ($)')
    ax.set_ylabel('Revenue R(p) ($)')
    ax.legend(loc='lower center')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def game_rules_figure(path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('Dynamic Pricing Competition (DPC): rules and market mechanics (illustration)',
                 fontsize=15, fontweight='bold')

    ax = axes[0, 0]
    ax.axis('off')
    tree = ("1 competition (player A vs player B)\n"
            "├── 2 simulations (same rival, new random demand model each)\n"
            "│   └── 100 selling seasons (demand model fixed within a simulation)\n"
            "│       └── 100 selling periods (+ period 101: feedback only)\n"
            "│           └── capacity: 80 seats per airline per season, $0 salvage\n"
            "└── winner: higher total revenue over all simulations")
    ax.text(0.0, 0.62, tree, family='monospace', fontsize=11, va='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#eff6ff', edgecolor=COLORS['accent']))
    ax.set_title('1. Competition hierarchy')

    ax = axes[0, 1]
    days = np.arange(1, 101)
    market = MarketDemandModel(avg_arrivals=92.0)
    lam = np.array([market.get_arrival_rate(t) for t in days])
    ax.fill_between(days, lam, color=COLORS['warn'], alpha=0.15)
    ax.plot(days, lam, color=COLORS['warn'], linewidth=3, label=f'λ(t), total ≈ {lam.sum():.0f} arrivals/season')
    ax.axvline(80, color=COLORS['rival'], linestyle='--', label='Late surge (days 80-100)')
    ax.set_title('2. Customer arrival intensity (shape random per simulation; level shown = simulator default)')
    ax.set_xlabel('Selling period (day)')
    ax.set_ylabel('Expected arrivals per day (whole market)')
    ax.legend(loc='upper left')

    ax = axes[1, 0]
    prices = np.linspace(0, 160, 300)
    for cap, color, style, label in [(50, COLORS['ours'], '--', 'Low-WTP segment (up to ~$50)'),
                                     (85, COLORS['amber'], '-.', 'Mid-WTP segment (up to ~$85)'),
                                     (150, COLORS['accent'], '-', 'High-WTP segment (up to ~$150)')]:
        ax.plot(prices, np.clip(1 - (prices / cap) ** 2, 0, 1), color=color, linestyle=style, linewidth=2.6, label=label)
    ax.set_title('3. Purchase probability by price for the three WTP segments (course spec)')
    ax.set_xlabel('Ticket price ($)')
    ax.set_ylabel('Purchase probability')
    ax.legend(loc='upper right')

    ax = axes[1, 1]
    ours = 80 - 0.5 * (days - 1)
    rival = np.clip(80 - 1.1 * (days - 1), 0, 80)
    sold_out = int(days[rival <= 0][0])
    ax.axvspan(sold_out, 100, color='#ede9fe', label='Monopoly phase')
    ax.plot(days, ours, color=COLORS['accent'], linewidth=3, label='Our remaining seats')
    ax.plot(days, rival, color=COLORS['amber'], linewidth=3, label='Rival remaining seats (sells out early)')
    ax.axvline(sold_out, color='#7c3aed', linestyle=':', linewidth=2.5,
               label=f'Rival sold out (day {sold_out}) → competitor_has_capacity = False')
    ax.set_title('4. Duopoly → monopoly transition')
    ax.set_xlabel('Selling period (day)')
    ax.set_ylabel('Remaining seats')
    ax.legend(loc='upper right', fontsize=8.5)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    apply_style()
    imgs = os.path.join(ROOT_DIR, 'imgs')
    os.makedirs(imgs, exist_ok=True)
    price_response_figure(os.path.join(imgs, 'price_response_plots.png'))
    game_rules_figure(os.path.join(imgs, 'dpc_game_rules_visualization.png'))
    print('Saved concept figures to imgs/')


if __name__ == '__main__':
    main()
