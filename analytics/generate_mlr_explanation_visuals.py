"""
analytics/generate_mlr_explanation_visuals.py
---------------------------------------------
Explanation figure for duopoly.py (v6.4). Every curve is computed with the functions and
constants inside duopoly.py, so the picture always matches the code:

1. Panel 1: the "revenue hill" (p - mu) * (A + B*p) for the average-demand prior, with the
   closed-form optimum p* = (A - B*mu) / (2*(-B)) and the band duopoly.py clips it to.
2. Panel 2: target booking curve 80*(0.85*tau + 0.15*tau^2) vs a linear pace.
3. Panel 3: capacity shadow cost mu(gap, burn rate) from _compute_capacity_shadow_cost.
4. Panel 4: monopoly price vs required burn rate per regime from _calculate_monopoly_price.

Saves output to: imgs/mlr_strategy_explained.png
"""

import os
import sys
from typing import Optional

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'analytics'))

from plot_style import COLORS, apply_style, plt  # noqa: E402
import duopoly  # noqa: E402


def generate_visuals(output_path: Optional[str] = None) -> str:
    apply_style()
    output_path = output_path or os.path.join(ROOT_DIR, 'imgs', 'mlr_strategy_explained.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5))

    # ------------------------------------------------------------------ Panel 1
    ax = axes[0, 0]
    prior = duopoly.REGIME_PRESETS_MLR['average_demand']
    rival, gap, burn = 50.0, 0.0, 0.8
    tau = (51 - 1) / 99.0  # day 51, exactly what _solve_closed_form_optimal_price uses
    B, C = prior['B'], max(duopoly.MIN_RIVAL_SLOPE_C, prior['C'])
    A = prior['intercept'] + C * rival + prior['b_t'] * tau
    mu = duopoly._compute_capacity_shadow_cost(gap, burn, False)
    prices = np.linspace(20, 110, 400)
    demand = np.maximum(0.0, A + B * prices)
    objective = (prices - mu) * demand
    p_star = (A - B * mu) / (2.0 * (-B))
    state = {'mlr_intercept': prior['intercept'], 'mlr_B': B, 'mlr_C': C, 'mlr_b_t': prior['b_t']}
    p_used = duopoly._solve_closed_form_optimal_price(51, gap, burn, rival, state)
    max_undercut = min(7.5, 1.20 + 0.35 * max(0.0, gap - 1.5))

    ax.plot(prices, objective, color=COLORS['accent'], linewidth=3, label='(p − μ) · expected demand(p)')
    ax.axvspan(rival - max_undercut, rival + 12.0, color='#dcfce7', alpha=0.7,
               label=f'Allowed band around rival: ${rival - max_undercut:.1f} to ${rival + 12:.0f}')
    ax.axvline(p_star, color=COLORS['warn'], linestyle=':', linewidth=2, label=f'Unconstrained optimum p* = ${p_star:.2f}')
    ax.axvline(p_used, color=COLORS['ours'], linewidth=2.4, label=f'Price after clipping = ${p_used:.2f} (inside band)')
    ax.axvline(rival, color=COLORS['rival'], linestyle='--', linewidth=1.5, label=f'Rival price ${rival:.0f}')
    ax.set_title('1. Closed-form price: maximise (p − μ)·(A + B·p)\n'
                 f'average-demand prior, rival $50, day 51, on pace (μ = ${mu:.0f})')
    ax.set_xlabel('Our price ($)')
    ax.set_ylabel('Objective ($ per period)')
    ax.legend(loc='lower center', fontsize=8.5)

    # ------------------------------------------------------------------ Panel 2
    ax = axes[0, 1]
    days = np.arange(1, 101)
    t = (days - 1) / 99.0
    target = 80 * (0.85 * t + 0.15 * t ** 2)
    linear = 80 * t
    ax.plot(days, target, color=COLORS['ours'], linewidth=3, label='Target: 80·(0.85τ + 0.15τ²)')
    ax.plot(days, linear, color=COLORS['rival'], linestyle='--', linewidth=2, label='Linear pace: 80·τ')
    ax.fill_between(days, target, linear, color=COLORS['ours'], alpha=0.15,
                    label='Seats held back vs linear (max 3 at day 50)')
    ax.annotate(f'Day 50: target {target[49]:.1f} sold vs {linear[49]:.1f} linear',
                xy=(50, target[49]), xytext=(8, 58), fontsize=9.5, color=COLORS['dark'],
                arrowprops=dict(arrowstyle='->', color=COLORS['dark']))
    ax.set_title('2. Starting target curve (bend 0.15; learned per market since v6.2) vs linear pace')
    ax.set_xlabel('Selling period (day)')
    ax.set_ylabel('Cumulative seats sold')
    ax.legend(loc='lower right')

    # ------------------------------------------------------------------ Panel 3
    ax = axes[1, 0]
    gaps = np.linspace(-15, 15, 300)
    for burn_rate, style in [(0.5, '-'), (0.8, '--'), (1.2, ':')]:
        mus = [duopoly._compute_capacity_shadow_cost(g, burn_rate, False) for g in gaps]
        ax.plot(gaps, mus, color=COLORS['accent'], linestyle=style, linewidth=2.6,
                label=f'Normal WTP, burn rate {burn_rate}')
    mus_low = [duopoly._compute_capacity_shadow_cost(g, 0.8, True) for g in gaps]
    ax.plot(gaps, mus_low, color=COLORS['amber'], linewidth=1.8, linestyle='-.', zorder=1,
            label='Low WTP, burn rate 0.8 (same line, floor $2 instead of $10)')
    ax.axvline(0, color=COLORS['rival'], linestyle=':')
    ax.text(-14.5, 4, '← ahead of pace (scarce seats)', fontsize=9, color=COLORS['dark'])
    ax.text(3, 14, 'behind pace (surplus seats) →', fontsize=9, color=COLORS['dark'])
    ax.set_title('3. Shadow cost μ = clip(26 − 16·(burn − 0.8) − 2.4·gap, floor, 56)')
    ax.set_xlabel('Inventory gap = remaining seats − target remaining')
    ax.set_ylabel('μ ($ per seat)')
    ax.legend(loc='upper right', fontsize=8.5)

    # ------------------------------------------------------------------ Panel 4
    ax = axes[1, 1]
    burns = np.linspace(0.0, 2.0, 200)
    for regime, color in [('high_demand', COLORS['accent']), ('average_demand', COLORS['ours']),
                          ('low_demand', COLORS['amber'])]:
        st = {'detected_regime': regime, 'detected_low_wtp': regime == 'low_demand'}
        ax.plot(burns, [duopoly._calculate_monopoly_price(b, 20, st) for b in burns],
                color=color, linewidth=2.8, label=regime.replace('_', ' '))
    st = {'detected_regime': 'average_demand', 'detected_low_wtp': True}
    ax.plot(burns, [duopoly._calculate_monopoly_price(b, 20, st) for b in burns],
            color=COLORS['ours'], linestyle='--', linewidth=2, label='average demand, truncated WTP (cap $60)')
    ax.set_title('4. Monopoly price once the rival is sold out\n(high-demand band is lower: v5 buyers had WTP capped near $78)')
    ax.set_xlabel('Required burn rate = remaining seats / remaining days')
    ax.set_ylabel('Price ($)')
    ax.set_ylim(20, 80)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved explanation figure to: {output_path}")
    return output_path


if __name__ == '__main__':
    generate_visuals()
