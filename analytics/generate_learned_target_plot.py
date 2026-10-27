"""
analytics/generate_learned_target_plot.py
-----------------------------------------
Figure for the learned target curve (v6.2).

Panel 1 (real data): average booking curve, as seats sold out of 80, of the seasons that sold
75+ seats in two v5 matches: 3UmBuN (books early) and Gq2CUL (books late), next to the fixed
v6.1 target curve (bend c = 0.15) and the curve with each match's own fitted bend.

Panel 2 (simulation): the bend duopoly.py uses in each season, as it learns online in the three
simulated markets (rival: bounded $40-$70, seed 2000).

Saves output to: imgs/learned_target_curve.png
"""

import glob
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'analytics'))

from plot_style import COLORS, apply_style, plt  # noqa: E402
import duopoly  # noqa: E402
from benchmark_3regime_models import MARKETS, RIVALS  # noqa: E402
from simulation_engine import DuopolySimulator  # noqa: E402

TAU = np.arange(100) / 99.0


def curve(bend: float) -> np.ndarray:
    return 80.0 * ((1.0 - bend) * TAU + bend * TAU ** 2)


def match_shape(competition_id: str):
    path = glob.glob(os.path.join(ROOT_DIR, 'data', 'v5', '*.csv'))[0]
    df = pd.read_csv(path)
    g = df[df.competition_id == competition_id].sort_values(['selling_season', 'selling_period'])
    shapes, bends = [], []
    for _, season in g.groupby('selling_season'):
        sales = season.demand.values
        if sales.sum() >= duopoly.TARGET_LEARN_MIN_SOLD and len(sales) == 100:
            shapes.append(np.concatenate([[0.0], np.cumsum(sales)[:-1]]) / sales.sum())
            bends.append(duopoly._fit_season_target_bend(sales))
    return 80.0 * np.mean(shapes, axis=0), float(np.mean(bends)), len(bends)


def learned_bends_by_season(market_name: str, n_seasons: int = 30) -> list:
    bends = []

    def recorder(**kwargs):
        price, state = duopoly.p(**kwargs)
        if kwargs['selling_period_in_current_season'] == 1:
            bends.append(state.get('target_c_now', duopoly.TARGET_C_DEFAULT))
        return price, state

    DuopolySimulator(recorder, MARKETS[market_name](seed=2000)).run(RIVALS['Bounded ($40-$70)'](), n_seasons=n_seasons)
    return bends


def create_plot(output_path: Optional[str] = None) -> str:
    apply_style()
    output_path = output_path or os.path.join(ROOT_DIR, 'imgs', 'learned_target_curve.png')
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    days = np.arange(1, 101)

    ax = axes[0]
    ax.plot(days, curve(duopoly.TARGET_C_DEFAULT), color=COLORS['dark'], linewidth=3, label='Fixed v6.1 plan (c = 0.15)')
    for comp, color in [('3UmBuN', COLORS['accent']), ('Gq2CUL', COLORS['warn'])]:
        avg, bend, n = match_shape(comp)
        ax.plot(days, avg, color=color, linewidth=3, label=f'{comp}: average real booking curve ({n} seasons)')
        ax.plot(days, curve(bend), color=color, linewidth=2, linestyle='--', label=f'{comp}: learned plan (c = {bend:.2f})')
    ax.annotate('Day 50: fixed plan 37 seats,\nlate market only 24', xy=(50, 25), xytext=(56, 8), fontsize=11,
                arrowprops=dict(arrowstyle='->', color=COLORS['dark']))
    ax.set_title('Real data: two v5 markets book very differently')
    ax.set_xlabel('Selling day')
    ax.set_ylabel('Seats sold (out of 80)')
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 82)
    ax.legend(loc='upper left', fontsize=10)

    ax = axes[1]
    styles = {'Low (~68 arr.)': (COLORS['amber'], ':'), 'Average (~95 arr.)': (COLORS['ours'], '-'),
              'High (~135 arr.)': (COLORS['accent'], '--')}
    for market, (color, style) in styles.items():
        bends = learned_bends_by_season(market)
        ax.plot(range(1, len(bends) + 1), bends, color=color, linestyle=style, linewidth=2.6, marker='o', markersize=4,
                label=f'{market} market')
    ax.axhline(duopoly.TARGET_C_DEFAULT, color=COLORS['dark'], linewidth=1.5, linestyle=':', label='Starting value 0.15')
    ax.axhline(duopoly.TARGET_C_MAX, color=COLORS['rival'], linewidth=1.5, linestyle='-.', label='Safety limit 0.8')
    ax.set_title('Simulation: the bend duopoly.py learns, season by season')
    ax.set_xlabel('Season')
    ax.set_ylabel('Bend c of the target curve')
    ax.set_ylim(0, 0.9)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f'Saved {output_path}')
    return output_path


if __name__ == '__main__':
    create_plot()
