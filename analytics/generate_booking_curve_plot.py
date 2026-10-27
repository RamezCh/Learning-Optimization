"""
analytics/generate_booking_curve_plot.py
----------------------------------------
Booking curve vs target curve on two REAL tournament seasons (data/v5):

- Left:  3UmBuN season 62 vs ImpartialGoldfish: the booking curve tracks the target and
         the flight sells out (80 seats).
- Right: Gq2CUL season 5 vs ConvivialCrane: the market is more back-loaded than the target.
         We run up to ~30 seats "behind pace", the gap pushes our price below the rival's,
         and the flight still fills in the last weeks: the discount was not needed.

Top row: remaining seats (start of day) vs target remaining seats.
Bottom row: our price vs rival price, with the inventory gap on a second axis.

Saves output to: imgs/booking_vs_target_real.png
"""

import glob
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(ROOT_DIR, 'analytics'))

from plot_style import COLORS, apply_style, plt  # noqa: E402

SEASONS = [
    ('v5', '3UmBuN', 62, 'On pace: booking curve follows the target'),
    ('v5', 'Gq2CUL', 5, 'Behind pace, yet the flight still fills'),
]


def season_frame(version: str, competition_id: str, season: int) -> pd.DataFrame:
    path = glob.glob(os.path.join(ROOT_DIR, 'data', version, '*.csv'))[0]
    df = pd.read_csv(path)
    g = df[(df.competition_id == competition_id) & (df.selling_season == season)].sort_values('selling_period').copy()
    tau = (g.selling_period.values - 1) / 99.0
    g['target_remaining'] = 80.0 * (1.0 - (0.85 * tau + 0.15 * tau ** 2))
    g['remaining'] = 80 - np.concatenate([[0], np.cumsum(g.demand.values)[:-1]])
    g['gap'] = g.remaining - g.target_remaining
    return g


def create_plot(output_path: Optional[str] = None) -> str:
    apply_style()
    output_path = output_path or os.path.join(ROOT_DIR, 'imgs', 'booking_vs_target_real.png')
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True,
                             gridspec_kw={'height_ratios': [1.35, 1]})

    for col, (version, comp, season, subtitle) in enumerate(SEASONS):
        g = season_frame(version, comp, season)
        rival = g.competitor_id.iloc[0]
        sold = int(g.demand.sum())
        rev = float((g.price * g.demand).sum())

        ax = axes[0, col]
        ax.plot(g.selling_period, g.target_remaining, color=COLORS['dark'], linewidth=2.6,
                label='Target curve (planned remaining seats)')
        ax.plot(g.selling_period, g.remaining, color=COLORS['accent'], linewidth=2.6,
                label='Booking curve (actual remaining seats)')
        ax.fill_between(g.selling_period, g.target_remaining, g.remaining,
                        where=g.remaining >= g.target_remaining, color=COLORS['warn'], alpha=0.15,
                        label='Behind pace (gap > 0)')
        ax.fill_between(g.selling_period, g.target_remaining, g.remaining,
                        where=g.remaining < g.target_remaining, color=COLORS['ours'], alpha=0.2,
                        label='Ahead of pace (gap < 0)')
        ax.set_title(f'{subtitle}\n{comp} vs {rival}, season {season}: {sold} seats sold, ${rev:,.0f}')
        ax.set_ylabel('Remaining seats')
        ax.set_ylim(0, 82)
        ax.legend(loc='lower left', fontsize=8.5)

        ax = axes[1, col]
        rival_price = g.price_competitor.where(g.price_competitor < 200)
        ax.plot(g.selling_period, rival_price, color=COLORS['rival'], linestyle='--', linewidth=1.8, label='Rival price')
        ax.plot(g.selling_period, g.price, color=COLORS['ours'], linewidth=2.2, label='Our price')
        ax.set_ylabel('Price ($)')
        ax.set_xlabel('Selling period (day)')
        ax2 = ax.twinx()
        ax2.plot(g.selling_period, g.gap, color=COLORS['warn'], linewidth=1.4, alpha=0.8, label='Inventory gap (seats)')
        ax2.axhline(0, color=COLORS['warn'], linewidth=0.8, linestyle=':')
        ax2.set_ylabel('Inventory gap (seats)')
        ax2.grid(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3,
                   fontsize=9, frameon=False)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f'Saved {output_path}')
    return output_path


if __name__ == '__main__':
    create_plot()
