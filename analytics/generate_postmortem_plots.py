"""
analytics/generate_postmortem_plots.py
--------------------------------------
Rebuilds the tournament post-mortem figures directly from the competition CSVs, so every
number and trajectory in them is real data:

- imgs/v1_performance_and_flaws.png    (data/v1)
- imgs/v3_vulnerabilities_postmortem.png (data/v3)
- imgs/v4_round4_postmortem.png         (data/v4)

Trajectory panels show one real season: the season whose revenue is closest to the
tournament's median season revenue (a typical season, not a cherry-picked one).
Rival prices >= 200 are the server's sold-out markers and are hidden; periods in which the
rival was sold out are shaded.
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

SOLDOUT_MARKER = 200.0


def load(version: str) -> pd.DataFrame:
    path = glob.glob(os.path.join(ROOT_DIR, 'data', version, '*.csv'))[0]
    df = pd.read_csv(path)
    df['revenue'] = df['price'] * df['demand']
    return df


def tournament_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby('competition_id')
    out = pd.DataFrame({
        'rival': g.competitor_id.first(),
        'revenue': g.revenue.sum(),
        'sold': g.demand.sum(),
        'seasons': g.selling_season.nunique(),
    })
    out['capacity'] = out.seasons * 80
    out['load_pct'] = 100.0 * out.sold / out.capacity
    out['spoiled'] = out.capacity - out.sold
    return out.sort_values('revenue', ascending=False)


def typical_season(df: pd.DataFrame, competition_id: str) -> pd.DataFrame:
    comp = df[df.competition_id == competition_id]
    season_rev = comp.groupby('selling_season').revenue.sum()
    season = int((season_rev - season_rev.median()).abs().idxmin())
    return comp[comp.selling_season == season].sort_values('selling_period')


def plot_trajectory(ax, season: pd.DataFrame, title: str) -> None:
    rival = season.price_competitor.where(season.price_competitor < SOLDOUT_MARKER)
    soldout = ~season.competitor_has_capacity.astype(bool)
    if soldout.any():
        first = int(season.selling_period[soldout].min())
        ax.axvspan(first - 0.5, 100.5, color='#ede9fe', alpha=0.8, label=f'Rival sold out (from day {first})')
    ax.plot(season.selling_period, season.price, color=COLORS['warn'], linewidth=2.4, label='Our price')
    ax.plot(season.selling_period, rival, color=COLORS['accent'], linestyle='--', linewidth=1.8, label='Rival price')
    sales = season[season.demand > 0]
    if len(sales):
        ax.scatter(sales.selling_period, sales.price, s=16 + 22 * sales.demand, color=COLORS['dark'], zorder=5,
                   label='Our sales (size = seats)')
    rev = season.revenue.sum()
    sold = int(season.demand.sum())
    ax.set_title(f"{title}\nseason {int(season.selling_season.iloc[0])}: {sold} seats sold, ${rev:,.0f} revenue")
    ax.set_xlabel('Selling period (day)')
    ax.set_ylabel('Price ($)')
    ax.set_xlim(0, 101)
    ax.legend(loc='best', fontsize=8.5)


def bar_revenue_load(ax, summary: pd.DataFrame, title: str, highlight: Optional[dict] = None) -> None:
    highlight = highlight or {}
    x = np.arange(len(summary))
    colors = [highlight.get(cid, COLORS['ours']) for cid in summary.index]
    bars = ax.bar(x - 0.2, summary.revenue / 1000.0, 0.4, color=colors, label='Revenue ($k)')
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, summary.load_pct, 0.4, color='#a5b4fc', label='Load factor (%)')
    ax2.set_ylim(0, 115)
    ax2.set_ylabel('Load factor (%)')
    ax2.grid(False)
    ax.set_zorder(ax2.get_zorder() + 1)  # keep revenue labels above the load-factor bars
    ax.patch.set_visible(False)
    for bar, v in zip(bars, summary.revenue):
        ax.annotate(f'${v / 1000:,.0f}k', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8.5, fontweight='bold')
    ax.set_ylim(0, max(summary.revenue.max() / 1000.0 * 1.18, 1))
    ax.set_xticks(x)
    ax.set_xticklabels([f'{cid} ({r})' for cid, r in zip(summary.index, summary.rival)],
                       fontsize=8.5, rotation=25, ha='right')
    ax.set_ylabel('Revenue ($k)')
    ax.set_title(title)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8.5)


def v1_figure(out: str) -> None:
    df = load('v1')
    summary = tournament_summary(df)
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.2))
    bar_revenue_load(axes[0], summary, f"v1.0: revenue and load factor per tournament\n"
                     f"total ${summary.revenue.sum():,.0f}; $0 in {(summary.revenue == 0).sum()} of {len(summary)} tournaments",
                     {cid: COLORS['warn'] for cid in summary.index[summary.revenue == 0]})
    plot_trajectory(axes[1], typical_season(df, '3XkNYY'), 'v1.0 vs ZealousMosquito (3XkNYY): fixed $85 / $105 schedule')

    ax = axes[2]
    phases = [('Days 1–50', df.selling_period <= 50), ('Days 51–70', df.selling_period.between(51, 70)),
              ('Days 71–100', df.selling_period >= 71)]
    total_cap = int(summary.capacity.sum())
    values = [int(df.loc[m, 'demand'].sum()) for _, m in phases] + [total_cap - int(df.demand.sum())]
    labels = [p for p, _ in phases] + ['Unsold (spoiled)']
    colors = [COLORS['ours'], COLORS['amber'], COLORS['accent'], COLORS['warn']]
    y = np.arange(len(values))
    ax.barh(y, values, color=colors)
    for yi, v in zip(y, values):
        ax.annotate(f'{v:,} seats ({100 * v / total_cap:.1f}%)', (v, yi), xytext=(4, 0), textcoords='offset points',
                    va='center', fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, total_cap * 1.25)
    ax.set_xlabel('Seats')
    ax.set_title(f'v1.0: what happened to all {total_cap:,} seats')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def v3_figure(out: str) -> None:
    df = load('v3')
    summary = tournament_summary(df)
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.2))
    bar_revenue_load(axes[0], summary, f"v3.0: revenue and load factor per tournament\ntotal ${summary.revenue.sum():,.0f}",
                     {'35zapi': COLORS['warn'], '3kGhWc': COLORS['amber']})
    plot_trajectory(axes[1], typical_season(df, '35zapi'), 'v3.0 vs RussetBat (35zapi): $58 holdout vs a ~$12 rival')
    plot_trajectory(axes[2], typical_season(df, '3kGhWc'), 'v3.0 vs LoutishMacaque (3kGhWc): undercutting a $66–$71 rival')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def v4_figure(out: str) -> None:
    df = load('v4')
    summary = tournament_summary(df)
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.2))
    bar_revenue_load(axes[0], summary, f"v4: revenue and load factor per tournament\ntotal ${summary.revenue.sum():,.0f}",
                     {'3d8tDh': COLORS['warn']})
    plot_trajectory(axes[1], typical_season(df, '3d8tDh'), 'v4 vs SageSloth (3d8tDh): stuck at the $27 floor')

    ax = axes[2]
    sp = summary.sort_values('spoiled')
    total = sp.spoiled.sum()
    y = np.arange(len(sp))
    ax.barh(y, sp.spoiled, color=[COLORS['warn'] if c == '3d8tDh' else COLORS['rival'] for c in sp.index])
    for yi, v in zip(y, sp.spoiled):
        ax.annotate(f'{v:,} ({100 * v / total:.1f}%)', (v, yi), xytext=(4, 0), textcoords='offset points',
                    va='center', fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{c} ({r})' for c, r in zip(sp.index, sp.rival)], fontsize=9)
    ax.set_xlim(0, sp.spoiled.max() * 1.3)
    ax.set_xlabel('Unsold seats over 100 seasons')
    ax.set_title(f'v4: unsold seats per tournament ({total:,} total)')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main() -> None:
    apply_style()
    imgs = os.path.join(ROOT_DIR, 'imgs')
    os.makedirs(imgs, exist_ok=True)
    for name, fn in [('v1_performance_and_flaws.png', v1_figure),
                     ('v3_vulnerabilities_postmortem.png', v3_figure),
                     ('v4_round4_postmortem.png', v4_figure)]:
        path = os.path.join(imgs, name)
        fn(path)
        print(f'Saved {path}')


if __name__ == '__main__':
    main()
