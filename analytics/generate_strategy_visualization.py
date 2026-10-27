"""
analytics/generate_strategy_visualization.py
---------------------------------------------
Strategy figure for duopoly.py (v6.2). Every panel is computed, not drawn by hand:

1. Panel 1: one simulated season of duopoly.py against a high-priced rival (~$70) in the
   low-demand market (valuations $20-$48), including the day the in-season low-demand
   test (SPRT) fires.
2. Panel 2: the same season's realized remaining seats against the target curve
   TargetRemaining(t) = 80 * [1 - (0.85*tau + 0.15*tau^2)].
3. Panel 3: calibration of the in-season low-demand detector on the real tournament CSVs:
   share of good seasons (>= 60 seats sold, data/v3-v5) it would wrongly flag, and share of
   low seasons (< 42 seats, data/mlr, v5.1, poisson_logit) it catches, for the old
   4-zero-day rule vs the SPRT at several thresholds.

Saves output to: imgs/dpc_strategy_visualization.png
"""

import glob
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'analytics'))

from plot_style import COLORS, apply_style, plt  # noqa: E402
import duopoly  # noqa: E402
from simulation_engine import BaseCompetitor, DuopolySimulator  # noqa: E402
from benchmark_3regime_models import LowDemandMarket  # noqa: E402


class HighPricedRival(BaseCompetitor):
    """Rival that ignores the market and prices around $70."""
    def __init__(self, seed: int = 5):
        self.rng = np.random.default_rng(seed)

    def get_price(self, season, period, our_last_price, our_last_demand, rival_has_capacity):
        return float(70.0 + 2.5 * np.sin(period / 9.0) + self.rng.uniform(-1.0, 1.0))


def simulate_trace(n_seasons: int = 1, seed: int = 11) -> pd.DataFrame:
    """Runs duopoly.p and records price, sales and the detector state each period."""
    trace: List[Dict] = []

    def recorder(**kwargs):
        price, state = duopoly.p(**kwargs)
        t = kwargs['selling_period_in_current_season']
        if t <= 100:
            trace.append({
                'season': kwargs['current_selling_season'], 'period': t, 'price': price,
                'regime': state.get('detected_regime'), 'llr': state.get('low_demand_llr', 0.0),
            })
        return price, state

    result = DuopolySimulator(recorder, LowDemandMarket(seed=seed)).run(HighPricedRival(), n_seasons=n_seasons)
    df = pd.DataFrame(trace)
    hist = result.df.rename(columns={'period': 'period'})
    return df.merge(hist[['season', 'period', 'rival_price', 'our_sales', 'our_capacity_remaining']],
                    on=['season', 'period'])


def _season_groups(paths: List[str]):
    frames = [pd.read_csv(p).assign(ds=os.path.basename(os.path.dirname(p))) for p in paths]
    data = pd.concat(frames, ignore_index=True)
    data['season_sales'] = data.groupby(['ds', 'competition_id', 'selling_season']).demand.transform('sum')
    return data


def _old_rule_fires(x: pd.DataFrame) -> bool:
    """v6.0 rule: 4 zero-sale periods in a row while cheaper than the rival at <= $55."""
    run = 0
    for p, r, d in zip(x.price.values, x.price_competitor.values, x.demand.values):
        if p <= 55 and r > p and d == 0:
            run += 1
            if run >= 4:
                return True
        elif d > 0:
            run = 0
    return False


def _sprt_fires(x: pd.DataFrame, threshold: float) -> bool:
    """v6.1 rule: SPRT over periods where we were cheaper than an in-stock rival at <= $55."""
    llr, n = 0.0, 0
    for p, r, d, t, cap in zip(x.price.values, x.price_competitor.values, x.demand.values,
                               x.selling_period.values, x.competitor_has_capacity.values):
        if cap and p <= 55 and r > p:
            rate_n = duopoly.SPRT_NORMAL_BASE + duopoly.SPRT_NORMAL_SLOPE * (t - 1) / 99.0
            rate_l = duopoly.SPRT_LOW_RATIO * rate_n
            llr = max(duopoly.SPRT_FLOOR, llr + d * np.log(rate_l / rate_n) - (rate_l - rate_n))
            n += 1
            if llr > threshold and n >= duopoly.SPRT_MIN_OBS:
                return True
    return False


def detector_calibration(data_dir: Optional[str] = None) -> pd.DataFrame:
    data_dir = data_dir or os.path.join(ROOT_DIR, 'data')
    data = _season_groups(glob.glob(os.path.join(data_dir, '*', '*.csv')))
    seasons = [(k, g.sort_values('selling_period')) for k, g in
               data.groupby(['ds', 'competition_id', 'selling_season'])]
    good = [g for k, g in seasons if k[0] in ('v3', 'v4', 'v5') and g.season_sales.iloc[0] >= 60]
    low = [g for k, g in seasons if k[0] in ('mlr', 'v5.1', 'poisson_logit') and g.season_sales.iloc[0] < 42]

    rules = {'Old: 4 zero days': _old_rule_fires}
    for label, thr in [('SPRT ln(19)', np.log(19)), ('SPRT ln(49)', np.log(49)), ('SPRT ln(99) (used)', np.log(99))]:
        rules[label] = (lambda thr_: (lambda g: _sprt_fires(g, thr_)))(thr)

    rows = []
    for name, rule in rules.items():
        rows.append({
            'rule': name,
            'false_pos': 100.0 * np.mean([rule(g) for g in good]),
            'detected': 100.0 * np.mean([rule(g) for g in low]),
            'n_good': len(good), 'n_low': len(low),
        })
    return pd.DataFrame(rows)


def create_visualization(output_path: Optional[str] = None) -> str:
    apply_style()
    output_path = output_path or os.path.join(ROOT_DIR, 'imgs', 'dpc_strategy_visualization.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    trace = simulate_trace()
    season = trace[trace.season == 1]
    latch = season[season.regime == 'low_demand']
    latch_day = int(latch.period.iloc[0]) if len(latch) else None

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))

    # Panel 1: price trajectory
    ax = axes[0]
    ax.plot(season.period, season.rival_price, color=COLORS['rival'], linestyle='--', linewidth=2, label='Rival price (~$70)')
    ax.plot(season.period, season.price, color=COLORS['ours'], linewidth=2.6, label='duopoly.py v6.2 price')
    sold = season[season.our_sales > 0]
    ax.scatter(sold.period, sold.price, s=14 + 18 * sold.our_sales, color=COLORS['dark'], zorder=5,
               label='Periods with sales (size = seats)')
    if latch_day:
        ax.axvline(latch_day, color=COLORS['warn'], linestyle=':', linewidth=2,
                   label=f'Low-demand test fires (day {latch_day})')
    ax.set_title('Panel 1: Simulated season vs a ~$70 rival\n(low-demand market, valuations $20-$48)')
    ax.set_xlabel('Selling period (day)')
    ax.set_ylabel('Price ($)')
    ax.set_xlim(1, 100)
    ax.set_ylim(15, 80)
    ax.legend(loc='upper right', fontsize=8.5)

    # Panel 2: target vs realized remaining seats
    ax = axes[1]
    days = np.arange(1, 101)
    tau = (days - 1) / 99.0
    target_remaining = 80.0 * (1.0 - (0.85 * tau + 0.15 * tau ** 2))
    remaining_start = 80 - season.our_sales.cumsum().shift(fill_value=0).values
    ax.fill_between(days, np.clip(target_remaining - 3, 0, 80), np.clip(target_remaining + 3, 0, 80),
                    color='#dcfce7', label='Target ± 3 seats')
    ax.plot(days, target_remaining, color=COLORS['dark'], linewidth=2.6,
            label='Target remaining: 80·[1 − (0.85τ + 0.15τ²)]')
    ax.plot(season.period, remaining_start, color=COLORS['accent'], linewidth=2.4,
            label='Realized remaining seats (start of day)')
    ax.set_title('Panel 2: Target curve vs realized booking curve\n(same simulated season)')
    ax.set_xlabel('Selling period (day)')
    ax.set_ylabel('Remaining seats')
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 82)
    ax.text(3, 6, 'Above the target = behind pace → shadow cost μ falls → lower price\n'
                  'Below the target = ahead of pace → μ rises → higher price',
            fontsize=8.5, color=COLORS['dark'], bbox=dict(facecolor='white', edgecolor=COLORS['grid']))
    ax.legend(loc='upper right', fontsize=8.5)

    # Panel 3: detector calibration on real data
    ax = axes[2]
    cal = detector_calibration()
    x = np.arange(len(cal))
    w = 0.38
    b1 = ax.bar(x - w / 2, cal.false_pos, w, color=COLORS['warn'], label=f"Good seasons wrongly flagged (n={cal.n_good.iloc[0]})")
    b2 = ax.bar(x + w / 2, cal.detected, w, color=COLORS['accent'], label=f"Low seasons caught in-season (n={cal.n_low.iloc[0]})")
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f'{bar.get_height():.1f}%', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cal.rule, fontsize=9)
    ax.set_ylabel('% of seasons')
    ax.set_ylim(0, max(cal.false_pos.max(), cal.detected.max()) * 1.2)
    ax.set_title('Panel 3: In-season low-demand detector, replayed on tournament CSVs\n'
                 '(low seasons where we were never cheaper than the rival cannot be caught in-season)')
    ax.legend(loc='upper right', fontsize=8.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(cal.round(1).to_string(index=False))
    print(f"Saved strategy visualization to: {output_path}")
    return output_path


if __name__ == '__main__':
    create_visualization()
