"""
analytics/ablation_study.py
---------------------------
Ablation study: switch parts of duopoly.py off, one at a time and all together, and measure
the change in revenue.

How it works: the source of duopoly.py is loaded and each part gets an on/off switch by a small
text substitution (see SWITCHES). The unmodified copy gives exactly the same prices as duopoly.py,
so every difference comes from the switched-off part. The script also counts how often each
guard actually changes the price.

Scenarios (17): 3 simulated markets x 4 rival types, plus 5 tournament rivals replayed from the
CSVs in a default market. Each scenario runs --seeds seeds x --seasons seasons; the same seeds
are used for every configuration (paired comparison).

Usage:
    python analytics/ablation_study.py                       # 10 seeds x 30 seasons, ~10 min on 2 cores
    python analytics/ablation_study.py --seeds 3             # quick look
    python analytics/ablation_study.py --compare old.py      # also compare with another duopoly.py

Outputs: analytics/ablation_results.csv (per scenario and seed) and imgs/ablation_study.png.
"""

import argparse
import glob
import os
import sys
import types
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path[:0] = [ROOT_DIR, os.path.join(ROOT_DIR, 'analytics')]

from benchmark_3regime_models import MARKETS, RIVALS  # noqa: E402
from simulation_engine import DuopolySimulator, HistoricalCSVCompetitor, MarketDemandModel  # noqa: E402

HISTORICAL = {
    'ZealousMosquito 3XkNYY': ('v1', '3XkNYY'), 'UncoveredElephant 3PTJwD': ('v1', '3PTJwD'),
    'ImpartialGoldfish 3UmBuN': ('v5', '3UmBuN'), 'ConvivialCrane Gq2CUL': ('v5', 'Gq2CUL'),
    'PinkDuck 3vG5Xe': ('v4', '3vG5Xe'),
}

# name -> (label, [(text in duopoly.py, replacement)])
SWITCHES = {
    'breaker': ('Circuit breaker (2 weak seasons, cheap rival)', [
        ("    if consecutive_low_sales >= 2 and rival_recent_avg < 38.0:",
         "    if 'breaker' not in ABL and consecutive_low_sales >= 2 and rival_recent_avg < 38.0:\n        _hit('breaker')")]),
    'dumper': ('Deep-dumper holdout (rival <= $19.50)', [
        ("    if latest_rival_price <= 19.5 and rival_recent_avg <= 19.5:",
         "    if 'dumper' not in ABL and latest_rival_price <= 19.5 and rival_recent_avg <= 19.5:\n        _hit('dumper')")]),
    'cheapfloor': ('Cheap-rival floor (rival < $38)', [
        ("    if target_price < 38.0 and rival_recent_avg < 38.0:\n        target_price = max(",
         "    if 'cheapfloor' not in ABL and target_price < 38.0 and rival_recent_avg < 38.0:\n"
         "        if min(rival_recent_avg - 0.5, 28.0) > target_price: _hit('cheapfloor')\n        target_price = max(")]),
    'highceil': ('High-rival ceiling (rival >= $58)', [
        ("        target_price = min(target_price, dyn_ceiling)",
         "        if 'highceil' not in ABL:\n            if dyn_ceiling < target_price: _hit('highceil')\n"
         "            target_price = min(target_price, dyn_ceiling)")]),
    'behindcap': ('Behind-pace cap ($64 / $48)', [
        ("        if target_price > duopoly_cap and inventory_gap > 2.0:",
         "        if 'behindcap' not in ABL and target_price > duopoly_cap and inventory_gap > 2.0:\n            _hit('behindcap')")]),
    'latehigh': ('Late cap under a high rival (t >= 75)', [
        ("    if rival_recent_avg >= 54.0 and required_burn_rate <= 1.15:",
         "    if 'latehigh' not in ABL and rival_recent_avg >= 54.0 and required_burn_rate <= 1.15:\n        _hit('latehigh')")]),
    'clearance': ('Late clearance discount', [
        ("    if inventory_gap > 2.5 or (period >= 90 and remaining_capacity >= 2):",
         "    if 'clearance' not in ABL and (inventory_gap > 2.5 or (period >= 90 and remaining_capacity >= 2)):\n"
         "        _hit('clearance')")]),
    'band': ('Price band around the rival', [
        ("    return float(np.clip(optimal_p, rival_price - max_undercut, rival_price + 12.0))",
         "    if 'band' in ABL:\n        return float(optimal_p)\n"
         "    clipped = float(np.clip(optimal_p, rival_price - max_undercut, rival_price + 12.0))\n"
         "    if clipped != optimal_p: _hit('band')\n    return clipped")]),
    'sprt': ('In-season low-demand test (SPRT)', [
        ("        if has_rival_capacity and my_prev_price <= 55.0 and latest_rival_price > my_prev_price:",
         "        if 'sprt' not in ABL and has_rival_capacity and my_prev_price <= 55.0 and latest_rival_price > my_prev_price:")]),
    'regimes': ('Regime classification (always average)', [
        ("    if (\n        consec_low >= 1", "    if 'regimes' not in ABL and (\n        consec_low >= 1"),
        ("    elif (\n        len(sales_hist) >= 2", "    elif 'regimes' not in ABL and (\n        len(sales_hist) >= 2")]),
    'floor18': ('$18 floor in low demand', [
        ("        LOW_DEMAND_MIN_PRICE\n        if (state and", "        LOW_DEMAND_MIN_PRICE\n        if 'floor18' not in ABL and (state and")]),
    'targetlearn': ('Learned target curve', [
        ("    observed = state.get(\"target_c_obs\", [])\n    if not observed:",
         "    observed = state.get(\"target_c_obs\", [])\n    if 'targetlearn' in ABL or not observed:")]),
}
ALL_GUARDS = [k for k in SWITCHES if k != 'targetlearn']

_MODULES = {}
_DATA = {}


def build_module(path: str, switchable: bool) -> types.ModuleType:
    src = open(path, encoding='utf-8').read().replace('\r\n', '\n')
    header = "ABL = set()\nCOUNTS = {}\ndef _hit(k):\n    COUNTS[k] = COUNTS.get(k, 0) + 1\n"
    if switchable:
        for name, (_, subs) in SWITCHES.items():
            for old, new in subs:
                if src.count(old) != 1:
                    raise RuntimeError(f"switch '{name}': pattern not found once in {path}; update SWITCHES")
                src = src.replace(old, new)
    module = types.ModuleType('duopoly_variant')
    exec(compile(header + src, path, 'exec'), module.__dict__)
    return module


def scenarios():
    return [(m, r) for m in MARKETS for r in RIVALS] + [('Replayed (~92 arr.)', h) for h in HISTORICAL]


def _environment(market, rival, seed):
    if market.startswith('Replayed'):
        version, comp = HISTORICAL[rival]
        if rival not in _DATA:
            _DATA[rival] = pd.read_csv(glob.glob(os.path.join(ROOT_DIR, 'data', version, '*.csv'))[0])
        return MarketDemandModel(seed=3000 + seed), HistoricalCSVCompetitor(_DATA[rival], comp)
    return MARKETS[market](seed=2000 + seed), RIVALS[rival]()


def _task(job):
    source, config, market, rival, seed, seasons = job
    key = (source, source != 'compare')
    if key not in _MODULES:
        _MODULES[key] = build_module(_SOURCES[source], switchable=key[1])
    module = _MODULES[key]
    module.ABL.clear()
    module.ABL.update(config)
    module.COUNTS.clear()
    periods = [0]

    def policy(**kwargs):
        if kwargs['selling_period_in_current_season'] <= 100:
            periods[0] += 1
        return module.p(**kwargs)

    demand, competitor = _environment(market, rival, seed)
    summary = DuopolySimulator(policy, demand).run(competitor, n_seasons=seasons).summary()
    name = 'compare' if source == 'compare' else ('+'.join(sorted(config)) or 'base')
    row = dict(config=name, market=market, rival=rival, seed=seed, revenue=summary['our_total_revenue'] / seasons,
               load=summary['our_load_factor_pct'], periods=periods[0])
    row.update({'fires_' + k: v for k, v in module.COUNTS.items()})
    return row


_SOURCES = {'main': os.path.join(ROOT_DIR, 'duopoly.py')}


def _init(sources):
    _SOURCES.update(sources)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(scenario=df.market.str.split(' ').str[0] + ' | ' + df.rival)
    wide = df.pivot_table(index=['scenario', 'seed'], columns='config', values='revenue')
    per_seed = wide.groupby(level=1).sum()
    per_scen = wide.groupby(level=0).mean()
    rows = []
    for config in wide.columns:
        if config == 'base':
            continue
        change = (per_seed[config] - per_seed['base']) / per_seed['base'] * 100
        scen = (per_scen[config] - per_scen['base']) / per_scen['base'] * 100
        rows.append(dict(config=config, change_pct=change.mean(),
                         ci95=1.96 * change.std(ddof=1) / np.sqrt(len(change)) if len(change) > 1 else np.nan,
                         seeds_better=int((change > 0).sum()), seeds=len(change),
                         worst_scenario=scen.idxmin(), worst_pct=scen.min(),
                         scenarios_worse_2pct=int((scen < -2).sum())))
    return pd.DataFrame(rows).sort_values('change_pct')


def firing_rates(df: pd.DataFrame) -> pd.Series:
    base = df[df.config == 'base']
    cols = [c for c in base.columns if c.startswith('fires_')]
    return (base[cols].sum() / base.periods.sum() * 100).rename(lambda c: c[6:]).sort_values(ascending=False)


def plot(summary: pd.DataFrame, rates: pd.Series, path: str) -> None:
    from matplotlib.patches import Patch
    from matplotlib.transforms import blended_transform_factory
    from plot_style import COLORS, apply_style, plt
    apply_style()
    labels = {k: v[0] for k, v in SWITCHES.items()}
    labels['ALL'] = 'All guards removed at once\n(learned target curve kept)'
    s = summary[summary.config != 'compare'].sort_values('change_pct', ascending=False).reset_index(drop=True)

    def label(config):
        text = labels.get(config, config)
        if config in rates.index:
            text += f'\nchanges the price on {rates[config]:.0f}% of days'
        return text

    def color(v):
        if v <= -1.0:
            return COLORS['dark']
        if v < 0.5:
            return COLORS['amber']
        return COLORS['accent']

    fig, ax = plt.subplots(figsize=(14, 0.78 * len(s) + 2.4))
    y = np.arange(len(s))
    ax.barh(y, s.change_pct, xerr=s.ci95, color=[color(v) for v in s.change_pct], capsize=3, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([label(c) for c in s.config], fontsize=11)
    ax.invert_yaxis()
    ax.axvline(0, color=COLORS['dark'], linewidth=1.2)
    ax.set_xlim(min(-5.0, s.change_pct.min() * 1.08), max(5.0, s.change_pct.max() + 1.5))
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.02, -0.9, 'Change', transform=trans, fontsize=11, fontweight='bold', ha='left')
    for yi, (v, ci) in enumerate(zip(s.change_pct, s.ci95)):
        ax.text(1.02, yi, f'{v:+.1f}% \u00b1{ci:.1f}', transform=trans, va='center', ha='left', fontsize=11)
    ax.set_xlabel('Change in total revenue when the part is switched off (%, 95% interval over seeds)')
    ax.set_title('Ablation study: switching off one part of duopoly.py at a time (17 scenarios, 10 seeds)')
    ax.legend(handles=[Patch(color=COLORS['dark'], label='Essential: removing it costs more than 1%'),
                       Patch(color=COLORS['amber'], label='Small: less than 1% either way'),
                       Patch(color=COLORS['accent'], label='Trade-off: more revenue overall without it,\n'
                                                           'but large losses in the low-demand market')],
              loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--seasons', type=int, default=30)
    parser.add_argument('--procs', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--compare', help='another duopoly.py to compare with the current one')
    args = parser.parse_args()

    sources = {}
    configs = [('main', frozenset())] + [('main', frozenset([k])) for k in SWITCHES]
    configs.append(('main', frozenset(ALL_GUARDS)))
    if args.compare:
        sources['compare'] = os.path.abspath(args.compare)
        configs.append(('compare', frozenset()))
    build_module(_SOURCES['main'], switchable=True)          # fail fast if a switch no longer matches

    jobs = [(src, cfg, m, r, seed, args.seasons) for src, cfg in configs for m, r in scenarios()
            for seed in range(args.seeds)]
    print(f'{len(configs)} configurations x 17 scenarios x {args.seeds} seeds = {len(jobs)} runs')
    with Pool(args.procs, initializer=_init, initargs=(sources,)) as pool:
        df = pd.DataFrame(pool.map(_task, jobs, chunksize=4))
    df.loc[df.config == '+'.join(sorted(ALL_GUARDS)), 'config'] = 'ALL'
    df.to_csv(os.path.join(ROOT_DIR, 'analytics', 'ablation_results.csv'), index=False)

    write_outputs(df)


def write_outputs(df: pd.DataFrame) -> None:
    summary = summarize(df)
    rates = firing_rates(df)
    pd.set_option('display.width', 200)
    print(summary.round(2).to_string(index=False))
    print('\nHow often each guard changes the price (% of priced days, base configuration):')
    print(rates.round(1).to_string())
    plot(summary, rates, os.path.join(ROOT_DIR, 'imgs', 'ablation_study.png'))
    print('Saved imgs/ablation_study.png')


if __name__ == '__main__':
    main()
