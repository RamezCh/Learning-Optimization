"""
analytics/benchmark_3regime_models.py
------------------------------------
Multi-seed evaluation of duopoly.py across 3 simulated market regimes.

Markets (expected total arrivals per season are now actually enforced by the simulator):
- Low Demand:     ~68 arrivals, valuations $20-$48
- Average Demand: ~95 arrivals, valuations $35-$92
- High Demand:    ~135 arrivals, valuations $25-$78

Rivals: Copycat, Aggressive Dumper ($18-$36), Bounded ($40-$70), old v1.0 algorithm.

Every (market, rival) pair is run with several independent seeds; results are reported as
mean revenue per season with a 95% confidence interval across seeds, next to a naive
"undercut the rival by $1" reference strategy.

Usage:
    python analytics/benchmark_3regime_models.py
    python analytics/benchmark_3regime_models.py --compare path/to/other_duopoly.py
    python analytics/benchmark_3regime_models.py --seeds 10 --seasons 40
"""

import argparse
import importlib.util
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ANALYTICS_DIR = os.path.join(ROOT_DIR, 'analytics')
sys.path.append(ROOT_DIR)
sys.path.append(ANALYTICS_DIR)

import duopoly  # noqa: E402

from simulation_engine import (  # noqa: E402
    DuopolySimulator,
    CopycatCompetitor,
    AggressiveDumperCompetitor,
    BoundedCompetitor,
    OldV1Competitor,
    MarketDemandModel,
)


class LowDemandMarket(MarketDemandModel):
    """Low-demand market: ~68 total arrivals, high price sensitivity, low WTP."""
    def __init__(self, seed=42):
        super().__init__(avg_arrivals=68, arrival_convexity=1.3, price_sensitivity=0.15, seed=seed)

    def sample_customer_wtp(self, n, t):
        return self.rng.uniform(20.0, 48.0, size=n)


class AverageDemandMarket(MarketDemandModel):
    """Balanced market: ~95 total arrivals, broad leisure/business valuations."""
    def __init__(self, seed=42):
        super().__init__(avg_arrivals=95, arrival_convexity=1.8, price_sensitivity=0.07, seed=seed)

    def sample_customer_wtp(self, n, t):
        return self.rng.uniform(35.0, 92.0, size=n)


class HighDemandMarket(MarketDemandModel):
    """High-volume market: ~135 total arrivals, convex late surge, WTP capped at $78."""
    def __init__(self, seed=42):
        super().__init__(avg_arrivals=135, arrival_convexity=2.0, price_sensitivity=0.11, seed=seed)

    def sample_customer_wtp(self, n, t):
        return self.rng.uniform(25.0, 78.0, size=n)


MARKETS: Dict[str, Callable[[int], MarketDemandModel]] = {
    "Low (~68 arr.)": LowDemandMarket,
    "Average (~95 arr.)": AverageDemandMarket,
    "High (~135 arr.)": HighDemandMarket,
}

RIVALS: Dict[str, Callable[[], Any]] = {
    "Copycat": lambda: CopycatCompetitor(initial_price=60.0, undercut=0.0, noise=0.5),
    "Aggressive Dumper": lambda: AggressiveDumperCompetitor(low=18.0, high=36.0),
    "Bounded ($40-$70)": lambda: BoundedCompetitor(mean_price=55.0, spread=12.0),
    "Old v1.0": lambda: OldV1Competitor(),
}


def naive_undercut(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season,
    demand_historical_in_current_season,
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump: Optional[Any] = None,
) -> Tuple[float, Any]:
    """Reference strategy: undercut the rival's last price by $1 (floor $25), $60 in monopoly."""
    if prices_historical_in_current_season is None:
        return 45.0, information_dump
    if not competitor_has_capacity_current_period_in_current_season:
        return 60.0, information_dump
    return float(np.clip(prices_historical_in_current_season[1, -1] - 1.0, 25.0, 90.0)), information_dump


def load_pricing_function(path: str) -> Callable:
    """Loads `p` from an arbitrary duopoly-style file (e.g. an older version for A/B tests)."""
    spec = importlib.util.spec_from_file_location(f"compare_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.p


def mean_ci(values: List[float]) -> Tuple[float, float]:
    """Mean and 95% half-width (normal approximation) across seeds."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr)))


def run_3regime_benchmark(
    algorithms: Optional[Dict[str, Callable]] = None,
    n_seeds: int = 5,
    n_seasons: int = 30,
) -> pd.DataFrame:
    algorithms = algorithms or {"duopoly.py": duopoly.p, "naive rival-$1": naive_undercut}

    records = []
    for market_name, market_cls in MARKETS.items():
        for rival_name, rival_factory in RIVALS.items():
            for algo_name, algo in algorithms.items():
                per_seed = []
                wins = 0
                for seed in range(n_seeds):
                    sim = DuopolySimulator(algorithm_func=algo, market_demand=market_cls(seed=1000 + seed))
                    summary = sim.run(rival_factory(), n_seasons=n_seasons, competitor_name=rival_name).summary()
                    per_seed.append(summary["our_total_revenue"] / n_seasons)
                    wins += int(summary["is_winner"])
                mean, half = mean_ci(per_seed)
                records.append({
                    "Market": market_name, "Rival": rival_name, "Algorithm": algo_name,
                    "Rev/season": mean, "CI95": half, "Match wins": f"{wins}/{n_seeds}",
                })
    return pd.DataFrame(records)


def print_report(df: pd.DataFrame) -> None:
    print("=" * 96)
    print(" MULTI-SEED BENCHMARK: mean revenue per season (± 95% CI across seeds)")
    print("=" * 96)
    table = df.copy()
    table["Revenue/season"] = table.apply(lambda r: f"${r['Rev/season']:,.0f} ± {r['CI95']:,.0f}", axis=1)
    wide = table.pivot_table(index=["Market", "Rival"], columns="Algorithm",
                             values="Revenue/season", aggfunc="first")
    print(wide.to_string())
    print("-" * 96)
    totals = df.groupby(["Market", "Algorithm"])["Rev/season"].sum().unstack()
    print(" Sum of mean revenue/season over the 4 rivals, per market:")
    print(totals.round(0).to_string())
    print("=" * 96)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seasons", type=int, default=30)
    parser.add_argument("--compare", action="append", default=[],
                        help="extra duopoly-style file to benchmark (repeatable)")
    args = parser.parse_args()

    algorithms: Dict[str, Callable] = {"duopoly.py": duopoly.p}
    for path in args.compare:
        algorithms[os.path.basename(path)] = load_pricing_function(path)
    algorithms["naive rival-$1"] = naive_undercut

    df = run_3regime_benchmark(algorithms, n_seeds=args.seeds, n_seasons=args.seasons)
    print_report(df)


if __name__ == "__main__":
    main()
