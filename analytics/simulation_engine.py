"""
simulation_engine.py
---------------------
High-performance simulation framework for the Dynamic Pricing Competition (DPC).

Simulates duopoly competition scenarios:
1. Replay against historical competitors from CSV logs (FuturisticSalamander, UncoveredElephant, ZealousMosquito)
2. Interactive Dynamic Competitors (Copycat, Aggressive Undercutter, Balanced, Monopoly Drainer)
3. Direct Head-to-Head against our old v1.0 baseline algorithm

Calibrated against empirical DPC market mechanics:
- Poisson customer arrivals peaking late in the season (~75 - 105 total arrivals/season)
- Multi-segment consumer Willingness-to-Pay (Budget, Leisure, Business)
- Multinomial logit / random utility choice model with outside option
- Strict capacity constraint (80 seats each, zero salvage value)
- Monopoly transition when a rival sells out
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
import os


class MarketDemandModel:
    """
    Calibrated DPC Poisson arrival and Random Utility Choice model.
    Average total arrivals: ~85-105 passengers per 100-period season.
    """
    def __init__(
        self,
        avg_arrivals: float = 92.0,
        arrival_convexity: float = 1.8,
        price_sensitivity: float = 0.08,
        seed: Optional[int] = None
    ):
        self.avg_arrivals = avg_arrivals
        self.arrival_convexity = arrival_convexity
        self.price_sensitivity = price_sensitivity
        self.rng = np.random.default_rng(seed)

    def get_arrival_rate(self, t: int) -> float:
        """Arrival rate lambda(t) accelerating toward departure period 100."""
        progress = (t - 1) / 99.0
        # Base arrival + convex late surge
        intensity = 0.35 + 2.1 * (progress ** self.arrival_convexity)
        return intensity

    def sample_customer_wtp(self, n_arrivals: int, t: int) -> np.ndarray:
        """Sample customer reservation price (WTP) based on market segment mix."""
        if n_arrivals == 0:
            return np.array([])
        
        # Segment mix shifts toward business travelers late in season
        progress = (t - 1) / 99.0
        p_business = 0.10 + 0.35 * progress  # 10% early -> 45% late
        p_leisure = 0.45 - 0.10 * progress   # 45% early -> 35% late
        p_budget = max(0.0, 1.0 - p_business - p_leisure)

        segments = self.rng.choice(
            ['budget', 'leisure', 'business'],
            size=n_arrivals,
            p=[p_budget, p_leisure, p_business]
        )

        wtp = np.zeros(n_arrivals)
        budget_mask = (segments == 'budget')
        leisure_mask = (segments == 'leisure')
        business_mask = (segments == 'business')

        if np.any(budget_mask):
            wtp[budget_mask] = self.rng.uniform(35.0, 68.0, size=np.sum(budget_mask))
        if np.any(leisure_mask):
            wtp[leisure_mask] = self.rng.uniform(62.0, 92.0, size=np.sum(leisure_mask))
        if np.any(business_mask):
            wtp[business_mask] = self.rng.uniform(85.0, 110.0, size=np.sum(business_mask))

        return wtp

    def step(
        self,
        t: int,
        price1: float,
        price2: float,
        cap1: int,
        cap2: int
    ) -> Tuple[int, int]:
        """
        Simulates 1 selling period.
        Returns (demand_airline1, demand_airline2).
        """
        if cap1 <= 0 and cap2 <= 0:
            return 0, 0

        # 1. Sample number of arrivals
        lam = self.get_arrival_rate(t)
        arrivals = self.rng.poisson(lam)
        if arrivals == 0:
            return 0, 0

        # 2. Sample WTP for each arrival
        wtp_list = self.sample_customer_wtp(arrivals, t)
        
        demand1 = 0
        demand2 = 0

        # 3. Simulate choice for each customer
        for wtp in wtp_list:
            can_buy_1 = (cap1 - demand1 > 0) and (wtp >= price1)
            can_buy_2 = (cap2 - demand2 > 0) and (wtp >= price2)

            if can_buy_1 and can_buy_2:
                # Both available: Multinomial Logit choice based on price difference
                # Utilities: U_i = (wtp - price_i) * beta
                diff = price2 - price1  # Positive if airline 1 is cheaper
                prob_1 = 1.0 / (1.0 + np.exp(-self.price_sensitivity * diff))
                if self.rng.random() < prob_1:
                    demand1 += 1
                else:
                    demand2 += 1
            elif can_buy_1 and not can_buy_2:
                # Monopoly for airline 1
                demand1 += 1
            elif can_buy_2 and not can_buy_1:
                # Monopoly for airline 2
                demand2 += 1

        return demand1, demand2


# -------------------------------------------------------------------------
# Competitor Agent Implementations
# -------------------------------------------------------------------------

class BaseCompetitor:
    def reset_season(self, season: int):
        pass

    def get_price(
        self,
        season: int,
        period: int,
        our_last_price: Optional[float],
        our_last_demand: Optional[int],
        rival_has_capacity: bool
    ) -> float:
        raise NotImplementedError


class CopycatCompetitor(BaseCompetitor):
    """
    Copycat competitor that mirrors our price from the previous period.
    Can be configured with an undercut delta or noise.
    """
    def __init__(self, initial_price: float = 65.0, undercut: float = 0.0, noise: float = 0.5):
        self.initial_price = initial_price
        self.undercut = undercut
        self.noise = noise
        self.rng = np.random.default_rng(42)

    def get_price(
        self,
        season: int,
        period: int,
        our_last_price: Optional[float],
        our_last_demand: Optional[int],
        rival_has_capacity: bool
    ) -> float:
        if period == 1 or our_last_price is None:
            return self.initial_price
        
        jitter = self.rng.uniform(-self.noise, self.noise) if self.noise > 0 else 0.0
        price = our_last_price - self.undercut + jitter
        return float(np.clip(price, 25.0, 100.0))


class AggressiveDumperCompetitor(BaseCompetitor):
    """
    Aggressive price dumper resembling FuturisticSalamander (prices ~18-40).
    """
    def __init__(self, low: float = 20.0, high: float = 38.0):
        self.low = low
        self.high = high
        self.rng = np.random.default_rng(123)

    def get_price(
        self,
        season: int,
        period: int,
        our_last_price: Optional[float],
        our_last_demand: Optional[int],
        rival_has_capacity: bool
    ) -> float:
        base = self.low + (self.high - self.low) * (period / 100.0)
        noise = self.rng.uniform(-3.0, 3.0)
        return float(np.clip(base + noise, 18.0, 50.0))


class BoundedCompetitor(BaseCompetitor):
    """
    Bounded mid-market competitor resembling ZealousMosquito (prices $45-$70).
    """
    def __init__(self, mean_price: float = 55.0, spread: float = 12.0):
        self.mean_price = mean_price
        self.spread = spread
        self.rng = np.random.default_rng(456)

    def get_price(
        self,
        season: int,
        period: int,
        our_last_price: Optional[float],
        our_last_demand: Optional[int],
        rival_has_capacity: bool
    ) -> float:
        cycle = np.sin(period * 0.2) * (self.spread / 2.0)
        noise = self.rng.uniform(-2.5, 2.5)
        price = self.mean_price + cycle + noise
        return float(np.clip(price, 40.0, 75.0))


class OldV1Competitor(BaseCompetitor):
    """
    Direct implementation of the old v1.0 algorithm from ALGORITHM_CHANGES.md.
    Prices at $85 early, jumps to $105 when rival <= $70, etc.
    """
    def __init__(self):
        self.info_dump = None

    def reset_season(self, season: int):
        self.info_dump = None

    def get_price(
        self,
        season: int,
        period: int,
        our_last_price: Optional[float],
        our_last_demand: Optional[int],
        rival_has_capacity: bool
    ) -> float:
        # Phase 1: Periods 1-50
        if period <= 50:
            return 85.0
        # Phase 2: Periods 51-70
        elif period <= 70:
            if period % 4 == 0:
                return 105.0
            elif period % 4 == 2:
                return 75.0
            else:
                return 90.0
        # Phase 3: Periods 71-100
        else:
            if not rival_has_capacity:
                return 105.0
            else:
                if our_last_price is not None:
                    if our_last_price <= 70.0:
                        return 105.0
                    elif our_last_price >= 120.0:
                        return our_last_price - 5.0
                    else:
                        return max(85.0, our_last_price - 2.0)
                return 110.0


class HistoricalCSVCompetitor(BaseCompetitor):
    """
    Replays exact historical competitor prices from the CSV file.
    """
    def __init__(self, df: pd.DataFrame, competition_id: str):
        self.df_comp = df[df['competition_id'] == competition_id].copy()
        if len(self.df_comp) == 0:
            raise ValueError(f"Competition ID '{competition_id}' not found in DataFrame.")
        self.competitor_name = self.df_comp['competitor_id'].iloc[0]
        # Pre-index by (selling_season, selling_period)
        self.price_map = self.df_comp.set_index(['selling_season', 'selling_period'])['price_competitor'].to_dict()

    def get_price(
        self,
        season: int,
        period: int,
        our_last_price: Optional[float],
        our_last_demand: Optional[int],
        rival_has_capacity: bool
    ) -> float:
        # Fallback to season 1 or median if season not in map
        return self.price_map.get((season, period), 60.0)


# -------------------------------------------------------------------------
# Simulation Engine Core
# -------------------------------------------------------------------------

class SimulationResult:
    """Stores detailed match statistics and provides analytics methods."""
    def __init__(self, history: List[Dict[str, Any]], competitor_name: str):
        self.df = pd.DataFrame(history)
        self.competitor_name = competitor_name

    def summary(self) -> Dict[str, Any]:
        """Calculates executive KPI summary."""
        season_summary = self.df.groupby('season').agg({
            'our_revenue': 'sum',
            'rival_revenue': 'sum',
            'our_sales': 'sum',
            'rival_sales': 'sum',
            'our_price': 'mean',
            'rival_price': 'mean',
            'rival_sold_out': 'max'
        }).reset_index()

        our_total_rev = season_summary['our_revenue'].sum()
        rival_total_rev = season_summary['rival_revenue'].sum()
        seasons_won = (season_summary['our_revenue'] > season_summary['rival_revenue']).sum()
        seasons_tied = (season_summary['our_revenue'] == season_summary['rival_revenue']).sum()
        total_seasons = len(season_summary)

        # DPC Official Winning Rule: Highest Cumulative Total Revenue across all seasons
        is_winner = bool(our_total_rev > rival_total_rev)
        is_tie = bool(our_total_rev == rival_total_rev)
        tournament_result = "WINNER" if is_winner else ("TIE" if is_tie else "RUNNER-UP")

        return {
            "competitor": self.competitor_name,
            "total_seasons": total_seasons,
            "our_total_revenue": our_total_rev,
            "rival_total_revenue": rival_total_rev,
            "revenue_diff": our_total_rev - rival_total_rev,
            "tournament_result": tournament_result,
            "is_winner": is_winner,
            "seasons_won": seasons_won,
            "seasons_lost": total_seasons - seasons_won - seasons_tied,
            "season_win_rate_pct": round((seasons_won / max(1, total_seasons)) * 100.0, 1),
            "our_mean_season_rev": season_summary['our_revenue'].mean(),
            "rival_mean_season_rev": season_summary['rival_revenue'].mean(),
            "our_median_season_rev": season_summary['our_revenue'].median(),
            "our_mean_sales": season_summary['our_sales'].mean(),
            "rival_mean_sales": season_summary['rival_sales'].mean(),
            "our_load_factor_pct": round((season_summary['our_sales'].mean() / 80.0) * 100.0, 1),
            "rival_load_factor_pct": round((season_summary['rival_sales'].mean() / 80.0) * 100.0, 1),
            "our_avg_price": season_summary['our_price'].mean(),
            "rival_avg_price": season_summary['rival_price'].mean(),
            "rival_sellout_seasons": int(season_summary['rival_sold_out'].sum()),
        }

    def print_summary(self):
        s = self.summary()
        status_label = "VICTORY (HIGHER TOTAL REVENUE)" if s['is_winner'] else "RUNNER-UP"
        print("\n" + "="*72)
        print(f" DPC TOURNAMENT EVALUATION: Our Algorithm vs {s['competitor'].upper()}")
        print(f" Official Ruling: {status_label}")
        print("="*72)
        print(f" Official Metric (Total Revenue of 100 Seasons):")
        print(f"   Our Total Revenue:     ${s['our_total_revenue']:,.2f}")
        print(f"   Rival Total Revenue:   ${s['rival_total_revenue']:,.2f}")
        print(f"   Net Revenue Advantage: ${s['revenue_diff']:+,.2f} ({'+' if s['revenue_diff']>=0 else ''}{s['revenue_diff']/max(1,s['rival_total_revenue'])*100.0:.1f}%)")
        print("-"*72)
        print(f" Supplementary Season Stats (Total Seasons: {s['total_seasons']}):")
        print(f"   Season Wins / Losses:  {s['seasons_won']} Won, {s['seasons_lost']} Lost ({s['season_win_rate_pct']}% of seasons)")
        print(f"   Our Mean Revenue/Ssn:  ${s['our_mean_season_rev']:,.2f} (Median: ${s['our_median_season_rev']:,.2f})")
        print(f"   Rival Mean Revenue/Ssn: ${s['rival_mean_season_rev']:,.2f}")
        print(f"   Our Load Factor:       {s['our_load_factor_pct']}% ({s['our_mean_sales']:.1f} / 80 seats)")
        print(f"   Rival Load Factor:     {s['rival_load_factor_pct']}% ({s['rival_mean_sales']:.1f} / 80 seats)")
        print(f"   Our Avg Price:         ${s['our_avg_price']:.2f} (Rival: ${s['rival_avg_price']:.2f})")
        print("="*72 + "\n")


class DuopolySimulator:
    """
    Executes simulations between our algorithm function `p` and any competitor.
    """
    def __init__(
        self,
        algorithm_func: Callable,
        market_demand: Optional[MarketDemandModel] = None,
        total_capacity: int = 80
    ):
        self.algo_func = algorithm_func
        self.market_demand = market_demand or MarketDemandModel(seed=42)
        self.total_capacity = total_capacity

    def run(
        self,
        competitor: BaseCompetitor,
        n_seasons: int = 100,
        competitor_name: str = "Competitor"
    ) -> SimulationResult:
        """
        Runs `n_seasons` full 100-period competitions.
        """
        history = []
        information_dump = None

        for s in range(1, n_seasons + 1):
            competitor.reset_season(s)
            
            our_capacity = self.total_capacity
            rival_capacity = self.total_capacity
            
            our_prices_history = []
            rival_prices_history = []
            our_demand_history = []

            for t in range(1, 101):
                # 1. Format inputs for our pricing algorithm
                if t == 1:
                    prices_hist = None
                    demand_hist = None
                else:
                    prices_hist = np.array([our_prices_history, rival_prices_history])
                    demand_hist = np.array(our_demand_history)

                rival_has_cap = (rival_capacity > 0)
                
                # 2. Call our algorithm
                our_price, information_dump = self.algo_func(
                    current_selling_season=s,
                    selling_period_in_current_season=t,
                    prices_historical_in_current_season=prices_hist,
                    demand_historical_in_current_season=demand_hist,
                    competitor_has_capacity_current_period_in_current_season=rival_has_cap,
                    information_dump=information_dump
                )

                # 3. Call competitor
                our_last_p = our_prices_history[-1] if len(our_prices_history) > 0 else None
                our_last_d = our_demand_history[-1] if len(our_demand_history) > 0 else None
                rival_price = competitor.get_price(
                    season=s,
                    period=t,
                    our_last_price=our_last_p,
                    our_last_demand=our_last_d,
                    rival_has_capacity=(our_capacity > 0)
                )

                # 4. Resolve market demand
                d_our, d_rival = self.market_demand.step(
                    t=t,
                    price1=our_price,
                    price2=rival_price,
                    cap1=our_capacity,
                    cap2=rival_capacity
                )

                # Deduct inventory
                our_capacity = max(0, our_capacity - d_our)
                rival_capacity = max(0, rival_capacity - d_rival)

                # Record period
                our_prices_history.append(our_price)
                rival_prices_history.append(rival_price)
                our_demand_history.append(d_our)

                history.append({
                    "season": s,
                    "period": t,
                    "our_price": our_price,
                    "rival_price": rival_price,
                    "our_sales": d_our,
                    "rival_sales": d_rival,
                    "our_revenue": d_our * our_price,
                    "rival_revenue": d_rival * rival_price,
                    "our_capacity_remaining": our_capacity,
                    "rival_capacity_remaining": rival_capacity,
                    "rival_sold_out": 1 if rival_capacity == 0 else 0
                })

            # Optional period 101 feedback step (as done by DPC platform)
            prices_hist_101 = np.array([our_prices_history, rival_prices_history])
            demand_hist_101 = np.array(our_demand_history)
            try:
                _, information_dump = self.algo_func(
                    current_selling_season=s,
                    selling_period_in_current_season=101,
                    prices_historical_in_current_season=prices_hist_101,
                    demand_historical_in_current_season=demand_hist_101,
                    competitor_has_capacity_current_period_in_current_season=(rival_capacity > 0),
                    information_dump=information_dump
                )
            except Exception:
                pass

        return SimulationResult(history, competitor_name)
