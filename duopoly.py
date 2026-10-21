"""
duopoly.py
----------
Autonomous Dynamic Pricing Engine for Airline Duopoly Competition (v5.1).

Architectural Overview:
1. Target Depletion Curve (Continuous Inventory Pacing):
   Maintains sales progress along TargetSold(t) = 80 * (0.85*tau + 0.15*tau^2).
2. Multi-Tier Competitor Archetype Profiling:
   Identifies rivals as deep dumper, dumper, aggressive, balanced, or high pricer.
3. Adaptive WTP-Calibrated Monopoly Engine:
   Dynamically identifies low-WTP environments and clears seats at the empirical $52–$64 sweet spot.
4. Robust Duopoly Pacing & Circuit Breakers:
   Smooth-scaled undercutting for balanced rivals and fallback triggers for persistent low-price deadlocks.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# -----------------------------------------------------------------------------
# Domain Constants
# -----------------------------------------------------------------------------
MIN_PRICE: float = 25.0
MAX_PRICE: float = 100.0
DEFAULT_TOTAL_CAPACITY: int = 80
TOTAL_SELLING_PERIODS: int = 100
SERVER_SOLDOUT_PRICE_THRESHOLD: float = 200.0
TERMINAL_SOLDOUT_PRICE: float = 95.0
DEFAULT_NEUTRAL_PRICE: float = 65.0


# -----------------------------------------------------------------------------
# Main Public Entrypoint (Competition Server Interface)
# -----------------------------------------------------------------------------
def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: Union[np.ndarray, List[float], None],
    demand_historical_in_current_season: Union[np.ndarray, List[int], None],
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump: Optional[Any] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Main dynamic pricing decision function called by the competition simulator each period.
    Adheres strictly to Clean Code Single Responsibility and Step-down Rules.
    """
    state = _initialize_or_get_state(information_dump)
    period = selling_period_in_current_season

    # 1. Terminal / End-of-Season Phase
    if period > TOTAL_SELLING_PERIODS:
        return _handle_end_of_season(
            prices_historical_in_current_season,
            demand_historical_in_current_season,
            state
        )

    # 2. Inventory & Capacity Status
    units_sold = _calculate_units_sold(demand_historical_in_current_season)
    remaining_capacity = max(0, state["total_capacity"] - units_sold)
    
    if remaining_capacity <= 0:
        return TERMINAL_SOLDOUT_PRICE, state

    remaining_periods = max(1, (TOTAL_SELLING_PERIODS + 1) - period)
    required_burn_rate = remaining_capacity / remaining_periods

    # 3. Process In-Season Environmental Signals
    _update_real_time_signals(
        period,
        prices_historical_in_current_season,
        demand_historical_in_current_season,
        competitor_has_capacity_current_period_in_current_season,
        state
    )

    # 4. Inventory Pacing Position
    inventory_gap = _calculate_inventory_gap(
        period,
        remaining_capacity,
        state["total_capacity"]
    )

    # 5. Pricing Engine (Monopoly vs Duopoly Regimes)
    if not competitor_has_capacity_current_period_in_current_season:
        raw_price = _calculate_monopoly_price(
            required_burn_rate,
            remaining_capacity,
            state
        )
    else:
        raw_price = _calculate_duopoly_price(
            period,
            current_selling_season,
            inventory_gap,
            required_burn_rate,
            remaining_capacity,
            state
        )

    # 6. Safety Bounds & State Persistence
    final_price = _clamp_price(raw_price)
    _persist_decision(final_price, period, state)

    return final_price, state


# -----------------------------------------------------------------------------
# 1. State Management & Initialization
# -----------------------------------------------------------------------------
def _initialize_or_get_state(information_dump: Optional[Any]) -> Dict[str, Any]:
    """Ensures information_dump exists and contains all required tracking structures."""
    if isinstance(information_dump, dict) and "total_capacity" in information_dump:
        return information_dump

    return {
        "total_capacity": DEFAULT_TOTAL_CAPACITY,
        "rival_season_medians": [],
        "rival_ema_price": DEFAULT_NEUTRAL_PRICE,
        "rival_profile": "balanced",
        "season_sales_history": [],
        "rival_sold_out_seasons": 0,
        "rival_sellout_periods": [],
        "my_last_price": DEFAULT_NEUTRAL_PRICE,
        "rival_last_price": DEFAULT_NEUTRAL_PRICE,
        "price_demand_stats": {},           # bucket -> [total_demand, count_periods]
        "current_season_sellout_t": None,
        "failed_holdout_seasons": 0,        # Count seasons where deep dumper holdout got < 15 sales
        "consecutive_low_sales": 0,         # Count seasons where sales were < 42
        "detected_low_wtp": False           # Dynamically set when high prices (> $70) fail to convert
    }


# -----------------------------------------------------------------------------
# 2. Real-Time Signal Extraction & History Tracking
# -----------------------------------------------------------------------------
def _calculate_units_sold(demand_history: Union[np.ndarray, List[int], None]) -> int:
    """Calculates total units sold so far in the current selling season."""
    if demand_history is None or len(demand_history) == 0:
        return 0
    return int(np.sum(demand_history))


def _extract_last_demand(demand_history: Union[np.ndarray, List[int], None]) -> float:
    """Extracts realized demand from the most recent completed period."""
    if demand_history is None or len(demand_history) == 0:
        return 0.0
    return float(demand_history[-1])


def _extract_latest_rival_price(
    prices_history: Union[np.ndarray, List[float], None]
) -> Optional[float]:
    """Safely extracts the most recent valid rival price from the 2xT price array."""
    if prices_history is None:
        return None

    if isinstance(prices_history, np.ndarray) and prices_history.ndim == 2:
        if prices_history.shape[1] > 0 and prices_history.shape[0] >= 2:
            raw_price = float(prices_history[1, -1])
            if raw_price < SERVER_SOLDOUT_PRICE_THRESHOLD:
                return raw_price
    elif isinstance(prices_history, (list, np.ndarray)) and len(prices_history) > 0:
        raw_price = float(prices_history[-1])
        if raw_price < SERVER_SOLDOUT_PRICE_THRESHOLD:
            return raw_price

    return None


def _update_real_time_signals(
    period: int,
    prices_history: Union[np.ndarray, List[float], None],
    demand_history: Union[np.ndarray, List[int], None],
    has_rival_capacity: bool,
    state: Dict[str, Any]
) -> None:
    """Updates EMA tracking, rival sellout timing, and price-response statistics."""
    # 1. Track first sellout period for competitor
    if not has_rival_capacity and state.get("current_season_sellout_t") is None:
        state["current_season_sellout_t"] = period

    # 2. Update rival price and EMA
    latest_rival = _extract_latest_rival_price(prices_history)
    if latest_rival is not None:
        state["rival_last_price"] = latest_rival
        if has_rival_capacity and latest_rival > 5.0:
            smoothing_factor = 0.25
            current_ema = state.get("rival_ema_price", DEFAULT_NEUTRAL_PRICE)
            updated_ema = smoothing_factor * latest_rival + (1.0 - smoothing_factor) * current_ema
            state["rival_ema_price"] = float(updated_ema)

    # 3. Log realized demand against our previous price bucket ($5 granularity)
    if (
        isinstance(prices_history, np.ndarray)
        and prices_history.ndim == 2
        and prices_history.shape[1] > 0
    ):
        our_prev_price = float(prices_history[0, -1])
        last_demand = _extract_last_demand(demand_history)
        bucket = int(round(our_prev_price / 5.0) * 5)
        stats = state["price_demand_stats"].get(bucket, [0.0, 0])
        stats[0] += last_demand
        stats[1] += 1
        state["price_demand_stats"][bucket] = stats


# -----------------------------------------------------------------------------
# 3. Inventory Pacing Calculus
# -----------------------------------------------------------------------------
def _calculate_inventory_gap(
    current_period: int,
    remaining_capacity: int,
    total_capacity: int
) -> float:
    """
    Computes the difference between actual remaining inventory and target booking curve:
    Gap > 0: Behind pace (surplus unsold seats) -> requires undercutting.
    Gap < 0: Ahead of pace (scarcity) -> permits price premium.
    """
    progress = (current_period - 1) / float(TOTAL_SELLING_PERIODS)
    target_depleted = total_capacity * (0.85 * progress + 0.15 * (progress ** 2))
    target_remaining = total_capacity - target_depleted
    return float(remaining_capacity - target_remaining)


# -----------------------------------------------------------------------------
# 4. Competitor Archetype Classification
# -----------------------------------------------------------------------------
def _classify_rival_archetype(median_price: float) -> str:
    """Maps historical competitor median price to a strategic archetype."""
    if median_price <= 20.0:
        return "deep_dumper"
    if median_price <= 38.0:
        return "dumper"
    if median_price <= 58.0:
        return "aggressive"
    if median_price <= 78.0:
        return "balanced"
    return "high"


# -----------------------------------------------------------------------------
# 5. Monopoly Regime Pricing Engine
# -----------------------------------------------------------------------------
def _calculate_monopoly_price(
    required_burn_rate: float,
    remaining_capacity: int,
    state: Dict[str, Any]
) -> float:
    """
    Prices dynamically when rival is out of stock.
    Adapts automatically between Standard WTP and Low-WTP market regimes.
    """
    is_deep_dumper = state.get("rival_profile") == "deep_dumper"
    failed_holdouts = state.get("failed_holdout_seasons", 0)
    is_low_wtp = state.get("detected_low_wtp", False)

    # Case 1: Post-Deep-Dumper Holdout Clearance
    if is_deep_dumper:
        if failed_holdouts >= 1:
            return 54.0 if required_burn_rate <= 1.0 else 46.0
        return 64.0 if required_burn_rate <= 1.2 else 58.0

    # Case 2: Low-WTP Empirical Sweet Spot (data/v5 Calibrated)
    if is_low_wtp:
        if required_burn_rate <= 0.40 and remaining_capacity <= 8:
            return 64.0
        if required_burn_rate <= 0.70:
            return 60.0
        if required_burn_rate <= 1.05:
            return 56.0
        if required_burn_rate <= 1.35:
            return 52.0
        return max(46.0, 52.0 - (required_burn_rate - 1.35) * 10.0)

    # Case 3: Standard Multi-Tier Monopoly
    if required_burn_rate <= 0.35 and remaining_capacity <= 8:
        return 92.0
    if required_burn_rate <= 0.55:
        return 84.0
    if required_burn_rate <= 0.85:
        return 78.0
    if required_burn_rate <= 1.25:
        return 72.0
    return max(48.0, 72.0 - (required_burn_rate - 1.25) * 14.0)


# -----------------------------------------------------------------------------
# 6. Duopoly Regime Pricing Engine
# -----------------------------------------------------------------------------
def _calculate_duopoly_price(
    period: int,
    season: int,
    inventory_gap: float,
    required_burn_rate: float,
    remaining_capacity: int,
    state: Dict[str, Any]
) -> float:
    """Computes asking price during active duopoly competition against stock-bearing rival."""
    rival_profile = state.get("rival_profile", "balanced")
    latest_rival = state.get("rival_last_price")
    rival_ema = state.get("rival_ema_price", DEFAULT_NEUTRAL_PRICE)
    failed_holdouts = state.get("failed_holdout_seasons", 0)
    consecutive_low_sales = state.get("consecutive_low_sales", 0)

    is_deep_dumper = (
        rival_profile == "deep_dumper"
        or (latest_rival is not None and latest_rival <= 19.5 and rival_ema <= 19.5)
    )

    # Branch 1: Deep Dumper Defense (FuturisticSalamander / RussetBat)
    if is_deep_dumper:
        if failed_holdouts >= 2:
            anchor = 34.0 if inventory_gap > 3.0 else 38.0
            return anchor - min(6.0, 1.0 + 0.3 * max(0.0, inventory_gap))
        return 58.0

    # Branch 2: Universal Low-Sales Circuit Breaker (Cures 3d8tDh SageSloth deadlock)
    if consecutive_low_sales >= 2:
        hist_med = (
            float(np.median(state["rival_season_medians"]))
            if len(state["rival_season_medians"]) > 0
            else rival_ema
        )
        breaker_price = min(hist_med - 0.5, rival_ema - 0.5)
        if inventory_gap > 3.0:
            breaker_price -= min(4.0, 0.35 * inventory_gap)
        return breaker_price

    # Branch 3: Standard Dynamic Pacing Engine
    anchor_price = _determine_duopoly_anchor(period, season, inventory_gap, state)
    target_price = _apply_pacing_differential(anchor_price, inventory_gap)

    # Guard 1: High-Price Ceiling Cap in Duopoly
    if target_price > 54.0 and inventory_gap > 2.5 and period >= 15:
        target_price = 52.0

    # Guard 2: Mid-Season Micro-Probing
    if 15 <= period <= 65 and abs(inventory_gap) <= 3.5:
        if period % 6 == 0:
            target_price += 3.5
        elif period % 6 == 3:
            target_price -= 2.5

    # Guard 3: Late-Season Opportunistic Exploitation & Spoilage Ramp (t >= 75)
    if period >= 75:
        target_price = _apply_late_season_clearance(
            period,
            target_price,
            anchor_price,
            rival_ema,
            inventory_gap,
            required_burn_rate,
            remaining_capacity
        )

    return target_price


def _determine_duopoly_anchor(
    period: int,
    season: int,
    inventory_gap: float,
    state: Dict[str, Any]
) -> float:
    """Computes smoothed rival price anchor with follower and anti-dumping guard."""
    rival_ema = state.get("rival_ema_price", DEFAULT_NEUTRAL_PRICE)
    rival_medians = state.get("rival_season_medians", [])

    if season == 1 and period <= 5:
        return 68.0

    if len(rival_medians) > 0:
        hist_med = float(np.median(rival_medians))
        anchor = 0.65 * rival_ema + 0.35 * hist_med
    else:
        anchor = rival_ema

    # Anti-Dumping & Copycat Guard
    if anchor < 38.0:
        anchor = float(np.clip(44.0 - 0.8 * max(0.0, inventory_gap - 1.5), 35.0, 44.0))

    return anchor


def _apply_pacing_differential(anchor_price: float, inventory_gap: float) -> float:
    """Applies smooth undercutting when behind pace and scarcity premium when ahead."""
    if inventory_gap > 1.5:
        undercut = min(7.50, 0.80 + 0.28 * (inventory_gap - 1.5))
        return anchor_price - undercut
    if inventory_gap < -2.5:
        premium = min(14.0, 1.0 + 0.45 * abs(inventory_gap))
        return anchor_price + premium
    return anchor_price - 0.80


def _apply_late_season_clearance(
    period: int,
    current_target: float,
    anchor_price: float,
    rival_ema: float,
    inventory_gap: float,
    required_burn_rate: float,
    remaining_capacity: int
) -> float:
    r"""
    Handles late clearance ($t \ge 75$):
    - Counters rivals that spike price late (e.g. ConvivialCrane) by holding $52–$54.
    - Accelerates liquidation when inventory is behind schedule to guarantee zero spoilage.
    """
    if rival_ema >= 54.0 and required_burn_rate <= 1.15:
        return min(54.0, rival_ema - 2.0)

    if inventory_gap > 3.0 or (period >= 90 and remaining_capacity >= 2):
        urgency_factor = (period - 74) / 26.0
        spoilage_discount = urgency_factor * max(0.0, (required_burn_rate - 0.80) * 12.0)
        return min(current_target, anchor_price - 1.5) - spoilage_discount

    return current_target


# -----------------------------------------------------------------------------
# 7. End-of-Season Inter-Season Processing (Period 101)
# -----------------------------------------------------------------------------
def _handle_end_of_season(
    prices_history: Union[np.ndarray, List[float], None],
    demand_history: Union[np.ndarray, List[int], None],
    state: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """Updates cross-season learning metrics, archetype classification, and low-WTP flags."""
    # 1. Update rival price medians
    if (
        isinstance(prices_history, np.ndarray)
        and prices_history.ndim == 2
        and prices_history.shape[0] >= 2
    ):
        rival_season_prices = prices_history[1, :]
        valid_prices = rival_season_prices[rival_season_prices < SERVER_SOLDOUT_PRICE_THRESHOLD]
        if len(valid_prices) > 0:
            season_med = float(np.median(valid_prices))
            state["rival_season_medians"].append(season_med)
            overall_med = float(np.median(state["rival_season_medians"]))
            state["rival_profile"] = _classify_rival_archetype(overall_med)

    # 2. Track season sales history and circuit breaker alarms
    units_sold = _calculate_units_sold(demand_history)
    state["season_sales_history"].append(units_sold)

    if state.get("rival_profile") == "deep_dumper":
        state["failed_holdout_seasons"] = (
            state.get("failed_holdout_seasons", 0) + 1 if units_sold < 15 else 0
        )

    state["consecutive_low_sales"] = (
        state.get("consecutive_low_sales", 0) + 1 if units_sold < 42 else 0
    )

    # 3. Track rival sellout timing
    sellout_period = (
        state["current_season_sellout_t"]
        if state.get("current_season_sellout_t") is not None
        else (TOTAL_SELLING_PERIODS + 1)
    )
    state["rival_sellout_periods"].append(sellout_period)
    if sellout_period <= TOTAL_SELLING_PERIODS:
        state["rival_sold_out_seasons"] = state.get("rival_sold_out_seasons", 0) + 1

    # 4. Check empirical WTP ceiling from transactions
    stats_dict = state.get("price_demand_stats", {})
    high_buckets = [b for b in stats_dict.keys() if b >= 70]
    high_periods = sum(stats_dict[b][1] for b in high_buckets)
    high_sales = sum(stats_dict[b][0] for b in high_buckets)
    
    rival_medians = state.get("rival_season_medians", [])
    if (high_periods >= 5 and high_sales == 0) or (
        len(rival_medians) >= 2
        and np.median(rival_medians) <= 53.0
        and max(rival_medians) <= 56.0
        and high_sales == 0
    ):
        state["detected_low_wtp"] = True

    state["current_season_sellout_t"] = None
    return DEFAULT_NEUTRAL_PRICE, state


# -----------------------------------------------------------------------------
# 8. Bounds Enforcement & Decision Persistence
# -----------------------------------------------------------------------------
def _clamp_price(price: float) -> float:
    """Clamps asking price strictly within permissible competition boundaries ($25–$100)."""
    return float(np.clip(price, MIN_PRICE, MAX_PRICE))


def _persist_decision(price: float, period: int, state: Dict[str, Any]) -> None:
    """Records the chosen price and period in state dictionary."""
    state["my_last_price"] = price
    state["last_period_priced"] = period
