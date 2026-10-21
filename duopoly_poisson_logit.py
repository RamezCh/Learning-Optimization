"""
duopoly_poisson_logit.py
------------------------
Autonomous Dynamic Pricing Engine: Poisson Arrivals & Logistic Discrete Choice + Opportunity Cost Pacing (v5.1).

Architectural Overview:
1. Poisson Arrival Process Estimation:
   Models time-dependent customer arrivals: λ(t) = exp(α₀ + α₁·τ^1.5) where τ = (t-1)/99.
2. Logistic Discrete Choice Conversion Model:
   Predicts conversion probability:
   P(sale | p_our, p_rival) = 1 / [1 + exp(-(γ₀ + γ_diff·(p_rival - p_our) - γ_own·(p_our - 30)))].
3. Dynamic Expected Surplus Maximization:
   Finds p* = argmax_p [ (p - μ(t, c)) · λ(t) · P(sale | p, p_rival) ] subject to capacity opportunity cost μ(t, c).
4. Adaptive Empirical Monopoly Engine & Dynamic WTP Ceiling:
   Adapts clearance prices to empirical WTP limits ($52–$64 sweet spot) when low-WTP environments are detected.
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
DEFAULT_RIVAL_PRICE: float = 50.0

# Calibrated priors from multi-round empirical data (v3 + v4 + v5)
PRIOR_ALPHA_0: float = -0.15     # Baseline arrival log-intensity
PRIOR_ALPHA_1: float = 1.30      # Arrival acceleration parameter
PRIOR_GAMMA_0: float = 2.10      # Choice model intercept
PRIOR_GAMMA_DIFF: float = 0.045  # Undercutting sensitivity
PRIOR_GAMMA_OWN: float = 0.055   # Absolute price resistance


# -----------------------------------------------------------------------------
# Main Public Entrypoint (Competition Server Interface)
# -----------------------------------------------------------------------------
def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: Union[np.ndarray, List[float], None],
    demand_historical_in_current_season: Union[np.ndarray, List[int], None],
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump: Optional[Dict[str, Any]] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Main Poisson-Logistic dynamic pricing function adhering to Clean Code design.
    """
    state = _initialize_or_get_state(information_dump, current_selling_season)
    period = selling_period_in_current_season

    # 1. Season Boundary Transition & Online Arrival Adaptation
    if state["last_season_seen"] != current_selling_season:
        _handle_season_transition(
            current_selling_season,
            demand_historical_in_current_season,
            state
        )

    # 2. Terminal Period Check (Period 101 Feedback Only)
    if period > TOTAL_SELLING_PERIODS:
        return _clamp_price(state.get("my_last_price", DEFAULT_RIVAL_PRICE)), state

    # 3. Inventory & Capacity Status
    units_sold = _calculate_units_sold(demand_historical_in_current_season)
    remaining_capacity = max(0, state["total_capacity"] - units_sold)

    if remaining_capacity <= 0:
        return TERMINAL_SOLDOUT_PRICE, state

    remaining_periods = max(1, (TOTAL_SELLING_PERIODS + 1) - period)
    required_burn_rate = remaining_capacity / remaining_periods

    # 4. Process Environmental Feedback & Signal Updates
    _update_real_time_signals(
        period,
        prices_historical_in_current_season,
        demand_historical_in_current_season,
        state
    )

    # 5. Inventory Pacing Position
    inventory_gap = _calculate_inventory_gap(
        period,
        remaining_capacity,
        state["total_capacity"]
    )

    # 6. Pricing Decision Engine (Monopoly vs Duopoly Regimes)
    if not competitor_has_capacity_current_period_in_current_season:
        raw_price = _calculate_monopoly_price(
            required_burn_rate,
            remaining_capacity,
            state
        )
    else:
        raw_price = _calculate_duopoly_price(
            period,
            inventory_gap,
            required_burn_rate,
            remaining_capacity,
            state
        )

    # 7. Safety Bounds & State Persistence
    final_price = _clamp_price(raw_price)
    _persist_decision(final_price, state)

    return final_price, state


# -----------------------------------------------------------------------------
# 1. State Management & Initialization
# -----------------------------------------------------------------------------
def _initialize_or_get_state(
    information_dump: Optional[Any],
    current_season: int
) -> Dict[str, Any]:
    """Ensures information_dump exists and contains all required structures."""
    if isinstance(information_dump, dict) and "total_capacity" in information_dump:
        return information_dump

    return {
        "initialized": True,
        "total_capacity": DEFAULT_TOTAL_CAPACITY,
        "last_season_seen": current_season,
        "training_history": [],            # Stores [p_our, p_rival, diff, norm_t, demand]
        "rival_recent_prices": [],
        "rival_season_medians": [],
        "consecutive_low_sales": 0,
        "my_last_price": DEFAULT_NEUTRAL_PRICE,
        "rival_last_price": DEFAULT_NEUTRAL_PRICE,
        "price_demand_stats": {},
        "detected_low_wtp": False,
        "alpha_0": PRIOR_ALPHA_0,
        "alpha_1": PRIOR_ALPHA_1,
        "gamma_0": PRIOR_GAMMA_0,
        "gamma_diff": PRIOR_GAMMA_DIFF,
        "gamma_own": PRIOR_GAMMA_OWN
    }


# -----------------------------------------------------------------------------
# 2. Season Boundary & Online Model Adaptation
# -----------------------------------------------------------------------------
def _handle_season_transition(
    new_season: int,
    demand_history: Union[np.ndarray, List[int], None],
    state: Dict[str, Any]
) -> None:
    """Updates low sales streak, evaluates WTP ceiling, and adapts baseline arrival rate."""
    prev_season_sales = _calculate_units_sold(demand_history)
    state["consecutive_low_sales"] = (
        state.get("consecutive_low_sales", 0) + 1 if prev_season_sales < 42 else 0
    )

    _evaluate_wtp_ceiling(state)
    _adapt_arrival_intensity(state)

    state["last_season_seen"] = new_season
    state["rival_recent_prices"] = []


def _evaluate_wtp_ceiling(state: Dict[str, Any]) -> None:
    """Checks whether high-priced tickets (>= $70) fail to convert in this tournament."""
    stats_dict = state.get("price_demand_stats", {})
    high_buckets = [b for b in stats_dict.keys() if b >= 70]
    high_periods = sum(stats_dict[b][1] for b in high_buckets)
    high_sales = sum(stats_dict[b][0] for b in high_buckets)
    if high_periods >= 5 and high_sales == 0:
        state["detected_low_wtp"] = True


def _adapt_arrival_intensity(state: Dict[str, Any]) -> None:
    """Online Bayesian / exponential smoothing update of baseline arrival log-intensity."""
    history = state.get("training_history", [])
    if len(history) >= 150:
        recent_demands = [h[4] for h in history[-150:]]
        obs_rate = float(np.mean(recent_demands))
        if obs_rate > 0.05:
            state["alpha_0"] = float(0.85 * state["alpha_0"] + 0.15 * np.log(obs_rate))


# -----------------------------------------------------------------------------
# 3. Environmental Feedback & Real-Time Signals
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
            if 0.0 < raw_price < SERVER_SOLDOUT_PRICE_THRESHOLD:
                return raw_price
    elif isinstance(prices_history, (list, np.ndarray)) and len(prices_history) > 0:
        raw_price = float(prices_history[-1])
        if 0.0 < raw_price < SERVER_SOLDOUT_PRICE_THRESHOLD:
            return raw_price

    return None


def _update_real_time_signals(
    period: int,
    prices_history: Union[np.ndarray, List[float], None],
    demand_history: Union[np.ndarray, List[int], None],
    state: Dict[str, Any]
) -> None:
    """Logs observed price differentials and appends observations to history."""
    if period <= 1 or prices_history is None:
        return

    latest_rival = _extract_latest_rival_price(prices_history)
    my_prev_price = state.get("my_last_price", DEFAULT_RIVAL_PRICE)
    last_demand = _extract_last_demand(demand_history)

    if latest_rival is not None:
        state["rival_last_price"] = latest_rival
        recent_prices = state.setdefault("rival_recent_prices", [])
        recent_prices.append(latest_rival)
        if len(recent_prices) > 15:
            recent_prices.pop(0)

        price_diff = latest_rival - my_prev_price
        norm_t = (period - 1) / float(TOTAL_SELLING_PERIODS - 1)
        state.setdefault("training_history", []).append([
            my_prev_price, latest_rival, price_diff, norm_t, last_demand
        ])

        bucket = int(round(my_prev_price / 5.0) * 5)
        stats = state["price_demand_stats"].get(bucket, [0.0, 0])
        stats[0] += last_demand
        stats[1] += 1
        state["price_demand_stats"][bucket] = stats


# -----------------------------------------------------------------------------
# 4. Inventory Pacing Calculus
# -----------------------------------------------------------------------------
def _calculate_inventory_gap(
    current_period: int,
    remaining_capacity: int,
    total_capacity: int
) -> float:
    """
    Computes difference between actual inventory and target booking curve:
    Gap > 0: Behind pace (surplus unsold seats).
    Gap < 0: Ahead of pace (scarcity).
    """
    progress = (current_period - 1) / float(TOTAL_SELLING_PERIODS)
    target_depleted = total_capacity * (0.85 * progress + 0.15 * (progress ** 2))
    target_remaining = total_capacity - target_depleted
    return float(remaining_capacity - target_remaining)


# -----------------------------------------------------------------------------
# 5. Monopoly Regime Pricing Engine
# -----------------------------------------------------------------------------
def _calculate_monopoly_price(
    required_burn_rate: float,
    remaining_capacity: int,
    state: Dict[str, Any]
) -> float:
    """Calibrates monopoly price based on required burn rate and detected WTP ceiling."""
    is_low_wtp = state.get("detected_low_wtp", False)

    # Case 1: Low-WTP Calibrated Sweet Spot (data/v5)
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

    # Case 2: Standard Multi-Tier Monopoly
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
# 6. Duopoly Regime: Poisson-Logistic Surplus Maximization
# -----------------------------------------------------------------------------
def _calculate_duopoly_price(
    period: int,
    inventory_gap: float,
    required_burn_rate: float,
    remaining_capacity: int,
    state: Dict[str, Any]
) -> float:
    """Computes asking price during active duopoly using Poisson-Logistic surplus maximization."""
    latest_rival = state.get("rival_last_price", DEFAULT_RIVAL_PRICE)
    rival_recent_avg = _get_recent_rival_average(state, latest_rival)
    consecutive_low_sales = state.get("consecutive_low_sales", 0)

    # Branch 1: Universal Low-Sales Circuit Breaker
    if consecutive_low_sales >= 2:
        breaker_price = min(latest_rival - 0.5, rival_recent_avg - 0.5)
        if inventory_gap > 3.0:
            breaker_price -= min(4.0, 0.35 * inventory_gap)
        return breaker_price

    # Branch 2: Deep Dumper Holdout
    if latest_rival <= 19.5 and rival_recent_avg <= 19.5:
        return 58.0

    # Branch 3: Poisson-Logistic Expected Surplus Optimization
    target_price = _solve_surplus_maximizing_price(
        period,
        inventory_gap,
        rival_recent_avg,
        state
    )

    # Guard 1: Anti-Dumping & Copycat Protection
    if rival_recent_avg < 38.0:
        target_price = max(
            target_price,
            float(np.clip(44.0 - 0.8 * max(0.0, inventory_gap - 1.5), 35.0, 44.0))
        )

    # Guard 2: High-Price Ceiling Cap
    if target_price > 54.0 and inventory_gap > 2.5 and period >= 15:
        target_price = 52.0

    # Guard 3: Late-Season Clearance & High-Rival Counter
    if period >= 75:
        target_price = _apply_late_season_clearance(
            period,
            target_price,
            rival_recent_avg,
            inventory_gap,
            required_burn_rate,
            remaining_capacity
        )

    return target_price


def _get_recent_rival_average(state: Dict[str, Any], fallback: float) -> float:
    """Returns average of the last 5 observed competitor prices."""
    recent = state.get("rival_recent_prices", [])
    if len(recent) > 0:
        return float(np.mean(recent[-5:]))
    return fallback


def _solve_surplus_maximizing_price(
    period: int,
    inventory_gap: float,
    rival_recent_avg: float,
    state: Dict[str, Any]
) -> float:
    """
    Optimizes expected surplus:
    p* = argmax_p [ (p - μ(t, c)) · λ(t) · P(sale | p, p_rival) ]
    """
    # 1. Capacity shadow price mu(inventory_gap)
    base_shadow_cost = 27.0
    if inventory_gap > 1.5:
        mu = max(10.0, base_shadow_cost - 2.8 * (inventory_gap - 1.5))
    elif inventory_gap < -2.5:
        mu = min(58.0, base_shadow_cost + 3.5 * abs(inventory_gap))
    else:
        mu = base_shadow_cost

    # 2. Arrival intensity lambda(t)
    alpha_0 = state["alpha_0"]
    alpha_1 = state["alpha_1"]
    norm_t = (period - 1) / float(TOTAL_SELLING_PERIODS - 1)
    arrival_intensity = float(np.exp(np.clip(alpha_0 + alpha_1 * (norm_t ** 1.5), -2.0, 1.5)))

    # 3. Discrete choice conversion probabilities over candidate prices
    is_low_wtp = state.get("detected_low_wtp", False)
    max_candidate_p = 68.0 if is_low_wtp else 82.0
    candidate_prices = np.arange(28.0, max_candidate_p, 1.0)

    gamma_0 = state["gamma_0"]
    gamma_diff = state["gamma_diff"]
    gamma_own = state["gamma_own"]

    price_diffs = rival_recent_avg - candidate_prices
    logits = gamma_0 + gamma_diff * price_diffs - gamma_own * (candidate_prices - 30.0)
    conversion_probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -10.0, 10.0)))

    # 4. Expected surplus evaluation
    expected_demand = arrival_intensity * conversion_probs
    expected_surplus = (candidate_prices - mu) * expected_demand

    best_index = int(np.argmax(expected_surplus))
    return float(candidate_prices[best_index])


def _apply_late_season_clearance(
    period: int,
    current_target: float,
    rival_recent_avg: float,
    inventory_gap: float,
    required_burn_rate: float,
    remaining_capacity: int
) -> float:
    r"""
    Handles late clearance ($t \ge 75$):
    - Exploits late price spikes by holding $52–$54 if competitor is high.
    - Accelerates clearance when inventory lags to prevent seat spoilage.
    """
    if rival_recent_avg >= 54.0 and required_burn_rate <= 1.15:
        return min(54.0, rival_recent_avg - 2.0)

    if inventory_gap > 3.0 or (period >= 90 and remaining_capacity >= 2):
        urgency = (period - 74) / 26.0
        spoilage_discount = urgency * max(0.0, (required_burn_rate - 0.80) * 12.0)
        return min(current_target, rival_recent_avg - 1.5) - spoilage_discount

    return current_target


# -----------------------------------------------------------------------------
# 7. Bounds Enforcement & Decision Persistence
# -----------------------------------------------------------------------------
def _clamp_price(price: float) -> float:
    """Clamps asking price strictly within permissible competition boundaries ($25–$100)."""
    return float(np.clip(price, MIN_PRICE, MAX_PRICE))


def _persist_decision(price: float, state: Dict[str, Any]) -> None:
    """Records the chosen price in state dictionary."""
    state["my_last_price"] = price
