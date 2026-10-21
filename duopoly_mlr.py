"""
duopoly_mlr.py
--------------
Autonomous Dynamic Pricing Engine: Multiple Linear Regression (MLR) + Target Depletion Pacing (v5.1).

Architectural Overview:
1. Online Multiple Linear Regression (OLS):
   Periodically fits an empirical price-response plane on observed season history:
   RealizedDemand = β₀ + β_p·p_our + β_riv·p_rival + β_diff·(p_rival - p_our) + β_t·(t/100).
2. Closed-Form Marginal Revenue Optimization:
   Solves directly for optimal ticket price incorporating capacity opportunity shadow cost:
   p* = argmax_p [ (p - μ(t, c)) · (A + B·p) ] = (A - B·μ) / (2·(-B)).
3. Target Depletion Curve (Booking Curve Pacing):
   Maintains inventory pace along TargetSold(t) = 80 * (0.85*tau + 0.15*tau^2).
4. Adaptive Empirical Monopoly Engine & Dynamic WTP Ceiling:
   Adapts clearance prices to empirical WTP limits ($52–$64 sweet spot) when low-WTP environments are detected.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# Domain Constants
# -----------------------------------------------------------------------------
MIN_PRICE: float = 25.0
MAX_PRICE: float = 100.0
DEFAULT_TOTAL_CAPACITY: int = 80
TOTAL_SELLING_PERIODS: int = 100
SERVER_SOLDOUT_PRICE_THRESHOLD: float = 200.0
TERMINAL_SOLDOUT_PRICE: float = 130.0
DEFAULT_NEUTRAL_PRICE: float = 65.0
DEFAULT_RIVAL_PRICE: float = 50.0
MIN_OBSERVATIONS_FOR_TRAINING: int = 80
TRAINING_WINDOW_SIZE: int = 2500

REGIME_PRESETS_MLR: Dict[str, Dict[str, Any]] = {
    "regime_1_high": {
        "intercept": 0.605, "b_p": -0.0146, "b_riv": 0.0138, "b_diff": 0.0284, "b_t": 0.439,
        "sweet_spot": (74.0, 84.0), "max_price": 95.0
    },
    "regime_2_medium": {
        "intercept": 1.245, "b_p": -0.0171, "b_riv": -0.0022, "b_diff": 0.0149, "b_t": 0.714,
        "sweet_spot": (64.0, 72.0), "max_price": 85.0
    },
    "regime_3_low": {
        "intercept": 2.233, "b_p": -0.0258, "b_riv": -0.0131, "b_diff": 0.0128, "b_t": 0.833,
        "sweet_spot": (52.0, 62.0), "max_price": 68.0
    }
}


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
    Main MLR dynamic pricing function adhering to Single Responsibility and Clean Code design.
    """
    state = _initialize_or_get_state(information_dump, current_selling_season)
    period = selling_period_in_current_season

    # 1. Season Boundary Transition & Online Model Refitting
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
        "mlr_model": None,                 # Fitted LinearRegression instance
        "mlr_intercept": 1.5,
        "mlr_b_p": -0.02,
        "mlr_b_riv": 0.005,
        "mlr_b_diff": 0.02,
        "mlr_b_t": 0.4,
        "rival_season_medians": [],
        "rival_recent_prices": [],
        "failed_holdout_seasons": 0,
        "consecutive_low_sales": 0,
        "my_last_price": DEFAULT_NEUTRAL_PRICE,
        "rival_last_price": DEFAULT_NEUTRAL_PRICE,
        "price_demand_stats": {},
        "detected_low_wtp": False,
        "detected_regime": "regime_2_medium"
    }


# -----------------------------------------------------------------------------
# 2. Season Boundary & Online Regression Training
# -----------------------------------------------------------------------------
def _handle_season_transition(
    new_season: int,
    demand_history: Union[np.ndarray, List[int], None],
    state: Dict[str, Any]
) -> None:
    """Updates circuit breaker streak, classifies macroeconomic regime, and refits the MLR model."""
    prev_season_sales = _calculate_units_sold(demand_history)
    state["consecutive_low_sales"] = (
        state.get("consecutive_low_sales", 0) + 1 if prev_season_sales < 42 else 0
    )

    _classify_and_update_regime(state)
    _fit_online_regression(state)

    state["last_season_seen"] = new_season
    state["rival_recent_prices"] = []


def _classify_and_update_regime(state: Dict[str, Any]) -> None:
    """Classifies the market into one of 3 empirical regimes based on cumulative transaction stats."""
    stats_dict = state.get("price_demand_stats", {})
    high_buckets = [b for b in stats_dict.keys() if b >= 75]
    high_periods = sum(stats_dict[b][1] for b in high_buckets)
    high_sales = sum(stats_dict[b][0] for b in high_buckets)

    # 1. Low-WTP Surge (Regime 3, e.g. data/v5): zero sales at >= $75 after 4+ tests
    if high_periods >= 4 and high_sales == 0:
        state["detected_regime"] = "regime_3_low"
        state["detected_low_wtp"] = True
    # 2. High-WTP Premium (Regime 1, e.g. data/v1/v4): > 25% conversion at >= $75
    elif high_periods >= 4 and (high_sales / float(high_periods)) >= 0.25:
        state["detected_regime"] = "regime_1_high"
        state["detected_low_wtp"] = False
    else:
        state["detected_regime"] = "regime_2_medium"
        state["detected_low_wtp"] = False

    preset = REGIME_PRESETS_MLR[state["detected_regime"]]
    # Apply prior weights if not enough online observations to fit
    if len(state.get("training_history", [])) < MIN_OBSERVATIONS_FOR_TRAINING:
        state["mlr_intercept"] = preset["intercept"]
        state["mlr_b_p"] = preset["b_p"]
        state["mlr_b_riv"] = preset["b_riv"]
        state["mlr_b_diff"] = preset["b_diff"]
        state["mlr_b_t"] = preset["b_t"]


def _fit_online_regression(state: Dict[str, Any]) -> None:
    """Fits an OLS multiple linear regression model on accumulated transaction history."""
    history = state.get("training_history", [])
    if len(history) < MIN_OBSERVATIONS_FOR_TRAINING:
        return

    try:
        hist_data = np.array(history[-TRAINING_WINDOW_SIZE:])
        feature_matrix = hist_data[:, :4]  # [p_our, p_rival, p_diff, norm_t]
        target_demand = hist_data[:, 4]    # realized demand

        model = LinearRegression(fit_intercept=True)
        model.fit(feature_matrix, target_demand)

        state["mlr_model"] = model
        state["mlr_intercept"] = float(model.intercept_)
        state["mlr_b_p"] = float(model.coef_[0])
        state["mlr_b_riv"] = float(model.coef_[1])
        state["mlr_b_diff"] = float(model.coef_[2])
        state["mlr_b_t"] = float(model.coef_[3])
    except Exception:
        pass


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
    """Logs observed price differentials and appends observations to training history."""
    if period <= 1 or prices_history is None:
        return

    latest_rival_price = _extract_latest_rival_price(prices_history)
    my_prev_price = state.get("my_last_price", DEFAULT_RIVAL_PRICE)
    last_demand = _extract_last_demand(demand_history)

    if latest_rival_price is not None:
        state["rival_last_price"] = latest_rival_price
        recent_prices = state.setdefault("rival_recent_prices", [])
        recent_prices.append(latest_rival_price)
        if len(recent_prices) > 15:
            recent_prices.pop(0)

        # Log training observation [p_our, p_rival, p_diff, norm_t, demand]
        price_diff = latest_rival_price - my_prev_price
        norm_t = (period - 1) / float(TOTAL_SELLING_PERIODS - 1)
        state.setdefault("training_history", []).append([
            my_prev_price, latest_rival_price, price_diff, norm_t, last_demand
        ])

        # Track price-demand conversion by bucket
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
# 5. Monopoly Regime Pricing Engine (Continuous MLR Shadow-Cost Optimization)
# -----------------------------------------------------------------------------
def _calculate_monopoly_price(
    required_burn_rate: float,
    remaining_capacity: int,
    state: Dict[str, Any]
) -> float:
    """Computes continuous monopoly price derived from detected regime bounds and capacity burn rate."""
    regime = state.get("detected_regime", "regime_2_medium")
    preset = REGIME_PRESETS_MLR.get(regime, REGIME_PRESETS_MLR["regime_2_medium"])
    p_low, p_high = preset["sweet_spot"]
    p_floor = 46.0 if state.get("detected_low_wtp", False) else 50.0

    # Scarcity bonus when capacity is nearly gone (<= 8 seats) and time remains
    if remaining_capacity <= 8 and required_burn_rate <= 0.40:
        return float(preset.get("max_price", p_high))

    # Continuous burn-rate clearance discount (zero hardcoded step ladders)
    burn_discount = min(p_high - p_floor, 18.0 * max(0.0, required_burn_rate - 0.45))
    return float(p_high - burn_discount)


# -----------------------------------------------------------------------------
# 6. Duopoly Regime: Closed-Form MLR Optimization
# -----------------------------------------------------------------------------
def _calculate_duopoly_price(
    period: int,
    inventory_gap: float,
    required_burn_rate: float,
    remaining_capacity: int,
    state: Dict[str, Any]
) -> float:
    """Computes asking price during active duopoly using closed-form MLR revenue maximization."""
    latest_rival_price = state.get("rival_last_price", DEFAULT_RIVAL_PRICE)
    rival_recent_avg = _get_recent_rival_average(state, latest_rival_price)
    consecutive_low_sales = state.get("consecutive_low_sales", 0)

    # Branch 1: Universal Low-Sales Circuit Breaker
    if consecutive_low_sales >= 2:
        breaker_price = min(latest_rival_price - 0.5, rival_recent_avg - 0.5)
        if inventory_gap > 3.0:
            breaker_price -= min(4.0, 0.35 * inventory_gap)
        return breaker_price

    # Branch 2: Deep Dumper Holdout
    if latest_rival_price <= 19.5 and rival_recent_avg <= 19.5:
        min_periods_needed = int(np.ceil(remaining_capacity / 1.85))
        safe_holdout_deadline = max(55, (TOTAL_SELLING_PERIODS + 1) - min_periods_needed)
        if period < safe_holdout_deadline:
            return 58.0

    # Branch 3: Closed-Form MLR Revenue Optimization
    target_price = _solve_closed_form_optimal_price(
        period,
        inventory_gap,
        required_burn_rate,
        rival_recent_avg,
        state
    )

    # Guard 1: Anti-Dumping & Copycat Protection
    if target_price < 38.0 and rival_recent_avg < 38.0:
        target_price = max(
            target_price,
            float(np.clip(44.0 - 0.8 * max(0.0, inventory_gap - 1.5), 35.0, 44.0))
        )

    # Guard 2: High-Price Ceiling Cap
    is_low_wtp = state.get("detected_low_wtp", False)
    duopoly_cap = 54.0 if is_low_wtp else 68.0
    if target_price > duopoly_cap and inventory_gap > 2.5 and period >= 15:
        target_price = duopoly_cap - 2.0

    # Guard 3: Late-Season Clearance & High-Rival Counter
    if period >= 75:
        target_price = _apply_late_season_clearance(
            period,
            target_price,
            rival_recent_avg,
            inventory_gap,
            required_burn_rate,
            remaining_capacity,
            is_low_wtp
        )

    return target_price


def _get_recent_rival_average(state: Dict[str, Any], fallback: float) -> float:
    """Returns average of the last 5 observed competitor prices."""
    recent = state.get("rival_recent_prices", [])
    if len(recent) > 0:
        return float(np.mean(recent[-5:]))
    return fallback


def _compute_capacity_shadow_cost(inventory_gap: float, required_burn_rate: float) -> float:
    """
    Computes capacity opportunity cost mu(t, c):
    - Higher when inventory is scarce (burn rate low, negative gap) -> raises asking price.
    - Lower when inventory is surplus (burn rate high, positive gap) -> discounts asking price.
    """
    base_mu = 28.0
    burn_penalty = 14.0 * (required_burn_rate - 0.80)
    gap_penalty = 1.8 * inventory_gap
    return float(np.clip(base_mu - burn_penalty - gap_penalty, 12.0, 56.0))


def _solve_closed_form_optimal_price(
    period: int,
    inventory_gap: float,
    required_burn_rate: float,
    rival_recent_avg: float,
    state: Dict[str, Any]
) -> float:
    """
    Solves the quadratic surplus optimization problem:
    p* = argmax_p [ (p - μ(t, c)) · (A + B·p) ]
    where B = β_p - β_diff (< 0) and A = β₀ + (β_riv + β_diff)·p_rival + β_t·τ.
    """
    beta_0 = state.get("mlr_intercept", 1.5)
    beta_p = min(-0.005, state.get("mlr_b_p", -0.02))
    beta_riv = state.get("mlr_b_riv", 0.005)
    beta_diff = max(0.005, state.get("mlr_b_diff", 0.02))
    beta_t = state.get("mlr_b_t", 0.4)

    # Aggregate demand slope B and intercept A
    B = beta_p - beta_diff
    norm_t = (period - 1) / float(TOTAL_SELLING_PERIODS - 1)
    A = beta_0 + (beta_riv + beta_diff) * rival_recent_avg + beta_t * norm_t

    # Dynamic capacity shadow cost mu(t, c)
    mu = _compute_capacity_shadow_cost(inventory_gap, required_burn_rate)

    if B < -0.001:
        optimal_p = (A - B * mu) / (2.0 * (-B))
    else:
        optimal_p = rival_recent_avg - 1.0

    # Restrict regression extrapolation
    return float(np.clip(optimal_p, rival_recent_avg - 8.0, rival_recent_avg + 12.0))


def _apply_late_season_clearance(
    period: int,
    current_target: float,
    rival_recent_avg: float,
    inventory_gap: float,
    required_burn_rate: float,
    remaining_capacity: int,
    is_low_wtp: bool = False
) -> float:
    r"""
    Handles late clearance ($t \ge 75$):
    - Exploits late price spikes by holding $52–$54 under low WTP or $78 under high WTP.
    - Accelerates clearance when inventory lags to prevent seat spoilage.
    """
    late_cap = 54.0 if is_low_wtp else 78.0
    if rival_recent_avg >= 54.0 and required_burn_rate <= 1.15:
        return min(late_cap, rival_recent_avg - 2.0)

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
