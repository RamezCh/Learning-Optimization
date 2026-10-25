"""
duopoly.py
----------
Autonomous Dynamic Pricing Engine for Airline Duopoly Competition (v6.0 Unified).
Combines Closed-Form Econometric Optimization (MLR) with Microeconomic Guardrails.

Architectural Overview:
1. 3-Regime Empirical Demand Priors:
   Calibrated across 377,570 active competition periods:
   - low_demand:     β₀=0.2145, β_t=0.2931, sweet spot $22-$30, ceiling $36.
   - average_demand: β₀=0.9696, β_t=0.3998, sweet spot $32-$44, ceiling $60.
   - high_demand:    β₀=2.2329, β_t=0.8327, sweet spot $44-$52, ceiling $64.
2. Fast In-Season & Cross-Season Low-Demand Detection:
   - Day-10 Sweet-Spot Check: 0 sales at $48-$55 across 8+ days -> low_demand.
   - Cheaper-than-Rival Lockout Check: 0 sales when cheaper than rival at <= $55 -> 100% low demand.
   - Cross-Season Latch: Season 1 sales < 42 seats permanently latches low_demand across seasons 2..100.
3. Closed-Form Marginal Revenue Optimization:
   Direct microeconomic first-order condition solution:
   p* = argmax_p [ (p - μ(t, c)) · (A + B·p) ] = (A - B·μ) / (2·(-B)).
4. Continuous Capacity Opportunity Shadow Cost (μ):
   Dynamic pacing without heuristic step-ladders:
   μ(t, c) smoothly discounts when behind schedule and extracts scarcity premium when ahead.
5. Essential Game-Theoretic & Boundary Guardrails:
   - Monopoly Forcing vs Volume Contesting against deep dumpers.
   - High-Rival Dynamic Ceiling ($26-$48) preventing consumer dead zones.
   - Terminal Clearance Acceleration at t >= 90 (perishable $0 salvage value).
   - Monotonicity & Elasticity Guard (B < -0.001).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# Domain Constants
# -----------------------------------------------------------------------------
MIN_PRICE: float = 25.0
MAX_PRICE: float = 100.0
LOW_DEMAND_MIN_PRICE: float = 18.0
DEFAULT_TOTAL_CAPACITY: int = 80
TOTAL_SELLING_PERIODS: int = 100
SERVER_SOLDOUT_PRICE_THRESHOLD: float = 200.0
TERMINAL_SOLDOUT_PRICE: float = 130.0
DEFAULT_NEUTRAL_PRICE: float = 65.0
DEFAULT_RIVAL_PRICE: float = 50.0
MIN_OBSERVATIONS_FOR_TRAINING: int = 80
TRAINING_WINDOW_SIZE: int = 2500

REGIME_PRESETS_MLR: Dict[str, Dict[str, Any]] = {
    "low_demand": {
        "intercept": 0.2145, "b_p": -0.0028, "b_riv": 0.0026, "b_diff": 0.0054, "b_t": 0.2931,
        "sweet_spot": (22.0, 30.0), "max_price": 36.0, "avg_demand": 0.347
    },
    "average_demand": {
        "intercept": 0.9696, "b_p": -0.0093, "b_riv": -0.0014, "b_diff": 0.0079, "b_t": 0.3998,
        "sweet_spot": (32.0, 44.0), "max_price": 60.0, "avg_demand": 0.477
    },
    "high_demand": {
        "intercept": 2.2329, "b_p": -0.0258, "b_riv": -0.0131, "b_diff": 0.0128, "b_t": 0.8327,
        "sweet_spot": (44.0, 52.0), "max_price": 64.0, "avg_demand": 0.803
    }
}
# Backward-compatibility aliases
REGIME_PRESETS_MLR["regime_1_high"] = REGIME_PRESETS_MLR["high_demand"]
REGIME_PRESETS_MLR["regime_2_medium"] = REGIME_PRESETS_MLR["average_demand"]
REGIME_PRESETS_MLR["regime_3_low"] = REGIME_PRESETS_MLR["low_demand"]


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
    Main dynamic pricing decision function called by the competition simulator each period.
    Adheres strictly to Clean Code Single Responsibility and Step-down Rules.
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
        return _handle_end_of_season(
            prices_historical_in_current_season,
            demand_historical_in_current_season,
            state
        )

    # 3. Inventory & Capacity Status
    units_sold = _calculate_units_sold(demand_historical_in_current_season)
    remaining_capacity = max(0, state["total_capacity"] - units_sold)

    if remaining_capacity <= 0:
        return TERMINAL_SOLDOUT_PRICE, state

    remaining_periods = max(1, (TOTAL_SELLING_PERIODS + 1) - period)
    required_burn_rate = remaining_capacity / remaining_periods

    # 4. Process Environmental Feedback & Fast In-Season Signal Updates
    _update_real_time_signals(
        period,
        prices_historical_in_current_season,
        demand_historical_in_current_season,
        competitor_has_capacity_current_period_in_current_season,
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
    final_price = _clamp_price(raw_price, state)
    _persist_decision(final_price, state)

    return final_price, state


# -----------------------------------------------------------------------------
# 1. State Management & Initialization
# -----------------------------------------------------------------------------
def _initialize_or_get_state(
    information_dump: Optional[Any],
    current_season: int
) -> Dict[str, Any]:
    """Ensures information_dump exists and contains all required tracking structures."""
    if isinstance(information_dump, dict) and "total_capacity" in information_dump:
        return information_dump

    return {
        "initialized": True,
        "total_capacity": DEFAULT_TOTAL_CAPACITY,
        "last_season_seen": current_season,
        "training_history": [],            # Stores [p_our, p_rival, diff, norm_t, demand]
        "mlr_model": None,                 # Fitted LinearRegression instance
        "mlr_intercept": 0.9696,
        "mlr_b_p": -0.0093,
        "mlr_b_riv": -0.0014,
        "mlr_b_diff": 0.0079,
        "mlr_b_t": 0.3998,
        "rival_season_medians": [],
        "rival_recent_prices": [],
        "rival_profile": "balanced",
        "season_sales_history": [],
        "rival_sold_out_seasons": 0,
        "failed_holdout_seasons": 0,
        "consecutive_low_sales": 0,
        "consecutive_zero_sales_sweet_spot": 0,
        "consecutive_zero_sales_cheaper": 0,
        "my_last_price": DEFAULT_NEUTRAL_PRICE,
        "rival_last_price": DEFAULT_NEUTRAL_PRICE,
        "rival_ema_price": DEFAULT_NEUTRAL_PRICE,
        "price_demand_stats": {},
        "detected_low_wtp": False,
        "detected_low_demand": False,
        "detected_regime": "average_demand"
    }


# -----------------------------------------------------------------------------
# 2. Season Boundary & Cross-Season Learning
# -----------------------------------------------------------------------------
def _handle_season_transition(
    new_season: int,
    demand_history: Union[np.ndarray, List[int], None],
    state: Dict[str, Any]
) -> None:
    """Updates sales history, classifies macroeconomic regime, and refits online MLR."""
    prev_season_sales = _calculate_units_sold(demand_history)
    state.setdefault("season_sales_history", []).append(prev_season_sales)
    
    state["consecutive_low_sales"] = (
        state.get("consecutive_low_sales", 0) + 1 if prev_season_sales < 42 else 0
    )

    # Cross-season latch: if Season 1 (or any season) had very low sales, permanently latch low_demand
    if prev_season_sales < 42:
        state["detected_low_demand"] = True
        state["detected_low_wtp"] = True

    _classify_and_update_regime(state)
    _fit_online_regression(state)

    state["last_season_seen"] = new_season
    state["rival_recent_prices"] = []
    state["consecutive_zero_sales_sweet_spot"] = 0
    state["consecutive_zero_sales_cheaper"] = 0


def _handle_end_of_season(
    prices_history: Union[np.ndarray, List[float], None],
    demand_history: Union[np.ndarray, List[int], None],
    state: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """Updates cross-season learning metrics, rival medians, and low-demand flags on period 101."""
    if (
        isinstance(prices_history, np.ndarray)
        and prices_history.ndim == 2
        and prices_history.shape[0] >= 2
    ):
        rival_prices = prices_history[1, :]
        valid_prices = rival_prices[rival_prices < SERVER_SOLDOUT_PRICE_THRESHOLD]
        if len(valid_prices) > 0:
            season_med = float(np.median(valid_prices))
            state["rival_season_medians"].append(season_med)
            overall_med = float(np.median(state["rival_season_medians"]))
            state["rival_profile"] = "deep_dumper" if overall_med <= 20 else ("dumper" if overall_med <= 38 else "balanced")

    units_sold = _calculate_units_sold(demand_history)
    state.setdefault("season_sales_history", []).append(units_sold)

    if units_sold < 42:
        state["consecutive_low_sales"] = state.get("consecutive_low_sales", 0) + 1
        state["detected_low_demand"] = True
        state["detected_low_wtp"] = True
    else:
        state["consecutive_low_sales"] = 0

    _classify_and_update_regime(state)
    _fit_online_regression(state)

    return DEFAULT_NEUTRAL_PRICE, state


def _classify_and_update_regime(state: Dict[str, Any]) -> None:
    """
    Classifies market into one of 3 empirical regimes based on consumer conversion:
    Decoupled from rival price: relies on sales volume and zero-conversion thresholds.
    """
    stats_dict = state.get("price_demand_stats", {})
    mid_high_buckets = [b for b in stats_dict.keys() if b >= 55]
    mid_high_periods = sum(stats_dict[b][1] for b in mid_high_buckets)
    mid_high_sales = sum(stats_dict[b][0] for b in mid_high_buckets)
    rival_medians = state.get("rival_season_medians", [])
    sales_hist = state.get("season_sales_history", [])
    consec_low = state.get("consecutive_low_sales", 0)
    is_low_demand_latched = state.get("detected_low_demand", False)

    # 1. Low-Demand Regime:
    # Triggered by: latched flag, zero sales at >= $55 across 4+ attempts, sales < 42, or rival <= $34
    if (
        is_low_demand_latched
        or consec_low >= 1
        or (mid_high_periods >= 4 and mid_high_sales == 0)
        or (len(rival_medians) >= 1 and np.median(rival_medians) <= 34.0)
        or (len(sales_hist) >= 2 and np.mean(sales_hist[-2:]) < 40.0)
    ):
        state["detected_regime"] = "low_demand"
        state["detected_low_demand"] = True
        state["detected_low_wtp"] = True

    # 2. High-Demand Regime (data/v5):
    # Surging arrival volume (mean sales >= 72) with truncated WTP (0 sales >= $55)
    elif (
        len(sales_hist) >= 2
        and np.mean(sales_hist[-2:]) >= 72.0
        and mid_high_sales == 0
    ):
        state["detected_regime"] = "high_demand"
        state["detected_low_wtp"] = True

    # 3. Average / Standard Demand Regime (data/v4, data/v3, data/v1):
    else:
        state["detected_regime"] = "average_demand"
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
    """Fits an OLS regression model on accumulated transaction history with elasticity guard."""
    history = state.get("training_history", [])
    if len(history) < MIN_OBSERVATIONS_FOR_TRAINING:
        return

    try:
        hist_data = np.array(history[-TRAINING_WINDOW_SIZE:])
        feature_matrix = hist_data[:, :4]  # [p_our, p_rival, p_diff, norm_t]
        target_demand = hist_data[:, 4]    # realized demand

        model = LinearRegression(fit_intercept=True)
        model.fit(feature_matrix, target_demand)

        # Monotonicity & Elasticity Guard:
        # Own price sensitivity must be negative (B = beta_p - beta_diff < 0).
        # If regression produces non-negative slope due to collinearity, fall back to regime prior.
        b_p_candidate = float(model.coef_[0])
        b_diff_candidate = float(model.coef_[2])
        if (b_p_candidate - b_diff_candidate) < -0.001:
            state["mlr_model"] = model
            state["mlr_intercept"] = float(model.intercept_)
            state["mlr_b_p"] = b_p_candidate
            state["mlr_b_riv"] = float(model.coef_[1])
            state["mlr_b_diff"] = b_diff_candidate
            state["mlr_b_t"] = float(model.coef_[3])
    except Exception:
        pass


# -----------------------------------------------------------------------------
# 3. Real-Time Signal Processing & Fast In-Season Detectors
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
    has_rival_capacity: bool,
    state: Dict[str, Any]
) -> None:
    """
    Logs transactions, updates EMA, and applies Fast In-Season Low-Demand Rules:
    1. Day-10 Sweet-Spot Zero-Sales Check ($48-$55 with 0 sales -> low demand).
    2. Cheaper-than-Rival Zero-Sales Check (cheaper at <= $55 with 0 sales -> low demand).
    """
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

        # Update EMA
        smoothing = 0.25
        current_ema = state.get("rival_ema_price", DEFAULT_NEUTRAL_PRICE)
        state["rival_ema_price"] = float(smoothing * latest_rival_price + (1.0 - smoothing) * current_ema)

        # FAST IN-SEASON DETECTOR 1: Sweet-Spot Zero-Sales (e.g. at ~$55 with 0 demand)
        if 48.0 <= my_prev_price <= 56.0 and last_demand == 0:
            state["consecutive_zero_sales_sweet_spot"] = state.get("consecutive_zero_sales_sweet_spot", 0) + 1
            if state["consecutive_zero_sales_sweet_spot"] >= 8:
                state["detected_low_demand"] = True
                state["detected_low_wtp"] = True
                state["detected_regime"] = "low_demand"
        elif last_demand > 0:
            state["consecutive_zero_sales_sweet_spot"] = 0

        # FAST IN-SEASON DETECTOR 2: Cheaper-than-rival at <= $55 but 0 sales
        # (If rival is higher, customers MUST choose us if they exist and WTP >= price)
        if my_prev_price <= 55.0 and latest_rival_price > my_prev_price and last_demand == 0:
            state["consecutive_zero_sales_cheaper"] = state.get("consecutive_zero_sales_cheaper", 0) + 1
            if state["consecutive_zero_sales_cheaper"] >= 4:
                state["detected_low_demand"] = True
                state["detected_low_wtp"] = True
                state["detected_regime"] = "low_demand"
        elif last_demand > 0:
            state["consecutive_zero_sales_cheaper"] = 0

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
    Target: 80 * (0.85*tau + 0.15*tau^2).
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
    """
    Solves monopoly pricing directly from detected regime sweet spot:
    Eliminates discrete step-ladders with continuous burn-rate discounting.
    """
    regime = state.get("detected_regime", "average_demand")
    preset = REGIME_PRESETS_MLR.get(regime, REGIME_PRESETS_MLR["average_demand"])
    p_low, p_high = preset["sweet_spot"]
    p_floor = 46.0 if state.get("detected_low_wtp", False) else 50.0

    # Scarcity bonus when capacity is nearly gone (<= 8 seats) and time remains
    if remaining_capacity <= 8 and required_burn_rate <= 0.40:
        return float(preset.get("max_price", p_high))

    # Continuous burn-rate clearance discount
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
    is_low_demand = state.get("detected_low_demand", False)
    is_low_wtp = state.get("detected_low_wtp", False)

    # Branch 1: Universal Low-Sales Circuit Breaker
    if consecutive_low_sales >= 2:
        breaker_price = min(latest_rival_price - 0.5, rival_recent_avg - 0.5)
        if inventory_gap > 3.0:
            breaker_price -= min(4.0, 0.35 * inventory_gap)
        return breaker_price

    # Branch 2: Deep Dumper Handling (Monopoly Forcing vs Volume Contesting)
    # If rival <= $19.50:
    # - In standard/high demand: Hold $58 to force dumper to exhaust stock and trigger monopoly.
    # - In low demand: Dumper will never sell out! Contest volume down to $18 to prevent $0 blowout.
    if latest_rival_price <= 19.5 and rival_recent_avg <= 19.5:
        min_periods_needed = int(np.ceil(remaining_capacity / 1.85))
        safe_holdout_deadline = max(55, (TOTAL_SELLING_PERIODS + 1) - min_periods_needed)
        if period < safe_holdout_deadline and not is_low_demand and consecutive_low_sales == 0:
            return 58.0
        return max(LOW_DEMAND_MIN_PRICE, min(latest_rival_price - 0.5, 24.0))

    # Branch 3: Closed-Form MLR Revenue Optimization: p* = (A - B*mu) / (2*(-B))
    target_price = _solve_closed_form_optimal_price(
        period,
        inventory_gap,
        required_burn_rate,
        rival_recent_avg,
        state
    )

    # Guard 1: Anti-Dumping & Copycat Protection
    if target_price < 38.0 and rival_recent_avg < 38.0:
        if is_low_demand or consecutive_low_sales >= 2:
            target_price = max(target_price, min(rival_recent_avg - 0.5, 28.0))
        else:
            target_price = max(
                target_price,
                float(np.clip(44.0 - 0.8 * max(0.0, inventory_gap - 1.5), 35.0, 44.0))
            )

    # Guard 2: High-Price Ceiling Cap & High-Rival Dynamic Exploitation
    # When rival >= $58: never price in the dead zone! Cap at $26-$48 to sweep 100% of arrivals.
    if rival_recent_avg >= 58.0:
        dyn_ceiling = float(np.clip(46.0 - 1.5 * max(0.0, inventory_gap), 26.0, 48.0))
        target_price = min(target_price, dyn_ceiling)
    else:
        duopoly_cap = 50.0 if is_low_wtp else 66.0
        if target_price > duopoly_cap and inventory_gap > 2.0:
            target_price = duopoly_cap - 2.0

    # Guard 3: Late-Season Clearance & Spoilage Ramp (t >= 75)
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


def _compute_capacity_shadow_cost(
    inventory_gap: float,
    required_burn_rate: float,
    is_low_wtp: bool = False
) -> float:
    """
    Computes capacity opportunity cost mu(t, c):
    - Higher when inventory is scarce (burn rate low, negative gap) -> raises asking price.
    - Lower when inventory is surplus (burn rate high, positive gap) -> aggressively discounts
      to guarantee liquidation before seats expire at $0 value.
    """
    base_mu = 26.0
    burn_penalty = 16.0 * (required_burn_rate - 0.80)
    gap_penalty = 2.4 * inventory_gap
    min_mu = 2.0 if is_low_wtp else 10.0
    return float(np.clip(base_mu - burn_penalty - gap_penalty, min_mu, 56.0))


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
    beta_0 = state.get("mlr_intercept", 0.9696)
    beta_p = min(-0.002, state.get("mlr_b_p", -0.0093))
    beta_riv = state.get("mlr_b_riv", -0.0014)
    beta_diff = max(0.004, state.get("mlr_b_diff", 0.0079))
    beta_t = state.get("mlr_b_t", 0.3998)

    # Aggregate demand slope B and intercept A
    B = beta_p - beta_diff
    norm_t = (period - 1) / float(TOTAL_SELLING_PERIODS - 1)
    A = beta_0 + (beta_riv + beta_diff) * rival_recent_avg + beta_t * norm_t

    # Dynamic capacity shadow cost mu(t, c)
    is_low_wtp = state.get("detected_low_wtp", False)
    mu = _compute_capacity_shadow_cost(inventory_gap, required_burn_rate, is_low_wtp)

    if B < -0.001:
        optimal_p = (A - B * mu) / (2.0 * (-B))
    else:
        optimal_p = rival_recent_avg - 1.0

    # Margin-Preserving Undercut Boundary:
    max_cap = 8.50 if is_low_wtp else 7.50
    max_undercut = min(max_cap, 1.20 + 0.35 * max(0.0, inventory_gap - 1.5))
    return float(np.clip(optimal_p, rival_recent_avg - max_undercut, rival_recent_avg + 12.0))


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
    - Exploits late price spikes by holding $46-$52 under low WTP.
    - Accelerates clearance when inventory lags to prevent seat spoilage.
    """
    late_cap = 52.0 if is_low_wtp else 72.0
    if rival_recent_avg >= 54.0 and required_burn_rate <= 1.15:
        return min(late_cap, rival_recent_avg - 2.0)

    if inventory_gap > 2.5 or (period >= 90 and remaining_capacity >= 2):
        urgency = (period - 74) / 26.0
        spoilage_discount = urgency * max(0.0, (required_burn_rate - 0.75) * 12.0)
        return min(current_target, rival_recent_avg - 1.5) - spoilage_discount

    return current_target


# -----------------------------------------------------------------------------
# 7. Bounds Enforcement & Decision Persistence
# -----------------------------------------------------------------------------
def _clamp_price(price: float, state: Optional[Dict[str, Any]] = None) -> float:
    """Clamps asking price strictly within permissible competition boundaries ($18–$100)."""
    min_p = (
        LOW_DEMAND_MIN_PRICE
        if (state and (state.get("detected_low_demand", False) or state.get("consecutive_low_sales", 0) >= 1))
        else MIN_PRICE
    )
    return float(np.clip(price, min_p, MAX_PRICE))


def _persist_decision(price: float, state: Dict[str, Any]) -> None:
    """Records the chosen price in state dictionary."""
    state["my_last_price"] = price
