"""
duopoly.py
----------
Autonomous Dynamic Pricing Engine for Airline Duopoly Competition (v6.2).
Combines Closed-Form Econometric Optimization (MLR) with Microeconomic Guardrails.

Architectural Overview:
1. 3-Regime Empirical Demand Priors:
   Calibrated across 377,570 active competition periods:
   - low_demand:     β₀=0.2145, β_t=0.2931, sweet spot $22-$30, ceiling $36.
   - average_demand: β₀=0.9696, β_t=0.3998, sweet spot $32-$44, ceiling $60.
   - high_demand:    β₀=2.2329, β_t=0.8327, sweet spot $44-$52, ceiling $64.
2. In-Season & Cross-Season Low-Demand Detection:
   - In-season: sequential likelihood-ratio test (SPRT) on sales observed while we are
     cheaper than the rival at <= $55. Flags low demand when the data are 99x more likely
     under a low-demand rate than a normal rate (~4% false positives on v3/v4/v5 data).
     The in-season flag resets every season.
   - Cross-season: regime re-classified at the end of every season from that season's
     sales (< 42 seats -> low_demand). Not permanent: a good season un-latches it.
3. Closed-Form Marginal Revenue Optimization:
   Direct microeconomic first-order condition solution:
   p* = argmax_p [ (p - μ(t, c)) · (A + B·p) ] = (A - B·μ) / (2·(-B)).
   Demand model: d = β₀ + β_p·p + β_diff·(p_rival - p) + β_t·τ  (features are linearly
   independent; the old [p, p_rival, diff, τ] design had rank 3, not 4).
4. Learned Target Curve (v6.2):
   The target booking curve is 80 * ((1 - c)*tau + c*tau^2). Its bend c starts at 0.15 and,
   after every season that sold >= 75 seats, is re-estimated from this match's own booking
   curves, blended with a demand-group prior that counts as 3 seasons of evidence:
       c = (3 * prior + sum(season bends)) / (3 + number of such seasons), clipped to [0, 0.8].
5. Continuous Capacity Opportunity Shadow Cost (μ):
   Dynamic pacing without heuristic step-ladders:
   μ(t, c) smoothly discounts when behind schedule and extracts scarcity premium when ahead.
6. Essential Game-Theoretic & Boundary Guardrails:
   - Monopoly Forcing vs Volume Contesting against deep dumpers.
   - High-Rival Dynamic Ceiling ($26-$48) preventing consumer dead zones.
   - Terminal Clearance Acceleration at t >= 90 (perishable $0 salvage value).
   - Monotonicity & Elasticity Guards: B <= -0.006, rival coefficient C >= 0.
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
LOW_SALES_SEASON_THRESHOLD: int = 42

# Identified demand-slope guards (see _solve_closed_form_optimal_price)
MAX_OWN_SLOPE_B: float = -0.006   # own-price slope B must be at most this (demand falls with price)
MIN_RIVAL_SLOPE_C: float = 0.0    # higher rival price must not reduce our demand

# In-season low-demand SPRT (calibrated on data/v3, v4, v5 vs mlr, v5.1, poisson_logit):
# normal sale rate when cheaper than rival at <= $55 is ~0.5 + 0.9*tau per period;
# low-demand seasons convert at ~45% of that.
SPRT_NORMAL_BASE: float = 0.5
SPRT_NORMAL_SLOPE: float = 0.9
SPRT_LOW_RATIO: float = 0.45
SPRT_THRESHOLD: float = 4.6       # ln(99): latch when low demand is 99x more likely
SPRT_FLOOR: float = -3.0          # cap accumulated evidence for "normal" so the test stays responsive
SPRT_MIN_OBS: int = 8

# Learned target curve (v6.2). Bend c of TargetSold(tau) = 80 * ((1 - c)*tau + c*tau^2).
# Group priors = mean fitted bend of tournament seasons that sold >= 75 seats
# (data/v3+v4: 0.19, data/v5: 0.60, data/mlr+v5.1+poisson_logit: 0.67).
TARGET_C_DEFAULT: float = 0.15
TARGET_C_PRIOR: Dict[str, float] = {"low_demand": 0.67, "average_demand": 0.19, "high_demand": 0.60}
TARGET_PRIOR_WEIGHT: float = 3.0           # the prior counts like 3 seasons of evidence
TARGET_C_MIN: float = 0.0
TARGET_C_MAX: float = 0.8                  # never plan a more back-loaded season than this
TARGET_LEARN_MIN_SOLD: int = 75            # only learn from seasons that (nearly) filled the plane

# Regime priors. Only the identified combinations of the original 4-feature fits are kept:
#   B = b_p - b_diff         (own-price slope of expected demand)
#   C = b_riv + b_diff       (rival-price slope of expected demand)
# These give exactly the same predictions as the original OLS fits.
# The priors are used until MIN_OBSERVATIONS_FOR_TRAINING own observations exist.
# "monopoly_band": (price at high burn rate, price at low burn rate) when the rival is sold out,
# calibrated on monopoly-period conversion in data/v4 and data/v5 (revenue/period peaks at
# $45-$55 and collapses above ~$70 in v5).
REGIME_PRESETS_MLR: Dict[str, Dict[str, Any]] = {
    "low_demand": {
        "intercept": 0.2145, "B": -0.0082, "C": 0.0080, "b_t": 0.2931,
        "sweet_spot": (22.0, 30.0), "monopoly_band": (28.0, 44.0), "avg_demand": 0.347
    },
    "average_demand": {
        "intercept": 0.9696, "B": -0.0172, "C": 0.0065, "b_t": 0.3998,
        "sweet_spot": (32.0, 44.0), "monopoly_band": (48.0, 72.0), "avg_demand": 0.477
    },
    "high_demand": {
        "intercept": 2.2329, "B": -0.0386, "C": 0.0, "b_t": 0.8327,
        "sweet_spot": (44.0, 52.0), "monopoly_band": (50.0, 66.0), "avg_demand": 0.803
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
        state["total_capacity"],
        state.get("target_c_now", TARGET_C_DEFAULT)
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
        "training_history": [],            # Stores [p_our, p_rival - p_our, norm_t, demand]
        "mlr_model": None,                 # Fitted LinearRegression instance
        "mlr_intercept": REGIME_PRESETS_MLR["average_demand"]["intercept"],
        "mlr_B": REGIME_PRESETS_MLR["average_demand"]["B"],
        "mlr_C": REGIME_PRESETS_MLR["average_demand"]["C"],
        "mlr_b_t": REGIME_PRESETS_MLR["average_demand"]["b_t"],
        "rival_season_medians": [],
        "rival_recent_prices": [],
        "rival_profile": "balanced",
        "season_sales_history": [],
        "rival_sold_out_seasons": 0,
        "failed_holdout_seasons": 0,
        "consecutive_low_sales": 0,
        "low_demand_llr": 0.0,             # in-season SPRT log-likelihood ratio
        "low_demand_obs": 0,               # number of SPRT observations this season
        "low_demand_in_season": False,     # in-season SPRT verdict (reset each season)
        "my_last_price": DEFAULT_NEUTRAL_PRICE,
        "rival_last_price": DEFAULT_NEUTRAL_PRICE,
        "rival_ema_price": DEFAULT_NEUTRAL_PRICE,
        "price_demand_stats": {},
        "detected_low_wtp": False,
        "detected_low_demand": False,
        "detected_regime": "average_demand",
        "target_c_obs": [],                # fitted bend of each season that sold >= 75 seats
        "target_c_now": TARGET_C_DEFAULT,  # bend used this season
    }


# -----------------------------------------------------------------------------
# 2. Season Boundary & Cross-Season Learning
# -----------------------------------------------------------------------------
def _handle_season_transition(
    new_season: int,
    demand_history: Union[np.ndarray, List[int], None],
    state: Dict[str, Any]
) -> None:
    """
    Resets per-season trackers at the first call of a new season.

    Season bookkeeping (sales history, low-sales counter, regime, refit) happens only in
    period 101 (_handle_end_of_season). In period 1 the server passes demand_history=None,
    so counting a season here would record a phantom 0-sale season every season.
    Fallback: if period 101 was never seen for the previous season, close it now without
    recording sales (they are unknown) so the regime is still re-evaluated.
    """
    if state.get("last_closed_season") != state.get("last_season_seen"):
        _classify_and_update_regime(state)
        _fit_online_regression(state)
        state["last_closed_season"] = state.get("last_season_seen")

    state["last_season_seen"] = new_season
    state["rival_recent_prices"] = []
    state["low_demand_llr"] = 0.0
    state["low_demand_obs"] = 0
    state["low_demand_in_season"] = False


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

    # Period 101 can be called more than once defensively; close each season only once.
    if state.get("last_closed_season") == state.get("last_season_seen"):
        return DEFAULT_NEUTRAL_PRICE, state

    units_sold = _calculate_units_sold(demand_history)
    state.setdefault("season_sales_history", []).append(units_sold)

    if units_sold < LOW_SALES_SEASON_THRESHOLD:
        state["consecutive_low_sales"] = state.get("consecutive_low_sales", 0) + 1
    else:
        state["consecutive_low_sales"] = 0

    if units_sold >= TARGET_LEARN_MIN_SOLD:
        season_bend = _fit_season_target_bend(demand_history)
        if season_bend is not None:
            state.setdefault("target_c_obs", []).append(season_bend)

    _classify_and_update_regime(state)
    _fit_online_regression(state)
    state["target_c_now"] = _current_target_bend(state)
    state["last_closed_season"] = state.get("last_season_seen")

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
    sales_hist = state.get("season_sales_history", [])
    consec_low = state.get("consecutive_low_sales", 0)

    # 1. Low-Demand Regime (re-evaluated every season, never permanently latched):
    # last season sold < 42 seats, or zero sales at >= $55 across 4+ attempts with weak volume.
    # Rival price level is deliberately NOT used: a cheap rival is not evidence of low demand
    # (and treating it as such disabled the deep-dumper holdout from season 2 on).
    weak_volume = len(sales_hist) >= 1 and sales_hist[-1] < 60
    if (
        consec_low >= 1
        or (mid_high_periods >= 4 and mid_high_sales == 0 and weak_volume)
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
        state["detected_low_demand"] = False
        state["detected_low_wtp"] = False

    if state["detected_regime"] == "high_demand":
        state["detected_low_demand"] = False

    preset = REGIME_PRESETS_MLR[state["detected_regime"]]
    # Apply prior weights if not enough online observations to fit
    if len(state.get("training_history", [])) < MIN_OBSERVATIONS_FOR_TRAINING:
        state["mlr_intercept"] = preset["intercept"]
        state["mlr_B"] = preset["B"]
        state["mlr_C"] = preset["C"]
        state["mlr_b_t"] = preset["b_t"]


def _fit_online_regression(state: Dict[str, Any]) -> None:
    """Fits an OLS regression model on accumulated transaction history with elasticity guard."""
    history = state.get("training_history", [])
    if len(history) < MIN_OBSERVATIONS_FOR_TRAINING:
        return

    try:
        hist_data = np.array(history[-TRAINING_WINDOW_SIZE:], dtype=float)
        if hist_data.ndim != 2 or hist_data.shape[1] != 4:
            return
        feature_matrix = hist_data[:, :3]  # [p_our, p_rival - p_our, norm_t]  (full rank)
        target_demand = hist_data[:, 3]    # realized demand

        # Need actual price variation, otherwise the slopes are not identified.
        if np.std(feature_matrix[:, 0]) < 0.5 or np.std(feature_matrix[:, 1]) < 0.5:
            return

        model = LinearRegression(fit_intercept=True)
        model.fit(feature_matrix, target_demand)

        # d = b0 + b_p*p + b_diff*(r - p) + b_t*tau = (b0 + b_diff*r + b_t*tau) + (b_p - b_diff)*p
        b_p, b_diff, b_t = (float(c) for c in model.coef_)
        own_slope_B = b_p - b_diff
        rival_slope_C = b_diff

        # Monotonicity guard: demand must fall with our own price; otherwise keep the prior.
        if own_slope_B < -0.001:
            state["mlr_model"] = model
            state["mlr_intercept"] = float(model.intercept_)
            state["mlr_B"] = own_slope_B
            state["mlr_C"] = rival_slope_C
            state["mlr_b_t"] = b_t
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
    Logs transactions, updates the rival EMA and runs the in-season low-demand SPRT.
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

        # IN-SEASON LOW-DEMAND TEST (SPRT)
        # Only periods where we were cheaper than an in-stock rival at <= $55 are informative:
        # there, a normal market sells ~0.5 + 0.9*tau seats/period to us.
        if has_rival_capacity and my_prev_price <= 55.0 and latest_rival_price > my_prev_price:
            _update_low_demand_sprt(period, last_demand, state)

        # Log training observation [p_our, p_rival - p_our, norm_t, demand]
        # (observation belongs to the previous period, period - 1)
        price_diff = latest_rival_price - my_prev_price
        norm_t = (period - 2) / float(TOTAL_SELLING_PERIODS - 1)
        state.setdefault("training_history", []).append([
            my_prev_price, price_diff, norm_t, last_demand
        ])

        # Track price-demand conversion by bucket
        bucket = int(round(my_prev_price / 5.0) * 5)
        stats = state["price_demand_stats"].get(bucket, [0.0, 0])
        stats[0] += last_demand
        stats[1] += 1
        state["price_demand_stats"][bucket] = stats


def _update_low_demand_sprt(period: int, last_demand: float, state: Dict[str, Any]) -> None:
    """
    Sequential probability ratio test: low-demand rate (45% of normal) vs normal rate.
    Each informative period adds  d*ln(r_low/r_norm) - (r_low - r_norm)  to the log-likelihood
    ratio. Latches the in-season low-demand flag once the ratio exceeds ln(99).
    Replay on the tournament CSVs: ~4% false positives in seasons that sold >= 60 seats
    (old 4-zero-day rule: ~44%).
    """
    tau = (period - 2) / float(TOTAL_SELLING_PERIODS - 1)
    rate_normal = SPRT_NORMAL_BASE + SPRT_NORMAL_SLOPE * max(0.0, tau)
    rate_low = SPRT_LOW_RATIO * rate_normal
    increment = last_demand * np.log(rate_low / rate_normal) - (rate_low - rate_normal)
    llr = max(SPRT_FLOOR, state.get("low_demand_llr", 0.0) + float(increment))
    state["low_demand_llr"] = llr
    state["low_demand_obs"] = state.get("low_demand_obs", 0) + 1

    if llr > SPRT_THRESHOLD and state["low_demand_obs"] >= SPRT_MIN_OBS:
        state["low_demand_in_season"] = True
        state["detected_low_demand"] = True
        state["detected_low_wtp"] = True
        state["detected_regime"] = "low_demand"


# -----------------------------------------------------------------------------
# 4. Inventory Pacing Calculus
# -----------------------------------------------------------------------------
def _calculate_inventory_gap(
    current_period: int,
    remaining_capacity: int,
    total_capacity: int,
    bend: float = TARGET_C_DEFAULT
) -> float:
    """
    Computes difference between actual inventory and the target booking curve:
    TargetSold(tau) = total_capacity * ((1 - bend)*tau + bend*tau^2), tau = (t - 1) / 99.
    Gap > 0: Behind pace (surplus unsold seats).
    Gap < 0: Ahead of pace (scarcity).
    """
    progress = (current_period - 1) / float(TOTAL_SELLING_PERIODS - 1)
    target_depleted = total_capacity * ((1.0 - bend) * progress + bend * (progress ** 2))
    target_remaining = total_capacity - target_depleted
    return float(remaining_capacity - target_remaining)


def _fit_season_target_bend(
    demand_history: Union[np.ndarray, List[int], None]
) -> Optional[float]:
    """
    Least-squares bend c of one season's booking shape.
    share_sold(tau) = seats sold before day t / seats sold in the season, fitted to
    (1 - c)*tau + c*tau^2. Uses the shape, not the level, so a season that sold 76 seats and
    one that sold 80 are comparable.
    """
    if demand_history is None:
        return None
    sales = np.asarray(demand_history, dtype=float)[:TOTAL_SELLING_PERIODS]
    total = float(sales.sum())
    if len(sales) < TOTAL_SELLING_PERIODS or total <= 0:
        return None
    share = np.concatenate([[0.0], np.cumsum(sales)[:-1]]) / total
    tau = np.arange(TOTAL_SELLING_PERIODS) / float(TOTAL_SELLING_PERIODS - 1)
    x = tau ** 2 - tau
    return float(np.sum((share - tau) * x) / np.sum(x * x))


def _current_target_bend(state: Dict[str, Any]) -> float:
    """
    Bend of the target curve for the next season.
    - No season of this match has sold >= 75 seats yet: keep the default 0.15.
    - Otherwise: blend the demand-group prior (weight 3) with this match's fitted bends.
    """
    observed = state.get("target_c_obs", [])
    if not observed:
        return TARGET_C_DEFAULT
    prior = TARGET_C_PRIOR.get(state.get("detected_regime", "average_demand"), TARGET_C_DEFAULT)
    bend = (TARGET_PRIOR_WEIGHT * prior + float(sum(observed))) / (TARGET_PRIOR_WEIGHT + len(observed))
    return float(np.clip(bend, TARGET_C_MIN, TARGET_C_MAX))


# -----------------------------------------------------------------------------
# 5. Monopoly Regime Pricing Engine
# -----------------------------------------------------------------------------
def _calculate_monopoly_price(
    required_burn_rate: float,
    remaining_capacity: int,
    state: Dict[str, Any]
) -> float:
    """
    Monopoly pricing once the rival is sold out.

    Price falls linearly with the required burn rate (seats left per day left):
      burn <= 0.30  -> top of the regime's monopoly band (few seats, plenty of time: skim)
      burn >= 1.20  -> bottom of the band (many seats, little time: clear inventory)
    Bands come from observed monopoly conversion (data/v4, data/v5): revenue per period
    peaks around $45-$55 and conversion collapses above ~$70 when WTP is truncated.
    """
    regime = state.get("detected_regime", "average_demand")
    preset = REGIME_PRESETS_MLR.get(regime, REGIME_PRESETS_MLR["average_demand"])
    p_low, p_high = preset["monopoly_band"]

    # Truncated willingness to pay (no sales observed at >= $55): cap the skim price.
    if state.get("detected_low_wtp", False) and regime != "low_demand":
        p_high = min(p_high, 60.0)

    pressure = float(np.clip((required_burn_rate - 0.30) / 0.90, 0.0, 1.0))
    return float(p_high - pressure * (p_high - p_low))


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
    # The rival's price today is forecast with YESTERDAY's price (v6.1, "A2"): on the tournament
    # logs it misses by $1.80 on average vs $2.31 for the 5-day average. The guardrails below keep
    # the steadier 5-day average so that one unusual rival day cannot switch them on and off.
    target_price = _solve_closed_form_optimal_price(
        period,
        inventory_gap,
        required_burn_rate,
        latest_rival_price,
        state
    )

    # Guard 1: Cheap-Rival Floor
    # Against a cheap rival (< $38) we follow just below it instead of holding a $35-$44 anchor:
    # in data/v3-v5, pricing below a < $38 rival earned ~$40/period vs ~$15-$20/period when
    # holding $2-$10 above it. The floor min(rival - 0.5, 28) stops a spiral to the bottom.
    if target_price < 38.0 and rival_recent_avg < 38.0:
        target_price = max(target_price, min(rival_recent_avg - 0.5, 28.0))

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
    """Returns average of the last 5 observed competitor prices (used by the guardrails)."""
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
    where B = β_p - β_diff (< 0) is the own-price slope and
          A = β₀ + C·p_rival + β_t·τ with rival slope C = β_diff (>= 0).
    Guards act on the identified quantities B and C directly.
    """
    prior = REGIME_PRESETS_MLR["average_demand"]
    beta_0 = state.get("mlr_intercept", prior["intercept"])
    B = min(MAX_OWN_SLOPE_B, state.get("mlr_B", prior["B"]))
    C = max(MIN_RIVAL_SLOPE_C, state.get("mlr_C", prior["C"]))
    beta_t = state.get("mlr_b_t", prior["b_t"])

    norm_t = (period - 1) / float(TOTAL_SELLING_PERIODS - 1)
    A = beta_0 + C * rival_recent_avg + beta_t * norm_t

    # Dynamic capacity shadow cost mu(t, c)
    is_low_wtp = state.get("detected_low_wtp", False)
    mu = _compute_capacity_shadow_cost(inventory_gap, required_burn_rate, is_low_wtp)

    optimal_p = (A - B * mu) / (2.0 * (-B))

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
    - Exploits late rival price spikes by pricing just below them, capped at $52 (low WTP)
      or $58 (normal WTP).
    - Accelerates clearance when inventory lags to prevent seat spoilage.
    """
    # Late-season revenue per period vs rivals >= $54 falls above ~$58 (data/v3-v5).
    late_cap = 52.0 if is_low_wtp else 58.0
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
