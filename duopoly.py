"""
duopoly.py
----------
Autonomous Dynamic Pricing Engine for Airline Duopoly Competition (v6.4).
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
4. Learned Target Curve (v6.2, simplified in v6.3):
   The target booking curve is 80 * ((1 - c)*tau + c*tau^2). Its bend c starts at 0.15 and,
   after every season that sold >= 75 seats, is re-estimated from this match's own booking
   curves; the starting value 0.15 counts as 3 seasons of evidence:
       c = (3 * 0.15 + sum(season bends)) / (3 + number of such seasons), clipped to [0, 0.8].
   (v6.2 used a different starting value per demand regime, fitted on old tournament data;
   the ablation study showed it added nothing measurable, so v6.3 removed it.)
5. Continuous Capacity Opportunity Shadow Cost (μ):
   Dynamic pacing without heuristic step-ladders:
   μ(t, c) smoothly discounts when behind schedule and extracts scarcity premium when ahead.
6. Market Sizing & Copying Rivals (v6.4, from the data/v6 tournaments):
   - A choice-demand model is fitted to this match's own sales (Poisson likelihood): buyers
     arrive at n*exp(k*tau) per day and buy from us with probability
     e^{b(m-p)} / (1 + e^{b(m-p)} + e^{b(m-r)}). It learns from zero-sale days too.
   - Every day: expected demand for the rest of the season at revenue-maximising prices vs
     seats left. If demand <= 90% of seats left, a seat has no scarcity value (mu = 0): price
     = argmax p x expected seats, and the plan, the gap and the gap-driven guards are skipped.
     Otherwise v6.3 prices as before.
   - Copying rival (its price = our price yesterday on >= 80% of the days we moved): hold one
     price for the season; between seasons step it up or down and keep the direction that
     earned more per day.
7. Essential Game-Theoretic & Boundary Guardrails:
   - Monopoly Forcing vs Volume Contesting against deep dumpers.
   - High-Rival Dynamic Ceiling ($26-$48) preventing consumer dead zones.
   - Terminal Clearance Acceleration at t >= 90 (perishable $0 salvage value).
   - Monotonicity & Elasticity Guards: B <= -0.006, rival coefficient C >= 0.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

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

# v6.4: market-sized pricing. When this market cannot fill the plane, a seat has no scarcity
# value (mu = 0) and the price only has to maximise price x expected seats.
CAPACITY_SLACK: float = 0.9              # mu = 0 when expected demand <= 90% of seats left
UNCONSTRAINED_FLOOR: float = 18.0        # lowest price in market-sized mode
HIGH_RIVAL_CAP: float = 48.0             # never price above this when the rival is >= $58
# v6.4: copying rival. Rival's price today equals our price yesterday.
COPIER_TOLERANCE: float = 0.6            # $ difference counted as a copy
COPIER_WINDOW: int = 20                  # informative days kept (days our price moved)
COPIER_SHARE: float = 0.8                # share of copies needed to call the rival a copier
COPIER_PRICE_RANGE: Tuple[float, float] = (18.0, 60.0)
COPIER_FIRST_STEP: float = 3.0           # $ change of the held price between seasons
COPIER_MIN_STEP: float = 1.0
COPIER_MIN_DAYS: int = 30                # days held in a season before its revenue counts
# v6.4: choice-demand model used in market-sized mode. Each day ~ Poisson(n*exp(k*tau)) buyers
# arrive; each buys from us with probability e^{b(m-p)} / (1 + e^{b(m-p)} + e^{b(m-r)})
# (m = willingness-to-pay midpoint, b = price sensitivity). Unlike the linear model it can
# learn from days with zero sales. Weak priors (penalty weights) keep it identified early on.
LOGIT_PRIOR: Tuple[float, float, float, float] = (np.log(0.35), np.log(35.0), np.log(0.25), 0.6)
LOGIT_PRIOR_WEIGHT: Tuple[float, float, float, float] = (1.0, 1.0, 5.0, 5.0)
LOGIT_WINDOW: int = 3000
LOGIT_REFIT_EVERY: int = 25            # in-season refit interval (days)
# Zero sales at high prices cannot tell "few buyers" from "buyers won't pay this much", and only
# the second lowers the best price. So after 7 days without a sale, cut $1 per extra dry day:
# sales at lower prices then teach the model where the buyers are.
DRY_DAYS_ALLOWED: int = 7
DRY_DAY_CUT: float = 1.0

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
# The starting value 0.15 is the hand-chosen v6.1 bend; it is also the prior for the update.
TARGET_C_DEFAULT: float = 0.15
TARGET_PRIOR_WEIGHT: float = 3.0           # the starting value counts like 3 seasons of evidence
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

    # 5b. v6.4: refresh the demand model every 25 days, then ask whether capacity binds:
    # expected demand for the rest of the season at revenue-maximising prices vs seats left.
    if period % LOGIT_REFIT_EVERY == 0:
        _fit_logit_demand(state)
    state["unconstrained"] = _capacity_is_slack(period, remaining_capacity, state)

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
        "unconstrained": False,            # v6.4: this market cannot fill the plane
        "copy_checks": [],                 # v6.4: 1 = rival copied our previous price
        "logit_theta": list(LOGIT_PRIOR),  # v6.4: [ln n, ln m, ln b, k] of the choice model
        "copier_hold": None,               # v6.4: held price vs a copying rival + search state
        "copier_rev": 0.0,                 # v6.4: revenue earned this season while holding
        "copier_days": 0,                  # v6.4: days held this season
        "dry_days": 0,                     # v6.4: days in a row without a sale (this season)
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
    state["dry_days"] = 0


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

    _fit_logit_demand(state)
    _update_copier_hold(state)

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

        # v6.4 copier check: does the rival's price equal our price the day before?
        # Only days on which we changed our price are informative (two flat prices match trivially).
        if (
            isinstance(prices_history, np.ndarray) and prices_history.ndim == 2
            and prices_history.shape[1] >= 3
        ):
            own = prices_history[0, :]
            if abs(own[-2] - own[-3]) > 0.3:
                checks = state.setdefault("copy_checks", [])
                checks.append(1 if abs(latest_rival_price - own[-2]) <= COPIER_TOLERANCE else 0)
                if len(checks) > COPIER_WINDOW:
                    checks.pop(0)

        state["dry_days"] = 0 if last_demand > 0 else state.get("dry_days", 0) + 1

        if _rival_is_copier(state):
            state["copier_rev"] = state.get("copier_rev", 0.0) + my_prev_price * last_demand
            state["copier_days"] = state.get("copier_days", 0) + 1

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
    - Otherwise: blend the starting value 0.15 (weight 3) with this match's fitted bends.
    """
    observed = state.get("target_c_obs", [])
    if not observed:
        return TARGET_C_DEFAULT
    bend = (TARGET_PRIOR_WEIGHT * TARGET_C_DEFAULT + float(sum(observed))) / (TARGET_PRIOR_WEIGHT + len(observed))
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

    # Branch 0 (v6.4): market-sized pricing or a copying rival. Capacity does not bind, so the
    # plan, the gap and the gap-driven guards below do not apply.
    if _rival_is_copier(state) or _is_unconstrained(state):
        return _calculate_revenue_max_price(period, latest_rival_price, state)

    # Branch 1: Low-Sales Circuit Breaker (only against a cheap rival, v6.3)
    # After 2+ weak seasons, undercut a rival priced < $38. Against an expensive rival this
    # rule priced us just under it, above what customers would pay, so we sold nothing and
    # the weak seasons never ended (low market vs $40-$70 rival: $201/season; $917 with the fix).
    if consecutive_low_sales >= 2 and rival_recent_avg < 38.0:
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


def _is_unconstrained(state: Dict[str, Any]) -> bool:
    """v6.4: True when capacity does not bind today (see _capacity_is_slack)."""
    return bool(state.get("unconstrained", False))


def _capacity_is_slack(period: int, remaining_capacity: int, state: Dict[str, Any]) -> bool:
    """
    v6.4: market sizing. Using the choice model fitted to this match, price every remaining day
    at its revenue-maximising price (rival assumed to keep yesterday's price) and add up the
    expected seats. If that is at most 90% of the seats left, a seat has no scarcity value
    (mu = 0). Needs a fitted model (>= 80 observations); until then v6.3 decides.
    """
    if not state.get("logit_fitted", False):
        return False
    theta = state["logit_theta"]
    rival = state.get("rival_last_price", DEFAULT_RIVAL_PRICE)
    taus = (np.arange(period, TOTAL_SELLING_PERIODS + 1) - 1) / float(TOTAL_SELLING_PERIODS - 1)
    grid = np.arange(UNCONSTRAINED_FLOOR, MAX_PRICE + 0.01, 1.0)
    rates = _logit_rate(theta, grid[:, None], rival, taus[None, :])
    best = np.argmax(grid[:, None] * rates, axis=0)
    expected_demand = float(rates[best, np.arange(len(taus))].sum())
    return expected_demand <= CAPACITY_SLACK * remaining_capacity


def _rival_is_copier(state: Dict[str, Any]) -> bool:
    """v6.4: True when the rival set its price to our previous price on >= 80% of recent checks."""
    checks = state.get("copy_checks", [])
    return len(checks) >= 10 and float(np.mean(checks)) >= COPIER_SHARE


def _logit_rate(theta: List[float], price, rival, tau):
    """Expected seats per day for us under the choice model (vectorised)."""
    ln_n, ln_m, ln_b, k = theta
    b, m = np.exp(ln_b), np.exp(ln_m)
    ours = np.exp(np.clip(b * (m - price), -50.0, 50.0))
    theirs = np.exp(np.clip(b * (m - rival), -50.0, 50.0))
    return np.exp(ln_n + k * tau) * ours / (1.0 + ours + theirs)


def _fit_logit_demand(state: Dict[str, Any]) -> None:
    """v6.4: Poisson maximum likelihood fit of the choice model on our own sales history."""
    history = state.get("training_history", [])
    if len(history) < MIN_OBSERVATIONS_FOR_TRAINING:
        return
    try:
        data = np.array(history[-LOGIT_WINDOW:], dtype=float)
        price, rival, tau, sold = data[:, 0], data[:, 0] + data[:, 1], data[:, 2], data[:, 3]
        prior = np.array(LOGIT_PRIOR)
        weight = np.array(LOGIT_PRIOR_WEIGHT)

        def neg_log_lik(theta):
            lam = _logit_rate(theta, price, rival, tau) + 1e-12
            return float(-(sold * np.log(lam) - lam).sum() + (weight * (theta - prior) ** 2).sum())

        start = np.array(state.get("logit_theta", LOGIT_PRIOR), dtype=float)
        result = minimize(neg_log_lik, start, method="Nelder-Mead",
                          options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-3})
        if np.all(np.isfinite(result.x)):
            state["logit_theta"] = [float(v) for v in result.x]
            state["logit_fitted"] = True
    except Exception:
        pass


def _update_copier_hold(state: Dict[str, Any]) -> None:
    """
    v6.4: season-to-season search for the best price to hold against a copying rival.
    Each season's revenue per day at the held price is a direct measurement (the rival always
    matches us), so no demand model is needed: keep moving in the direction that earned more,
    turn around and take smaller steps when it earned less.
    """
    hold = state.get("copier_hold")
    days, rev = state.get("copier_days", 0), state.get("copier_rev", 0.0)
    state["copier_rev"], state["copier_days"] = 0.0, 0
    if hold is None or days < COPIER_MIN_DAYS:
        return
    rate = rev / days
    if hold["best_rev"] is None or rate >= hold["best_rev"]:
        hold["best_rev"], hold["best_price"] = rate, hold["price"]
    else:
        hold["direction"] = -hold["direction"]
        hold["step"] = max(COPIER_MIN_STEP, hold["step"] * 0.6)
    hold["price"] = float(np.clip(hold["best_price"] + hold["direction"] * hold["step"], *COPIER_PRICE_RANGE))


def _calculate_revenue_max_price(period: int, rival_price: float, state: Dict[str, Any]) -> float:
    """
    v6.4: price that maximises price x expected seats when capacity does not bind (mu = 0),
    using the choice model fitted to this match. Against a copying rival the rival's price
    equals ours, so we maximise p x seats(p, r = p) and hold that price.
    """
    theta = state.get("logit_theta", list(LOGIT_PRIOR))
    tau = (period - 1) / float(TOTAL_SELLING_PERIODS - 1)
    grid = np.arange(UNCONSTRAINED_FLOOR, MAX_PRICE + 0.01, 0.5)
    if _rival_is_copier(state):
        hold = state.get("copier_hold")
        if hold is None:
            # First season against a copier: start from the choice model's best common price.
            revenue = grid * _logit_rate(theta, grid, grid, 0.5)
            start = float(np.clip(grid[int(np.argmax(revenue))], *COPIER_PRICE_RANGE))
            hold = {"price": start, "best_price": None, "best_rev": None,
                    "direction": -1.0, "step": COPIER_FIRST_STEP}
            state["copier_hold"] = hold
        return float(hold["price"])
    revenue = grid * _logit_rate(theta, grid, rival_price, tau)
    price = float(grid[int(np.argmax(revenue))])
    dry_cut = DRY_DAY_CUT * max(0, state.get("dry_days", 0) - DRY_DAYS_ALLOWED)
    return max(UNCONSTRAINED_FLOOR, price - dry_cut)


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
    rival_price: float,
    state: Dict[str, Any]
) -> float:
    """
    Solves the quadratic surplus optimization problem:
    p* = argmax_p [ (p - μ(t, c)) · (A + B·p) ]
    where B = β_p - β_diff (< 0) is the own-price slope and
          A = β₀ + C·p_rival + β_t·τ with rival slope C = β_diff (>= 0).
    Guards act on the identified quantities B and C directly.
    rival_price is the rival's price from yesterday (v6.1 "A2"); the band is centred on it.
    """
    prior = REGIME_PRESETS_MLR["average_demand"]
    beta_0 = state.get("mlr_intercept", prior["intercept"])
    B = min(MAX_OWN_SLOPE_B, state.get("mlr_B", prior["B"]))
    C = max(MIN_RIVAL_SLOPE_C, state.get("mlr_C", prior["C"]))
    beta_t = state.get("mlr_b_t", prior["b_t"])

    norm_t = (period - 1) / float(TOTAL_SELLING_PERIODS - 1)
    A = beta_0 + C * rival_price + beta_t * norm_t

    # Dynamic capacity shadow cost mu(t, c)
    is_low_wtp = state.get("detected_low_wtp", False)
    mu = _compute_capacity_shadow_cost(inventory_gap, required_burn_rate, is_low_wtp)

    optimal_p = (A - B * mu) / (2.0 * (-B))

    # Margin-Preserving Undercut Boundary:
    max_cap = 8.50 if is_low_wtp else 7.50
    max_undercut = min(max_cap, 1.20 + 0.35 * max(0.0, inventory_gap - 1.5))
    return float(np.clip(optimal_p, rival_price - max_undercut, rival_price + 12.0))


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
    if state and _is_unconstrained(state) and not _rival_is_copier(state):
        min_p = min(min_p, UNCONSTRAINED_FLOOR)
    return float(np.clip(price, min_p, MAX_PRICE))


def _persist_decision(price: float, state: Dict[str, Any]) -> None:
    """Records the chosen price in state dictionary."""
    state["my_last_price"] = price
