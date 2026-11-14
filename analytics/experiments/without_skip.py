"""
duopoly.py
----------
Autonomous Dynamic Pricing Engine for Airline Duopoly Competition (v6.6).
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
8. Data-Driven Seat Value (v6.5, from the data/v6.4 tournaments):
   Once the choice model of component 6 is fitted, the hand-tuned mu of component 5 is replaced
   by dynamic programming on that model. With D ~ Poisson(choice-model rate), rival held at its
   5-day average price, prices $18-$100 in $1 steps:
       V(t, s) = max_p E[ p * min(D, s) + V(t+1, s - min(D, s)) ],   V(101, s) = 0
       mu      = V(t+1, s) - V(t+1, s-1)      (value of one more seat after today)
   and today's price is the maximising p (the DP policy), not the regression formula. The DP is
   recomputed every 10 days, each new season and whenever the model is refitted (~5 ms).
   While the DP price is active, the hand-made +$12 / -undercut band around the rival, guards
   1-2 and the late-season clearance are skipped; the monopoly rule (rival sold out) and the
   market-sizing / copier branches still apply. Switches DP_* at the end of the file;
   ALGORITHM_CHANGES.md section 22 has the on/off tests (+10.9% on the v6.4 test bed). Before the choice model
   is fitted (first season), the hand formula of component 5 is still used.
9. Learning approach with linear and nonlinear optimisation (v6.6, from the data/v6.5 tournaments;
   after Bertsimas & Perakis 2001 and the lecture "A learning approach to dynamic pricing"):
   a. Rival price forecast (lecture step 3 + time series). At the end of every season:
        profile[d] = median rival price on day d over the last 10 seasons,
        phi        = share of today's gap to the profile that is still there tomorrow (AR(1)),
        reaction   = (a, c, b) of rival_t = a + c*rival_{t-1} + b*our_{t-1}, fitted by least absolute
                     deviations as a LINEAR PROGRAMME on the last 3 seasons; used only if its error is
                     <= 70% of the error of "same price as yesterday".
      Forecast for the rest of the season: rival[d] = profile[d] + phi^(d-t) * (rival now - profile[t]).
   b. Dynamic programme on the forecast path. The DP of component 8 held the rival at its 5-day
      average until day 100; it now uses the forecast path (market sizing, component 6, too).
   c. Today's price as a NONLINEAR PROGRAMME (lecture step 4, with capacity):
        max_p E[ p*min(D, s) + V(t+1, s - min(D, s)) ],  D ~ Poisson(rate(p, rival forecast for today))
      solved every day with a bounded scalar solver; the DP table (rebuilt every 10 days) supplies V.
   d. Demand data: a sold-out rival still posts a price. Those days are now left out of the demand
      data; before, the demand models read the posted price as competition.
   e. The choice model is first fitted on day 50 of season 1 (was day 100).
   Switches at the end of the file (RIVAL_*, NLP_PRICE). Tested and NOT adopted (EVALUATION.md):
   rival sell-out share inside the DP, the lecture's linear demand model fitted by LP,
   several starting points for the choice-model fit, logging sold-out-rival days as "no rival"
   instead of leaving them out, a $10 price floor.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog, minimize, minimize_scalar

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
LOGIT_MIN_OBS: int = 45                # days of own data before the choice model is fitted (v6.5: 80,
                                       # i.e. not before day 100 of season 1; now day 50 of season 1)
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

    _record_rival_capacity(period, competitor_has_capacity_current_period_in_current_season, state)

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
    state["seats_left_now"] = remaining_capacity
    state["period_now"] = period

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
        "rival_cap_days": [],              # v6.6: rival had capacity on day 1..t of this season
        "rival_paths": [],                 # v6.6: rival price per day of the last seasons (nan = not selling)
        "own_paths": [],                   # v6.6: our price per day of the same seasons
        "rival_model": None,               # v6.6: fitted rival forecast (profile, phi, reaction, selling share)
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
    state["rival_cap_days"] = []


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
    _store_rival_path(prices_history, state)
    _fit_rival_model(state)

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
        # v6.6: a sold-out rival still posts a price ($0.10 to $999 in data/v6.5). On such a day we
        # were alone in the market, so its posted price is not competition: the day is left out of
        # the demand data. (Up to v6.5 it was logged as an ordinary duopoly day.)
        if not (SKIP_SOLD_OUT_RIVAL_DAYS and not _rival_was_selling(period - 1, state)):
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
    if DP_MONOPOLY:
        dp = _dp_monopoly_price(state.get("period_now", 1), remaining_capacity, state)
        if dp is not None:
            return dp
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

    if DP_SKIP_GUARDS and state.get("dp_active"):
        if period >= 75 and not DP_SKIP_CLEARANCE:
            return _apply_late_season_clearance(period, target_price, rival_recent_avg, inventory_gap,
                                                required_burn_rate, remaining_capacity, is_low_wtp)
        return target_price

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
    if period >= 75 and not (DP_SKIP_CLEARANCE and state.get("dp_active")):
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
    if RIVAL_PATH_IN_SIZING:
        rival = _forecast_rival_path(period, state)[0][period:TOTAL_SELLING_PERIODS + 1][None, :]
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
    """v6.4: Poisson maximum likelihood fit of the choice model on our own sales history.
    v6.6: days on which the rival was sold out are not in the history (see _update_real_time_signals)."""
    history = state.get("training_history", [])
    if len(history) < LOGIT_MIN_OBS:
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
    if RIVAL_REACTION_TODAY:
        rival_price = _forecast_rival_today(state, rival_price)
    revenue = grid * _logit_rate(theta, grid, rival_price, tau)
    price = float(grid[int(np.argmax(revenue))])
    if NLP_PRICE:
        # Lecture step 4: one-dimensional revenue maximisation with a standard solver.
        res = minimize_scalar(lambda q: -q * float(_logit_rate(theta, q, rival_price, tau)),
                              bounds=(max(UNCONSTRAINED_FLOOR, price - 0.5), min(MAX_PRICE, price + 0.5)),
                              method="bounded", options={"xatol": 0.02})
        if np.isfinite(res.x):
            price = round(float(res.x), 1)
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
    mu_dp = _dp_seat_value(period, state.get("seats_left_now", 0), rival_price, state)
    if mu_dp is not None:
        mu = mu_dp

    optimal_p = (A - B * mu) / (2.0 * (-B))
    state["dp_active"] = mu_dp is not None
    if mu_dp is not None and "dp_price" in state:
        optimal_p = state["dp_price"]
        if not DP_BAND:
            return float(optimal_p)

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


# -----------------------------------------------------------------------------
# v6.5: seat value from dynamic programming on the fitted choice model
# Switches below were tested on/off on the v6.4 test bed (ALGORITHM_CHANGES.md section 22).
# -----------------------------------------------------------------------------
DP_PRICE_GRID = np.arange(18.0, 100.01, 1.0)
DP_KMAX = 7
DP_REFRESH_DAYS = 10
DP_BAND = False            # True: clip the DP price to [rival - undercut, rival + 12]
DP_SKIP_GUARDS = True      # True: skip guards 1-2 when the DP price is active
DP_SKIP_CLEARANCE = True   # True: skip the late-season clearance when the DP price is active
DP_MONOPOLY = False        # True: DP price (no rival) when the rival is sold out


def _dp_values(theta, rival, start_day, selling_share=None):
    """V[t][s] = expected revenue from day t to 100 with s seats, pricing optimally each day.
    Demand per day ~ Poisson(choice-model rate), capped at seats left.
    rival: one price held for all days, or an array indexed by day (the forecast rival path).
    selling_share: optional array indexed by day, P(rival still sells on that day); on the other
    days we are alone in the market."""
    fact = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880], dtype=float)[:DP_KMAX + 1]
    S = DEFAULT_TOTAL_CAPACITY
    seats = np.arange(S + 1)
    k = np.arange(DP_KMAX + 1)
    sold = np.minimum(k[None, :], seats[:, None])            # (S+1, K+1)
    idx = seats[:, None] - sold
    rival_by_day = np.broadcast_to(np.asarray(rival, dtype=float), (TOTAL_SELLING_PERIODS + 2,)) \
        if np.ndim(rival) == 0 else np.asarray(rival, dtype=float)
    V = np.zeros((TOTAL_SELLING_PERIODS + 2, S + 1))
    POL = np.zeros((TOTAL_SELLING_PERIODS + 2, S + 1))
    for t in range(TOTAL_SELLING_PERIODS, start_day - 1, -1):
        tau = (t - 1) / float(TOTAL_SELLING_PERIODS - 1)
        pmf = _poisson_pmf(_logit_rate(theta, DP_PRICE_GRID, rival_by_day[t], tau), k, fact)
        if selling_share is not None and selling_share[t] < 0.999:
            alone = _poisson_pmf(_logit_rate(theta, DP_PRICE_GRID, 1000.0, tau), k, fact)
            pmf = selling_share[t] * pmf + (1.0 - selling_share[t]) * alone
        exp_sold = pmf @ sold.T                               # (P, S+1)
        cont = pmf @ V[t + 1][idx].T                          # (P, S+1)
        tot = DP_PRICE_GRID[:, None] * exp_sold + cont
        V[t] = tot.max(axis=0); POL[t] = DP_PRICE_GRID[tot.argmax(axis=0)]
    return V, POL


def _poisson_pmf(lam, k, fact):
    """P(D = k) for k = 0..K with the tail lumped into K. lam: array of rates (one per price)."""
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    pmf = np.exp(-lam)[:, None] * lam[:, None] ** k[None, :] / fact[None, :]
    pmf[:, -1] = np.maximum(0.0, 1.0 - pmf[:, :-1].sum(axis=1))
    return pmf


def _dp_monopoly_price(period, seats_left, state):
    """Rival sold out: DP price with no rival (it stays sold out for the rest of the season)."""
    if not state.get("logit_fitted", False) or seats_left <= 0:
        return None
    cache = state.get("dp_mono_cache")
    if cache is None or cache["season"] != state.get("last_season_seen") or cache["theta"] != state["logit_theta"]:
        V, POL = _dp_values(np.array(state["logit_theta"]), 1000.0, period)
        cache = {"season": state.get("last_season_seen"), "theta": list(state["logit_theta"]), "POL": POL}
        state["dp_mono_cache"] = cache
    return float(cache["POL"][period][seats_left])


def _dp_seat_value(period, seats_left, rival_price, state):
    """mu = value of keeping one more seat after today: V[t+1][s] - V[t+1][s-1].
    Also sets state["dp_price"], today's price."""
    if not state.get("logit_fitted", False) or seats_left <= 0:
        return None
    cache = state.get("dp_cache")
    stale = (cache is None or cache["season"] != state.get("last_season_seen") or period - cache["day"] >= DP_REFRESH_DAYS
             or cache["theta"] != state["logit_theta"])
    if stale:
        if RIVAL_PATH_IN_DP:
            rival, share = _forecast_rival_path(period, state)
            if not RIVAL_SELLOUT_IN_DP:
                share = None
        else:
            recent = state.get("rival_recent_prices", [])
            rival, share = (float(np.mean(recent[-5:])) if recent else float(rival_price)), None
        V, POL = _dp_values(np.array(state["logit_theta"]), rival, period, share)
        cache = {"season": state.get("last_season_seen"), "day": period, "theta": list(state["logit_theta"]), "V": V, "POL": POL}
        state["dp_cache"] = cache
    V = cache["V"]
    state["dp_price"] = float(cache["POL"][period][seats_left])
    if NLP_PRICE:
        today = _forecast_rival_today(state, rival_price) if RIVAL_REACTION_TODAY else rival_price
        state["dp_price"] = _solve_price_today(period, seats_left, today, V[period + 1], state)
    return float(V[period + 1][seats_left] - V[period + 1][seats_left - 1])


def _solve_price_today(period, seats_left, rival_today, value_tomorrow, state):
    """
    Lecture step 4 with capacity: today's price as a one-dimensional nonlinear programme
        max_p  E[ p * min(D, s) + V(t+1, s - min(D, s)) ],   D ~ Poisson(rate(p, rival today))
    The $1 grid finds the right hill, a bounded scalar solver then finds its top. Unlike the cached
    DP policy (rebuilt every DP_REFRESH_DAYS days) this uses today's forecast of the rival price.
    """
    theta = state["logit_theta"]
    tau = (period - 1) / float(TOTAL_SELLING_PERIODS - 1)
    fact = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880], dtype=float)[:DP_KMAX + 1]
    k = np.arange(DP_KMAX + 1)
    sold = np.minimum(k, seats_left)
    future = value_tomorrow[seats_left - sold]

    def value(price):
        pmf = _poisson_pmf(_logit_rate(theta, price, rival_today, tau), k, fact)
        return pmf @ (np.atleast_1d(price)[:, None] * sold[None, :] + future[None, :]).T \
            if np.ndim(price) else float(pmf[0] @ (price * sold + future))

    on_grid = np.einsum("pk,pk->p", _poisson_pmf(_logit_rate(theta, DP_PRICE_GRID, rival_today, tau), k, fact),
                        DP_PRICE_GRID[:, None] * sold[None, :] + future[None, :])
    start = float(DP_PRICE_GRID[int(np.argmax(on_grid))])
    try:
        res = minimize_scalar(lambda q: -value(float(q)), method="bounded", options={"xatol": 0.02},
                              bounds=(max(DP_PRICE_GRID[0], start - 1.0), min(DP_PRICE_GRID[-1], start + 1.0)))
        if np.isfinite(res.x) and -res.fun >= on_grid.max() - 1e-9:
            return round(float(res.x), 1)
    except Exception:
        pass
    return start


# -----------------------------------------------------------------------------
# v6.6: rival price forecast (time series + linear programme), after Bertsimas & Perakis (2001)
# -----------------------------------------------------------------------------
RIVAL_PATH_SEASONS = 10        # seasons of rival prices kept for the per-day profile
RIVAL_MIN_SEASONS = 2          # seasons needed before the forecast replaces the 5-day average
RIVAL_REACT_SEASONS = 3        # seasons used to fit the one-day-ahead reaction model
RIVAL_REACT_MAX_RATIO = 0.7    # reaction model is used only if its error is <= 70% of "yesterday's price"
RIVAL_PATH_IN_DP = True        # DP uses the forecast rival path instead of one held price
RIVAL_SELLOUT_IN_DP = False    # DP also uses the share of seasons the rival was still selling
RIVAL_PATH_IN_SIZING = True    # market sizing uses the forecast path
RIVAL_REACTION_TODAY = True    # today's rival price from the reaction model (else yesterday's)
NLP_PRICE = True               # today's price from a scalar solver on today's forecast
SKIP_SOLD_OUT_RIVAL_DAYS = False  # days on which the rival was sold out are left out of the demand data


def _rival_was_selling(day, state):
    """True if the rival had capacity on that day of this season (unknown days count as selling)."""
    days = state.get("rival_cap_days", [])
    return bool(days[day - 1]) if 1 <= day <= len(days) else True


def _record_rival_capacity(period, has_capacity, state):
    """Remembers for every day of this season whether the rival was selling."""
    if period > TOTAL_SELLING_PERIODS:
        return
    days = state.setdefault("rival_cap_days", [])
    if period == 1:
        del days[:]
    while len(days) < period - 1:
        days.append(True)
    if len(days) == period - 1:
        days.append(bool(has_capacity))


def _store_rival_path(prices_history, state):
    """End of season: keep the rival's and our price per day (nan where the rival was not selling)."""
    if not (isinstance(prices_history, np.ndarray) and prices_history.ndim == 2
            and prices_history.shape[0] >= 2 and prices_history.shape[1] >= TOTAL_SELLING_PERIODS):
        return
    own = np.asarray(prices_history[0, :TOTAL_SELLING_PERIODS], dtype=float)
    rival = np.asarray(prices_history[1, :TOTAL_SELLING_PERIODS], dtype=float)
    days = list(state.get("rival_cap_days", []))[:TOTAL_SELLING_PERIODS]
    days += [True] * (TOTAL_SELLING_PERIODS - len(days))
    selling = np.array(days, dtype=bool) & (rival > 0.0) & (rival < SERVER_SOLDOUT_PRICE_THRESHOLD)
    state.setdefault("rival_paths", []).append(np.where(selling, rival, np.nan).tolist())
    state.setdefault("own_paths", []).append(own.tolist())
    del state["rival_paths"][:-RIVAL_PATH_SEASONS]
    del state["own_paths"][:-RIVAL_PATH_SEASONS]


def _least_absolute_fit(X, y, bounds):
    """
    min sum |y - X b|  as a linear programme (lecture step 1 / 3):
        min sum(D)   s.t.   y - X b <= D,   -(y - X b) <= D,   D >= 0
    """
    n, m = X.shape
    cost = np.r_[np.zeros(m), np.ones(n)]
    A = np.block([[-X, -np.eye(n)], [X, -np.eye(n)]])
    res = linprog(cost, A_ub=A, b_ub=np.r_[-y, y], bounds=list(bounds) + [(0, None)] * n, method="highs")
    return res.x[:m] if res.status == 0 else None


def _fit_rival_model(state):
    """
    Fits the rival forecast on the stored seasons of this match:
    1. profile[d]  = median rival price on day d (the least-absolute-deviation level per day),
    2. phi         = how much of today's gap to the profile is still there tomorrow (AR(1), in [0, 1]),
    3. share[d]    = share of seasons in which the rival was still selling on day d,
    4. reaction    = (a, c, b) of  rival_t = a + c * rival_{t-1} + b * our_{t-1}, least absolute
                     deviations by linear programming on the last 3 seasons (c, b >= 0).
    """
    paths = np.array(state.get("rival_paths", []), dtype=float)
    if paths.ndim != 2 or paths.shape[0] < RIVAL_MIN_SEASONS:
        return
    try:
        seen = np.isfinite(paths)
        share = seen.mean(axis=0)
        days = np.arange(paths.shape[1])
        has = seen.any(axis=0)
        if has.sum() < 10:
            return
        med = np.array([np.median(paths[seen[:, d], d]) if has[d] else np.nan for d in days])
        profile = np.interp(days, days[has], med[has])
        gap = paths - profile[None, :]
        x, y = gap[:, :-1].ravel(), gap[:, 1:].ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        phi = float(np.clip((x[ok] * y[ok]).sum() / max(1e-9, (x[ok] ** 2).sum()), 0.0, 1.0)) if ok.sum() > 20 else 0.0

        reaction = None
        own = np.array(state.get("own_paths", []), dtype=float)[-RIVAL_REACT_SEASONS:]
        riv = paths[-RIVAL_REACT_SEASONS:]
        r_now, r_prev, p_prev = riv[:, 1:].ravel(), riv[:, :-1].ravel(), own[:, :-1].ravel()
        ok = np.isfinite(r_now) & np.isfinite(r_prev) & (p_prev < TERMINAL_SOLDOUT_PRICE - 1.0)
        if ok.sum() >= 60:
            X = np.c_[np.ones(ok.sum()), r_prev[ok], p_prev[ok]]
            coef = _least_absolute_fit(X, r_now[ok], [(None, None), (0.0, 1.5), (0.0, 1.5)])
            if coef is not None and np.all(np.isfinite(coef)):
                # Keep it only if it clearly beats "same price as yesterday" on these seasons.
                miss_model = np.abs(r_now[ok] - X @ coef).mean()
                miss_naive = np.abs(r_now[ok] - r_prev[ok]).mean()
                if miss_model <= RIVAL_REACT_MAX_RATIO * miss_naive:
                    reaction = [float(v) for v in coef]
        lo, hi = float(np.nanmin(paths)), float(np.nanmax(paths))
        state["rival_model"] = {"profile": profile.tolist(), "phi": phi, "share": share.tolist(),
                                "reaction": reaction, "range": [lo, hi]}
    except Exception:
        pass


def _forecast_rival_today(state, fallback):
    """Rival price today: reaction model on yesterday's two prices; else yesterday's rival price."""
    model = state.get("rival_model")
    if not model or not model.get("reaction"):
        return float(fallback)
    a, c, b = model["reaction"]
    guess = a + c * state.get("rival_last_price", fallback) + b * state.get("my_last_price", fallback)
    lo, hi = model["range"]
    return float(np.clip(guess, lo, hi))


def _forecast_rival_path(period, state):
    """
    Forecast of the rival's price for every day (array indexed by day 1..100; entries before
    `period` are unused) and the share of seasons the rival was still selling, given it sells today.
        rival[d] = profile[d] + phi^(d - t) * (rival today - profile[t])
    Before RIVAL_MIN_SEASONS seasons are stored: the 5-day average held flat (as in v6.5).
    """
    T = TOTAL_SELLING_PERIODS
    recent = state.get("rival_recent_prices", [])
    last = float(state.get("rival_last_price", DEFAULT_RIVAL_PRICE))
    model = state.get("rival_model")
    path = np.full(T + 2, float(np.mean(recent[-5:])) if recent else last)
    share = np.ones(T + 2)
    if not model:
        return path, share
    profile = np.r_[np.nan, model["profile"], model["profile"][-1]]      # index = day
    today = last if (recent and period > 1) else float(profile[min(max(period, 1), T)])
    days = np.arange(T + 2)
    ahead = np.maximum(0, days - period)
    here = min(max(period - 1, 1), T) if (recent and period > 1) else min(max(period, 1), T)
    path = profile + model["phi"] ** np.maximum(1, days - here) * (today - profile[here])
    path[0] = path[1]
    path = np.clip(path, 0.1, SERVER_SOLDOUT_PRICE_THRESHOLD - 1.0)
    sell = np.r_[1.0, model["share"], model["share"][-1]]
    base = max(sell[min(max(period, 1), T)], 1e-6)
    share = np.where(ahead > 0, np.clip(sell / base, 0.0, 1.0), 1.0)
    return path, share
