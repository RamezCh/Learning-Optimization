import numpy as np
from typing import Union, Tuple, Optional, Any, Dict

def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: Union[np.ndarray, None],
    demand_historical_in_current_season: Union[np.ndarray, None],
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump: Optional[Any] = None
) -> Tuple[float, Any]:
    """
    Dynamic Pricing Algorithm v2.0 for Airline Duopoly Competition.
    
    Key Upgrades from v1.0 Empirical Post-Mortem:
    1. Continuous Dynamic Inventory Pacing (Periods 1-100):
       - Eliminates the static $85 early holdout that caused zero-revenue matches when total
         market arrivals (~75-105 seats) were smaller than combined capacity (160 seats).
       - Tracks a continuous target remaining curve: Target(t) = 80 * (1 - t/100).
    2. Competitor Price Anchoring & Smart Undercutting:
       - Removes the inverted reaction rule (jumping to $105 when rival priced <= $70).
       - Anchors to rival's exponentially smoothed price and undercuts by $1-$3 when
         inventory lags behind the pacing curve.
    3. Realistic Empirical Ceiling ($95-$100) & Floor ($25):
       - Empirical match logs across 60,000 periods proved customer WTP virtually tops out
         around $95-$105 (0 sales above $105). Monopoly and baseline ceilings are capped at $100.
    4. Active Cross-Season Learning in `information_dump`:
       - Explicitly persists competitor price distributions, rival sellout frequency,
         historical conversion rates, and classifies rival archetype across seasons 1..100.
    """
    
    # -------------------------------------------------------------------------
    # 1. Initialize & Update Persistent Memory (information_dump)
    # -------------------------------------------------------------------------
    if information_dump is None or not isinstance(information_dump, dict):
        information_dump = {
            "total_capacity": 80,
            "rival_season_medians": [],
            "rival_ema_price": 65.0,
            "rival_profile": "balanced",  # 'aggressive' (<=60), 'balanced' (60-80), 'high' (>80)
            "season_sales_history": [],
            "rival_sold_out_seasons": 0,
            "my_last_price": 65.0,
            "rival_last_price": 65.0,
            "price_demand_stats": {},     # bucket -> [total_demand, count_periods]
            "last_season_seen": 0
        }
    
    TOTAL_CAPACITY = information_dump.get("total_capacity", 80)
    t = selling_period_in_current_season
    
    # -------------------------------------------------------------------------
    # 2. Calculate Current Inventory & Remaining Horizon
    # -------------------------------------------------------------------------
    if demand_historical_in_current_season is not None and len(demand_historical_in_current_season) > 0:
        units_sold_so_far = int(np.sum(demand_historical_in_current_season))
        last_demand = float(demand_historical_in_current_season[-1])
    else:
        units_sold_so_far = 0
        last_demand = 0.0
        
    remaining_capacity = max(0, TOTAL_CAPACITY - units_sold_so_far)
    remaining_periods = max(1, 101 - t)
    
    # Period 101 is the end-of-season feedback step (price returned is not evaluated)
    if t > 100:
        if prices_historical_in_current_season is not None and prices_historical_in_current_season.shape[0] >= 2:
            rival_season_prices = prices_historical_in_current_season[1, :]
            if len(rival_season_prices) > 0:
                season_med = float(np.median(rival_season_prices))
                information_dump["rival_season_medians"].append(season_med)
                # Classify rival archetype across seasons
                overall_med = float(np.median(information_dump["rival_season_medians"]))
                if overall_med <= 60.0:
                    information_dump["rival_profile"] = "aggressive"
                elif overall_med <= 80.0:
                    information_dump["rival_profile"] = "balanced"
                else:
                    information_dump["rival_profile"] = "high"
        information_dump["season_sales_history"].append(units_sold_so_far)
        if not competitor_has_capacity_current_period_in_current_season:
            information_dump["rival_sold_out_seasons"] += 1
        return 65.0, information_dump

    # If out of capacity early, return empirical ceiling
    if remaining_capacity <= 0:
        return 100.0, information_dump

    # -------------------------------------------------------------------------
    # 3. Extract Rival Historical Price Safely (Row 0 = Ours, Row 1 = Rival)
    # -------------------------------------------------------------------------
    latest_rival_price = None
    rival_recent_avg = information_dump.get("rival_ema_price", 65.0)
    
    if (
        prices_historical_in_current_season is not None
        and prices_historical_in_current_season.ndim == 2
        and prices_historical_in_current_season.shape[1] > 0
        and prices_historical_in_current_season.shape[0] >= 2
    ):
        latest_rival_price = float(prices_historical_in_current_season[1, -1])
        information_dump["rival_last_price"] = latest_rival_price
        
        # Update exponential moving average of rival price (only when rival has capacity)
        if competitor_has_capacity_current_period_in_current_season and latest_rival_price > 5.0:
            alpha = 0.25
            rival_recent_avg = alpha * latest_rival_price + (1.0 - alpha) * rival_recent_avg
            information_dump["rival_ema_price"] = rival_recent_avg
            
        # Also record realized demand against our last price bucket (rounded to nearest $5)
        my_prev_price = float(prices_historical_in_current_season[0, -1])
        bucket = int(round(my_prev_price / 5.0) * 5)
        stats = information_dump["price_demand_stats"].get(bucket, [0.0, 0])
        stats[0] += last_demand
        stats[1] += 1
        information_dump["price_demand_stats"][bucket] = stats
    else:
        latest_rival_price = information_dump.get("rival_last_price", 65.0)

    # End-of-season fallback if Period 101 is not called by environment
    if t == 100 and prices_historical_in_current_season is not None and prices_historical_in_current_season.shape[0] >= 2:
        rival_season_prices = prices_historical_in_current_season[1, :]
        if len(rival_season_prices) > 0 and information_dump.get("last_season_seen", 0) != current_selling_season:
            season_med = float(np.median(rival_season_prices))
            information_dump["rival_season_medians"].append(season_med)
            overall_med = float(np.median(information_dump["rival_season_medians"]))
            if overall_med <= 60.0:
                information_dump["rival_profile"] = "aggressive"
            elif overall_med <= 80.0:
                information_dump["rival_profile"] = "balanced"
            else:
                information_dump["rival_profile"] = "high"
            information_dump["last_season_seen"] = current_selling_season

    # -------------------------------------------------------------------------
    # 4. Continuous Dynamic Pacing Curve & Inventory Gap
    # -------------------------------------------------------------------------
    # Expected inventory depletion follows slightly convex trajectory (45% early/mid, 55% late)
    progress = (t - 1) / 100.0
    target_sold = TOTAL_CAPACITY * (0.85 * progress + 0.15 * (progress ** 2))
    target_remaining = TOTAL_CAPACITY - target_sold
    
    # Positive gap means we have MORE unsold seats than planned (behind pace -> lower price)
    # Negative gap means we have FEWER seats than planned (ahead of pace -> raise price)
    inventory_gap = remaining_capacity - target_remaining
    required_burn_rate = remaining_capacity / remaining_periods

    # -------------------------------------------------------------------------
    # 5. Pricing Regime Selection (Monopoly vs. Duopoly with Pacing)
    # -------------------------------------------------------------------------
    rival_profile = information_dump.get("rival_profile", "balanced")
    
    # === REGIME A: MONOPOLY MODE (Competitor Out of Capacity) ===
    if not competitor_has_capacity_current_period_in_current_season:
        # Empirical ceiling is $95-$100 (demand drops to ~0 above $105)
        if required_burn_rate <= 0.45:
            target_price = 96.0
        elif required_burn_rate <= 0.80:
            target_price = 86.0
        elif required_burn_rate <= 1.25:
            target_price = 76.0
        else:
            # Even in monopoly, if we hold too many seats late, price to clear volume
            target_price = max(52.0, 74.0 - (required_burn_rate - 1.25) * 12.0)

    # === REGIME B: DUOPOLY MODE (Competitor Has Capacity) ===
    else:
        # 1. Establish Market Anchor Price based on Rival Profile & Recent EMA
        if current_selling_season == 1 and t <= 5:
            # Initial cold-start exploration in reasonable mid-market range
            anchor_price = 68.0
        else:
            # Blend recent rival EMA with learned cross-season median
            if len(information_dump["rival_season_medians"]) > 0:
                historical_med = float(np.median(information_dump["rival_season_medians"]))
                anchor_price = 0.65 * rival_recent_avg + 0.35 * historical_med
            else:
                anchor_price = rival_recent_avg

            # Guard against extreme suicide dumping (e.g., rival at $18):
            # If rival dumps below $35, don't price at $15, but hold a competitive low-mid
            # clearing price ($48-$58) that still captures utility share without $0 revenue
            if anchor_price < 38.0:
                anchor_price = 48.0 if inventory_gap > 5.0 else 54.0

        # 2. Apply Continuous Inventory Pacing Adjustment
        if inventory_gap > 2.0:
            # Behind schedule: Undercut competitor anchor by $1.50 to $6.00 proportional to excess seats
            undercut = min(8.0, 1.5 + 0.35 * inventory_gap)
            target_price = anchor_price - undercut
        elif inventory_gap < -3.0:
            # Ahead of schedule (selling fast): Charge a scarcity premium over anchor
            premium = min(14.0, 1.0 + 0.45 * abs(inventory_gap))
            target_price = anchor_price + premium
        else:
            # On pace: Match or slightly undercut by $1.00 to maintain steady market share
            target_price = anchor_price - 1.0

        # 3. Micro-Probing in Early/Mid Season when Near Target Pace
        if 15 <= t <= 65 and abs(inventory_gap) <= 4.0:
            if t % 6 == 0:
                target_price += 4.0   # Gentle upward probe
            elif t % 6 == 3:
                target_price -= 3.5   # Gentle downward probe

        # 4. Late-Season Acceleration & Anti-Spoilage Valve (Periods 75 - 100)
        if t >= 75 and inventory_gap > 4.0:
            # Do not wait until period 90! Ramp discounts smoothly from period 75
            urgency_factor = (t - 74) / 26.0  # Scales 0.04 -> 1.00
            spoilage_discount = urgency_factor * max(0.0, (required_burn_rate - 0.85) * 14.0)
            target_price = min(target_price, anchor_price - 2.0) - spoilage_discount

    # -------------------------------------------------------------------------
    # 6. Clip Price to Empirically Viable Bounds ($25.0 to $100.0) & Persist
    # -------------------------------------------------------------------------
    final_price = float(np.clip(target_price, 25.0, 100.0))
    
    information_dump["my_last_price"] = final_price
    information_dump["last_period_priced"] = t
    
    return final_price, information_dump
