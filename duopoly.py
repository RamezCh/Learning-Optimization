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
    Dynamic Pricing Algorithm v5.0 for Airline Duopoly Competition.
    
    Key Strategic Pillars:
    1. Multi-Tier Rival Archetype Profiling & Counter-Pricing:
       - 'deep_dumper' (<= $20.00, e.g. FuturisticSalamander at $18.40)
       - 'dumper' ($20.00 - $38.00, e.g. SageSloth at $25.80, FierceDuck at $34.50)
       - 'aggressive' ($38.00 - $58.00, e.g. PinkDuck, ZealousMosquito)
       - 'balanced' ($58.00 - $78.00, e.g. Winner_2023, ScrupulousZebu)
       - 'high' (> $78.00, e.g. LoutishMacaque, Old v1.0 Baseline)
    2. Deep Dumper Defense with Automated Circuit Breaker:
       - Holds $58.00 in duopoly to avoid "Death Valley", forcing the rival to cap out at $1,472/season.
       - If Season 1 yields < 15 sales (e.g. low-demand dumper markets like 35zapi), the circuit
         breaker lowers monopoly clearing to $54.00, and if failure persists, contests duopoly share
         ($32-$36) to prevent zero-revenue blowouts.
    3. Universal Low-Sales Circuit Breaker (Cures the 3d8tDh Deadlock):
       - If holdout or pacing produces consecutive low-sales seasons (< 42 seats sold), breaks out
         of artificial anchors and contests market volume at rival_med - $0.50 to clear capacity.
    4. Continuous Dynamic Anchor Smoothing:
       - Replaces discontinuous step-jumps with a smooth linear decay against inventory gap,
         preventing reactive rivals from cascading downward into price wars.
    5. Calibrated Empirical Monopoly Engine:
       - Anchors peak monopoly yield at $78-$84, with high-conversion clearance ($52-$56) when
         inventory burn-rate is elevated, eliminating seat spoilage.
    6. Unconditional Late-Season Anti-Spoilage Clearance:
       - At t >= 90 with remaining capacity >= 2, applies clearance discounting regardless of gap
         to ensure zero seat spoilage at period 100.
    7. Clean Data Hygiene:
       - Filters out server 999.0 sold-out markers when computing rival historical medians.
    """
    
    # -------------------------------------------------------------------------
    # 1. Initialize & Update Persistent Memory (information_dump)
    # -------------------------------------------------------------------------
    if information_dump is None or not isinstance(information_dump, dict):
        information_dump = {
            "total_capacity": 80,
            "rival_season_medians": [],
            "rival_ema_price": 65.0,
            "rival_profile": "balanced",
            "season_sales_history": [],
            "rival_sold_out_seasons": 0,
            "rival_sellout_periods": [],
            "my_last_price": 65.0,
            "rival_last_price": 65.0,
            "price_demand_stats": {},           # bucket -> [total_demand, count_periods]
            "current_season_sellout_t": None,
            "failed_holdout_seasons": 0,        # Count seasons where deep dumper holdout got < 15 sales
            "consecutive_low_sales": 0          # Count seasons where sales were < 42
        }
    
    TOTAL_CAPACITY = information_dump.get("total_capacity", 80)
    t = selling_period_in_current_season
    
    # -------------------------------------------------------------------------
    # 2. Calculate Current Inventory & Demand
    # -------------------------------------------------------------------------
    if demand_historical_in_current_season is not None and len(demand_historical_in_current_season) > 0:
        units_sold_so_far = int(np.sum(demand_historical_in_current_season))
        last_demand = float(demand_historical_in_current_season[-1])
    else:
        units_sold_so_far = 0
        last_demand = 0.0
        
    remaining_capacity = max(0, TOTAL_CAPACITY - units_sold_so_far)
    remaining_periods = max(1, 101 - t)
    required_burn_rate = remaining_capacity / remaining_periods

    # Track real-time moment competitor runs out of capacity
    if not competitor_has_capacity_current_period_in_current_season:
        if information_dump.get("current_season_sellout_t") is None:
            information_dump["current_season_sellout_t"] = t
            
    # -------------------------------------------------------------------------
    # 3. Period 101: End-of-Season Inter-Season Analytics
    # -------------------------------------------------------------------------
    if t > 100:
        if prices_historical_in_current_season is not None and prices_historical_in_current_season.shape[0] >= 2:
            rival_season_prices = prices_historical_in_current_season[1, :]
            # Exclude server sold-out placeholder values (999.0)
            valid_rival_prices = rival_season_prices[rival_season_prices < 200.0]
            if len(valid_rival_prices) > 0:
                season_med = float(np.median(valid_rival_prices))
                information_dump["rival_season_medians"].append(season_med)
                overall_med = float(np.median(information_dump["rival_season_medians"]))

                if overall_med <= 20.0:
                    information_dump["rival_profile"] = "deep_dumper"
                elif overall_med <= 38.0:
                    information_dump["rival_profile"] = "dumper"
                elif overall_med <= 58.0:
                    information_dump["rival_profile"] = "aggressive"
                elif overall_med <= 78.0:
                    information_dump["rival_profile"] = "balanced"
                else:
                    information_dump["rival_profile"] = "high"

        information_dump["season_sales_history"].append(units_sold_so_far)
        
        # Track failed holdout against deep dumpers (e.g. 35zapi)
        if information_dump.get("rival_profile") == "deep_dumper":
            if units_sold_so_far < 15:
                information_dump["failed_holdout_seasons"] += 1
            else:
                information_dump["failed_holdout_seasons"] = 0

        # Track general low sales for circuit breaker (cures 3d8tDh deadlock)
        if units_sold_so_far < 42:
            information_dump["consecutive_low_sales"] += 1
        else:
            information_dump["consecutive_low_sales"] = 0

        # Record rival sellout timing
        sellout_t = information_dump["current_season_sellout_t"] if information_dump["current_season_sellout_t"] is not None else 101
        information_dump["rival_sellout_periods"].append(sellout_t)
        if sellout_t <= 100:
            information_dump["rival_sold_out_seasons"] += 1

        information_dump["current_season_sellout_t"] = None
        return 65.0, information_dump

    if remaining_capacity <= 0:
        return 95.0, information_dump

    # -------------------------------------------------------------------------
    # 4. Extract Rival Historical Price Safely (Row 0 = Ours, Row 1 = Rival)
    # -------------------------------------------------------------------------
    latest_rival_price = None
    rival_recent_avg = information_dump.get("rival_ema_price", 65.0)
    
    if (
        prices_historical_in_current_season is not None
        and prices_historical_in_current_season.ndim == 2
        and prices_historical_in_current_season.shape[1] > 0
        and prices_historical_in_current_season.shape[0] >= 2
    ):
        raw_rival_price = float(prices_historical_in_current_season[1, -1])
        if raw_rival_price < 200.0:
            latest_rival_price = raw_rival_price
            information_dump["rival_last_price"] = latest_rival_price
            
            if competitor_has_capacity_current_period_in_current_season and latest_rival_price > 5.0:
                alpha = 0.25
                rival_recent_avg = alpha * latest_rival_price + (1.0 - alpha) * rival_recent_avg
                information_dump["rival_ema_price"] = rival_recent_avg
                
            # Record realized demand against our last price bucket ($5 granularity)
            my_prev_price = float(prices_historical_in_current_season[0, -1])
            bucket = int(round(my_prev_price / 5.0) * 5)
            stats = information_dump["price_demand_stats"].get(bucket, [0.0, 0])
            stats[0] += last_demand
            stats[1] += 1
            information_dump["price_demand_stats"][bucket] = stats
    else:
        latest_rival_price = information_dump.get("rival_last_price", 65.0)

    # -------------------------------------------------------------------------
    # 5. Continuous Dynamic Pacing Curve & Inventory Gap
    # -------------------------------------------------------------------------
    progress = (t - 1) / 100.0
    target_sold = TOTAL_CAPACITY * (0.85 * progress + 0.15 * (progress ** 2))
    target_remaining = TOTAL_CAPACITY - target_sold
    inventory_gap = remaining_capacity - target_remaining

    # -------------------------------------------------------------------------
    # 6. Pricing Strategy Selection
    # -------------------------------------------------------------------------
    rival_profile = information_dump.get("rival_profile", "balanced")
    is_deep_dumper = (rival_profile == "deep_dumper" or (latest_rival_price is not None and latest_rival_price <= 19.5 and rival_recent_avg <= 19.5))
    failed_holdouts = information_dump.get("failed_holdout_seasons", 0)
    low_sales_streak = information_dump.get("consecutive_low_sales", 0)

    # === REGIME A: MONOPOLY MODE (Competitor Sold Out) ===
    if not competitor_has_capacity_current_period_in_current_season:
        if is_deep_dumper:
            # If holdout previously produced 0 sales at $64, scale down to empirical conversion zone ($54)
            if failed_holdouts >= 1:
                target_price = 54.0 if required_burn_rate <= 1.0 else 46.0
            else:
                target_price = 64.0 if required_burn_rate <= 1.2 else 58.0
        else:
            # Realigned monopoly tiers matching customer WTP distribution
            if required_burn_rate <= 0.35 and remaining_capacity <= 8:
                target_price = 92.0  # Harvest "Up to High" buyers when seats are scarce
            elif required_burn_rate <= 0.55:
                target_price = 84.0  # Upper leisure tier
            elif required_burn_rate <= 0.85:
                target_price = 78.0  # Core leisure yield
            elif required_burn_rate <= 1.25:
                target_price = 72.0  # Mid leisure transition
            else:
                # High burn rate: smooth descent into budget conversion band ($52-$56)
                target_price = max(48.0, 72.0 - (required_burn_rate - 1.25) * 14.0)

    # === REGIME B: DUOPOLY MODE (Competitor Has Stock) ===
    else:
        if is_deep_dumper:
            # Deep Dumper Defense with Circuit Breaker:
            if failed_holdouts >= 2:
                # Break out of the $0 trap: contest volume at $32-$36 to earn $200k+ instead of $0
                anchor_price = 34.0 if inventory_gap > 3.0 else 38.0
                target_price = anchor_price - min(6.0, 1.0 + 0.3 * max(0.0, inventory_gap))
            else:
                # Standard holdout: hold $58.00 during duopoly to let rival cap out at $1,472
                target_price = 58.0
        elif low_sales_streak >= 2:
            # Universal Circuit Breaker for persistent low-price deadlocks (e.g. SageSloth 3d8tDh)
            hist_med = float(np.median(information_dump["rival_season_medians"])) if len(information_dump["rival_season_medians"]) > 0 else rival_recent_avg
            target_price = min(hist_med - 0.5, rival_recent_avg - 0.5)
            if inventory_gap > 3.0:
                target_price -= min(4.0, 0.35 * inventory_gap)
        else:
            if current_selling_season == 1 and t <= 5:
                anchor_price = 68.0
            else:
                if len(information_dump["rival_season_medians"]) > 0:
                    hist_med = float(np.median(information_dump["rival_season_medians"]))
                    anchor_price = 0.65 * rival_recent_avg + 0.35 * hist_med
                else:
                    anchor_price = rival_recent_avg

                # Anti-Dumping & Copycat Guard: smooth continuous transition
                if anchor_price < 38.0:
                    anchor_price = np.clip(44.0 - 0.8 * max(0.0, inventory_gap - 1.5), 35.0, 44.0)

            # Dynamic Pacing Undercut / Premium
            if inventory_gap > 2.0:
                undercut = min(8.0, 1.5 + 0.35 * inventory_gap)
                target_price = anchor_price - undercut
            elif inventory_gap < -3.0:
                premium = min(14.0, 1.0 + 0.45 * abs(inventory_gap))
                target_price = anchor_price + premium
            else:
                target_price = anchor_price - 1.0

            # High-Price Ceiling Guard in Duopoly (The LoutishMacaque / Old V1 Fix)
            # If competitor prices high ($70+), quoting $64+ yields near-zero demand in early/mid season.
            # Cap target_price at $52-$54 if behind pace to stimulate volume and prevent late-season panic dumping.
            if target_price > 54.0 and inventory_gap > 3.0 and t >= 15:
                target_price = 52.0

            # Micro-Probing in early/mid season
            if 15 <= t <= 65 and abs(inventory_gap) <= 4.0:
                if t % 6 == 0:
                    target_price += 4.0
                elif t % 6 == 3:
                    target_price -= 3.5

            # Late-Season Anti-Spoilage Clearance (Periods 75 - 100)
            if t >= 75 and (inventory_gap > 3.5 or (t >= 90 and remaining_capacity >= 2)):
                urgency_factor = (t - 74) / 26.0
                spoilage_discount = urgency_factor * max(0.0, (required_burn_rate - 0.80) * 14.0)
                target_price = min(target_price, anchor_price - 2.0) - spoilage_discount

    # -------------------------------------------------------------------------
    # 7. Clip to Viable Bounds ($25.0 - $100.0) & Persist State
    # -------------------------------------------------------------------------
    final_price = float(np.clip(target_price, 25.0, 100.0))
    information_dump["my_last_price"] = final_price
    information_dump["last_period_priced"] = t
    
    return final_price, information_dump
