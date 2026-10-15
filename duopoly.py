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
    Dynamic Pricing Algorithm v2.1 for Airline Duopoly Competition.
    
    Key Strategic Pillars:
    1. Cross-Season Market Demand & Sellout Tracking (Demand-Conditioned Reasoning):
       - Identifies whether the constant market demand model in the simulation is High,
         Moderate, or Low based on empirical competitor inventory exhaustion periods.
       - High Demand Market (Rival sells out early <= Period 75): Holds high margins,
         lets the dumper burn through their 80 seats, and extracts pure monopoly surplus ($86-$96).
       - Low/Moderate Demand Market (Rival rarely sells out or sells out late >= Period 95):
         Actively contests the market volume ($28-$34) to avoid getting starved of sales.
    2. Multi-Tier Rival Archetype Profiling:
       - 'deep_dumper' (<= $24, e.g. FuturisticSalamander 3DRpAU at $18.40)
       - 'dumper' ($24 - $38, e.g. Aggressive Undercutter)
       - 'aggressive' ($38 - $58, e.g. ZealousMosquito, UncoveredElephant)
       - 'balanced' ($58 - $78, e.g. FuturisticSalamander D9Le8X)
       - 'high' (> $78, e.g. Old v1.0 Baseline)
    3. Continuous Dynamic Inventory Pacing:
       - Tracks convex depletion curve: TargetSold(t) = 80 * (0.85*tau + 0.15*tau^2).
       - Adjusts reference anchor with smart undercut ($1.50-$8.00) or premium (+$1-$14).
    4. Empirical Monopoly Ceiling ($86-$96) & Spoilage Valve ($25-$100):
       - Respects observed customer WTP bounds with zero external library overhead.
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
            "rival_sellout_periods": [],        # Period when rival sold out (1..100) or 101 if never
            "my_last_price": 65.0,
            "rival_last_price": 65.0,
            "price_demand_stats": {},           # bucket -> [total_demand, count_periods]
            "last_season_seen": 0,
            "current_season_sellout_t": None,  # Real-time sellout tracker within season
            "demand_regime": "unknown",         # 'high', 'moderate', 'low'
            "estimated_market_size": 95.0       # Running estimate of total arrivals per season
        }
    
    TOTAL_CAPACITY = information_dump.get("total_capacity", 80)
    t = selling_period_in_current_season
    
    # -------------------------------------------------------------------------
    # 2. Calculate Current Inventory & Track Real-Time Sellout
    # -------------------------------------------------------------------------
    if demand_historical_in_current_season is not None and len(demand_historical_in_current_season) > 0:
        units_sold_so_far = int(np.sum(demand_historical_in_current_season))
        last_demand = float(demand_historical_in_current_season[-1])
    else:
        units_sold_so_far = 0
        last_demand = 0.0
        
    remaining_capacity = max(0, TOTAL_CAPACITY - units_sold_so_far)
    remaining_periods = max(1, 101 - t)

    # Track real-time moment competitor runs out of capacity
    if not competitor_has_capacity_current_period_in_current_season:
        if information_dump.get("current_season_sellout_t") is None:
            information_dump["current_season_sellout_t"] = t
    
    # -------------------------------------------------------------------------
    # 3. Period 101: End-of-Season Inter-Season Machine Learning / Analytics
    # -------------------------------------------------------------------------
    if t > 100:
        if prices_historical_in_current_season is not None and prices_historical_in_current_season.shape[0] >= 2:
            rival_season_prices = prices_historical_in_current_season[1, :]
            if len(rival_season_prices) > 0:
                season_med = float(np.median(rival_season_prices))
                information_dump["rival_season_medians"].append(season_med)
                overall_med = float(np.median(information_dump["rival_season_medians"]))
                
                # Classify rival archetype using early-season pricing (periods 1-10) to distinguish
                # inherent dumpers (who open low <= $38) from followers/copycats who start high (~$65)
                early_prices = rival_season_prices[:10] if len(rival_season_prices) >= 10 else rival_season_prices
                early_med = float(np.median(early_prices))

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
        
        # Record rival sellout timing
        sellout_t = information_dump["current_season_sellout_t"] if information_dump["current_season_sellout_t"] is not None else 101
        information_dump["rival_sellout_periods"].append(sellout_t)
        if sellout_t <= 100:
            information_dump["rival_sold_out_seasons"] += 1

        # Classify market demand regime across seasons (demand is constant across seasons)
        sellout_arr = np.array(information_dump["rival_sellout_periods"])
        med_sellout = float(np.median(sellout_arr))
        sold_out_freq = float(np.mean(sellout_arr <= 100))

        if med_sellout <= 75 and sold_out_freq >= 0.5:
            information_dump["demand_regime"] = "high"
            information_dump["estimated_market_size"] = 125.0
        elif med_sellout >= 95 or sold_out_freq < 0.45:
            information_dump["demand_regime"] = "low"
            information_dump["estimated_market_size"] = 88.0
        else:
            information_dump["demand_regime"] = "moderate"
            information_dump["estimated_market_size"] = 102.0

        # Reset seasonal tracker
        information_dump["current_season_sellout_t"] = None
        return 65.0, information_dump

    # If out of capacity early, return empirical ceiling
    if remaining_capacity <= 0:
        return 100.0, information_dump

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
        latest_rival_price = float(prices_historical_in_current_season[1, -1])
        information_dump["rival_last_price"] = latest_rival_price
        
        # Update EMA only when rival has capacity
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
    required_burn_rate = remaining_capacity / remaining_periods

    # -------------------------------------------------------------------------
    # 6. Pricing Strategy Selection
    # -------------------------------------------------------------------------
    rival_profile = information_dump.get("rival_profile", "balanced")
    is_deep_dumper = (rival_profile == "deep_dumper" or (latest_rival_price is not None and latest_rival_price <= 19.5 and rival_recent_avg <= 19.5))

    # === REGIME A: MONOPOLY MODE (Competitor Sold Out) ===
    if not competitor_has_capacity_current_period_in_current_season:
        if is_deep_dumper:
            # When deep dumper exhausts capacity, clearing at $64 captures both leisure
            # and business arrivals without pricing out mid-WTP travelers ($62-$80)
            target_price = 64.0
        else:
            if required_burn_rate <= 0.45:
                target_price = 96.0
            elif required_burn_rate <= 0.80:
                target_price = 86.0
            elif required_burn_rate <= 1.25:
                target_price = 76.0
            else:
                target_price = max(52.0, 74.0 - (required_burn_rate - 1.25) * 12.0)

    # === REGIME B: DUOPOLY MODE (Competitor Has Stock) ===
    else:
        if is_deep_dumper:
            # Deep Dumper (e.g. FuturisticSalamander 3DRpAU at $18.40):
            # Avoid the $25-$35 "Death Valley" where the rival still undercuts and we sacrifice margin.
            # Hold $58.00: rival burns through 80 seats with an absolute revenue cap of $1,472/season.
            # We capture residual demand and monopoly surplus to cleanly beat them.
            target_price = 58.0
        else:
            if current_selling_season == 1 and t <= 5:
                anchor_price = 68.0
            else:
                if len(information_dump["rival_season_medians"]) > 0:
                    historical_med = float(np.median(information_dump["rival_season_medians"]))
                    anchor_price = 0.65 * rival_recent_avg + 0.35 * historical_med
                else:
                    anchor_price = rival_recent_avg

                # Calibrated Anti-Dumping & Demand Contestation Guard:
                # If rival dumps below $38 (e.g. $20-$35), anchor at $35 when behind pace to contest
                # market share, or $44 when on pace to prevent self-induced price collapse with followers.
                if anchor_price < 38.0:
                    anchor_price = 35.0 if inventory_gap > 4.0 else 44.0

            # Dynamic Pacing Undercut / Premium
            if inventory_gap > 2.0:
                undercut = min(8.0, 1.5 + 0.35 * inventory_gap)
                target_price = anchor_price - undercut
            elif inventory_gap < -3.0:
                premium = min(14.0, 1.0 + 0.45 * abs(inventory_gap))
                target_price = anchor_price + premium
            else:
                target_price = anchor_price - 1.0

            # Micro-Probing in early/mid season
            if 15 <= t <= 65 and abs(inventory_gap) <= 4.0:
                if t % 6 == 0:
                    target_price += 4.0
                elif t % 6 == 3:
                    target_price -= 3.5

            # Late-Season Anti-Spoilage Ramp (Periods 75 - 100)
            if t >= 75 and inventory_gap > 4.0:
                urgency_factor = (t - 74) / 26.0
                spoilage_discount = urgency_factor * max(0.0, (required_burn_rate - 0.85) * 14.0)
                target_price = min(target_price, anchor_price - 2.0) - spoilage_discount

    # -------------------------------------------------------------------------
    # 7. Clip to Viable Bounds ($25.0 - $100.0) & Persist State
    # -------------------------------------------------------------------------
    final_price = float(np.clip(target_price, 25.0, 100.0))
    information_dump["my_last_price"] = final_price
    information_dump["last_period_priced"] = t
    
    return final_price, information_dump
