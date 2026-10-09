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
    Dynamic Pricing Algorithm for Airline Duopoly Competition.
    
    Strategy Highlights:
    1. Periods 1-70 (Probing & Capacity Reservation):
       - Preserve ~70% of capacity (56 seats) for high-demand final 30 days.
       - Test prices in $75-$105 range to probe customer WTP and build elasticity history.
    2. Periods 71-100 (High-Demand Exploitation):
       - Customer Poisson arrivals peak; WTP ranges up to $150.
       - MONOPOLY MODE (Competitor Sold Out): Charge premium rates ($110-$150) depending on remaining inventory.
       - DUOPOLY MODE (Competitor Has Capacity): Price strategically around rival without entering destructive price wars.
    3. Safety Pacing Valve:
       - Dynamically adjusts price in periods 90-100 if unsold seats remain to avoid spoilage (unsold seats = $0).
    4. Information Dump Memory:
       - Tracks cross-season learnings and rival behaviors.
    """
    
    # -------------------------------------------------------------------------
    # 1. Initialize Persistent Memory (information_dump)
    # -------------------------------------------------------------------------
    if information_dump is None or not isinstance(information_dump, dict):
        information_dump = {
            "total_capacity": 80,
            "season_history": {},
            "learned_wtp_level": "medium", # 'low', 'medium', 'high'
            "rival_avg_price_late": 100.0
        }
    
    TOTAL_CAPACITY = information_dump.get("total_capacity", 80)
    
    # -------------------------------------------------------------------------
    # 2. Calculate Inventory & Historical Sales
    # -------------------------------------------------------------------------
    if demand_historical_in_current_season is not None and len(demand_historical_in_current_season) > 0:
        units_sold_so_far = int(np.sum(demand_historical_in_current_season))
    else:
        units_sold_so_far = 0
        
    remaining_capacity = max(0, TOTAL_CAPACITY - units_sold_so_far)
    remaining_periods = 101 - selling_period_in_current_season
    
    # Edge case: If completely out of capacity, price arbitrarily high
    if remaining_capacity <= 0:
        return 150.0, information_dump

    # -------------------------------------------------------------------------
    # 3. Analyze Rival Historical Pricing (if available)
    # -------------------------------------------------------------------------
    latest_rival_price = None
    if prices_historical_in_current_season is not None and prices_historical_in_current_season.shape[1] > 0:
        # Assuming row 1 is competitor prices (or checking available rows)
        if prices_historical_in_current_season.shape[0] >= 2:
            latest_rival_price = float(prices_historical_in_current_season[1, -1])
        else:
            latest_rival_price = float(prices_historical_in_current_season[0, -1])

    # -------------------------------------------------------------------------
    # 4. Phase-Based Pricing Logic
    # -------------------------------------------------------------------------
    target_price = 90.0

    # === PHASE 1: Early Season (Periods 1 - 50) ===
    # Reserve inventory, filter out low-value bargain hunters
    if selling_period_in_current_season <= 50:
        # Price high enough to preserve seats, but low enough to capture high early WTP
        target_price = 85.0
        
    # === PHASE 2: Mid-Season WTP Probing (Periods 51 - 70) ===
    elif selling_period_in_current_season <= 70:
        # Alternating probing prices to measure market sensitivity
        if selling_period_in_current_season % 4 == 0:
            target_price = 105.0  # High probe
        elif selling_period_in_current_season % 4 == 2:
            target_price = 75.0   # Low probe
        else:
            target_price = 90.0   # Baseline

    # === PHASE 3: Late Season High-Demand Surge (Periods 71 - 100) ===
    else:
        # Scenario A: MONOPOLY MODE (Competitor Out of Capacity)
        if not competitor_has_capacity_current_period_in_current_season:
            if remaining_capacity <= 15:
                # High scarcity + high late demand = Max WTP extraction
                target_price = 145.0
            elif remaining_capacity <= 35:
                target_price = 125.0
            else:
                # Higher remaining inventory -> Keep price attractive to sell out
                target_price = 105.0
                
        # Scenario B: DUOPOLY MODE (Competitor Has Capacity)
        else:
            if latest_rival_price is not None:
                if latest_rival_price >= 120.0:
                    # Rival is pricing high -> Undercut slightly to capture high-WTP surge
                    target_price = latest_rival_price - 5.0
                elif latest_rival_price <= 70.0:
                    # Rival is dumping prices -> Don't follow down! Let rival sell out fast.
                    target_price = 105.0
                else:
                    # Rival is moderately priced -> Match or slightly beat
                    target_price = max(85.0, latest_rival_price - 2.0)
            else:
                target_price = 110.0

        # === Safety Pacing Valve (Periods 90 - 100) ===
        # Avoid spoilage: If we still have lots of seats relative to remaining days
        seats_per_remaining_day = remaining_capacity / max(1, remaining_periods)
        if selling_period_in_current_season >= 90 and seats_per_remaining_day > 1.5:
            # Drop price moderately to clear remaining inventory before period 100
            target_price = max(60.0, target_price - (seats_per_remaining_day * 10.0))

    # -------------------------------------------------------------------------
    # 5. Bound Price to Competition Limits ($0 to $150) & Return
    # -------------------------------------------------------------------------
    final_price = float(np.clip(target_price, 30.0, 150.0))
    
    # Store current period state in information_dump
    information_dump["last_period_priced"] = selling_period_in_current_season
    
    return final_price, information_dump
