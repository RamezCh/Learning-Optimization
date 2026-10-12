# Dynamic Pricing Algorithm Evolution & Post-Mortem Report (`v1.0` $\to$ `v2.0`)

This document archives our **initial baseline algorithm (`v1.0`)**, presents the **empirical post-mortem findings** from 60,000 competition periods across 6 tournaments, and details the **engineering and economic changes implemented in `v2.0`** ([`duopoly.py`](file:///c:/Users/ramiz/OneDrive/Desktop/BHT/BHT%20-%20Assignments/Semester%203/Learning%20Optimization/duopoly.py)) along with the rationale behind each modification.

---

## 1. Executive Summary of `v1.0` Tournament Performance

In our first round of competition matches, the baseline strategy (`v1.0`) generated **\$0.00 in total revenue in 4 out of 6 tournaments** and captured only partial volume in the remaining 2 tournaments:

| Tournament ID | Opponent | Total Sales | Total Revenue | Our Avg Price | Rival Avg Price | Periods Rival Sold Out |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `3DRpAU` | `FuturisticSalamander` | 0 | \$0.00 | \$87.05 | \$18.61 | 81 |
| `3PTJwD` | `UncoveredElephant` | 2,041 | \$132,877.00 | \$86.61 | \$51.37 | 2,481 |
| `3XkNYY` | `ZealousMosquito` | 0 | \$0.00 | \$87.05 | \$54.89 | 0 |
| `3n5PCY` | `ZealousMosquito` | 0 | \$0.00 | \$87.05 | \$55.09 | 0 |
| `D9Le8X` | `FuturisticSalamander` | 2,818 | \$203,615.00 | \$84.84 | \$69.78 | 25 |
| `HhTiQz` | `UncoveredElephant` | 0 | \$0.00 | \$84.01 | \$73.92 | 144 |

### Sales Breakdown by Phase Across All 60,000 Periods
* **Phase 1 (Periods 1–50):** 994 seats sold (20.5% of total sales)
* **Phase 2 (Periods 51–70):** 523 seats sold (10.8% of total sales)
* **Phase 3 (Periods 71–100):** 3,342 seats sold (68.8% of total sales)
* **Total Seats Sold:** 4,859 out of 48,000 available capacity units (**10.1% overall load factor**; **89.9% seat spoilage**).

---

## 2. Archive of Previous Algorithm (`v1.0`)

Below is the exact reference implementation of our first-generation algorithm (`v1.0`) prior to the empirical revision:

```python
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
    # 1. Initialize Persistent Memory (information_dump)
    if information_dump is None or not isinstance(information_dump, dict):
        information_dump = {
            "total_capacity": 80,
            "season_history": {},
            "learned_wtp_level": "medium",
            "rival_avg_price_late": 100.0
        }
    
    TOTAL_CAPACITY = information_dump.get("total_capacity", 80)
    
    # 2. Calculate Inventory & Historical Sales
    if demand_historical_in_current_season is not None and len(demand_historical_in_current_season) > 0:
        units_sold_so_far = int(np.sum(demand_historical_in_current_season))
    else:
        units_sold_so_far = 0
        
    remaining_capacity = max(0, TOTAL_CAPACITY - units_sold_so_far)
    remaining_periods = 101 - selling_period_in_current_season
    
    if remaining_capacity <= 0:
        return 150.0, information_dump

    # 3. Analyze Rival Historical Pricing
    latest_rival_price = None
    if prices_historical_in_current_season is not None and prices_historical_in_current_season.shape[1] > 0:
        if prices_historical_in_current_season.shape[0] >= 2:
            latest_rival_price = float(prices_historical_in_current_season[1, -1])
        else:
            latest_rival_price = float(prices_historical_in_current_season[0, -1])

    # 4. Phase-Based Pricing Logic
    target_price = 90.0

    # PHASE 1: Early Season (Periods 1 - 50)
    if selling_period_in_current_season <= 50:
        target_price = 85.0
        
    # PHASE 2: Mid-Season WTP Probing (Periods 51 - 70)
    elif selling_period_in_current_season <= 70:
        if selling_period_in_current_season % 4 == 0:
            target_price = 105.0
        elif selling_period_in_current_season % 4 == 2:
            target_price = 75.0
        else:
            target_price = 90.0

    # PHASE 3: Late Season High-Demand Surge (Periods 71 - 100)
    else:
        if not competitor_has_capacity_current_period_in_current_season:
            if remaining_capacity <= 15:
                target_price = 145.0
            elif remaining_capacity <= 35:
                target_price = 125.0
            else:
                target_price = 105.0
        else:
            if latest_rival_price is not None:
                if latest_rival_price >= 120.0:
                    target_price = latest_rival_price - 5.0
                elif latest_rival_price <= 70.0:
                    target_price = 105.0
                else:
                    target_price = max(85.0, latest_rival_price - 2.0)
            else:
                target_price = 110.0

        # Safety Pacing Valve (Periods 90 - 100)
        seats_per_remaining_day = remaining_capacity / max(1, remaining_periods)
        if selling_period_in_current_season >= 90 and seats_per_remaining_day > 1.5:
            target_price = max(60.0, target_price - (seats_per_remaining_day * 10.0))

    # 5. Bound Price & Return
    final_price = float(np.clip(target_price, 30.0, 150.0))
    information_dump["last_period_priced"] = selling_period_in_current_season
    
    return final_price, information_dump
```

---

## 3. Root-Cause Diagnosis: Why `v1.0` Failed

### Flaw 1: The "Capacity Drain / Holdout" Fallacy (Market Size < Combined Capacity)
* **What `v1.0` Assumed:** Combined capacity is $80 + 80 = 160$ seats, and we assumed total market demand was large enough ($>160$ passengers) that an aggressive low-priced rival would quickly sell out their 80 seats by period 60–70, handing us an uncontested late-season monopoly.
* **What the Data Showed:** Total market arrivals across an entire 100-period season only average **75 to 105 passengers**:
  * In `3DRpAU`, `FuturisticSalamander` dumped seats at an average price of **\$18.61**. Even at giveaway prices, they only sold out in **41 out of 100 seasons**, and their average sellout period was **Period 99.02**—leaving us with 1 period to clear 80 seats.
  * In `3XkNYY` and `3n5PCY`, `ZealousMosquito` priced between \$40 and \$70 (mean \$55) and **never sold out once across 200 seasons** (`periods_rival_sold_out = 0`).
  * In `3PTJwD`, when `UncoveredElephant` sold out at period ~73.7 (80 seats sold), we captured all remaining demand and sold an average of **20.0 seats** ($80 + 20 = 100$ total market demand).
* **Consequence:** Holding out at \$85.00 during Periods 1–50 forfeited the first half of the season, frequently entering Period 51 with 0 sales and all 80 seats still unsold.

### Flaw 2: Inverted Phase 3 Duopoly Reaction (`latest_rival_price <= 70.0 -> 105.0`)
* **What `v1.0` Did:** When the rival priced at or below \$70.00 in Periods 71–100, `v1.0` raised our price to **\$105.00** ("don't follow them down; let them sell out").
* **What the Data Showed:**
  * `ZealousMosquito`'s price distribution in Phase 3 was strictly bounded between **\$40.00 and \$70.00** (mean \$55.11, max \$70.00).
  * Because `latest_rival_price <= 70.0` evaluated to `True` in **100% of periods**, our algorithm locked itself at **\$105.00** from Period 71 to 89 in every single season, resulting in **0 sales across 200 seasons**.

### Flaw 3: Unrealistic Upper Price Ceiling (\$105–\$150)
* **What `v1.0` Assumed:** Late-booking business travelers had willingness to pay (WTP) up to \$145–\$150.
* **What the Data Showed:** Across all 60,000 periods in the dataset, realized sales by price point were:
  * **\$60.00:** 1,628 periods with sales
  * **\$85.00:** 1,235 periods with sales
  * **\$75.00:** 281 periods with sales
  * **\$90.00:** 161 periods with sales
  * **\$105.00:** Only **28 periods** with sales (out of thousands of attempts)
  * **Above \$105.00:** **0 sales**
* **Consequence:** Pricing at \$105–\$145 choked off customer demand even when the rival was out of capacity. Over 85% of all successful transactions occurred at or below **\$85.00**.

### Flaw 4: Dead `information_dump` Memory & Circular Price Fallback Bug
* **Memory Never Updated:** `v1.0` initialized keys (`season_history`, `learned_wtp_level`, `rival_avg_price_late`) in `information_dump`, but **only ever wrote `last_period_priced`** at the end of the function. Every season started completely blind to the previous season's prices and demand.
* **Row Indexing Bug:** In Section 3 of `v1.0`, the `else` branch `latest_rival_price = float(prices_historical_in_current_season[0, -1])` read row `0` (our own historical price) if `shape[0] < 2`, risking self-referential price loops.

### Flaw 5: Late & Floor-Constrained Safety Valve (Period 90, \$60 Floor)
* **What `v1.0` Did:** Waited until Period 90 (`remaining_periods <= 11`) to trigger inventory liquidation, and enforced a hard floor of **\$60.00**.
* **Consequence:** With only ~10 customer arrivals left in the final 10 periods, it is mathematically impossible to clear 60–80 stranded seats. Furthermore, a \$60.00 floor could not undercut rivals pricing at \$40–\$55.

---

## 4. Summary of Changes Applied in `v2.0` and Rationale

| Component | Previous Algorithm (`v1.0`) | Updated Algorithm (`v2.0`) | Rationale / Empirical Justification |
| :--- | :--- | :--- | :--- |
| **Inventory Pacing** | Static \$85 holdout in Periods 1–50; no pacing until Period 90. | **Continuous Dynamic Pacing Curve** from Period 1 to 100: $\text{TargetSold}(t) = 80 \cdot (0.85\tau + 0.15\tau^2)$ where $\tau = \frac{t-1}{100}$. | Because total market demand is only ~80–100 seats per season, seats must be sold steadily throughout the entire season rather than hoarded for a non-existent late-season windfall. |
| **Competitor Reaction (Duopoly)** | If rival $\le \$70$, jump to **\$105.00**; only undercut if rival $\ge \$120$. | **Competitor EMA Anchoring + Pacing Undercut:** Track rival's Exponential Moving Average (`rival_ema_price`) and cross-season median. When behind target pace (`inventory_gap > 2`), undercut rival anchor by **\$1.50–\$8.00**. | Directly fixes the 0-revenue failure against `ZealousMosquito` (\$40–\$70) and `UncoveredElephant` (\$51–\$74). Undercutting by \$1–\$3 when behind pace immediately captures market share. |
| **Extreme Dumping Guard** | Retreated to \$105 whenever rival priced low. | If rival dumps below \$38 (e.g., `FuturisticSalamander` at \$18), floor the anchor at **\$48–\$54** rather than matching \$18 or retreating to \$105. | Prevents destructive near-zero margin sales while staying low enough within the random utility window to capture steady volume. |
| **Price Ceiling & Monopoly Mode** | Ceiling of **\$150.00**; Monopoly prices of **\$105 / \$125 / \$145**. | **Hard Ceiling of \$100.00**; Monopoly prices tiered at **\$76 / \$86 / \$96** based on `required_burn_rate` (and lower if high inventory remains). | Empirical data showed **zero demand above \$105** and virtually all demand concentrated at $\le \$90$. Capping monopoly extraction at \$96 maximizes expected revenue $p \cdot d(p)$. |
| **Anti-Spoilage Liquidation** | Triggered at Period 90 with a hard **\$60.00** floor. | Smooth urgency ramp starting at **Period 75** (`urgency_factor = (t - 74) / 26`), with global floor lowered to **\$25.00**. | Starting the discount ramp at Period 75 provides 26 periods (where Poisson arrivals peak) to clear excess stock. Lowering the floor to \$25.00 guarantees competitiveness against \$40–\$55 rivals. |
| **Cross-Season Memory (`information_dump`)** | Initialized dictionary keys but never updated them across seasons. | Actively records `rival_season_medians`, `rival_ema_price`, `rival_profile` (`aggressive`, `balanced`, `high`), `season_sales_history`, `rival_sold_out_seasons`, and `price_demand_stats`. | Enables Season $k$ to immediately start with the learned competitor median price from Seasons $1 \dots k-1$, eliminating cold-start mispricing after Season 1. |
| **Historical Array Indexing** | Fallback read `prices_historical[0, -1]` (own price) as rival price. | Explicitly checks `ndim == 2` and `shape[0] >= 2` for row `1` (rival) and row `0` (own), falling back to persisted `information_dump["rival_last_price"]`. | Eliminates self-referential pricing bugs on edge-case array shapes. |
