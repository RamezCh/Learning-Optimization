# Dynamic Pricing in Airline Duopoly: Learning Optimization

**Berliner Hochschule für Technik (BHT)**  
**Course:** Learning Optimization  
**Project:** Duopoly Pricing Challenge (DPC)  
**Date:** October 2026

---

# 1. Current Status of the Algorithm(s)

### Codebase Implementations on the `regression` Branch:
1. **Master Algorithm ([`duopoly.py`](file:///c:/Users/ramiz/OneDrive/Desktop/BHT/BHT%20-%20Assignments/Semester%203/Learning%20Optimization/duopoly.py) – v5.0):**
   * The production competition entry.
   * Closed-loop dynamic pacing controller using an Exponential Moving Average (EMA) price anchor, a universal low-sales circuit breaker, a duopoly high-price ceiling guard ($52), and an empirical monopoly sweet-spot harvester ($52–$56).
   * **Cumulative Benchmark Revenue:** **$1,209.9k** across 500 simulated seasons (**87.4% win rate**, **85.6% load factor**).
2. **Multiple Linear Regression Variant ([`duopoly_mlr.py`](file:///c:/Users/ramiz/OneDrive/Desktop/BHT/BHT%20-%20Assignments/Semester%203/Learning%20Optimization/duopoly_mlr.py)):**
   * Online Ordinary Least Squares (OLS) model trained on prior seasons of the active simulation.
   * Dynamically estimates simulation-specific baseline demand ($\beta_0$) and cross-price sensitivity ($\beta_{\Delta}$), adjusting target anchor prices.
3. **Poisson / Logistic Discrete Choice Variant ([`duopoly_poisson_logit.py`](file:///c:/Users/ramiz/OneDrive/Desktop/BHT/BHT%20-%20Assignments/Semester%203/Learning%20Optimization/duopoly_poisson_logit.py)):**
   * Probabilistic engine modeling time-dependent Poisson passenger arrival intensity $\lambda(t)$ and logistic discrete choice conversion probability, optimizing ticket price against capacity opportunity cost $\mu(t, c)$.

### Verification, Testing & Runtime Status:
* **Contract Compliance:** All three algorithms strictly implement the required competition signature: `p(current_selling_season, selling_period, prices_history, demand_history, competitor_has_capacity, information_dump)`.
* **Latency & Resources:** Average execution latency is **~0.015s** (hard ceiling is $\le 0.20$s); memory footprint is **< 65 MB** (limit is $\le 512$ MB).
* **Price Bounds:** All prices are strictly clipped to $[25.00, 100.00]$.
* **Test Suite:** 100% pass rate across automated pytest unit tests in [`tests/test_algorithm.py`](file:///c:/Users/ramiz/OneDrive/Desktop/BHT/BHT%20-%20Assignments/Semester%203/Learning%20Optimization/tests/test_algorithm.py) and [`tests/test_regression_variants.py`](file:///c:/Users/ramiz/OneDrive/Desktop/BHT/BHT%20-%20Assignments/Semester%203/Learning%20Optimization/tests/test_regression_variants.py).

---

# 2. Desired Goals

### Primary Optimization Objective:
Maximize cumulative total revenue across 100 independent flight departures (selling seasons) under an uncertain, randomized market demand model:
$$\max_{\mathbf{p}} \sum_{s=1}^{100} \sum_{t=1}^{100} p_{s,t} \cdot d_{s,t}$$

### Perishable Inventory Constraints:
* **Fixed Capacity Constraint:** Exactly 80 seats per season. Inventory cannot be replenished.
* **Zero Salvage Value:** At period $t=100$, any unsold seat expires with exactly $\$0$ salvage value ($V(0) = \$0.00$).
* **Seat Spoilage Target:** Minimize unsold inventory at period 100 ($\le 1.0$ seat/season average; $\ge 98.7\%$ load factor in stable markets).

### Competitive & Behavioral Objectives:
* **Monopoly Extraction:** When the rival exhausts their 80 seats, exploit unconstrained monopoly pricing along empirical consumer reservation price tiers ($72–$92) without risk of being undercut.
* **Adversarial Robustness:** Defend against extreme competitor archetypes:
  * *Deep dumpers* ($12–$19) without falling into zero-sales holdout traps.
  * *Persistent low-price deadlocks* ($22–$27) through automated circuit breakers.
  * *Copycat / undercutting rivals* without triggering destructive downward price wars.
  * *High-price anchor traps* ($70–$95) by observing customer reservation ceilings.
* **Academic Rubric Compliance:** Direct implementation and evaluation of the **Target Depletion Curve**, **Booking Curve Pacing**, and **Regression Demand Modeling**.

---

# 3. Implementation Details

### The Strategic Dilemma:
* **Ideal Strategy (Monopoly Induction):**
  If the rival prices low, allow them to rapidly burn their 80 seats ($80 \times p_{\text{rival}} \approx \$1,472$). Once they deplete inventory, they exit the market. We become the monopoly seller and harvest remaining demand at $\$72–\$92$.
* **Market Reality (The Capacity Trap):**
  Total market arrivals average only **75 to 105 passengers** across 100 periods, while combined airline capacity is 160 seats. If the rival prices at $\$35–\$55$, they will never sell out. Passively waiting leads to stranded seats and zero revenue.
* **Dynamic Response:**
  If the rival is not selling out and our booking pace falls behind schedule, the algorithm detects that market demand is insufficient. It immediately shifts from passive holdout to actively contesting volume via dynamic undercutting ($-\$1.50$ to $-\$8.00$).

### Decision Architecture & Opponent Profiling:
```
1. Capacity Check:
   IF competitor_has_capacity == False:
      -> Monopoly Engine: Price along burn-rate curve
         (BurnRate <= 0.35: $92 | BurnRate <= 0.85: $78 | BurnRate > 1.25: $52-$56)
   ELSE:
      -> Proceed to Duopoly Logic

2. Universal Circuit Breaker:
   IF consecutive_low_sales >= 2 (Seasons with < 42 sales):
      -> Contest Market Share: p = min(hist_med, ema) - $0.50

3. Archetype Classification & Pacing:
   CASE Deep Dumper (p <= $20):
      -> Hold $58 duopoly; sweep $54-$64 monopoly.
      -> IF failed >= 2 seasons: contest volume at $32-$36.
   CASE High Pricer (p >= $62):
      -> Ceiling Guard: Cap price at $52-$54 (prevent customer walkaways).
   CASE Balanced / Follower:
      -> Anchor = clip(44.0 - 0.8 * (inventory_gap - 1.5), 35.0, 44.0)
      -> Gap > +2.0 (Behind Pace): Undercut anchor by -$1.50 to -$8.00
      -> Gap < -3.0 (Ahead of Pace): Markup anchor by +$1.00 to +$14.00
      -> t >= 90 & remaining_capacity >= 2: Force clearance discount
```

---

# 4. Which Ideas Worked Well and Which Didn't

*(Derived directly from real tournament postmortems in [`ALGORITHM_CHANGES.md`](file:///c:/Users/ramiz/OneDrive/Desktop/BHT/BHT%20-%20Assignments/Semester%203/Learning%20Optimization/ALGORITHM_CHANGES.md))*

### What Failed (Postmortems & Traps):
1. **v1.0 Market Size Fallacy (\$0.00 in 4 of 6 Tournaments):**
   * Assumed arrivals $> 160$. Held $\$85$ early, raised to $\$105$ in Phase 3. Rivals captured all arrivals. **89.9% of seats spoiled unsold** (43,141 seats lost).
2. **v2.0 "Death Valley" Trap (FuturisticSalamander at \$18.40):**
   * Undercutting to $\$28–\$35$ against an $\$18$ dumper lost both volume and margin. Rival still sold out 80 seats, while we sold few seats at low prices (~$\$980$/season).
3. **v3.0 Passive Holdout Blowout (`35zapi` $\to$ \$0.00 Total Revenue):**
   * Held $\$58$ against $\$12$ dumper `RussetBat`. Rival only sold out at period 98; in the remaining 2 periods, customer WTP was below $\$64$. Sold **0 seats across 100 seasons**.
4. **v4.0 Dumper Deadlock (`3d8tDh` $\to$ 4,342 Spoiled Seats):**
   * `SageSloth` priced at median $\$25.80$. Classified as dumper, anchor snapped to $\$35$, price frozen at $\$27$. Rival never sold out, freezing our bot for 100 seasons (74.6% of round 4 spoilage).
5. **Unconstrained Myopic Regression:**
   * Raising prices based purely on single-period regression caused rivals to undercut, dropping conversion to zero and leaving 10% lower capacity load factor.

### What Worked Well (Key Breakthroughs):
1. **Universal Low-Sales Circuit Breaker:**
   * Automatically breaks out of failed holdouts and dumper freezes by contesting volume when sales fall below 42 for 2 seasons. Cured both `35zapi` and `3d8tDh`.
2. **High-Price Ceiling Guard (\$52.00):**
   * When rival prices at $\$70+$, quoting $\$63–\$67$ caused demand collapse because customer reservation prices are capped at $\$50–\$85$. Cap at $\$52$ cured `3kGhWc` 40% seat spoilage.
3. **Monopoly Sweet Spot (\$52–\$56):**
   * Quoting $>\$65$ in monopoly produced an 80% zero-conversion rate. Pricing high-burn monopoly at $\$52–\$56$ yields **peak \$70.28/period** and converts 1.45 seats/period.
4. **Continuous Anchor Smoothing (\$35–\$44):**
   * Replaced discontinuous anchor cliffs with smooth linear scaling: `clip(44 - 0.8*(gap-1.5), 35, 44)`, preventing undercutting cascades.
5. **Unconditional Late Clearance at $t \ge 90$:**
   * Liquidating remaining inventory regardless of gap eliminated perishable seat spoilage.

---

# 5. Performance Evaluation: Target Curve, Booking Curve & Regression

### Target Curve & Booking Pacing Formulation:
* **Target Depletion Curve:**
  $$\text{TargetSold}(t) = 80 \cdot \left(0.85\tau + 0.15\tau^2\right), \quad \tau = \frac{t-1}{100}$$
  Convex pacing aligns with accelerating Poisson arrival intensity $\lambda(t)$ near departure, avoiding premature discounting early in the season.
* **Booking Curve Pacing:**
  $$\text{InventoryGap}(t) = \text{RemainingCapacity}(t) - \text{TargetRemaining}(t)$$
  The real-time gap governs dynamic undercuts ($-\$1.50$ to $-\$8.00$) or scarcity premiums ($+\$1.00$ to $+\$14.00$).

### Econometric Regression Analysis (Chronological 80/20 Split):
Trained on **97,219 periods** (Seasons 1–80), evaluated on **24,258 blind holdout periods** (Seasons 81–100) from 13 tournaments:
| Model Type | Key Estimated Coefficient | Holdout Test Performance (Seasons 81–100) |
| :--- | :--- | :--- |
| **Multiple Linear Regression (OLS)** | $\beta_{\text{our}} = -0.0147, \; \beta_{\Delta} = +0.0153, \; \beta_t = +0.5407$ | $R^2 = +0.1494$, $\text{RMSE} = 0.9025$ |
| **Poisson Regressor (GLM)** | $\alpha_{\text{our}} = -0.0292, \; \alpha_t = +2.0710$ | $\text{RMSE} = 0.9014$, $\text{MAE} = 0.6713$ |
| **Logistic Regressor (Choice)** | $\gamma_{\Delta} = +0.0672, \; \gamma_{\text{our}} = -0.0590$ | **$\text{ROC-AUC} = \mathbf{0.7268}$**, $\text{Accuracy} = 65.63\%$ |

* **Empirical Takeaway:** Undercutting sensitivity ($\beta_{\Delta} = +0.015$) dominates absolute price sensitivity ($\beta_{\text{our}} = -0.0147$). Being $\$1.00$ cheaper matters more than absolute price levels.

### Head-to-Head Benchmark Comparison (1,500 Seasons Simulated):
| Competitor Archetype | Master v5.0 ([`duopoly.py`](file:///c:/Users/ramiz/OneDrive/Desktop/BHT/BHT%20-%20Assignments/Semester%203/Learning%20Optimization/duopoly.py)) | Multiple Linear Reg ([`duopoly_mlr.py`](file:///c:/Users/ramiz/OneDrive/Desktop/BHT/BHT%20-%20Assignments/Semester%203/Learning%20Optimization/duopoly_mlr.py)) | Poisson/Logistic ([`duopoly_poisson_logit.py`](file:///c:/Users/ramiz/OneDrive/Desktop/BHT/BHT%20-%20Assignments/Semester%203/Learning%20Optimization/duopoly_poisson_logit.py)) |
| :--- | :---: | :---: | :---: |
| **Copycat Rival** | $\$147.6\text{k}$ (51% Win) | $\$138.1\text{k}$ (47% Win) | $\$138.8\text{k}$ (50% Win) |
| **Aggressive Dumper ($\$18-\$38$)** | $\$174.5\text{k}$ (88% Win) | **$\mathbf{\$180.6\text{k}}$** (85% Win) | $\$179.7\text{k}$ (83% Win) |
| **Bounded Mid-Market ($\$40-$70)** | $\$298.1\text{k}$ (100% Win) | $\$297.5\text{k}$ (98% Win) | $\$299.0\text{k}$ (97% Win) |
| **Hist: ZealousMosquito (`3XkNYY`)** | $\$299.2\text{k}$ (100% Win) | $\$293.6\text{k}$ (100% Win) | $\$295.2\text{k}$ (100% Win) |
| **Hist: UncoveredElephant (`3PTJwD`)** | $\$290.5\text{k}$ (98% Win) | $\$259.3\text{k}$ (94% Win) | $\$259.1\text{k}$ (94% Win) |
| **Total Revenue (500 Seasons)** | **$\mathbf{\$1,209.9\text{k}}$** | $\$1,169.1\text{k}$ | $\$1,171.8\text{k}$ |
| **Mean Capacity Load Factor** | **85.6%** | $75.8\%$ | $75.9\%$ |

### Why Closed-Loop Pacing (DP) Won Overall & Where Regression Excelled:
* **Why DP Won Overall:** $+10\%$ higher load factor ($85.6\%$ vs $75.8\%$) avoided stranded inventory. No cold-start latency when simulations re-randomize demand parameters.
* **Where MLR Beat DP:** Against aggressive dumpers ($\$180.6\text{k}$ vs $\$174.5\text{k}$), MLR's dynamic elasticity dropped the price anchor faster and more fluidly than static heuristic rules.
