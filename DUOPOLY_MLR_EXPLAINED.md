# The Airline Ticket Battle: How Our Pricing Algorithm Works
### A Plain-English Guide to `duopoly.py` (No Math or Business Background Required!)

![Rule-Based MLR Strategy Overview](imgs/mlr_strategy_explained.png)

---

## 1. The Big Picture: What Are We Actually Doing?

Imagine you own a small airline called **SkyBlue**. 
- You fly a plane every single day from **Berlin to New York**.
- Your plane has exactly **80 seats**.
- Tickets go on sale **100 days before takeoff** (Period 1 to 100).
- Across the terminal, there is another airline called **RivalAir** that also has **80 seats** on the exact same route on the exact same day.

### The Two Harsh Realities of Selling Flight Tickets:
1. **Seats Expire at Zero Value:** Once the plane wheels leave the runway on Day 100, any unsold seat is worth **\$0.00**. You cannot sell it tomorrow.
2. **You Can't Restock:** You cannot add an 81st seat. If you sell all 80 tickets on Day 20 for \$30 each, you make \$2,400. But if rich business travelers show up on Day 80 willing to pay \$90, you have nothing left to sell!

Your goal is simple: **Make the absolute maximum total dollars across all flights.**

---

## 2. The Core Problem: The Pricing Seesaw

Pricing a ticket is like a **seesaw**:

```text
       HIGH PRICE ($80)                       LOW PRICE ($35)
              ▲                                      ▲
              │                                      │
   Few passengers buy                       Huge crowd buys
   (High profit per ticket,                 (Lots of tickets sold,
    but plane flies empty!)                  but no profit per ticket!)
              │                                      │
              └───────────────▲──────────────────────┘
                              │
                    THE SWEET SPOT ($50-$55)
                  Maximum Dollars in the Bank!
```

- If your ticket is **too expensive (\$85)**: You make a high margin on each ticket, but only 10 people buy. The other 70 seats take off empty. **You lose.**
- If your ticket is **too cheap (\$30)**: Your plane fills up immediately in 15 days, but you gave away seats for peanuts. **You lose.**
- **The Golden Rule:** There is a curved "hill" of revenue. Our job is to stand exactly at the **peak of the hill**.

---

## 3. Our Strategy: The "Two-Brain" System

Instead of guessing prices, [`duopoly.py`](duopoly.py) uses a **Two-Brain System**:

```mermaid
flowchart TD
    CustomerArrival["Passenger Wants a Flight Ticket"] --> Bouncer{"Brain 1: The Bouncer (Safety Rules)<br/>Is the opponent doing something crazy?"}
    
    Bouncer -- "YES (Dumping, Bluffing, Sold Out)" --> TakeAction["Execute Instant Tactical Defense<br/>- Hold out at $58<br/>- Break volume deadlock<br/>- Post $130 when sold out"]
    
    Bouncer -- "NO (Normal Competition)" --> Economist["Brain 2: The Economist (MLR Model)<br/>1. Check how many seats we have left<br/>2. Predict passenger choice<br/>3. Solve exact sweet-spot price p*"]
    
    TakeAction --> FinalTicket["Post Ticket Price on Travel Website"]
    Economist --> FinalTicket
```

---

## 4. Brain 1: The Bouncer (Rules That Protect Us from Dirty Tricks)

Computers that use only math equations get fooled by irrational human competitors. **The Bouncer** enforces street-smart economic safeguards:

### Scenario A: The Crazy Price Dumper
* **The Situation:** RivalAir goes crazy and lists tickets at **\$18.00** (below operating cost).
* **What a dumb bot does:** Matches them at \$17.50, triggering a death spiral. Both airlines go broke.
* **What our Bouncer does (The Holdout):**
  1. We say: *"Go ahead, sell at \$18!"* We hold our price at **\$58.00**.
  2. All the bargain hunters buy from RivalAir. RivalAir burns through all 80 seats by Day 42.
  3. The moment RivalAir sells out, **we are the only seller left**. Monopoly prices then sit in a band (\$48–\$72 in a normal market): higher when we have few seats per remaining day, lower when we have many.
* **The "Safe Deadline" Guard:** What if demand is slow and the dumper hasn't sold out?
  - The code assumes we can sell about 1.85 seats per day, so 80 seats need about 44 days. The holdout therefore stops at day 57 at the latest (never before day 55).
  - After the deadline, or whenever demand looks low, we contest the market just below the dumper (at most \$24, never under \$18).
  - Real lesson (v3.0, `35zapi`): a rival at ~\$12 typically sold out only on day 98, and a holdout without a deadline sold **0 seats in 100 seasons**.

### Scenario B: The Low-Demand Alarm (a statistical test, not a fixed day)
* **The Situation:** We are cheaper than the rival, yet hardly anyone buys.
* **The trap:** A few empty days early in the season are *normal*, because only about 0.3–0.5 customers arrive per day. The old v6.0 rule ("4 empty days in a row → low demand") fired in 44% of perfectly good seasons.
* **What our Bouncer does now:** Every day we are cheaper than the rival, it compares two explanations: "normal market" (we should sell about 0.5–1.4 seats per day) or "low-demand market" (about 45% of that). It only sounds the alarm once low demand is **99 times more likely**. On past tournament data that means about 4% false alarms.
* **After the alarm:** The \$18 floor is unlocked and the monopoly band drops to \$28–\$44. The alarm resets every season; if a whole season sells fewer than 42 seats, the next season starts in low-demand mode.

### Scenario C: The High-Rival Dynamic Ceiling Guard
* **The Situation:** In a low-demand market where customers won't pay more than \$35, RivalAir foolishly posts **\$72.00**.
* **What a naive bot does:** Quotes \$65–\$68 because it's still cheaper than the rival. But passengers refuse to pay \$65, so nobody buys!
* **What our Bouncer does:** Whenever the rival's recent average is \$58 or more, it caps our price at **\$26–\$48** (lower the further behind pace we are), in any market. Real lesson (v3.0, `3kGhWc`): pricing about \$62–\$68 just under a \$71 rival sold only about 3 seats in days 1–75.

### Scenario D: The Circuit Breaker (Two Weak Seasons Against a Cheap Rival)
* **The Situation:** We sold fewer than 42 seats in each of the last two seasons, and the rival keeps pricing cheap (recent average under \$38).
* **What our Bouncer does:** It stops trusting the model and simply prices **\$0.50 under the rival** (a bit more when far behind pace), to win back volume. Real lesson (v4, `3d8tDh`): frozen at \$27 against a cheaper rival, we spoiled 4,342 seats.
* **Why only against a cheap rival (v6.3):** Undercutting a rival priced \$40–\$70 in a market whose customers only pay up to \$48 means pricing above what anyone pays. We sold almost nothing, the weak seasons continued, and the breaker never switched off (\$201 per season in simulation). Since v6.3, against an expensive rival the normal model and the high-rival ceiling decide instead (\$917 per season).

### Scenario E: Sold Out
* **The Situation:** We sold our 80th seat on Day 82. We have 0 seats left.
* **What our Bouncer does:** We post **\$130.00**. With no seats, the price can't earn or lose anything. A high number simply avoids giving a rival that copies our price a low anchor. (We have no evidence that this changed any rival's behaviour.)

---

## 5. Brain 2: The Economist (How the Math Actually Works)

When there is no emergency, **The Economist** takes over. It uses **Multiple Linear Regression (MLR)**.

### What is Multiple Linear Regression?
Think of it like a smart scale. We feed it 3 pieces of information every day, together with how many seats we sold:
1. **Our Price ($p$):** If we raise our price by \$1, fewer people buy from us.
2. **The Price Difference ($p_{\text{rival}} - p$):** The cheaper we are than the rival, the more passengers choose us. (The rival's price itself is not a separate input: together with our own price it would only repeat the same information, which is why v6.1 removed it.)
3. **Season Progress ($\tau$, 0 on day 1 and 1 on day 100):** More customers arrive close to departure, so demand rises with $\tau$.

The computer fits a predictive equation:
$$\text{Expected Seats Sold Today} = \text{Base} + \text{Own-Price Slope} \times \text{Our Price} + \text{Gap Slope} \times \text{Price Difference} + \text{Time Slope} \times \tau$$

The own-price slope is negative (higher price, fewer seats) and is kept at or below $-0.006$; the effect of a more expensive rival is kept at or above zero. The model is refit once per season on our latest 2,500 days of data.

---

### Target Curve vs. Booking Curve vs. Capacity Utilization

To understand how our Economist paces the flight, keep these three concepts distinct:

| Concept | Plain-English Meaning | Analogy | Real Numbers at Day 50 |
| :--- | :--- | :--- | :---: |
| **Target Curve** | **The Master Plan:** Where we *ideally want* sales to be. | The navigation route planned before takeoff. | $\text{Target} = \mathbf{37 \text{ seats sold}}$ ($43$ left) |
| **Booking Curve** | **The Reality:** How many seats we have *actually sold* so far. | The actual flight path the plane has flown. | E.g. We sold $\mathbf{32 \text{ seats}}$ ($48$ left) |
| **Capacity Utilization** | **The Seat Counter:** What percentage of our 80-seat plane is full. | The passenger manifest ($\text{Sold} / 80$). | $\frac{32}{80} = \mathbf{40.0\% \text{ Load Factor}}$ |
| **Inventory Gap** | **The Error Signal:** Are we lagging behind or flying ahead? | Being off-course by $+5$ miles. | $48 - 43 = \mathbf{+5 \text{ seats behind pace}}$ |

---

### Why $\tau^2$ and Why $0.15$ in the Target Formula?

Our planned Target Curve starts as:
$$\text{Target Cumulative Seats}(\tau) = 80 \times \left(0.85\tau + 0.15\tau^2\right), \quad \tau = \frac{\text{Day} - 1}{99}$$

1. **Why not just a flat line ($80 \times \tau$)?**
   - A flat pace would mean selling $0.80$ seats every single day.
   - At Day 50, a flat line says you must have sold **40 seats**.
   - But in real life, **passengers do not book at a flat rate!** Far more customers show up in the last weeks than in the first (in our own data, our sales per day roughly double from day 1 to day 85).
   - If you push to sell 40 seats by Day 50, you have to price low early on, and then have fewer seats for the busy final weeks.
2. **What does $\tau^2$ do?**
   - Squaring the time progress ($\tau^2$) creates a **gentle upward curve (convexity)**.
   - At Day 50 ($\tau = 0.5$), $\tau^2 = 0.25$. The formula asks for only **37 seats sold** instead of 40.
   - It deliberately saves about 3 extra seats for the final weeks (a small effect).
3. **Why $0.15$?**
   - **85% ($0.85\tau$):** Gives a steady, reliable baseline of ticket sales so we don't fall too far behind.
   - **15% ($0.15\tau^2$):** Shifts sales a little later. At most it is 3 seats behind a linear pace (at mid-season), not 12. The value 0.15 was chosen by hand, not fitted.
   - At Day 100 ($\tau = 1$): $0.85(1) + 0.15(1)^2 = 1.00 \implies 80 \times 1.00 = \mathbf{80 \text{ seats}}$ (100% sold out!).
4. **Since v6.2, 0.15 is only where we start.** Written with a "bend" $c$, the plan is $80 \times ((1-c)\tau + c\tau^2)$. Real markets book very differently: in one `v5` match customers came early ($c \approx 0.01$), in another very late ($c \approx 0.82$: only 24 seats sold by day 50, and still sold out). So after every season in which we sell **75 or more seats**, the algorithm measures that season's bend and averages it with the starting value 0.15, which counts like 3 seasons:
   $$c = \frac{3 \times 0.15 + \text{sum of measured bends}}{3 + \text{number of such seasons}}, \quad \text{kept between 0 and 0.8}$$
   In a late-booking market the plan therefore moves later, and being "behind" the old plan no longer triggers needless discounts. In simulation this earns about 2% more (see `imgs/learned_target_curve.png`).

---

### How the Inventory Gap Steers the Price (Two Simple Examples)

$$\text{Inventory Gap} = \text{Our Remaining Seats} - \text{Target Remaining Seats}$$

#### Story 1: Behind Pace (Too Many Empty Seats at Day 50)
- **Target Plan:** We should have 43 seats left (37 sold).
- **Reality:** We have 48 seats left (only 32 sold).
- **Inventory Gap:** $48 - 43 = \mathbf{+5 \text{ seats behind pace}}$.
- **What happens:** 
  - Having 5 extra empty seats means we are in danger of empty seats at departure.
  - The Economist lowers the opportunity shadow price $\mu$ by $2.4 \times 5 = \$12$.
  - The price formula moves by half of that: about **\$6 lower** (unless the rival-relative band stops it).
  - **Result:** Bargain shoppers see the price drop, bookings pick up, and within a few days we are back on schedule!

#### Story 2: Ahead of Pace (Selling Out Too Fast at Day 67)
- **Target Plan:** We should have 29 seats left (51 sold).
- **Reality:** We only have 22 seats left (58 sold!).
- **Inventory Gap:** $22 - 29 = \mathbf{-7 \text{ seats ahead of pace}}$.
- **What happens:** 
  - Our seats are running out too quickly! At this rate the plane would be full well before the busy last weeks, when most customers arrive.
  - The Economist raises the opportunity shadow price $\mu$ by $2.4 \times 7 = \$16.80$.
  - The price moves up by half of that: about **\$8.40 higher** (unless the band stops it).
  - **Result:** Sales slow down and the remaining seats sell at a higher price.

---

### The Secret Weapon: Capacity Opportunity Cost ($\mu$)

Here is the biggest insight in the entire algorithm:
> **The value of an empty seat changes every single day!**

We calculate **Opportunity Cost** ($\mu$):
- **When we have few seats and lots of days (Scarcity):**
  A seat in inventory is valuable. Its opportunity value can rise to **$\mu = \$56$** (the cap), which pushes our price up.
- **When we have 30 unsold seats and only 15 days left (Surplus):**
  If those seats don't sell, they become **\$0.00**. Their opportunity value drops to the floor, **\$10** (or **\$2** in a low-WTP market), and prices fall to reduce waste.

### The Math Magic (Closed-Form Solution):
Because Revenue is a parabola (a smooth upside-down U-curve), calculus allows us to calculate the **exact peak of the curve in one step**:

$$p^* = \frac{A - B \cdot \mu}{2(-B)}$$

The formula gives the price that maximizes expected (price − μ) × demand **under our fitted linear demand model**. The model is rough: per-day demand is mostly 0, 1 or 2 seats, and the fit explains only about 15% of the variation. So the code then keeps the price within a band around the rival's price from yesterday: from \$1.20 below it when we are on pace, up to \$7.50 below (\$8.50 with low willingness to pay) when far behind, and at most \$12 above. In simulation that band changes the price on about 60% of days.

---

## 6. The 3 Passenger Worlds (Demand Regimes, 377,570 duopoly periods)

The algorithm re-checks at the end of every season which world it is in:

| Passenger World | How it is detected | Price floor | Monopoly price band |
| :--- | :--- | :---: | :---: |
| **Low Demand** | Last season sold < 42 seats, or the in-season alarm fired | **\$18** | \$28 – \$44 |
| **Average Demand** | Default | **\$25** | \$48 – \$72 |
| **High Demand** | Last two seasons averaged ≥ 72 seats and nothing sold at ≥ \$55 | **\$25** | \$50 – \$66 |

The regression starting values (priors) for each world come from pooled tournament data (`mlr`/`v5.1`/`poisson_logit`, `v3`/`v4`, and `v1`). They are only used until about one season of our own data exists; after that the model is refit on the current competition.

---

## 7. Concrete Walkthrough of a 100-Day Season

The example below is a first season in a normal market, before anything has been learned (plan bend 0.15). Prices are illustrative.

#### Phase 1: Days 1 to 15 (Starting Out)
- **Day 1:** The plane is empty (80 seats). The rival posts \$55.00.
- **Action:** The regression still uses the average-market starting values (the "prior"), and the price stays within the band around the rival, e.g. \$53.50.
- **Meanwhile:** The low-demand alarm (Scenario B) checks every day on which we are cheaper than the rival.

#### Phase 2: Days 16 to 50 (Pacing Against the Plan)
- **Day 35:** We have sold 28 seats; the plan says 25. We are 3 seats ahead, so the seat value $\mu$ rises a little.
- **Action:** The formula gives e.g. \$51.20 against the rival's \$52.00.
- **Goal:** Sell steadily without panic discounts.

#### Phase 3: Days 51 to 75 (The Rival Sells Out)
- **Day 52:** The rival has no seats left (`competitor_has_capacity = False`).
- **Action:** Monopoly mode. With 40 seats and 49 days left (0.82 seats per day) the monopoly price is about \$58; with fewer seats per day it rises toward \$72.

#### Phase 4: Days 76 to 100 (Landing the Plane With No Empty Seats)
- **Day 92:** 4 seats left, 9 days to go (0.44 seats per day).
- **Action:** In monopoly that burn rate gives about \$68. If we were behind the plan instead, the late clearance discount would lower the price to avoid empty seats.

#### After the Season (Day 101)
- The server reports the season's sales. The regime is re-checked, the regression is refit, and if we sold 75+ seats, the season's booking curve updates the plan's bend for the next season.

---

## 8. Presentation Guide: How to Pitch This to a Non-Technical Audience

If you have to present this project to professors, executives, or classmates who know **nothing about math or code**, use this slide-by-slide playbook:

### Slide 1: The Challenge (The 80-Seat Problem)
- **Visual:** A graphic of an airplane with half of its seats empty in red.
- **Talking Point:** *"In airline pricing, seats are perishable. If a plane leaves with 10 empty seats, that revenue is gone. Our mission was to build an autonomous algorithm that maximizes revenue while keeping empty seats to a minimum."*

### Slide 2: The Two Brains (Street Smarts + Economic Science)
- **Visual:** A graphic showing **The Bouncer** on the left (shield icon) and **The Economist** on the right (brain/calculator icon).
- **Talking Point:** *"Pure math bots fail because opponents play dirty. Pure rule bots fail because they can't optimize. We built a Two-Brain system: Street-smart defense rules to handle crazy rivals, paired with a regression model that estimates the profit-maximizing price."*

### Slide 3: The Revenue Hill (Finding the Peak)
- **Visual:** Display **Panel 1 from `imgs/mlr_strategy_explained.png`** (The upside-down U-curve).
- **Talking Point:** *"If you charge \$30, your plane fills up, but you make no profit. If you charge \$85, you make high margins, but the plane flies empty. Our algorithm uses calculus to find the top of that hill under its demand estimate, then keeps the price close to the rival's."*

### Slide 4: Target Curve vs. Booking Curve (Pacing the Flight)
- **Visual:** Display **Panel 2 from `imgs/dpc_strategy_visualization.png`** (Target Curve vs Realized Booking Curve).
- **Talking Point:** *"We distinguish between the Target Curve (the planned convex benchmark that holds a few seats back for the late arrival surge) and the Booking Curve (actual sales). If actual sales fall behind target, the algorithm drops the opportunity cost of seats to accelerate bookings, so the flight lands with as few empty seats as possible."*

### Slide 5: The Scorecard & Live Day-by-Day Playback
- **Visual:** The multi-seed benchmark table from `README.md` §4.5 (v6.3 earns 1.5–2.8× a naive "undercut by \$1" strategy in every simulated market), the ablation chart `imgs/ablation_study.png` from §4.6, and a screenshot of the **Streamlit Day-by-Day Simulation Studio**.
- **Closing Punchline:** *"Every rule in our algorithm comes from a real tournament failure, and every number in our figures can be regenerated from the data. In the Streamlit app you can watch each pricing decision day by day."*

---

### Key Takeaway Summary Table (Cheat-Sheet for Questions)

| Question You Might Be Asked | Simple 1-Sentence Answer |
| :--- | :--- |
| **"Why not just undercut the rival by \$1 every day?"** | *"In our simulations that earns 35–65% less, because it gives away margin against expensive rivals and never sells at monopoly prices. (Against a cheap dumper it is competitive, which is why we follow cheap rivals too.)"* |
| **"What is the difference between the Target Curve and Booking Curve?"** | *"The Target Curve is the planned convex benchmark holding seats for late surges; the Booking Curve is actual cumulative sales over time."* |
| **"What happens if demand is unexpectedly low?"** | *"A statistical test runs during the season. Once low demand is 99× more likely than normal demand (about 4% false alarms on past data), we unlock an \$18 floor and lower monopoly prices."* |
| **"Why not use deep learning or a neural network?"** | *"Neural networks are black boxes that take thousands of rounds to learn; Multiple Linear Regression refits every season on the current competition and gives a price we can explain line by line."* |
| **"Why linear regression and not a Poisson model?"** | *"We tested it on each competition's own data: with ~25 seasons both are about equally accurate, but after one season (when the model is first refit) the Poisson model's predictions explode. The linear model stays stable and gives the price in one formula (ALGORITHM_CHANGES §17.5)."* |
| **"Do you need all those rules?"** | *"We switched each one off and measured: two are essential, most are each worth under 1% but protect each other, and the test found one bug, which v6.3 fixes (+2.1%)."* |
