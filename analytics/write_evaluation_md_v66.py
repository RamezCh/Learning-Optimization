"""
analytics/write_evaluation_md_v66.py
------------------------------------
ARCHIVE: writes EVALUATION_v6.6.md (the evaluation of v6.6 on data/v6.5) from analytics/v66_evaluation.json, so that every number in the text is the
number the figures were drawn from. Run analytics/evaluate_v66.py first.

EVALUATION.md always describes the CURRENT duopoly.py. For a new version: re-run the test bed,
evaluate_*.py and this script, and edit the wording below where the findings changed.
"""

import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E = json.load(open(os.path.join(ROOT, "analytics", "v66_evaluation.json")))
SIM = json.load(open(os.path.join(ROOT, "analytics", "v66_simulator_check.json")))
ON, DM, RF, CA, TB = E["online"], E["demand_models"], E["rival_forecast"], E["causation"], E["testbed"]
V, R, AB = TB["validity"], TB["result"], TB["ablation"]
RB = R["robust"]
NEW = "duopoly.py"

INTERACTION_NOTE = (
    "**A result that had to be questioned.** On set A, logging sold-out-rival days as \"no rival\" looked like the best "
    "single change (+1.7 points). Looking at the matches: almost all of it came from one match (`khLeM2`, +\\$34,501 of "
    "+\\$39,923). On the check matches repairing these days is worth only {skip_alone} alone. Is it noise? Taking it out of "
    "the full version lowers the check-match result by {skip_inside} points, so it is real but mostly works together with "
    "the other parts (why exactly has not been isolated). Two ways to repair the data were then compared on "
    "both sets: log the day as \"no rival\" ({norival}, worst match {norival_worst}) or leave the day out ({full}, worst "
    "match {full_worst}). Leaving it out is as good and loses less in the worst match, and the choice model is known to be "
    "poor without a rival (section 9, point 8), so v6.6 leaves these days out. This last choice between two near-equal "
    "variants used the check matches; the headline number is therefore very slightly optimistic."
)

# Development runs (set A: every 3rd match, seasons 1-40, 30 matches), copied from the test-bed
# output while v6.6 was built. Baseline v6.5 = $2,374,262. Columns: change %, better, worse, worst %.
DEV = [
    ("All new switches off (must equal v6.5)", "+0.00%", 0, 0, "0.0%", "check passed: identical revenue"),
    ("Rival path in the DP", "+1.45%", 12, 0, "−0.1%", "adopted"),
    ("Today's price by solver (yesterday's rival price)", "+0.73%", 11, 1, "−0.8%", "adopted"),
    ("Path + solver", "+1.86%", 14, 0, "−0.1%", "adopted"),
    ("Path + solver + choice model from day 50", "+2.26%", 24, 0, "−0.1%", "adopted"),
    ("... + reaction model for today's rival price (used only if error <= 70% of naive)", "+2.27%", 24, 0, "−0.1%",
     "adopted, worth ~0 here"),
    ("... + reaction model always used", "+2.27%", 20, 0, "−0.4%", "rejected: more matches unchanged or slightly down"),
    ("... + rival path also in market sizing", "+2.49%", 23, 0, "−0.2%", "adopted"),
    ("... + sold-out rival days logged as \"no rival\"", "+4.17%", 23, 0, "−0.2%",
     "replaced by the next row after the check described in 8.2"),
    ("... + sold-out rival days left out of the demand data (= v6.6)", "+4.15%", 22, 0, "−0.3%", "adopted"),
    ("Rival sell-out share inside the DP", "−0.49 points vs the same build without it", 10, 4, "−5.9%", "rejected"),
    ("Several starting points for the choice-model fit", "−0.33%", 0, 1, "−4.1%", "rejected"),
    ("DP table rebuilt every 5 days instead of 10", "+0.05 points vs the same build with 10 days", 12, 2, "−2.0%", "rejected: no gain"),
    ("Lecture's linear demand model by LP in season 1", "+0.11 points vs the same build without it", 16, 2, "−3.6%",
     "rejected: within noise, larger worst case"),
    ("Price floor \\$10 instead of \\$18", "+2.11 points vs the same build with \\$18", 19, 3, "−4.5%",
     "not adopted: extrapolation below any price we ever charged (section 9)"),
]


def pct(x, digits=0):
    return f"{100 * x:.{digits}f}%"


def money(x):
    return f"\\${x:,.0f}"


def signed(x, digits=2):
    return f"{x:+.{digits}f}%".replace("-", "−")


def main():
    o, t = ON["overall"], ON["by_type"]
    lines = []
    w = lines.append

    w("# Evaluation of the current algorithm (`duopoly.py` v6.6)")
    w("")
    w("> **Archived evaluation of v6.6 (written before the v6.6 tournaments). The current evaluation is "
      "[`EVALUATION.md`](EVALUATION.md); section 5 there checks these predictions against data/v6.6.**")
    w(">")
    w("> This file described the version of `duopoly.py` current at the time. It answers: what does the algorithm do, how do "
      "we measure it, how good are its models, what is correlation and what is cause, how does the offline simulation work "
      "and how far can it be trusted, where are the weak points and how were they found. "
      "History of all versions: [`ALGORITHM_CHANGES.md`](ALGORITHM_CHANGES.md).")
    w(">")
    w("> Every number and figure below is produced by `analytics/evaluate_v66.py` from `data/v6.5` and the test-bed result "
      "files, stored in `analytics/v66_evaluation.json`, and written into this file by `analytics/write_evaluation_md.py`. "
      "Text and plots therefore show the same numbers. To refresh: section 11.")
    w("")
    w("## 1. Summary")
    w("")
    w("| Question | Answer |")
    w("| :--- | :--- |")
    w(f"| What changed in v6.6? | The method of Bertsimas & Perakis (2001) / the lecture: forecast the rival's price (linear "
      f"programme + time series), put the forecast into the dynamic programme, solve today's price as a one-dimensional "
      f"nonlinear programme. Plus two data fixes found on the way. |")
    w(f"| One quality number | Revenue on the offline test bed, same rivals and same random buyers for both versions: "
      f"**{signed(R['change'])}** vs v6.5 over {R['matches']} matches x 100 seasons "
      f"(95% interval {signed(R['ci95'][0])} to {signed(R['ci95'][1])}). |")
    w(f"| How consistent? | {R['better']} matches better, {R['worse']} worse (beyond ±0.5%), worst match "
      f"{signed(R['worst_match'], 1)}, median match {signed(R['median_match'], 1)}. |")
    w(f"| On matches not used to choose the changes | **{signed(AB['files'][NEW]['change'])}** on the {AB['matches']} check "
      f"matches ({AB['files'][NEW]['better']} better / {AB['files'][NEW]['worse']} worse). |")
    w(f"| How robust is that number? | {signed(RB['trusted_change'])} on the {RB['trusted_matches']} matches where the test bed "
      f"reproduces v6.5's real revenue within ±10%; {signed(RB['change_without_top5'])} without the 5 matches that gain most "
      f"(they are {pct(RB['top5_share_of_gain'])} of the gain); median match {signed(R['median_match'], 1)}. |")
    w(f"| With a different demand model (old simulator) | **{signed(SIM['total_change'], 1)}**: no gain. Three of the four "
      f"changes are neutral there; fitting the choice model on day 50 costs revenue when the model family is wrong "
      f"(section 8.4). |")
    w(f"| Honest expectation for the tournament | Between the two offline worlds: somewhere from about 0% to +3%. |")
    w(f"| Is the offline number a forecast of the tournament? | No. It is an upper estimate: the test bed draws buyers from "
      f"the same model family the algorithm fits, and its rivals replay logged prices and do not react to us (section 7). |")
    w(f"| Biggest open weak point | {t['dumper']['matches']} of {o['matches']} matches ({pct(t['dumper']['share_matches'])}) "
      f"are against rivals below our \\$18 floor; they bring {pct(t['dumper']['share_revenue'])} of our revenue (section 9). |")
    w("")
    w("## 2. The algorithm in one table")
    w("")
    w("The four steps of the lecture *A learning approach to dynamic pricing using nonlinear and linear optimization* "
      "and where each one lives in `duopoly.py`:")
    w("")
    w("| Lecture step | In the lecture | In `duopoly.py` v6.6 | Optimisation type | Since |")
    w("| :--- | :--- | :--- | :--- | :---: |")
    w("| 1. Estimate our demand model | Linear $d = \\beta_0 + \\beta_1 p_1 + \\beta_2 p_2$, least absolute deviations by LP, "
      "coefficients vary slowly | Choice model: buyers/day $\\sim$ Poisson$(n e^{k\\tau})$, each buys from us with probability "
      "$\\frac{e^{b(m-p)}}{1+e^{b(m-p)}+e^{b(m-r)}}$. Fitted by maximum likelihood on all our days. **The lecture's linear LP "
      "model was tested and not adopted (section 5.1).** | Nonlinear (Nelder–Mead on the likelihood) | v6.4, first fit now "
      "day 50 of season 1 |")
    w("| 2. Coefficients for the next period | Average of the last $N$ periods | Refit every 25 days and every season end on the "
      "last 3,000 days | – | v6.4 |")
    w("| 3. Estimate the rival | Assume the rival maximises revenue, fit its model by LP | **Rival price forecast.** Tomorrow: "
      "$r_t = a + c\\,r_{t-1} + b\\,p_{t-1}$ fitted by **least absolute deviations as a linear programme** (`scipy.linprog`, "
      "last 3 seasons, $b, c \\ge 0$). Rest of the season: $r_d = \\text{profile}_d + \\varphi^{\\,d-t}(r_t - \\text{profile}_t)$, "
      "profile = median price per day over the last 10 seasons, $\\varphi$ = AR(1) | **Linear programme** + time series | "
      "**v6.6** |")
    w("| 4. Our optimal price | $\\max_p\\ p\\,(\\beta_0+\\beta_1 p+\\beta_2 \\hat p_2)$, one-dimensional, standard solver | "
      "$\\max_p\\ E[\\,p\\min(D,s) + V(t+1, s-\\min(D,s))\\,]$ with $D \\sim$ Poisson(rate$(p, \\hat r_t)$): the same "
      "one-dimensional problem plus the value of the seats kept. Solved every day with a bounded scalar solver "
      "(`scipy.minimize_scalar`) | **Nonlinear programme** | **v6.6** |")
    w("| (paper, section 2) Dynamic programming | $J_s(c_s) = \\max_p E[p\\min(d,c_s) + J_{s+1}(c_s - \\min(d,c_s))]$ | "
      "$V(t,s)$ by backward induction over 100 days x 80 seats x prices \\$18–\\$100. v6.5 held the rival at its 5-day "
      "average until day 100; **v6.6 uses the forecast path** | Dynamic programme | v6.5, path in v6.6 |")
    w("")
    w("Unchanged from v6.5: market sizing (seat value 0 when the market cannot fill the plane), the copier rule, the "
      "monopoly rule when the rival is sold out, the first-season regression with regime priors (now only until day 50).")
    w("")
    w("## 3. How we measure")
    w("")
    w("| Item | Definition |")
    w("| :--- | :--- |")
    w("| Quality measure | **Revenue per season** (the competition ranks by total revenue). Selling out is not a goal: "
      "selling out early means the price was too low, unsold seats mean it was too high. |")
    w("| Baseline | v6.5, the version that played the `data/v6.5` tournaments (`analytics/experiments/duopoly_v65.py`). "
      "Every change is measured **with and without** it, everything else equal. |")
    w("| Pairing | Both versions play the same match, same rival prices, same random number per (match, season, day). "
      "A difference is therefore caused by the code, not by luck in the draw. |")
    w("| Reported per comparison | Total revenue change in %, matches better / worse (beyond ±0.5%), worst match, and a 95% "
      "interval from resampling matches 4,000 times (bootstrap). |")
    w("| Choosing vs checking | Changes were chosen on set A (every 3rd match, seasons 1–40, 30 matches). They were then "
      "checked on the other 60 matches, which had not been looked at until then (section 8.2; one late choice between "
      "two near-equal variants did use them and is marked there). |")
    w("| Model quality | Out of sample only: fit on seasons 1–50, score on seasons 51–100. $R^2$, adjusted $R^2$, and "
      "$R^2$ as a share of the best value Poisson noise allows (section 5). |")
    w("| Sanity check of the switches | With all new switches off the file must reproduce v6.5 to the cent. It does. |")
    w("")
    w("## 4. Online: what happened with v6.5")
    w("")
    w(f"`data/v6.5`: 3 tournaments, {o['matches']} matches, 100 seasons x 100 days each. Our ranks: "
      + ", ".join(str(v["rank"]) for v in ON["by_tournament"].values()) + ".")
    w("")
    w("![v6.5 online scoreboard](imgs/v66_online_scoreboard.png)")
    w("")
    w("| Tournament | Rank | Matches | Revenue / season (mean) | (median) | Seats sold of 80 | We sold out | Rival sold out | "
      "Our / rival median price |")
    w("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for name, v in ON["by_tournament"].items():
        w(f"| `{name}` | {v['rank']} | {v['matches']} | {money(v['rev_mean'])} | {money(v['rev_median'])} | {v['seats']:.1f} | "
          f"{pct(v['sellout'])} of seasons | {pct(v['rival_sellout'])} | {money(v['our_median'])} / {money(v['rival_median'])} |")
    w(f"| **All** | – | {o['matches']} | {money(o['rev_mean'])} | {money(o['rev_median'])} | {o['seats']:.1f} | "
      f"{pct(o['sellout'])} | {pct(o['rival_sellout'])} | – |")
    w("")
    w("**Comparison with the market.** The logs contain our sales only, never the rival's sales or revenue, so "
      "\"our revenue vs the rival's revenue\" cannot be computed from them. What can be compared is the price position "
      "(our median price is within about \\$1.50 of the rival's in all three tournaments) and the sell-out rate "
      f"(ours {pct(o['sellout'])} of seasons, rivals {pct(o['rival_sellout'])}). The only direct market comparison is the "
      "tournament rank itself. Ranks of different days are not comparable either: the rivals and the random markets change "
      "every day (v6.4 ranked 1 to 33 with the same code).")
    w("")
    w("### 4.1 Rival behaviour and what it did to our revenue")
    w("")
    w("Each rival is labelled from its logged prices only (rules in `classify_rival`, checked in this order): **copier** = its "
      "price equals our price of the day before on ≥ 80% of days; **dumper** = median price below our \\$18 floor; "
      "**seasonal** = the median price per day of seasons 1–50 explains ≥ 50% of its price variance in seasons 51–100; "
      "**steady** = moves less than \\$1 per day on average; **erratic** = the rest.")
    w("")
    w("| Rival behaviour | Matches | Our revenue / season | Share of our total revenue | Seats sold | We sold out | "
      "Rival sold out | Our / rival median price | Days at our \\$18 floor |")
    w("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for name in ("dumper", "copier", "seasonal", "steady", "erratic"):
        v = t[name]
        w(f"| {name} | {v['matches']} ({pct(v['share_matches'])}) | {money(v['rev_mean'])} | {pct(v['share_revenue'])} | "
          f"{v['seats']:.1f} | {pct(v['sellout'])} | {pct(v['rival_sellout'])} | {money(v['our_median'])} / "
          f"{money(v['rival_median'])} | {pct(v['at_floor'])} |")
    w("")
    w("**What caused what?**")
    w("")
    w(f"- Our revenue per season rises with the rival's median price (r = {o['corr_rev_rival_price']:.2f}, panel A). This is "
      f"not \"an expensive rival makes us rich\". Both follow the market: each match has its own random demand level and "
      f"willingness to pay, and both players adapt to it (our median price and the rival's: r = {o['corr_our_rival_price']:.2f}).")
    w(f"- Dumpers: the rival's low price is not what takes our revenue away. A rival at a median of "
      f"{money(t['dumper']['rival_median'])} sold out in only {pct(t['dumper']['rival_sellout'])} of seasons. If many buyers "
      f"existed, 80 seats at that price would be gone every season. So these are small markets with low willingness to pay; "
      f"the rival's price and our low revenue are both effects of that. Lowering our floor could win a part of these few "
      f"buyers, not create a normal season (section 9).")
    w(f"- {o['matches_below_500']} matches earn less than \\$500 per season and together bring "
      f"{pct(o['revenue_share_below_500'], 1)} of our revenue. The ranking is decided in the other matches: steady and erratic "
      f"rivals are {t['steady']['matches'] + t['erratic']['matches']} matches and "
      f"{pct(t['steady']['share_revenue'] + t['erratic']['share_revenue'])} of our revenue. v6.6 changes mostly those.")
    w("")
    w("## 5. Are the models good?")
    w("")
    w("### 5.1 Demand model")
    w("")
    w("![Demand model quality](imgs/v66_demand_model_quality.png)")
    w("")
    w(f"Four models, each fitted per match on seasons 1–50 and scored on seasons 51–100 ({DM['matches']} matches, "
      f"{DM['test_days']:,} days). One day sells {DM['mean_sales_per_day']:.2f} seats on average and "
      f"{pct(DM['zero_days'])} of days sell none.")
    w("")
    w("| Model | $R^2$ daily (median match) | Adjusted $R^2$ | Middle half of matches | Share of the best possible $R^2$ | "
      "$R^2$ of 10-day totals | Level error (median) | Best model in (matches) |")
    w("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for key, label in (("ols", "Linear, least squares (the v6.0 regression)"),
                       ("lad_day", "Linear, LP, one row per day (lecture step 1 as written)"),
                       ("lad_block", "Linear, LP, 5-day blocks (lecture, adapted)"),
                       ("choice", "**Choice model, Poisson (used by the DP)**")):
        v = DM[key]
        w(f"| {label} | {v['r2_median']:.3f} | {v['adj_r2_median']:.3f} | {v['r2_q25']:.3f} to {v['r2_q75']:.3f} | "
          f"{pct(v['share_of_ceiling'])} | {v['block_r2_median']:.2f} | {pct(v['abs_bias_median'], 1)} | "
          f"{v['best_in_matches']} |")
    w("")
    w("How to read this:")
    w("")
    w(f"- **A daily $R^2$ of {DM['choice']['r2_median']:.2f} looks terrible and is close to the maximum.** Buyers arrive at "
      f"random (Poisson). Even a model that knew the true expected sales of every day could not explain more than "
      f"$1 - \\text{{mean}}/\\text{{variance}}$ of daily sales, which is {DM['ceiling_median']:.3f} in the median match. The "
      f"choice model reaches {pct(DM['choice']['share_of_ceiling'])} of that limit, the least-squares line "
      f"{pct(DM['ols']['share_of_ceiling'])}. Adjusted $R^2$ equals $R^2$ to three digits because each test set has "
      f"thousands of days and 3–4 parameters.")
    w(f"- Added up over 10 days the noise shrinks and $R^2$ rises to {DM['choice']['block_r2_median']:.2f} (panel B). "
      f"Panel C: when the choice model forecasts x seats per day, x seats are sold (points on the diagonal).")
    w(f"- **The lecture's LP model as written fails on this data.** Least absolute deviations fits the median. The median "
      f"of daily sales is 0, so in {DM['lad_day_all_zero_matches']} of {DM['matches']} matches the LP returns all "
      f"coefficients = 0 and forecasts no sales at all ($R^2$ {DM['lad_day']['r2_median']:.3f}). The paper's example sells "
      f"about 50 units per period; we sell 0.5. Summing 5 days per row repairs it ($R^2$ {DM['lad_block']['r2_median']:.3f}) "
      f"but it stays behind the choice model, and a line cannot learn from days without sales where the price was too high.")
    w("- Consequence: the LP is used where its strength counts, on the rival's prices (step 3: prices are never mostly zero, "
      "and they have spikes that least squares would chase). The demand model stays the choice model.")
    w("")
    w("### 5.2 Rival price forecast")
    w("")
    w("![Rival forecast quality](imgs/v66_rival_forecast.png)")
    w("")
    w("Every forecast uses only earlier days and earlier seasons of the same match.")
    w("")
    w("| Forecast | Mean absolute error | Median match |")
    w("| :--- | :---: | :---: |")
    w(f"| **Rest of the season**, flat 5-day average (v6.5's DP assumption) | \\${RF['many_flat']['mean']:.2f} | "
      f"\\${RF['many_flat']['median']:.2f} |")
    w(f"| **Rest of the season**, profile + AR(1) path (v6.6) | **\\${RF['many_path']['mean']:.2f}** | "
      f"\\${RF['many_path']['median']:.2f} |")
    w(f"| Tomorrow, 5-day average | \\${RF['one_avg5']['mean']:.2f} | \\${RF['one_avg5']['median']:.2f} |")
    w(f"| Tomorrow, yesterday's price (v6.1–v6.5) | \\${RF['one_yesterday']['mean']:.2f} | \\${RF['one_yesterday']['median']:.2f} |")
    w(f"| Tomorrow, reaction model by LP (v6.6) | **\\${RF['one_reaction']['mean']:.2f}** | \\${RF['one_reaction']['median']:.2f} |")
    w("")
    w(f"- The path forecast cuts the error for the rest of the season by {pct(RF['path_reduction'])}: better in "
      f"{RF['path_better']} matches, worse in {RF['path_worse']} (by more than \\$0.25). For seasonal rivals the error falls "
      f"from \\${RF['by_type']['seasonal']['flat']:.2f} to \\${RF['by_type']['seasonal']['path']:.2f}, for erratic rivals from "
      f"\\${RF['by_type']['erratic']['flat']:.2f} to \\${RF['by_type']['erratic']['path']:.2f}.")
    w(f"- The reaction model cuts the next-day error by {pct(RF['reaction_reduction'])} on average, but the median match "
      f"does not improve: the gain comes from the rivals that follow our price. `duopoly.py` therefore uses it only when its "
      f"error is at most 70% of the error of \"same price as yesterday\"; that is the case in about "
      f"{RF['reaction_gate_matches']} of {RF['matches']} matches.")
    w("")
    w("## 6. Correlation or causation?")
    w("")
    w("![Correlation vs causation](imgs/v66_correlation_vs_causation.png)")
    w("")
    w("Three places where a correlation in the logs must not be read as a cause:")
    w("")
    w(f"**1. Our price and our sales (panels A, B).** In {CA['raw_positive']} of {CA['matches']} matches a higher price goes "
      f"with *more* seats sold. Nobody buys more because it is dearer. First guess: the day of the season (we charge more "
      f"late, when more buyers come). **That guess was wrong**: holding the day fixed, {CA['positive_given_day_only']} matches "
      f"still show a positive correlation, and across matches the day explains nothing (r = {CA['day_hypothesis_r']:.2f}). "
      f"Holding the *rival's price* fixed leaves {CA['positive_given_rival_only']}. The third variable is the rival's price: "
      f"when the rival is expensive we charge more (median correlation {CA['price_rival_median']:.2f}) *and* we sell more "
      f"(positive in {CA['sales_rival_positive']} matches). The product of these two links predicts the raw correlation with "
      f"r = {CA['rival_hypothesis_r']:.2f} (panel B). With rival price and day held fixed the price effect is negative in "
      f"{CA['partial_negative']} of {CA['matches']} matches. This is why every demand model here contains the rival's price, "
      f"and why a model with our price alone would recommend raising prices for the wrong reason.")
    w("")
    w(f"**2. Their price and ours (panel C).** The rival's price today correlates with our price yesterday (median "
      f"{CA['they_follow_median']:.2f}). Our price today correlates just as much with their price yesterday (median "
      f"{CA['we_follow_median']:.2f}); both are above 0.5 in {CA['both_follow_above_half']} matches. Our algorithm is built to "
      f"respond to the rival, and both respond to the same market, so the logs cannot tell who follows whom. Only the "
      f"{CA['copiers']} copiers are certain: their price *equals* ours of the day before (top of panel C), which no common "
      f"cause produces. The reaction coefficient $b$ of step 3 is therefore used as a **forecast** of tomorrow's price, not "
      f"as proof that we can steer the rival. Proof would need an experiment: change our price on random days and watch.")
    w("")
    w(f"**3. Rival price level and our revenue (section 4.1).** r = {o['corr_rev_rival_price']:.2f} across matches, caused by "
      f"the market behind both.")
    w("")
    w("What makes the test-bed result causal *inside the test bed*: both versions face identical rivals and identical random "
      "buyers, so the code is the only difference. Whether the test bed resembles the tournament is a separate question "
      "(next section).")
    w("")
    w("## 7. Offline simulation: how it works and how far to trust it")
    w("")
    w("`analytics/v65_testbed.py`, same method as the v6 and v6.4 test beds:")
    w("")
    w("1. **Demand** per match: the choice model fitted to all 10,000 logged days of that match (8 starting points, best "
      "likelihood kept).")
    w("2. **Rival**: replays its logged price and its logged sold-out days, day by day. Copiers are simulated as copiers "
      "(their price = our price the day before).")
    w("3. **Play**: a pricing file plays every match for 100 seasons. Sales per day are drawn from the fitted model with a "
      "fixed random number per (match, season, day), identical for every file.")
    w("")
    w("![Test bed vs reality](imgs/v66_testbed_validity.png)")
    w("")
    w("| Check | Result |")
    w("| :--- | :--- |")
    w(f"| v6.5 replayed on the test bed vs v6.5 in the tournament, total revenue | {money(V['replay_total'])} vs "
      f"{money(V['actual_total'])} ({signed(100 * V['total_error'], 1)}) |")
    w(f"| Same, per match | $R^2$ = {V['r2_levels']:.3f}; median error {pct(V['median_abs_error_big'], 1)} in the "
      f"{V['big_matches']} matches above \\$500 per season; {V['within_10pct']} of them within ±10%; worst {pct(V['worst'])} |")
    w(f"| Demand model out of sample (section 5.1) | {pct(DM['choice']['share_of_ceiling'])} of the achievable daily $R^2$, "
      f"level error {pct(DM['choice']['abs_bias_median'], 1)} |")
    w("")
    w("What the test bed can and cannot show:")
    w("")
    w("| | |")
    w("| :--- | :--- |")
    w("| Can | Rank two versions against the real rival price paths and sold-out days of 90 matches, without luck. Find "
      "matches where a change hurts. |")
    w("| Cannot: rival reactions | The rival's prices are a recording. If our new prices would make a real rival answer "
      "differently, the test bed does not see it (only copiers react). |")
    w("| Cannot: wrong model family | Buyers are drawn from the model family the algorithm itself fits, so the algorithm's "
      "model is \"right\" by construction. Real gains are smaller. The simulator check in section 8.4 uses a different family. |")
    w("| Cannot: prices never tried | Below \\$18 and far above the logged prices the demand curve is the model's "
      "extrapolation, not data. |")
    w("| Fit is in-sample here | The validity row above compares with the same logs the demand model was fitted on. The "
      "out-of-sample evidence is section 5.1. |")
    w("")
    w("One error was found in the test bed itself while questioning a result: with a single starting point the fit of 7 of "
      "90 matches ended in a flat region (\"price does not matter below \\$1,000\") with a much worse likelihood; in such a "
      "match any version that prices higher wins. The fit now uses 8 starting points; no match is degenerate "
      "(largest willingness-to-pay midpoint \\$83).")
    w("")
    w("## 8. Results: v6.6 against v6.5")
    w("")
    w("![Test bed results](imgs/v66_testbed_results.png)")
    w("")
    w("### 8.1 All 90 matches x 100 seasons")
    w("")
    w("| | v6.5 | v6.6 | Change |")
    w("| :--- | :---: | :---: | :---: |")
    w(f"| Total revenue | {money(R['old_total'])} | {money(R['new_total'])} | **{signed(R['change'])}** "
      f"(95%: {signed(R['ci95'][0])} to {signed(R['ci95'][1])}) |")
    w(f"| Revenue per season | {money(R['old_per_season'])} | {money(R['new_per_season'])} | "
      f"+{money(R['new_per_season'] - R['old_per_season'])} |")
    w(f"| Seasons sold out | {pct(R['sellout_old'], 1)} | {pct(R['sellout_new'], 1)} | |")
    w(f"| Matches better / worse (beyond ±0.5%) | | | {R['better']} / {R['worse']} |")
    w(f"| Median / worst / best match | | | {signed(R['median_match'], 1)} / {signed(R['worst_match'], 1)} / "
      f"{signed(R['best_match'], 1)} |")
    w("")
    w("**Questioning the headline.** The gain is not spread evenly:")
    w("")
    w("| Cut | Matches | Change vs v6.5 |")
    w("| :--- | :---: | :---: |")
    w(f"| All matches | {R['matches']} | {signed(R['change'])} |")
    w(f"| Matches where the test bed reproduces v6.5's real revenue within ±10% (and ≥ \\$500 per season) | "
      f"{RB['trusted_matches']} | **{signed(RB['trusted_change'])}** (95%: {signed(RB['trusted_ci95'][0])} to "
      f"{signed(RB['trusted_ci95'][1])}; {RB['trusted_better']} better / {RB['trusted_worse']} worse; median "
      f"{signed(RB['trusted_median'], 1)}) |")
    w(f"| Without the 5 matches that gain most | {R['matches'] - 5} | {signed(RB['change_without_top5'])} |")
    w("")
    w(f"- The 5 biggest winners are {pct(RB['top5_share_of_gain'])} of the total gain. The biggest one (`{RB['top1_match']}`, "
      f"{pct(RB['top1_share_of_gain'])} of the gain) is also the match the test bed reproduces worst "
      f"({signed(100 * RB['top1_testbed_error'], 0)} against reality), so its gain should not be believed.")
    w(f"- The +{R['best_match']:.0f}% match is a copier match in which v6.5 earned \\$40 per season on the test bed. The one "
      f"match that gets worse ({signed(R['worst_match'], 1)}) is against a steady rival; the cause has not been traced.")
    w("- Read the result as: **about +3% where the test bed can be trusted, +1% in a typical match**, driven by a "
      "minority of matches with large gains.")
    w("")
    w("By rival behaviour:")
    w("")
    w("| Rival behaviour | Matches | Change | Per season |")
    w("| :--- | :---: | :---: | :---: |")
    for name in ("dumper", "copier", "seasonal", "steady", "erratic"):
        if name in R["by_type"]:
            v = R["by_type"][name]
            w(f"| {name} | {v['matches']} | {signed(v['change'])} | {'+' if v['dollars_per_season'] >= 0 else '−'}"
              f"\\${abs(v['dollars_per_season']):,.0f} |")
    w("")
    w("The copier gain comes entirely from the earlier fit of the choice model (8.2) and depends on the test bed's "
      "copier (an exact copy of our price of the day before, every day). Dumper matches hardly change: the \\$18 floor "
      "still binds.")
    w("")
    w("### 8.2 With / without each component, on the 60 check matches")
    w("")
    w(f"These {AB['matches']} matches were not used to build the version (60 seasons each). \"Alone\" = v6.5 plus this one "
      f"component. \"Removed\" = v6.6 minus this one component; the last column is what v6.6 loses without it.")
    w("")
    w("| Component | Alone: revenue vs v6.5 | Better / worse | Worst match | v6.6 without it: revenue vs v6.5 | "
      "Contribution inside v6.6 |")
    w("| :--- | :---: | :---: | :---: | :---: | :---: |")
    full = AB["files"][NEW]
    parts = (("path", "Rival path forecast in the DP and in market sizing"),
             ("nlp", "Today's price by scalar solver, rival from the reaction model"),
             ("early", "Choice model fitted from day 50 of season 1"),
             ("skip", "Sold-out rival days left out of the demand data"))
    for key, label in parts:
        a, b = AB["files"][f"only_{key}.py"], AB["files"][f"without_{key}.py"]
        w(f"| {label} | {signed(a['change'])} | {a['better']} / {a['worse']} | {signed(a['worst_match'], 1)} | "
          f"{signed(b['change'])} | {full['change'] - b['change']:+.2f} points |")
    w(f"| **All four = v6.6** | **{signed(full['change'])}** (95%: {signed(full['ci95'][0])} to {signed(full['ci95'][1])}) | "
      f"{full['better']} / {full['worse']} | {signed(full['worst_match'], 1)} | – | – |")
    w("")
    w(INTERACTION_NOTE.format(skip_alone=signed(AB["files"]["only_skip.py"]["change"]),
                              skip_inside=f"{full['change'] - AB['files']['without_skip.py']['change']:.2f}",
                              norival=signed(AB["files"]["logged_as_no_rival.py"]["change"]),
                              norival_worst=signed(AB["files"]["logged_as_no_rival.py"]["worst_match"], 1),
                              full=signed(full["change"]), full_worst=signed(full["worst_match"], 1)))
    w("")
    w("### 8.3 How the version was built (set A, 30 matches, seasons 1–40)")
    w("")
    w("| Variant | Revenue vs v6.5 | Better / worse (vs v6.5) | Worst match | Decision |")
    w("| :--- | :---: | :---: | :---: | :--- |")
    for label, change, better, worse, worst, decision in DEV:
        w(f"| {label} | {change} | {better} / {worse} | {worst} | {decision} |")
    w("")
    w("### 8.4 Check with a different demand model (old simulator)")
    w("")
    w("`analytics/benchmark_3regime_models.py` draws buyers with uniform willingness to pay (not the choice model) against "
      "four scripted rivals; 5 seeds x 30 seasons; the number is the sum over the four rivals of mean revenue per season. "
      "Here the algorithm's demand model is wrong on purpose.")
    w("")
    w("| Market (simulated) | v6.5 | v6.6 | Change | v6.6 without the day-50 fit | without path | without solver | "
      "without the data fix |")
    w("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for row in SIM["rows"]:
        w(f"| {row['market']} | {money(row['old'])} | {money(row['new'])} | {signed(row['change'], 1)} | "
          f"{money(row['without_early'])} | {money(row['without_path'])} | {money(row['without_nlp'])} | "
          f"{money(row['without_skip'])} |")
    w("")
    w(f"- **v6.6 does not gain here ({signed(SIM['total_change'], 1)} in total).** The differences per rival are inside the "
      f"95% intervals of the seeds except one cell: {SIM['worst_cell'].replace('$', chr(92) + '$')}.")
    w("- Which change is responsible? Removing the day-50 fit returns v6.5's numbers almost exactly; removing any of the "
      "other three changes nothing. So the path forecast, the solver and the data fix are neutral in this world (its "
      "scripted rivals have no season path worth learning), and **the early fit is the one change with evidence in both "
      "directions**: +1.4% alone on the test bed, of which the 10 copier matches carry +11.4% and the other 50 matches "
      "+0.4%; "
      f"{signed(SIM['rows'][1]['change'], 1)} and {signed(SIM['rows'][2]['change'], 1)} in the average and high simulated "
      "markets, where a choice model fitted on 48 days of data from "
      "a different kind of buyer prices the rest of season 1 worse than the priors did.")
    w("- It stays on: real sales fit the choice model well (section 5.1), unlike these simulated buyers, and the possible "
      "loss is limited to season 1 of 100 (the simulator has 30 seasons, so one bad season weighs three times more "
      "there). It is the first thing to switch off if a tournament disappoints: `LOGIT_MIN_OBS = 80` in `duopoly.py`.")
    w("")
    w("## 9. Weak points: how they were found, what was done")
    w("")
    w("| # | Weak point | How it was found | Evidence | Status |")
    w("| :---: | :--- | :--- | :--- | :--- |")
    w(f"| 1 | The DP assumed the rival keeps one price until day 100 | Scored that assumption against the logs | Error "
      f"\\${RF['many_flat']['mean']:.2f} per day; a path from earlier seasons: \\${RF['many_path']['mean']:.2f} | "
      f"**Fixed in v6.6** |")
    w("| 2 | The DP price was up to 10 days old | Read the code: the price table is rebuilt every 10 days from the 5-day "
      "average at that moment | Next-day error of the 5-day average is "
      f"\\${RF['one_avg5']['mean']:.2f}, of yesterday's price \\${RF['one_yesterday']['mean']:.2f} | **Fixed in v6.6** "
      "(price solved every day) |")
    w("| 3 | A sold-out rival's posted price was treated as competition in the demand data | Checked what the server logs "
      "on days with `competitor_has_capacity = False`: the rival still posts prices from \\$0.10 to \\$999 | "
      f"Rivals sold out in {pct(o['rival_sellout'])} of seasons. Worth "
      f"{AB['files'][NEW]['change'] - AB['files']['without_skip.py']['change']:.2f} points inside v6.6 (8.2) | **Fixed in v6.6** "
      "(days left out) |")
    w("| 4 | The whole first season ran on priors | Counted: the fit needed 80 days and was tried on days 25/50/75/100, so "
      "it first succeeded on day 100 | Fit on day 50: "
      f"{signed(AB['files']['only_early.py']['change'])} alone on the check matches (8.2), but a loss on the old "
      "simulator (8.4) | **Changed in v6.6, evidence mixed** |")
    w("| 5 | Test-bed fits degenerate in 7 matches | A match lost 10% for no visible reason; traced it day by day: sales "
      "did not depend on price at all | Section 7 | **Fixed** (test bed) |")
    w(f"| 6 | \\$18 price floor against dumpers | Online scoreboard by rival behaviour | {t['dumper']['matches']} matches, "
      f"{pct(t['dumper']['at_floor'])} of days at the floor, {money(t['dumper']['rev_mean'])} per season. Test bed with a "
      f"\\$10 floor: +2.1% total revenue | **Open.** We have never charged less than \\$18, so the gain is the model's guess. "
      f"Needs one tournament with a lower floor to know. The loss is bounded: these matches are "
      f"{pct(t['dumper']['share_revenue'])} of revenue |")
    w("| 7 | Rival reactions are not in the DP | Section 6, point 2 | Cannot be measured offline | **Open.** Copiers are "
      "handled by their own rule |")
    w("| 8 | The choice model without a rival (monopoly days) prices too high | v6.5 tests (ALGORITHM_CHANGES §22) and "
      "again here: rival sell-out share inside the DP loses | 8.3 | **Kept off**; the hand-made monopoly rule stays |")
    w("")
    w("## 10. What we do not know")
    w("")
    w(f"- **The tournament effect.** {signed(R['change'])} is an offline number with the limits of section 7. v6.5 was "
      f"+10.9% on its test bed; its tournaments cannot be compared with v6.4's because rivals and markets differ by day. "
      f"A clean online comparison would need both versions in the same tournament.")
    w("- **Whether rivals will repeat their season paths against different prices of ours.** The path forecast assumes "
      "the rival's curve mostly depends on the day; for copiers and reactive rivals it does not, and the forecast then "
      "falls back on the AR(1) part and the reaction model.")
    w("- **Demand below \\$18 and above the prices we tried.**")
    w("- **Speed on the server.** Locally v6.6 needs "
      f"{SIM['latency_new_ms']:.1f} ms per call on average (v6.5: {SIM['latency_old_ms']:.1f} ms) and "
      f"{SIM['latency_new_max_ms']:.0f} ms for the slowest call (v6.5: {SIM['latency_old_max_ms']:.0f} ms), measured on "
      f"{SIM['latency_note']}. Limits: 200 ms on average, 5 s per call. v6.5 online: 18 ms on average, 0.61 s slowest call, "
      "so about 1.6 times that is expected.")
    w("")
    w("## 11. Reproduce")
    w("")
    w("```bash")
    w("python -m pytest tests -q                                   # 51 tests")
    w("# test bed: all 90 matches, then one component at a time on the 60 check matches")
    w("python analytics/v65_testbed.py --out analytics/v66_testbed_all.csv --compare analytics/experiments/duopoly_v65.py")
    w("python analytics/v65_testbed.py --sample 3 --holdout --seasons 60 --out analytics/v66_testbed_ablation.csv \\")
    w("    --compare analytics/experiments/duopoly_v65.py $(for f in only without; do for c in path nlp early skip; do \\")
    w("      echo --compare analytics/experiments/${f}_$c.py; done; done) --compare analytics/experiments/logged_as_no_rival.py")
    w("python analytics/benchmark_3regime_models.py --compare analytics/experiments/duopoly_v65.py \\")
    w("    $(for c in path nlp early skip; do echo --compare analytics/experiments/without_$c.py; done)")
    w("# then copy the simulator totals into analytics/v66_simulator_check.json")
    w("python analytics/evaluate_v66.py          # numbers -> analytics/v66_evaluation.json, figures -> imgs/v66_*.png")
    w("python analytics/write_evaluation_md.py   # this file")
    w("```")
    w("")
    w("Do not edit `duopoly.py` while the test bed runs (it reloads the file per match).")
    w("")
    w("| File | Content |")
    w("| :--- | :--- |")
    w("| `duopoly.py` | v6.6. New code: section \"v6.6: rival price forecast\" at the end, `_solve_price_today`, `_dp_values` |")
    w("| `tests/test_v66_learning_approach.py` | 11 tests for the new parts |")
    w("| `analytics/v65_testbed.py`, `v65_demand_fits.json` | Test bed built from `data/v6.5` |")
    w("| `analytics/evaluate_v66.py`, `v66_evaluation.json`, `v66_match_table.csv` | All numbers and figures of this file |")
    w("| `analytics/experiments/` | `duopoly_v65.py` (baseline), `only_*.py` (v6.5 + one component), `without_*.py` "
      "(v6.6 minus one component) |")
    w("")
    w("Sources: D. Bertsimas, G. Perakis, *Dynamic Pricing: A Learning Approach* (2001); T. Winter, *A learning approach to "
      "dynamic pricing using nonlinear and linear optimization* (Learning Optimization, BHT, lecture slides).")
    w("")
    open(os.path.join(ROOT, "EVALUATION_v6.6.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("EVALUATION_v6.6.md written,", len(lines), "lines")


if __name__ == "__main__":
    main()
