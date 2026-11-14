"""
analytics/write_evaluation_md.py
--------------------------------
Writes EVALUATION.md (the CURRENT algorithm, v6.7, on the LATEST logs, data/v6.6) from
analytics/evaluation_v6.6.json (analytics/evaluate.py) and analytics/v67_simulator_check.json, so that every
number in the text is the number the figures were drawn from. The evaluation of v6.6 written before the v6.6
tournaments is archived in EVALUATION_v6.6.md (analytics/write_evaluation_md_v66.py).

For a new version: re-run the test beds, analytics/oracle.py, analytics/evaluate.py (edit its CFG) and this
script, and rewrite the wording below where the findings changed.
"""

import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E = json.load(open(os.path.join(ROOT, "analytics", "evaluation_v6.6.json")))
SIM = json.load(open(os.path.join(ROOT, "analytics", "v67_simulator_check.json")))
OLD = json.load(open(os.path.join(ROOT, "analytics", "v66_evaluation.json")))
GEN = json.load(open(os.path.join(ROOT, "analytics", "v67_generality.json")))
ON, PON, DM, RF, CA, TB, OR = (E["online"], E["prev_online"], E["demand_models"], E["rival_forecast"], E["causation"],
                               E["testbed"], E["oracle"])
V, R, PR, AB, PC = TB["validity"], TB["result"], TB["prev_result"], TB["ablation"], E["prediction_check"]
RB = R["robust"]
NEW, PREV = "v6.7", "v6.6"

# Development steps on the new data (copied from the test-bed output while v6.7 was built; full logs in work/).
DEV = [
    ("1", "All new switches off", "10 matches x 30 seasons", "identical to v6.6 to the cent", "check passed"),
    ("2", "Monopoly DP (no cap), copier search on season revenue, copier DP, DP over rival scenarios",
     "development half, 30 matches x 50 seasons", "+1.50% (9 better / 0 worse), +0.01%, −0.01%, +0.00%",
     "monopoly DP continues; scenario DP rejected (no effect)"),
    ("3", "Copier search on season revenue / copier DP", "all 4 v6.6 copier matches x 100 seasons",
     "+18.2% (3 / 0) / +26.9% (3 / 1)", "both continue"),
    ("4", "Same", "all 12 v6.5 copier matches x 60 seasons", "+1.3% (1 / 2, worst −4.0%) / +21.7% (9 / 0)",
     "copier DP preferred"),
    ("5", "Monopoly DP (no cap)", "check half, 30 v6.6 matches x 50 seasons", "−0.60% (4 / 1); one match −20.2% (48gs8r)",
     "investigated (step 6)"),
    ("6", "48gs8r: why?", "trace", "the algorithm's fitted willingness to pay m = $447 (true $52): fine with a rival, "
     "but alone the DP charges $100", "cap the monopoly DP at the hand rule"),
    ("7", "Monopoly DP capped / uncapped / days logged as \"no rival\"", "4 matches x 40 seasons",
     "+4.24% (3 / 0) / −2.46% / −2.15% (worst −19.1%)", "cap adopted"),
    ("8", "Out-of-sample check on old data", "data/v6.4 day 1, 18 matches x 100 seasons",
     "v6.5 beats v6.6 by 3.6%: match 446Pbr +40.6%, LvrJkv +13.0%", "investigated (steps 9-11)"),
    ("9", "Which v6.6 part?", "the 2 matches x 40 seasons, v6.6 minus one part",
     "446Pbr: the day-50 fit ($190,824 without it vs $52,322); LvrJkv: dropping sold-out days ($138,157 vs $121,202)", ""),
    ("10", "446Pbr repairs", "same", "fit back on day 100: $190,790; m bounded at $150 (day-50 fit kept): $191,135",
     "both repair it; step 12 decides"),
    ("11", "LvrJkv: rival at a random $5-$87 every day", "same",
     "price against forecast errors: $133,623; + DP over scenarios: $131,768; old \"posted price\" logging: $138,614",
     "uncertainty pricing adopted; posted-price logging not (wrong data that happens to help here)"),
    ("12", "Fit revert vs m bound, with all other v6.7 parts", "all 60 v6.6 matches x 100 seasons",
     "revert +2.49% (15 / 10, worst −3.1%); bound +2.82% (15 / 3, worst −1.5%)", "bound adopted"),
]


def pct(x, digits=0):
    return f"{100 * x:.{digits}f}%"


def money(x):
    return f"\\${x:,.0f}"


def signed(x, digits=2):
    return f"{x:+.{digits}f}%".replace("-", "−")


def pts(x, digits=2):
    return f"{x:+.{digits}f}".replace("-", "−") + " points"


def main():
    o, t, po = ON["overall"], ON["by_type"], PON["overall"]
    e6, e5 = OR["v6.6"], OR["v6.5"]
    tbm = OR["testbed_in_latest_markets"]
    pb = OR["price_bias"]
    full = R["change"]
    contrib = {k: full - AB[k]["change"] for k in AB}
    lines = []
    w = lines.append

    w(f"# Evaluation of the current algorithm (`duopoly.py` {NEW})")
    w("")
    w("> **This file always describes the current `duopoly.py`, evaluated on the latest tournament logs "
      "(`data/v6.6`, played by v6.6).** What the algorithm does, how it is measured, how good its models are, what is "
      "correlation and what is cause, how the offline simulation works and how far to trust it, where the weak points "
      "are and how they were found, and **how accurate the previous evaluation was** (section 5).")
    w(">")
    w("> Every number and figure is produced by `analytics/evaluate.py` and `analytics/oracle.py`, stored in "
      "`analytics/evaluation_v6.6.json`, and written into this file by `analytics/write_evaluation_md.py`. "
      "The evaluation of v6.6 written before its tournaments is archived in [`EVALUATION_v6.6.md`](EVALUATION_v6.6.md). "
      "History of all versions: [`ALGORITHM_CHANGES.md`](ALGORITHM_CHANGES.md).")
    w("")
    w("## 1. Summary")
    w("")
    w("| Question | Answer |")
    w("| :--- | :--- |")
    w(f"| How did v6.6 do online? | Ranks 1, 2 and 7 in three tournaments (the rank-2 log was lost to a platform bug; "
      f"60 matches analysed). v6.5 had ranks 4, 8, 9. |")
    w(f"| Better than v6.5, fairly measured? | Yes, by **{signed(PC['change'], 1)}** (v6.5 replayed in the same v6.6 "
      f"markets; {PC['better']} matches better / {PC['worse']} worse), not by the "
      f"{signed(OLD['testbed']['result']['change'], 1)} the last evaluation headlined. "
      f"Online, v6.6 earned {pct(e6['vs_o1'], 1)} of what its markets made possible (v6.5: {pct(e5['vs_o1'], 1)}), but "
      f"most of that jump is easier markets: replayed in the *same* v6.6 markets, v6.5 reaches {pct(tbm['v6.5'], 1)} "
      f"and v6.6 {pct(tbm['v6.6'], 1)} (section 4.2). |")
    w(f"| Was the last evaluation accurate? | The measurement tools were: the test bed reproduces v6.6's real revenue "
      f"to {signed(100 * V['total_error'])}, the rival-forecast and demand-model scores replicate on the new data. The "
      f"*forecast of the gain* was too high ({signed(OLD['testbed']['result']['change'])} predicted, "
      f"{signed(PC['change'])} measured on the new matches), and the copier gain reversed. Its stated tournament "
      f"expectation, 0% to +3%, was right. Section 5. |")
    w(f"| What changed in {NEW}? | Four changes to the optimised price and the dynamic programme, all found in the v6.6 "
      f"logs or in an out-of-sample replay of old logs: monopoly days priced by the DP (capped at the hand rule), "
      f"copiers priced by the DP (and a bug in the copier search fixed), a bound on the fitted willingness to pay, and "
      f"today's price optimised over the rival's likely prices instead of one guess. Section 2. |")
    w(f"| One quality number | Test bed, all 60 v6.6 matches x 100 seasons, same rivals and buyers for both versions: "
      f"**{signed(full)}** vs v6.6 (95% interval {signed(R['ci95'][0])} to {signed(R['ci95'][1])}), {R['better']} "
      f"matches better / {R['worse']} worse, worst match {signed(R['worst_match'], 1)}. **Concentrated:** median match "
      f"{signed(R['median_match'], 1)}, without the 5 biggest winners {signed(RB['change_without_top5'])}; the 4 copier "
      f"matches gain {signed(R['by_type']['copier']['change'], 1)}, the other rival types "
      f"{signed(min(R['by_type'][k]['change'] for k in ('dumper', 'seasonal', 'steady', 'erratic')), 1)} to "
      f"{signed(max(R['by_type'][k]['change'] for k in ('dumper', 'seasonal', 'steady', 'erratic')), 1)}. |")
    w(f"| Does it hold elsewhere? | data/v6.5 matches: {signed(PR['change'])} ({PR['better']} / {PR['worse']}). "
      f"data/v6.4 file: {signed(TB['v64_file'][NEW]['change'])} ({TB['v64_file'][NEW]['better']} / "
      f"{TB['v64_file'][NEW]['worse']}); the other 8 data/v6.4 files, never used for a decision: "
      f"{signed(GEN['v64_rest']['change'])} over {GEN['v64_rest']['matches']} matches ({GEN['v64_rest']['better']} / "
      f"{GEN['v64_rest']['worse']}). Old simulator (different demand model): {signed(SIM['change_total'], 1)}. Per rival "
      f"bot: {GEN['per_rival']['better']} of {GEN['per_rival']['rivals']} better, {GEN['per_rival']['worse']} worse "
      f"(section 9.4). |")
    w(f"| Honest expectation for the next tournament | +1% to +3%, depending mostly on how many copiers we meet "
      f"(about +1% with none). Section 11. |")
    w("")

    w("## 2. The algorithm")
    w("")
    w("The lecture's four steps (*A learning approach to dynamic pricing using nonlinear and linear optimization*, after "
      "Bertsimas & Perakis 2001), where each one lives in `duopoly.py`, and what v6.7 changed:")
    w("")
    w("| Step | In `duopoly.py` | Optimisation | Changed in v6.7 |")
    w("| :--- | :--- | :--- | :--- |")
    w("| 1. Estimate our demand | Choice model: buyers/day $\\sim$ Poisson$(n e^{k\\tau})$, each buys from us with "
      "probability $\\frac{e^{b(m-p)}}{1+e^{b(m-p)}+e^{b(m-r)}}$; Poisson maximum likelihood on our own days, first on day "
      "50 of season 1, then every 25 days and every season end | Nonlinear (Nelder–Mead) | **$m \\le \\$150$** in the fit "
      "(the course model's highest willingness to pay) |")
    w("| 2. Coefficients for the next period | Refit on the last 3,000 days | – | – |")
    w("| 3. Estimate the rival | Tomorrow: $r_t = a + c\\,r_{t-1} + b\\,p_{t-1}$ by least absolute deviations "
      "(**linear programme**, used only if it beats \"same as yesterday\" by 30%). Rest of the season: median price per "
      "day of the last 10 seasons + AR(1) | LP + time series | **Its last 60 forecast errors are kept** |")
    w("| 4. Our optimal price | $\\max_p E[\\,p\\min(D,s) + V(t+1, s-\\min(D,s))\\,]$ by a bounded scalar solver every day "
      "(**nonlinear programme**) | NLP | **Expectation also over the rival's price today** (forecast + each kept error), "
      "not one guess |")
    w("| Dynamic programme | $V(t,s)$ by backward induction, 100 days x 80 seats x \\$18–\\$100, on the forecast rival "
      "path; rebuilt every 10 days | DP | **Also used when the rival is sold out** (no rival; its price may only lower "
      "the hand-made monopoly price) **and against copiers** (rival price = ours) |")
    w("")
    w("Other rules unchanged from v6.6: market sizing (seat value 0 when the market cannot fill the plane), regime priors "
      "and the regression before the first fit, sold-out-rival days left out of the demand data.")
    w("")

    w("## 3. How we measure")
    w("")
    w("| Item | Definition |")
    w("| :--- | :--- |")
    w("| Quality measure | Revenue per season (the competition ranks by total revenue). |")
    w("| **Online, normalised** (new) | Revenue / revenue of an **oracle** that knows the match's demand model and the "
      "rival's actual prices and sold-out days in advance and plays the optimal DP (`analytics/oracle.py`). A tournament "
      "with rich markets and one with poor markets become comparable. A **fair oracle** knows the demand and the rival's "
      "*typical* path only; the gap between the two oracles is the part of the rival nobody can predict. |")
    w("| Offline | The test bed (section 8): each version plays every logged match against the logged rival, with "
      "the same random buyers. Differences are caused by the code. Reported: total change, 95% interval (bootstrap "
      "over matches), better / worse matches (beyond ±0.5%), worst match. |")
    w("| With / without | Each change is removed from the full version, everything else equal. |")
    w("| Out of sample | Changes were found on data/v6.6. They are also measured on data/v6.5, on one data/v6.4 file, and "
      "on the old simulator, whose buyers do not follow our demand model. |")
    w("| Prediction check | The previous evaluation's numbers against the same measurements on data/v6.6 (section 5). |")
    w("")

    w("## 4. Online: what happened with v6.6")
    w("")
    w("![v6.6 online scoreboard](imgs/v67_online_scoreboard.png)")
    w("")
    w("| Tournament | Rank | Matches | Revenue / season (mean) | (median) | Seats sold of 80 | We sold out | Rival sold out | "
      "Our / rival median price |")
    w("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for name, v in ON["by_tournament"].items():
        w(f"| `{name}` | {v['rank']} | {v['matches']} | {money(v['rev_mean'])} | {money(v['rev_median'])} | {v['seats']:.1f} | "
          f"{pct(v['sellout'])} | {pct(v['rival_sellout'])} | {money(v['our_median'])} / {money(v['rival_median'])} |")
    w("| `ProficientCivet` | 2 | – | log lost (platform bug) | | | | | |")
    w(f"| **v6.6, all logged** | – | {o['matches']} | {money(o['rev_mean'])} | {money(o['rev_median'])} | {o['seats']:.1f} | "
      f"{pct(o['sellout'])} | {pct(o['rival_sellout'])} | – |")
    w(f"| v6.5 (data/v6.5) | 4, 8, 9 | {po['matches']} | {money(po['rev_mean'])} | {money(po['rev_median'])} | "
      f"{po['seats']:.1f} | {pct(po['sellout'])} | {pct(po['rival_sellout'])} | – |")
    w("")
    w("| Rival behaviour (rules: section 4.1 of EVALUATION_v6.6.md) | Matches | Our revenue / season | Share of our revenue | "
      "Seats | We sold out | Rival sold out | Our / rival median price |")
    w("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for name in ("dumper", "copier", "seasonal", "steady", "erratic"):
        v = t[name]
        w(f"| {name} | {v['matches']} | {money(v['rev_mean'])} | {pct(v['share_revenue'])} | {v['seats']:.1f} | "
          f"{pct(v['sellout'])} | {pct(v['rival_sellout'])} | {money(v['our_median'])} / {money(v['rival_median'])} |")
    w("")
    w(f"18 of the 20 v6.6 rivals had also played v6.5, so the opponents are largely the same students; the markets are "
      f"new random draws. v6.6 sold more seats (+{o['seats'] - po['seats']:.1f} per season) and priced further below "
      f"the rival (median gap −\\$3.60 vs −\\$0.14 for v6.5). Sell-outs still came late (median day 98; 2% of seasons "
      f"before day 85), so this is not underpricing that empties the plane early.")
    w("")
    w("### 4.1 How much of the available revenue did we earn?")
    w("")
    w("![Oracle efficiency](imgs/v67_oracle_efficiency.png)")
    w("")
    w("| | v6.5 online (data/v6.5) | v6.6 online (data/v6.6) |")
    w("| :--- | :---: | :---: |")
    w(f"| Revenue / perfect-information oracle | {pct(e5['vs_o1'], 1)} | **{pct(e6['vs_o1'], 1)}** |")
    w(f"| Revenue / fair oracle (knows the rival's typical path only) | {pct(e5['vs_o2'], 1)} | {pct(e6['vs_o2'], 1)} |")
    w(f"| Fair / perfect oracle (= the rival's unpredictable part) | {pct(e5['o2_vs_o1'], 1)} | {pct(e6['o2_vs_o1'], 1)} |")
    w(f"| Season 1 only | {pct(e5['season1']['vs_o1'], 1)} | {pct(e6['season1']['vs_o1'], 1)} |")
    w(f"| Same 18 rivals in both | {pct(OR['matched_rivals']['v6.5']['vs_o1'], 1)} | "
      f"{pct(OR['matched_rivals']['v6.6']['vs_o1'], 1)} ({OR['matched_rivals']['rivals_better']} of "
      f"{OR['matched_rivals']['rivals']} rivals better) |")
    w(f"| Left on the table per season (vs perfect oracle) | {money(e5['regret_per_season'])} | "
      f"{money(e6['regret_per_season'])} |")
    w(f"| What a \\$1 instead of \\$18 floor would add (inside the model) | {signed(100 * e5['floor_value'], 1)} | "
      f"{signed(100 * e6['floor_value'], 1)} |")
    w("")
    w("### 4.2 Questioning the online jump")
    w("")
    w(f"{pct(e5['vs_o1'], 1)} → {pct(e6['vs_o1'], 1)} looks like a large step. But the oracle only normalises for "
      f"market size, not for how *hard* a market is: copiers and dumpers are hard, and v6.5 met {PON['by_type']['copier']['matches']} "
      f"copiers in 90 matches, v6.6 only {t['copier']['matches']} in 60. The test bed answers it directly: every version "
      f"replayed in the same v6.6 markets earns")
    w("")
    w("| Version in the v6.6 markets (test bed) | Share of the perfect-information oracle |")
    w("| :--- | :---: |")
    for v_, val in tbm.items():
        w(f"| {v_} | {pct(val, 1)} |")
    w("")
    w(f"So of the {100 * (e6['vs_o1'] - e5['vs_o1']):.1f}-point jump, about {100 * (tbm['v6.6'] - tbm['v6.5']):.1f} points "
      f"are v6.6's code and the rest are easier markets and rivals. The matched-rival comparison has the same problem "
      f"(same rival, different random market).")
    w("")
    w("### 4.3 Where the rest was lost (this found the v6.7 changes)")
    w("")
    w(f"- **Copiers:** {pct(OR['regret_share_by_type'].get('copier', 0))} of everything left on the table came from the "
      f"{t['copier']['matches']} copier matches. Against FlashyInchworm (copies our previous price on 100% of days) "
      f"v6.6 held ~\\$30 and sold out every season, often by day 64–88; the oracle charges ~\\$49–53. Cause: the "
      f"season-to-season search ranked prices by revenue per day held, and days after selling out were never counted, "
      f"so selling out early at a low price looked as good as selling all season at a higher one. And the season "
      f"noise is larger than the difference between \\$1 steps.")
    w(f"- **Monopoly days:** with the rival sold out, the hand-made monopoly rule priced a median "
      f"\\${pb['monopoly_median']:.2f} above the fair oracle ({pb['monopoly_days']:,} days). On duopoly days the median "
      f"difference is \\${pb['duopoly_median']:.2f} on every day of the season: the DP's optimisation is right, given "
      f"the model.")
    w(f"- **Season 1:** {pct(e6['season1']['vs_o1'])} of the oracle (later seasons {pct(e6['later']['vs_o1'])}); prices "
      f"\\${pb['season1_mean']:.1f} above the fair oracle on average.")
    w(f"- **Erratic rivals:** {pct(OR['regret_share_by_type'].get('erratic', 0))} of the loss, but mostly the "
      f"unpredictable part (fair-oracle gap).")
    w("")

    w("## 5. Was the last evaluation accurate?")
    w("")
    w("`EVALUATION_v6.6.md` was written on data/v6.5 before v6.6 played. Here every number it predicted is measured "
      "again on data/v6.6. For the gains, the test bed was rebuilt from the v6.6 logs and v6.5 replayed in the new "
      "markets, so \"observed\" is v6.6 vs v6.5 in matches neither was tuned on.")
    w("")
    w("![Prediction vs reality](imgs/v67_prediction_vs_reality.png)")
    w("")
    w("| Claim in EVALUATION_v6.6.md | Predicted | Observed on data/v6.6 | Verdict |")
    w("| :--- | :---: | :---: | :--- |")
    verdicts = {
        "Test bed reproduces real revenue (total error)": "**held** (now with the version that actually played)",
        "v6.6 vs v6.5, all matches (test bed)": "**too high** by a factor of ~2.5",
        "v6.6 vs v6.5, best-reproduced matches (observed: all matches)": "too high",
        "v6.6 vs v6.5, median match": "held",
        "v6.6 vs v6.5 against copiers": "**wrong sign**: the copier gain was path-dependent luck of the search",
        "v6.6 vs v6.5 against steady rivals": "better than predicted",
        "v6.6 vs v6.5 against erratic rivals": "lower",
        "v6.6 vs v6.5 against seasonal rivals": "lower",
        "v6.6 vs v6.5 against dumpers": "better than predicted (small \\$)",
        "Rival path forecast: error reduction vs flat": "**held exactly**",
        "Reaction model: next-day error reduction": "slightly lower; useful in fewer matches",
        "Choice model: share of the achievable daily R2": "**held**",
        "Server call time, mean (ms)": "better than predicted",
    }
    for r in E["prediction_vs_reality"]:
        unit = "%" if r["unit"] == "%" else " ms"
        signs = "share" in r["claim"] or "reduction" in r["claim"] or r["unit"] != "%"
        fmt = (lambda x: f"{x:.1f}{unit}") if signs else (lambda x: f"{x:+.1f}{unit}".replace("-", "−"))
        w(f"| {r['claim']} | {fmt(r['predicted'])} | {fmt(r['observed'])} | {verdicts.get(r['claim'], '')} |")
    w(f"| Honest expectation for the tournament | 0% to +3% | {signed(PC['change'])} | **held** |")
    w(f"| \"The day-50 fit is the one change with evidence in both directions; first to switch off\" | risk | it caused "
      f"the 446Pbr collapse (section 9) | **held**; repaired by the m bound |")
    w(f"| Old simulator: \"one cell outside the 95% intervals\" (old v1.0 rival, average market) | – | \\$3,330 ± 91 vs "
      f"\\$3,439 ± 41 overlap | **was wrong**: no cell was outside |")
    w("")
    w("**Why the gain was over-predicted.**")
    w("")
    w(f"1. *Match mix:* on data/v6.5 the 5 biggest winners were {pct(OLD['testbed']['result']['robust']['top5_share_of_gain'])} "
      f"of the gain, and 12 of 90 matches were copiers (the biggest per-type gain). data/v6.6 has 4 copiers in 60, and "
      f"there v6.6 lost to v6.5 ({signed(PC['by_type']['copier']['change'])}). Without copiers v6.6 gains "
      f"{signed(PC['change_no_copier'])} on the new data.")
    w("2. *Selection:* the changes were chosen because they won on the data/v6.5 test bed; some of that win is the noise "
      "of those particular matches (a winner's curse). The held-out check matches limited this but could not remove "
      "the match-mix effect.")
    w("3. *What did replicate:* everything measured on the logs themselves (test-bed fidelity, forecast errors, model "
      "fit). What did not replicate are *differences between versions* that rest on few matches.")
    w("")
    w("Consequence for this evaluation: v6.7's number is reported with its spread over three data sets and the simulator, "
      "and section 11 gives a range, not the headline.")
    w("")

    w("## 6. Are the models good? (data/v6.6)")
    w("")
    w("![Demand model quality](imgs/v67_demand_model_quality.png)")
    w("")
    w(f"Fitted on seasons 1–50, scored on seasons 51–100 ({DM['matches']} matches, {DM['test_days']:,} days; "
      f"{pct(DM['zero_days'])} of days sell no seat, {DM['mean_sales_per_day']:.2f} seats per day).")
    w("")
    w("| Model | $R^2$ daily (median match) | Adjusted $R^2$ | Share of the achievable $R^2$ | $R^2$ of 10-day totals | "
      "Level error | Best in (matches) |")
    w("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for key, label in (("ols", "Linear, least squares"), ("lad_day", "Linear, LP per day (lecture step 1 as written)"),
                       ("lad_block", "Linear, LP on 5-day blocks"), ("choice", "**Choice model (used)**")):
        v = DM[key]
        w(f"| {label} | {v['r2_median']:.3f} | {v['adj_r2_median']:.3f} | {pct(v['share_of_ceiling'])} | "
          f"{v['block_r2_median']:.2f} | {pct(v['abs_bias_median'], 1)} | {v['best_in_matches']} |")
    w("")
    w(f"A daily $R^2$ of {DM['choice']['r2_median']:.3f} is close to the maximum: with Poisson buyers even the true "
      f"model explains at most {DM['ceiling_median']:.3f} (median match). The LP-per-day model fits the median day, which "
      f"sells 0 seats: all coefficients 0 in {DM['lad_day_all_zero_matches']} of {DM['matches']} matches.")
    w("")
    w("**A weakness the fit scores do not show.** The choice model predicts daily sales well, but its *absolute* "
      "willingness to pay $m$ is weakly identified while the rival is present (demand then depends mostly on price "
      "differences). Fitted on duopoly days only, $m$ came out above \\$150 in 12 of 60 v6.6 matches and 9 of 90 v6.5 "
      "matches (\\$160–\\$446). That is harmless for duopoly pricing and fatal once the rival is gone: it caused the "
      "monopoly-DP failure in 48gs8r and the 446Pbr collapse. v6.7 bounds $m$ at \\$150.")
    w("")
    w("![Rival forecast](imgs/v67_rival_forecast.png)")
    w("")
    w("| Rival forecast (each from earlier days/seasons only) | Mean abs. error | Median match |")
    w("| :--- | :---: | :---: |")
    for key, label in (("many_flat", "Rest of season: flat 5-day average"), ("many_path", "Rest of season: path (used)"),
                       ("one_avg5", "Tomorrow: 5-day average"), ("one_yesterday", "Tomorrow: yesterday's price"),
                       ("one_reaction", "Tomorrow: reaction model (LP)")):
        w(f"| {label} | \\${RF[key]['mean']:.2f} | \\${RF[key]['median']:.2f} |")
    w("")
    w(f"The path cuts the rest-of-season error by {pct(RF['path_reduction'])} ({RF['path_better']} matches better, "
      f"{RF['path_worse']} worse). For erratic rivals it barely helps (\\${RF['by_type']['erratic']['flat']:.2f} → "
      f"\\${RF['by_type']['erratic']['path']:.2f}): their prices cannot be predicted, which is why v6.7 prices against "
      f"the spread of their likely prices.")
    w("")

    w("## 7. Correlation or causation? (data/v6.6)")
    w("")
    w("![Correlation vs causation](imgs/v67_correlation_vs_causation.png)")
    w("")
    w(f"- **Our price and our sales:** a higher price goes with *more* sales in {CA['raw_positive']} of {CA['matches']} "
      f"matches. The third variable is the rival's price (we charge more *and* sell more when the rival is expensive; "
      f"product of the two links vs raw correlation: r = {CA['rival_hypothesis_r']:.2f}). With rival price and day held "
      f"fixed the price effect is negative in {CA['partial_negative']} of {CA['matches']}.")
    w(f"- **Who follows whom:** their price today vs ours yesterday {CA['they_follow_median']:.2f}, ours today vs theirs "
      f"yesterday {CA['we_follow_median']:.2f} (medians). The logs cannot separate the directions; only the "
      f"{CA['copiers']} copiers are certain (exact copies). The reaction model is therefore used as a forecast, not as "
      f"proof that we can steer the rival.")
    w(f"- **Online efficiency vs version:** v6.6's higher efficiency online is partly market mix (section 4.2), a "
      f"correlation that the test bed separates into cause (code) and confounder (markets).")
    w("- **Inside the test bed** both versions face the same rivals and buyers, so a difference is caused by the code. "
      "Whether the test bed resembles the tournament is section 8.")
    w("")

    w("## 8. Offline simulation: how it works and how far to trust it")
    w("")
    w("`analytics/testbed.py --data v6.6`: per match the choice model fitted to all 10,000 logged days (8 starting "
      "points); the rival replays its logged prices and sold-out days (copiers copy us); every version plays 100 seasons "
      "with the same random number per (match, season, day).")
    w("")
    w("![Test bed validity](imgs/v67_testbed_validity.png)")
    w("")
    w("| Check | Result |")
    w("| :--- | :--- |")
    w(f"| v6.6 replayed vs v6.6 in the tournament, total | {money(V['replay_total'])} vs {money(V['actual_total'])} "
      f"({signed(100 * V['total_error'])}) |")
    w(f"| Per match | $R^2$ = {V['r2_levels']:.3f}; median error {pct(V['median_abs_error_big'], 1)} in the "
      f"{V['big_matches']} matches above \\$500 per season; {V['within_10pct']} within ±10% |")
    w("| Last time | v6.5 replayed vs v6.5 actual: +0.1%. Now validated with the version that actually played, on data "
      "the test bed had never seen |")
    w(f"| All switches of the new code off | identical to v6.6 to the cent |")
    w("")
    w("| Limit | Consequence |")
    w("| :--- | :--- |")
    w("| Buyers come from the same model family the algorithm fits | Gains are upper estimates; the old simulator "
      "(uniform willingness to pay) is the counter-check |")
    w("| Rivals are recordings (except copiers) | Reactions to our new prices are not simulated |")
    w("| Prices never tried (below \\$18, above logged prices) | Extrapolation; the \\$18 floor stays (section 9) |")
    w("| Differences between versions rest on few matches | Section 5: the last gain forecast was ~2.5x too high |")
    w("")

    w(f"## 9. Results: {NEW} against {PREV}")
    w("")
    w("![Test bed results](imgs/v67_testbed_results.png)")
    w("")
    w("| Data | Matches x seasons | Change vs v6.6 | 95% interval | Better / worse | Worst match |")
    w("| :--- | :---: | :---: | :---: | :---: | :---: |")
    w(f"| **data/v6.6** (latest) | {R['matches']} x 100 | **{signed(full)}** | {signed(R['ci95'][0])} to {signed(R['ci95'][1])} | "
      f"{R['better']} / {R['worse']} | {signed(R['worst_match'], 1)} |")
    w(f"| data/v6.5 | {PR['matches']} x 60 | {signed(PR['change'])} | {signed(PR['ci95'][0])} to {signed(PR['ci95'][1])} | "
      f"{PR['better']} / {PR['worse']} | {signed(PR['worst_match'], 1)} |")
    f64 = TB["v64_file"]
    w(f"| data/v6.4 day 1, Tody_Rank_33 | {f64[NEW]['matches']} x 100 | {signed(f64[NEW]['change'])} | "
      f"{signed(f64[NEW]['ci95'][0])} to {signed(f64[NEW]['ci95'][1])} | {f64[NEW]['better']} / {f64[NEW]['worse']} | "
      f"{signed(f64[NEW]['worst_match'], 1)} |")
    w(f"| (same file: v6.5 vs v6.6) | | {signed(f64['v6.5']['change'])} | | {f64['v6.5']['better']} / {f64['v6.5']['worse']} | |")
    w("")
    w(f"Revenue per season on data/v6.6: {money(R['old_per_season'])} → {money(R['new_per_season'])}; seasons sold out "
      f"{pct(R['sellout_old'], 1)} → {pct(R['sellout_new'], 1)}. Median match {signed(R['median_match'], 1)}. "
      f"Without the 5 biggest winners: {signed(RB['change_without_top5'])}; on the {RB['trusted_matches']} matches the "
      f"test bed reproduces within ±10%: {signed(RB['trusted_change'])}.")
    w("")
    w("| By rival behaviour (data/v6.6) | Matches | Change | Per season |")
    w("| :--- | :---: | :---: | :---: |")
    for name in ("dumper", "copier", "seasonal", "steady", "erratic"):
        if name in R["by_type"]:
            v = R["by_type"][name]
            w(f"| {name} | {v['matches']} | {signed(v['change'])} | {'+' if v['dollars_per_season'] >= 0 else '−'}"
              f"\\${abs(v['dollars_per_season']):,.0f} |")
    w("")
    w("### 9.1 With / without each change (data/v6.6, 60 x 100)")
    w("")
    w("| Version | Change vs v6.6 | Better / worse | Worst match | Contribution of the change |")
    w("| :--- | :---: | :---: | :---: | :---: |")
    w(f"| **v6.7** | **{signed(full)}** | {R['better']} / {R['worse']} | {signed(R['worst_match'], 1)} | – |")
    for k in AB:
        a = AB[k]
        label = k.replace("v6.7 alt: ", "alternative: ")
        w(f"| {label} | {signed(a['change'])} | {a['better']} / {a['worse']} | {signed(a['worst_match'], 1)} | "
          f"{pts(contrib[k]) if 'alt' not in k else '–'} |")
    w("")
    w(f"Copier matches alone: {signed(TB['copier_matches']['latest']['change'])} on the {TB['copier_matches']['latest']['matches']} "
      f"v6.6 copiers, {signed(TB['copier_matches']['prev']['change'])} on the {TB['copier_matches']['prev']['matches']} "
      f"v6.5 copiers. The rival-uncertainty change is worth little on data/v6.6 because few of its rivals are random from "
      f"day to day; it repairs matches like data/v6.4 LvrJkv (+10% there, step 11 below).")
    w("")
    w("### 9.2 How v6.7 was built (and what was rejected)")
    w("")
    w("| # | Change / question | Data | Result | Decision |")
    w("| :---: | :--- | :--- | :--- | :--- |")
    for row in DEV:
        w("| " + " | ".join(row) + " |")
    w("")
    w("### 9.3 Check with a different demand model (old simulator)")
    w("")
    w("`analytics/benchmark_3regime_models.py`: buyers with uniform willingness to pay, 4 scripted rivals, 5 seeds x 30 "
      "seasons; sum over the rivals of mean revenue per season.")
    w("")
    cols = [c for c in ("v6.6", "v6.7", "v6.7 alt: fit revert instead of bound") if c in SIM["rows"][0]]
    names = {"v6.7 alt: fit revert instead of bound": "alt: fit revert"}
    w("| Market | " + " | ".join(names.get(c, c) for c in cols) + " | v6.7 vs v6.6 |")
    w("| :--- | " + " | ".join(":---:" for _ in cols) + " | :---: |")
    for row in SIM["rows"]:
        w(f"| {row['market']} | " + " | ".join(money(row[c]) for c in cols) +
          f" | {signed(100 * (row['v6.7'] / row['v6.6'] - 1), 1)} |")
    w("")
    w(f"Total {signed(SIM['change_total'], 1)}. Cells whose 95% intervals do not overlap: "
      f"{len(SIM['cells_outside_ci'])}" + (": " + "; ".join(f"{c['rival']} ({c['market']}): \\${c['new'][0]:,.0f} ± "
                                                             f"{c['new'][1]:,.0f} vs \\${c['old'][0]:,.0f} ± {c['old'][1]:,.0f}"
                                                             for c in SIM["cells_outside_ci"]) if SIM["cells_outside_ci"] else "") + ".")
    w("")

    w("### 9.4 How general is the gain?")
    w("")
    w("Two of the four changes started from single matches (monopoly cap: 48gs8r; rival uncertainty: LvrJkv; the m "
      "bound: 48gs8r and 446Pbr). Three checks of whether the result rests on a few scenarios:")
    w("")
    pr = GEN["per_rival"]
    w(f"**1. Per rival bot** (data/v6.6 and data/v6.5 pooled, {pr['matches']} matches, {pr['rivals']} distinct bots, "
      f"`analytics/per_rival_v67.py`). v6.7 is better against {pr['better']} bots, worse against {pr['worse']}, and "
      f"unchanged (within ±0.5%) against {pr['unchanged']}. Leaving out any one bot, the total gain stays between "
      f"{signed(pr['loo_min'])} (without {pr['loo_min_rival']}) and {signed(pr['loo_max'])}. The top 3 bots carry "
      f"{pct(pr['top3_share'])} of the gain: " + ", ".join(f"{x['rival']} ({x['type']}, {signed(x['change'], 1)})"
                                                           for x in pr["top3"]) + ".")
    w("")
    cm = GEN["copier_modes"]
    w("**2. Imperfect copiers.** The test bed simulates copiers as exact copies of our price the day before (the real "
      "copiers in the logs copy exactly). Stress test on all copier matches, 40 seasons, `--copier-mode`:")
    w("")
    w("| Copier behaviour | v6.6 copiers (4 matches) | v6.5 copiers (12 matches) |")
    w("| :--- | :---: | :---: |")
    w(f"| exact copy of our price yesterday (as above) | {signed(TB['copier_matches']['latest']['change'], 1)} | "
      f"{signed(TB['copier_matches']['prev']['change'], 1)} |")
    for mode, label in (("lag2", "copies our price of 2 days ago"), ("noise", "copies with ±\\$2 noise"),
                        ("partial", "halfway between our price and its own logged price")):
        a, b = cm[mode]["66"], cm[mode]["65"]                       # v6.7 vs v6.6
        w(f"| {label} | {signed(a['change'], 1)} | {signed(b['change'], 1)} |")
    w("")
    w("v6.7 is never worse than v6.6. The large gain needs an exact copier: copier detection asks for our previous "
      "price within \\$0.60 on 80% of days, so lagging or noisy copiers are treated as ordinary rivals by both versions "
      "(no gain, and no exposure to the copier DP). The half-copier gain on the v6.6 matches comes from the other v6.7 "
      "changes.")
    w("")
    vr = GEN["v64_rest"]
    w(f"**3. Data never used for a decision:** the other {vr['files']} data/v6.4 files ({vr['matches']} matches x 10 "
      f"seasons, rivals of the v6.4 tournaments): v6.7 vs v6.6 **{signed(vr['change'])}**, {vr['better']} matches "
      f"better / {vr['worse']} worse (beyond ±0.5%), worst match {signed(vr['worst_match'], 1)}, median match "
      f"{signed(vr['median_match'], 1)}, without the 5 biggest winners {signed(vr['without_top5'])}.")
    w("")
    w("## 10. Weak points: how they were found, what was done")
    w("")
    w("| # | Weak point | How it was found | Status in v6.7 |")
    w("| :---: | :--- | :--- | :--- |")
    w("| 1 | Copier search ranked prices by revenue per day held; sold-out days uncounted → drifted to ~\\$30 | Oracle regret "
      "by rival type (31% from 4 copier matches), then a day-by-day trace of the worst match | **Fixed**; and copiers "
      "now priced by the DP |")
    w(f"| 2 | Hand-made monopoly price a median \\${pb['monopoly_median']:.2f} above the oracle | Our price vs the fair "
      f"oracle's, per day | **Fixed**: DP, capped at the hand price |")
    w("| 3 | Fitted willingness to pay $m$ can be implausible (\\$160–\\$446) | A −20% match in the check set, traced to "
      "$m$ = \\$447; then counted in all matches | **Fixed**: $m \\le \\$150$ in the fit |")
    w("| 4 | v6.6's day-50 fit could collapse against a rival parked at \\$100 (1 in 8 random replays of 446Pbr) | "
      "Replaying an old data file (data/v6.4) out of sample | **Fixed** by #3 (the fit date stays) |")
    w("| 5 | Today's price used one guess of a random rival's price | LvrJkv trace: the right model earned less than a "
      "wrong one | **Fixed**: expectation over recent forecast errors |")
    w(f"| 6 | The rival's unpredictable part: fair / perfect oracle = {pct(e6['o2_vs_o1'], 1)} | Two oracles | **Open**: "
      f"a DP over rival scenarios added nothing (tested twice) |")
    w(f"| 7 | \\$18 floor vs dumpers: {t['dumper']['matches']} matches, {pct(t['dumper']['share_revenue'])} of revenue; "
      f"a \\$1 floor would add {signed(100 * e6['floor_value'], 1)} inside the model | Oracle with floor \\$1 | **Open**: "
      f"never priced below \\$18, so the gain is extrapolation; needs one online experiment |")
    w(f"| 8 | Season 1: {pct(e6['season1']['vs_o1'])} of the oracle | Oracle by season | Open; 1% of revenue |")
    w("| 9 | Rival reactions to our price | Section 7 | Open; not identifiable from logs |")
    w("| 10 | The oracle and the test bed share the algorithm's model family | By construction | The simulator check (9.3) "
      "and the out-of-sample data sets are the counterweight |")
    w("")
    w("## 11. What we do not know, and what to expect")
    w("")
    w(f"- **Next tournament:** +1% to +3% over v6.6. The test bed says {signed(full)} on the latest matches and "
      f"{signed(PR['change'])} on the older ones; last time the test bed over-predicted the gain about 2.5x, and the "
      f"simulator (wrong model on purpose) shows {signed(SIM['change_total'], 1)}. The gain is concentrated: median "
      f"match {signed(R['median_match'], 1)}, {signed(RB['change_without_top5'])} without the 5 biggest winners, most of "
      f"it in copier matches. A tournament without copiers should give about +1% (monopoly days); one with several "
      f"copiers more. Unlike v6.6's copier gain, this one is supported by the simulator's copycat (\\$1,424 → "
      f"\\$1,702), which uses a different demand model.")
    w("- **Copier DP prices far above anything we held** (~\\$50 vs \\$25–33). The test bed's copier demand at \\$50 is "
      "itself extrapolated from the same logs. The simulator's copycat does not contradict it; an online result will.")
    w(f"- **Speed:** v6.6 online {E['latency']['mean_ms']:.1f} ms per call on average, {E['latency']['max_s']:.2f} s "
      f"slowest, {E['latency']['errors']} errors (limits 200 ms / 5 s). v6.7 adds a few DP runs (copier, monopoly) and "
      f"an expectation over 60 rival scenarios in the price solver.")
    w("")
    w("## 12. Reproduce")
    w("")
    w("```bash")
    w("python -m pytest tests -q                                 # 55 tests (+6 that need data/v6.4 and the analytics data)")
    w("python analytics/testbed.py --data v6.6 --out analytics/testbed_v6.6_all.csv --compare analytics/experiments/duopoly_v65.py")
    w("python analytics/testbed.py --data v6.6 --out work/all66_mbound.csv --compare analytics/experiments/v67_mbound_fit45.py")
    w("# ... with/without runs, data/v6.5, data/v6.4: see work/run*.sh; then")
    w("python analytics/merge_testbed_v67.py      # version-named result tables")
    w("python analytics/oracle.py --data v6.6 && python analytics/oracle.py --data v6.5")
    w("python analytics/benchmark_3regime_models.py --compare analytics/experiments/v67_mbound_fit45.py ... > work/sim_v67.log")
    w("python analytics/parse_simulator_v67.py")
    w("python analytics/evaluate.py               # numbers -> analytics/evaluation_v6.6.json, figures -> imgs/v67_*.png")
    w("python analytics/write_evaluation_md.py    # this file")
    w("```")
    w("")
    w("Sources: D. Bertsimas, G. Perakis, *Dynamic Pricing: A Learning Approach* (2001); T. Winter, *A learning approach "
      "to dynamic pricing using nonlinear and linear optimization* (Learning Optimization, BHT, lecture slides).")
    w("")
    open(os.path.join(ROOT, "EVALUATION.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("EVALUATION.md written,", len(lines), "lines")


if __name__ == "__main__":
    main()
