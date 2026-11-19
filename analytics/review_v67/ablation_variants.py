"""Writes the ablation variants of duopoly.py (one rule or a combination switched off) to analytics/experiments/ablation/.
Then: python analytics/testbed.py --data v6.7 --seasons 20 --sample 2 --compare analytics/experiments/ablation/<v>.py ...
LP variant (capacity-constrained linear programme instead of the DP): analytics/experiments/duopoly_v68_lp.py."""
import os, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
base = open(os.path.join(ROOT, "duopoly.py")).read()

def flag(name, val):
    def f(s):
        s2, n = re.subn(rf"^{name}( *(:[^=]+)?)= *(True|False)", rf"{name}\1= {val}", s, flags=re.M)
        assert n == 1, name
        return s2
    return f

def patch(old, new):
    def f(s):
        assert s.count(old) == 1, old[:50]
        return s.replace(old, new)
    return f

NO_DP = patch('''    Also sets state["dp_price"], today's price."""
    if not state.get("logit_fitted", False) or seats_left <= 0:''', '''    Also sets state["dp_price"], today's price."""
    return None
    if not state.get("logit_fitted", False) or seats_left <= 0:''')
NO_SIZING = patch('''    if not state.get("logit_fitted", False):
        return False
    theta = state["logit_theta"]
    rival = state.get("rival_last_price", DEFAULT_RIVAL_PRICE)''', '''    return False
    theta = state["logit_theta"]
    rival = state.get("rival_last_price", DEFAULT_RIVAL_PRICE)''')
NO_COPIER = patch('''    checks = state.get("copy_checks", [])
    return len(checks) >= 10 and float(np.mean(checks)) >= COPIER_SHARE''', '''    return False''')
FORECASTS_OFF = [flag("RIVAL_PATH_IN_DP", "False"), flag("RIVAL_PATH_IN_SIZING", "False"),
                 flag("RIVAL_UNCERTAINTY_TODAY", "False"), flag("RIVAL_REACTION_TODAY", "False")]
VARIANTS = {
    "no_dp": [NO_DP], "no_nlp": [flag("NLP_PRICE", "False")],
    "no_rival_path": [flag("RIVAL_PATH_IN_DP", "False"), flag("RIVAL_PATH_IN_SIZING", "False")],
    "no_rival_uncertainty": [flag("RIVAL_UNCERTAINTY_TODAY", "False")], "no_reaction_today": [flag("RIVAL_REACTION_TODAY", "False")],
    "no_market_sizing": [NO_SIZING], "no_monopoly_dp": [flag("DP_MONOPOLY", "False")], "no_copier_logic": [NO_COPIER],
    "no_coop": [flag("COOP_PRICING", "False")],
    "guards_back": [flag("DP_BAND", "True"), flag("DP_SKIP_GUARDS", "False"), flag("DP_SKIP_CLEARANCE", "False")],
    "keep_soldout_days": [flag("SKIP_SOLD_OUT_RIVAL_DAYS", "False")],
    "C_flat_rival_dp": [flag("NLP_PRICE", "False")] + FORECASTS_OFF, "C_no_forecasting": FORECASTS_OFF,
    "C_hand_rules_only": [NO_DP, NO_SIZING],
    "C_no_special_cases": [flag("DP_MONOPOLY", "False"), NO_COPIER, flag("COOP_PRICING", "False")],
}
if __name__ == "__main__":
    out = os.path.join(ROOT, "analytics", "experiments", "ablation"); os.makedirs(out, exist_ok=True)
    for name, fs in VARIANTS.items():
        s = base
        for f in fs:
            s = f(s)
        open(os.path.join(out, name + ".py"), "w").write(s)
    print(len(VARIANTS), "variants in", out)
