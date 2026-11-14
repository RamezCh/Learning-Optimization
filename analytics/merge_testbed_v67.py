"""
analytics/merge_testbed_v67.py
------------------------------
Test-bed result files name their columns after the pricing FILE that played (duopoly.py was v6.6 when
these runs were made). This writes the same numbers with VERSION names, which evaluate.py reads:
    analytics/v67_testbed_v66data.csv   60 data/v6.6 matches x 100 seasons: v6.5, v6.6, v6.7, v6.7 without each change
    analytics/v67_testbed_v65data.csv   90 data/v6.5 matches x 60 seasons: v6.6, v6.7
    analytics/v67_testbed_v64file.csv   18 matches of data/v6.4 day 1 Tody_Rank_33 x 100 seasons: v6.5, v6.6, v6.7
All runs use the same random draws per (match, season, day), so columns are paired.
"""
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
W = os.path.join(ROOT, "work")
NAMES = {"duopoly.py": "v6.6", "duopoly_v65.py": "v6.5", "v67_mbound_fit45.py": "v6.7",
         "v67.py": "v6.7 alt: fit revert instead of bound",
         "v67_without_fit_revert.py": "v6.7 without WTP bound",          # = all v6.7 parts, day-50 fit unbounded
         "v67f_without_monopoly.py": "v6.7 without monopoly DP", "v67f_without_copier.py": "v6.7 without copier DP",
         "v67f_without_uncertainty.py": "v6.7 without rival uncertainty",
         # removals measured on the fit-revert variant (development; not used in EVALUATION.md):
         "v67_without_monopoly.py": "alt without monopoly DP", "v67_without_copier.py": "alt without copier DP",
         "v67_without_uncertainty.py": "alt without rival uncertainty"}


def rename(frame):
    out = {}
    for c in frame.columns:
        base, _, suffix = c.partition(":")
        out[c] = NAMES.get(base, base) + (":" + suffix if suffix else "")
    return frame.rename(columns=out)


def merge(files, out):
    frames = [rename(pd.read_csv(os.path.join(W, f))) for f in files]
    t = frames[0]
    for f in frames[1:]:
        assert (f.key.values == t.key.values).all()
        for c in f.columns:
            if c not in t.columns:
                t[c] = f[c].values
            elif c not in ("key", "actual"):
                assert (abs(t[c] - f[c]) < 0.01).all(), c          # same file, same draws -> same revenue
    t.to_csv(os.path.join(ROOT, "analytics", out), index=False)
    print(out, list(t.columns))


if __name__ == "__main__":
    import shutil
    shutil.copy(os.path.join(ROOT, "analytics", "testbed_v6.6_all.csv"), os.path.join(W, "testbed_v6.6_all.csv"))
    merge(["testbed_v6.6_all.csv", "all66_mbound.csv", "all66_v67.csv", "all66_abl_final.csv"], "v67_testbed_v66data.csv")
    merge(["all65_v67.csv"], "v67_testbed_v65data.csv")
    merge(["tody33_v67.csv", "tody33_mbound.csv"], "v67_testbed_v64file.csv")
