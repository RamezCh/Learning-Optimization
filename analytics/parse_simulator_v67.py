"""
analytics/parse_simulator_v67.py
--------------------------------
Reads the old simulator's output (analytics/experiments/simulator_v67.txt, from analytics/benchmark_3regime_models.py) and writes
analytics/v67_simulator_check.json: per market the sum over the 4 rivals of mean revenue per season, per
version, plus every rival cell whose 95% intervals do not overlap between v6.7 and v6.6.
"""
import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAMES = {"duopoly.py": "v6.6", "v67_mbound_fit45.py": "v6.7", "v67.py": "v6.7 alt: fit revert instead of bound"}


NEW, OLD = "v67_mbound_fit45.py", "duopoly.py"


def main(log=os.path.join(ROOT, "work", "sim_v67.log"), out_path=os.path.join(ROOT, "analytics", "v67_simulator_check.json")):
    text = open(log).read()
    block = text[text.index("MULTI-SEED BENCHMARK"):]
    lines = block.splitlines()
    header = next(l for l in lines if l.startswith("Algorithm"))
    names = header_names(header)
    cells, market = [], None
    for l in lines[lines.index(header) + 2:]:
        if l.startswith("---"):
            break
        m = re.match(r"^(\S.*?\))\s+(.*)$", l)
        if m and "arr." in m.group(1):
            market, rest = m.group(1), m.group(2)
        else:
            rest = l.strip()
        values = [(float(a.replace(",", "")), float(b.replace(",", ""))) for a, b in re.findall(r"\$([\d,]+) ± ([\d,]+)", rest)]
        cells.append(dict(market=market, rival=re.split(r"\s+\$", rest)[0].strip(), values=dict(zip(names, values))))
    totals = {}
    for c in cells:
        for name, (mean, ci) in c["values"].items():
            totals.setdefault(c["market"], {}).setdefault(name, 0.0)
            totals[c["market"]][name] += mean
    rows = []
    for mk, t in totals.items():
        rows.append(dict(market=mk, **{NAMES.get(k, k): v for k, v in t.items()}))
    differ = []
    for c in cells:
        a, b = c["values"].get(NEW), c["values"].get(OLD)
        if a and b and (a[0] - a[1] > b[0] + b[1] or a[0] + a[1] < b[0] - b[1]):
            differ.append(dict(market=c["market"], rival=c["rival"], new=a, old=b))
    total = {NAMES.get(k, k): sum(t[k] for t in totals.values()) for k in next(iter(totals.values()))}
    out = dict(source=f"{os.path.relpath(log, ROOT)} (benchmark_3regime_models.py, 5 seeds x 30 seasons)", rows=rows,
               total=total, change_total=100 * (total[NAMES[NEW]] / total[NAMES[OLD]] - 1), cells_outside_ci=differ, cells=cells)
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("rows", "total", "change_total", "cells_outside_ci")}, indent=1))


def header_names(header):
    """Column names of the table; "naive rival-$1" is the only name with a space."""
    tokens, names, i = header[len("Algorithm"):].split(), [], 0
    while i < len(tokens):
        if tokens[i] == "naive":
            names.append("naive rival-$1"); i += 2
        else:
            names.append(tokens[i]); i += 1
    return names


if __name__ == "__main__":
    main()
