"""
Does the rival react to our price? For every match: on days where OUR price moved by more than the band since the day
before (informative days, rival selling, both prices real), compare the rival's price today with two targets:
  copy:  our price yesterday
  BR:    the rival's best reply to our price yesterday, argmax_r r * rate(theta, r, p_{t-1}, tau)
         (match's fitted choice model, symmetric for both airlines, capacity ignored)
A hit = rival within +-band (relative) of the target. Placebo: the same test with our price from the NEXT season on the
same day (a rival that ignores us hits the placebo as often as the real target). Reaction = hit rate - placebo rate.
Usage: python analytics/review_v67/reaction_rule.py v6.7 [v6.6 v6.5]
"""
import sys, os, json, glob, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "analytics"))
import testbed as tb
BANDS = (0.01, 0.02, 0.03)
GRID = np.arange(1.0, 120.01, 0.25)

def best_reply(theta, ours, tau):
    rev = GRID[:, None] * tb.rate(theta, GRID[:, None], ours[None, :], tau[None, :])
    return GRID[rev.argmax(axis=0)]

def main(versions):
    rows = []
    for v in versions:
        tb.set_data(v); D = tb.load(); F = json.load(open(tb.FITS))
        for key, g in D.groupby("key"):
            if key not in F: continue
            th = np.array(F[key]["theta"])
            P = g.price.values.reshape(-1, 100); R = g.price_competitor.values.reshape(-1, 100)
            cap = g.cap.values.reshape(-1, 100).astype(bool)
            n = P.shape[0]
            p_prev, p_prev2 = P[:, :-2 + 1][:, :-1], None
            pp = P[:, 1:-1]; pp2 = P[:, :-2]; r = R[:, 2:]; c = cap[:, 2:]
            plac = np.roll(P, -1, axis=0)[:, 1:-1]                          # our price, next season, same day
            tau = np.broadcast_to((np.arange(2, 100) / 99.0)[None, :], r.shape)
            ok = c & (pp < 120) & (pp2 < 120) & (r < 120) & (r > 0) & (plac < 120)
            out = dict(data=v, key=key, rival=g.competitor_id.iloc[0], copier_fit=bool(F[key]["copier"]))
            br = best_reply(th, pp[ok], tau[ok]); brp = best_reply(th, plac[ok], tau[ok]); rr = r[ok]
            moved_all = np.abs(pp - pp2)[ok] / pp2[ok]
            for b in BANDS:
                m = moved_all > b
                out[f"n_{b:.2f}"] = int(m.sum())
                if m.sum() < 30:
                    continue
                out[f"copy_{b:.2f}"] = float((np.abs(rr[m] - pp[ok][m]) <= b * pp[ok][m]).mean())
                out[f"copyP_{b:.2f}"] = float((np.abs(rr[m] - plac[ok][m]) <= b * plac[ok][m]).mean())
                out[f"br_{b:.2f}"] = float((np.abs(rr[m] - br[m]) <= b * br[m]).mean())
                out[f"brP_{b:.2f}"] = float((np.abs(rr[m] - brp[m]) <= b * brp[m]).mean())
            # direction test: does the rival move the same way we moved yesterday? (all moved days, 2% band)
            m = moved_all > 0.02
            dr = (r[ok] - R[:, 1:-1][ok])[m]; dp = (pp[ok] - pp2[ok])[m]
            out["same_dir"] = float((np.sign(dr) == np.sign(dp))[dr != 0].mean()) if (dr != 0).sum() >= 30 else np.nan
            # placebo: our move on the same day of the next season (shared time trends give the same score)
            plac2 = np.roll(P, -1, axis=0)[:, :-2][ok][m]
            dpp = plac[ok][m] - plac2
            k = (dr != 0) & (dpp != 0)
            out["same_dirP"] = float((np.sign(dr) == np.sign(dpp))[k].mean()) if k.sum() >= 30 else np.nan
            out["rival_moves"] = float((np.abs(dr) > 0.01 * R[:, 1:-1][ok][m]).mean()) if m.sum() else np.nan
            rows.append(out)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "analytics", "review_v67", "reaction_rule.csv"), index=False)
    return df

if __name__ == "__main__":
    df = main(sys.argv[1:] or ["v6.7"])
    for b in ("0.01", "0.02", "0.03"):
        x = df.dropna(subset=[f"copy_{b}"])
        print(f"band {b}: matches {len(x)}  copy hit {x[f'copy_{b}'].mean():.3f} (placebo {x[f'copyP_{b}'].mean():.3f})"
              f"  BR hit {x[f'br_{b}'].mean():.3f} (placebo {x[f'brP_{b}'].mean():.3f})")
