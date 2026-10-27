"""
analytics/compare_demand_models.py
----------------------------------
Why duopoly.py uses a linear demand model (MLR) and not a Poisson regression.

The comparison mimics how the model is used online: per competition, train only on that
competition's own early seasons, then predict the following seasons.

- Setting A: train on seasons 1-25 (~2,500 days, the size of duopoly.py's training window),
  test on seasons 26-35.
- Setting B: train on season 1 only (~100 days, what we have after the first refit),
  test on seasons 2-11.

Features in both models: our price, (rival price - our price), season progress tau.
Only duopoly days (rival in stock, valid rival price) are used.

Usage:
    python analytics/compare_demand_models.py
"""

import glob
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.metrics import mean_poisson_deviance

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCALE = np.array([[100.0, 100.0, 1.0]])  # keeps the Poisson solver well conditioned


def load() -> pd.DataFrame:
    frames = [pd.read_csv(p).assign(ds=os.path.basename(os.path.dirname(p)))
              for p in glob.glob(os.path.join(ROOT_DIR, 'data', '*', '*.csv'))]
    d = pd.concat(frames, ignore_index=True)
    d = d[d.competitor_has_capacity.astype(bool) & (d.price_competitor < 200)].copy()
    d['diff'] = d.price_competitor - d.price
    d['tau'] = (d.selling_period - 1) / 99.0
    return d


def features(d: pd.DataFrame) -> np.ndarray:
    return d[['price', 'diff', 'tau']].values


def compare(d: pd.DataFrame, train_last: int, test_last: int, min_rows: int, min_sales: int) -> pd.DataFrame:
    rows = []
    for (ds, comp), g in d.groupby(['ds', 'competition_id']):
        train = g[g.selling_season <= train_last]
        test = g[(g.selling_season > train_last) & (g.selling_season <= test_last)]
        if len(train) < min_rows or len(test) < 100 or train.demand.sum() < min_sales:
            continue
        t0 = time.perf_counter()
        mlr = LinearRegression().fit(features(train), train.demand)
        t1 = time.perf_counter()
        poisson = PoissonRegressor(alpha=1e-6, max_iter=300).fit(features(train) / SCALE, train.demand)
        t2 = time.perf_counter()
        y_mlr = mlr.predict(features(test))
        y_poi = poisson.predict(features(test) / SCALE)
        rows.append({
            'ds': ds, 'competition': comp,
            'rmse_mlr': float(np.sqrt(np.mean((y_mlr - test.demand) ** 2))),
            'rmse_poisson': float(np.sqrt(np.mean((y_poi - test.demand) ** 2))),
            'dev_mlr': float(mean_poisson_deviance(test.demand, np.clip(y_mlr, 1e-3, None))),
            'dev_poisson': float(mean_poisson_deviance(test.demand, y_poi)),
            'mlr_negative_share': float((y_mlr < 0).mean()),
            'fit_ms_mlr': 1000 * (t1 - t0), 'fit_ms_poisson': 1000 * (t2 - t1),
        })
    return pd.DataFrame(rows)


def report(name: str, r: pd.DataFrame) -> None:
    n = len(r)
    print(f'\n{name}: {n} competitions')
    print(f"  RMSE      MLR {r.rmse_mlr.mean():.3f} | Poisson {r.rmse_poisson.mean():.3f}"
          f" | MLR better in {(r.rmse_mlr < r.rmse_poisson).sum()}/{n}")
    print(f"  Deviance  MLR {r.dev_mlr.mean():.3f} | Poisson {r.dev_poisson.mean():.3f}"
          f" | MLR better in {(r.dev_mlr < r.dev_poisson).sum()}/{n}")
    print(f"  MLR predicts negative sales on {100 * r.mlr_negative_share.mean():.1f}% of test days")
    print(f"  Median fit time: MLR {r.fit_ms_mlr.median():.1f} ms | Poisson {r.fit_ms_poisson.median():.1f} ms")


def main() -> None:
    warnings.filterwarnings('ignore')
    d = load()
    report('A) ~25 seasons of training data', compare(d, 25, 35, min_rows=500, min_sales=20))
    report('B) 1 season of training data', compare(d, 1, 11, min_rows=60, min_sales=5))


if __name__ == '__main__':
    main()
