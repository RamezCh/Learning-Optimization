import pandas as pd, numpy as np, glob, os
out=[]
for p in sorted(glob.glob("data/v6.7/*/*.csv")):
    d=pd.read_csv(p,usecols=["competition_id","selling_season","selling_period","competitor_id","price_competitor","price","demand"])
    d=d.sort_values(["competition_id","selling_season","selling_period"])
    g=d.groupby(["competition_id","selling_season"])
    d["own_prev"]=g.price.shift(1); d["own_prev2"]=g.price.shift(2)
    inf=(d.own_prev-d.own_prev2).abs()>0.3
    d["cp"]=((d.price_competitor-d.own_prev).abs()<=0.6)
    d["flat60"]=((d.price-60).abs()<0.05)&(d.selling_period<=30)
    s=d.groupby(["competition_id","selling_season"]).agg(flat60=("flat60","sum"),comp=("competitor_id","first"))
    for c,x in d.groupby("competition_id"):
        xi=x[inf.loc[x.index]]
        f60=s.loc[c].flat60
        first=f60[f60>=27].index.min() if (f60>=27).any() else None
        out.append(dict(file=os.path.basename(p),match=c,rival=x.competitor_id.iloc[0],copy_share_informative=xi.cp.mean() if len(xi) else np.nan,
           copy_share_all=x.cp.mean(),seasons_flat60=int((f60>=27).sum()),first_flat60=first))
o=pd.DataFrame(out)
print(o[o.seasons_flat60>0].sort_values("seasons_flat60",ascending=False).to_string())
print("matches stuck at $60:",(o.seasons_flat60>0).sum())
o.to_csv("analytics/review_v67/v67_copier_matches.csv",index=False)
