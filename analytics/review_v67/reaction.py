import pandas as pd, numpy as np, glob, os
fr=[]
for p in sorted(glob.glob("data/v6.7/*/*.csv")):
    d=pd.read_csv(p,usecols=["competition_id","selling_season","selling_period","competitor_id","price_competitor","price","demand","competitor_has_capacity"])
    d["key"]=os.path.basename(os.path.dirname(p))+"/"+os.path.basename(p)+"/"+d.competition_id; fr.append(d)
d=pd.concat(fr).sort_values(["key","selling_season","selling_period"])
g=d.groupby(["key","selling_season"])
d["own_l1"]=g.price.shift(1); d["riv_l1"]=g.price_competitor.shift(1)
d["dr"]=d.price_competitor-d.riv_l1; d["do"]=d.own_l1-g.price.shift(2)
ok=(d.price<120)&(d.own_l1<120)&(d.price_competitor<120)&(d.riv_l1<120)&d.competitor_has_capacity&d.do.notna()&(d.do.abs()>0.3)
rows=[]
for k,x in d[ok].groupby("key"):
    if len(x)<200: rows.append(dict(key=k,n=len(x),beta=np.nan)); continue
    X=np.c_[np.ones(len(x)),x.do.values]; b=np.linalg.lstsq(X,x.dr.values,rcond=None)[0]
    rows.append(dict(key=k,n=len(x),beta=b[1],rival=x.competitor_id.iloc[0]))
r=pd.DataFrame(rows)
print(r.beta.describe())
print("rivals with reaction slope >0.3 (follow our price changes the next day):",(r.beta>0.3).sum(),"of",r.beta.notna().sum())
print(r[r.beta>0.3].groupby("rival").beta.agg(["count","mean"]).sort_values("mean",ascending=False).round(2).to_string())
r.to_csv("analytics/review_v67/v67_reaction.csv",index=False)
