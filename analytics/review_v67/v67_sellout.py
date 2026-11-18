import pandas as pd, glob, os, numpy as np
fr=[]
for p in sorted(glob.glob("data/v6.7/*/*.csv")):
    f=pd.read_csv(p,usecols=["competition_id","selling_season","selling_period","competitor_id","price_competitor","price","demand","competitor_has_capacity"])
    f["file"]=os.path.basename(os.path.dirname(p))+"/"+os.path.basename(p); fr.append(f)
d=pd.concat(fr).sort_values(["file","competition_id","selling_season","selling_period"])
g=d.groupby(["file","competition_id","selling_season"])
d["cum"]=g.demand.cumsum(); d["left_before"]=80-(d.cum-d.demand)
s=g.agg(sold=("demand","sum"),rev=("price",lambda x:0)).reset_index()
d["rev"]=d.price*d.demand
s["rev"]=g.rev.sum().values
so=d[d.cum>=80].groupby(["file","competition_id","selling_season"]).selling_period.min().rename("so_day")
s=s.merge(so.reset_index(),how="left")
s["so"]=s.so_day.notna(); s["stockout_days"]=np.where(s.so,100-s.so_day,0)
print("sell-out seasons %.1f%%"%(100*s.so.mean()))
print("sell-out day distribution:\n",s.so_day.value_counts(bins=[0,50,70,80,90,95,97,98,99,100]).sort_index())
print("share of all season-days spent sold out: %.2f%%"%(100*s.stockout_days.sum()/(len(s)*100)))
print("seats left at end (unsold) distribution:"); print((80-s.sold).describe())
print("unsold seats mean %.1f, seasons with >=20 unsold %.1f%%"%((80-s.sold).mean(),100*((80-s.sold)>=20).mean()))
# price on the sell-out day and last 5 days before sellout vs our typical last-days price in seasons not sold out
last=d[d.selling_period>=91]
last=last.merge(s[["file","competition_id","selling_season","so"]])
print("mean price days 91-100 (while seats left), sold-out seasons vs not:")
print(last[last.left_before>0].groupby("so").price.describe())
# how many seats sold on the sellout day: excess demand indicator -> if demand==left_before exactly on so day, demand may have been censored
sod=d[(d.cum>=80)&(d.cum-d.demand<80)]
print("sell-out day: seats bought == seats left (censored?) share:",(sod.demand==sod.left_before).mean())
print("rival capacity: share of seasons rival sold out:", d.groupby(["file","competition_id","selling_season"]).competitor_has_capacity.min().eq(False).mean())
s.to_csv("analytics/review_v67/v67_seasons.csv",index=False)
# by season index (learning): sellout rate by season bucket
s["sb"]=pd.cut(s.selling_season,[0,1,5,20,50,100])
print(s.groupby("sb").agg(so=("so","mean"),soday=("so_day","median"),rev=("rev","mean"),sold=("sold","mean")))
