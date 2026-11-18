"""For each v6.7 match and season: O1 (perfect info) and O2 (typical rival path) policies on the match's fitted model.
Forward distribution of seats left -> P(sell out), P(sell out by day 90), expected days sold out, expected unsold seats."""
import json, sys, os, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0,"analytics"); sys.path.insert(0,".")
import oracle as o, testbed as tb
T,S,K=o.T,o.S,o.K
SEAS=int(os.environ.get("SEAS","20"))
def forward(theta, rival, POL, copier):
    dist=np.zeros(S+1); dist[S]=1.0
    so_by=np.zeros(T+1); stock_days=0.0; rev=0.0
    k=np.arange(K+1)
    for t in range(1,T+1):
        stock_days+=dist[0]
        price=POL[t]; r=price if copier else rival[t-1]
        pmf=o.pmf_table(tb.rate(theta,price,r,(t-1)/99.0))  # (S+1,K+1)
        new=np.zeros(S+1)
        for s in range(1,S+1):
            sold=np.minimum(k,s); 
            rev+=dist[s]*(pmf[s]*price[s]*sold).sum()
            np.add.at(new,s-sold,dist[s]*pmf[s])
        new[0]+=dist[0]; dist=new; so_by[t]=dist[0]
    return dict(rev=rev,p_so=dist[0],p_so90=so_by[90],stock_days=stock_days,unsold=(dist*np.arange(S+1)).sum())
def one(args):
    key,fit=args
    tb.set_data("v6.7"); D=tb.load() if not hasattr(one,"D") else one.D; one.D=D
    g=D[D.key==key]; th=np.array(fit["theta"]); cp=bool(fit["copier"])
    R=g.rival_eff.values.reshape(-1,T)[:SEAS]
    typ=np.nanmedian(np.where(R<200,R,np.nan),axis=0); typ=pd.Series(typ).ffill().bfill().fillna(1000.0).values
    _,PT=o.dp_solve(th,typ,o.GRID,cp)
    rows=[]
    for s in range(R.shape[0]):
        _,P1=o.dp_solve(th,R[s],o.GRID,cp)
        a=forward(th,R[s],P1,cp); b=forward(th,R[s],PT,cp)
        rows.append(dict(key=key,season=s+1,**{"o1_"+k:v for k,v in a.items()},**{"o2_"+k:v for k,v in b.items()}))
    return rows
if __name__=="__main__":
    F=json.load(open("analytics/demand_fits_v6.7.json"))
    with Pool(2) as p: out=p.map(one,sorted(F.items()),chunksize=2)
    df=pd.DataFrame([r for o_ in out for r in o_]); df.to_csv("analytics/review_v67/oracle_sellout_v67.csv",index=False)
    print(df.mean(numeric_only=True))
