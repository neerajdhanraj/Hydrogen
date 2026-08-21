#!/usr/bin/env python3
from pathlib import Path
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results'/'reduced'
rh=pd.read_csv(D/'limited_foresight_top6_stress.csv');m=pd.read_csv(D/'mechanism_test_ensemble.csv')
years=sorted(rh.year.unique());ann=m[m.year.isin(years)].pivot(index='year',columns='scenario',values='eue_gwh');ann_benefit=ann['Electricity only']-ann['H2 + reconversion']
rows=[];detail=[]
for h,g in rh.groupby('lookahead_h'):
 p=g.pivot(index='year',columns='system',values='eue_gwh').sort_index();p['avoided_gwh']=p.electricity-p.h2;p['annual_pf_avoided_gwh']=ann_benefit.reindex(p.index);p['retention_fraction']=p.avoided_gwh/p.annual_pf_avoided_gwh
 for y,r in p.iterrows(): detail.append(dict(year=y,lookahead_h=h,rolling_baseline_eue_gwh=r.electricity,rolling_h2_eue_gwh=r.h2,rolling_eue_avoided_gwh=r.avoided_gwh,annual_pf_eue_avoided_gwh=r.annual_pf_avoided_gwh,benefit_retention_fraction=r.retention_fraction))
 rows.append(dict(lookahead_h=h,n_years=len(p),improved_years=int((p.avoided_gwh>1e-6).sum()),worse_years=int((p.avoided_gwh<-1e-6).sum()),mean_eue_avoided_gwh=float(p.avoided_gwh.mean()),sum_eue_avoided_gwh=float(p.avoided_gwh.sum()),annual_pf_sum_eue_avoided_gwh=float(p.annual_pf_avoided_gwh.sum()),aggregate_benefit_retention_fraction=float(p.avoided_gwh.sum()/p.annual_pf_avoided_gwh.sum())))
pd.DataFrame(detail).to_csv(D/'operational_foresight_detail.csv',index=False);pd.DataFrame(rows).to_csv(D/'operational_foresight_summary.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
