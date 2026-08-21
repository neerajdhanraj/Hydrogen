#!/usr/bin/env python3
"""Assemble cross-weather storage/reconversion phase slices and uncertainty summaries."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results'/'reduced'
rng=np.random.default_rng(20260814)

def ci(x,n=20000):
    x=np.asarray(x,float); b=rng.choice(x,(n,len(x)),replace=True).mean(1)
    return float(x.mean()),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))

m=pd.read_csv(D/'mechanism_test_ensemble.csv')
b=m[m.scenario=='Electricity only'][['year','eue_gwh']].rename(columns={'eue_gwh':'reference_eue_gwh'})
snew=pd.read_csv(D/'cross_weather_storage_slice_new.csv');pnew=pd.read_csv(D/'cross_weather_power_slice_new.csv')

def get_existing(name): return m[m.scenario==name][['year','eue_gwh']]
# Storage slice at 15 GW reconversion
parts=[]
for s,df in [
 (0.0,get_existing('H2 rigid load')),
 (0.5,snew[snew.h2_storage_gwh==500][['year','eue_gwh']]),
 (1.0,get_existing('H2 + reconversion')),
 (2.0,snew[snew.h2_storage_gwh==2000][['year','eue_gwh']]),
]:
 z=b.merge(df,on='year');z['storage_twh']=s;z['reconversion_gw']=15.0;z['eue_avoided_gwh']=z.reference_eue_gwh-z.eue_gwh;parts.append(z)
storage=pd.concat(parts,ignore_index=True).sort_values(['year','storage_twh'])
storage.to_csv(D/'cross_weather_storage_phase.csv',index=False)
# Power slice at 1 TWh storage
parts=[]
for p,df in [
 (0.0,get_existing('H2 flexible production')),
 (3.0,pnew[pnew.fuelcell_total_gw==3][['year','eue_gwh']]),
 (9.0,pnew[pnew.fuelcell_total_gw==9][['year','eue_gwh']]),
 (15.0,get_existing('H2 + reconversion')),
]:
 z=b.merge(df,on='year');z['storage_twh']=1.0;z['reconversion_gw']=p;z['eue_avoided_gwh']=z.reference_eue_gwh-z.eue_gwh;parts.append(z)
power=pd.concat(parts,ignore_index=True).sort_values(['year','reconversion_gw'])
power.to_csv(D/'cross_weather_power_phase.csv',index=False)

summ=[]
for kind,df,xcol in [('storage',storage,'storage_twh'),('power',power,'reconversion_gw')]:
 for x,g in df.groupby(xcol):
    mean,lo,hi=ci(g.eue_avoided_gwh)
    summ.append(dict(slice=kind,value=float(x),mean_eue_gwh=float(g.eue_gwh.mean()),mean_eue_avoided_gwh=mean,ci95_low_gwh=lo,ci95_high_gwh=hi,
                     improved_years=int((g.eue_avoided_gwh>1e-6).sum()),worse_years=int((g.eue_avoided_gwh<-1e-6).sum()),neutral_years=int((g.eue_avoided_gwh.abs()<=1e-6).sum()),
                     zero_shortage_years=int((g.eue_gwh<=1e-6).sum())))
pd.DataFrame(summ).to_csv(D/'cross_weather_phase_summary.csv',index=False)

# Grid-resolved sign and saturation brackets for the 12 years with reference shortage.
rows=[]
for y in sorted(b.year):
    e0=float(b.loc[b.year==y,'reference_eue_gwh'].iloc[0])
    gs=storage[storage.year==y].sort_values('storage_twh')
    gp=power[power.year==y].sort_values('reconversion_gw')
    if e0>1e-6:
        benef=gs[gs.eue_avoided_gwh>1e-6]
        if len(benef):
            hi=float(benef.storage_twh.iloc[0]); prev=gs[gs.storage_twh<hi]
            lo=float(prev.storage_twh.iloc[-1]) if len(prev) else 0.0
        else: lo=hi=np.nan
        zero=gs[gs.eue_gwh<=1e-6]; zero_s=float(zero.storage_twh.iloc[0]) if len(zero) else np.nan
        final=float(gp[gp.reconversion_gw==15].eue_avoided_gwh.iloc[0])
        sat=gp[(final-gp.eue_avoided_gwh).abs()<=max(1.0,0.01*max(final,1.0))]
        sat_p=float(sat.reconversion_gw.iloc[0]) if len(sat) else np.nan
    else:
        lo=hi=zero_s=sat_p=np.nan
    rows.append(dict(year=int(y),reference_eue_gwh=e0,storage_benefit_bracket_low_twh=lo,storage_benefit_bracket_high_twh=hi,
                     minimum_tested_storage_zero_shortage_twh=zero_s,reconversion_saturation_grid_gw=sat_p))
br=pd.DataFrame(rows);br.to_csv(D/'cross_weather_threshold_brackets.csv',index=False)
print(pd.read_csv(D/'cross_weather_phase_summary.csv').to_string(index=False))
print('\nThreshold brackets for stressed years:\n',br[br.reference_eue_gwh>0].to_string(index=False))
