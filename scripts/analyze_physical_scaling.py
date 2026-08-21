#!/usr/bin/env python3
from __future__ import annotations
import os, sys, importlib.util
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, linregress
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('h2lp',ROOT/'scripts'/'reduced_model.py')
m=importlib.util.module_from_spec(spec);sys.modules['h2lp']=m;spec.loader.exec_module(m)
DATA_PATH=ROOT/'data'/'real'/'MERRA2_subset_demand_wind_solar.csv'
DATA=None
COMMON=dict(transmission_scale=3.5,extra_firm_gw=5.0)

def init():
    global DATA; DATA=m.load_data(DATA_PATH)

def longest_true_run(x):
    best=cur=0
    for v in np.asarray(x,bool):
        cur=cur+1 if v else 0; best=max(best,cur)
    return int(best)

def max_event_stats(short):
    a=np.asarray(short,float); runs=[]; st=None
    for i,v in enumerate(a>1e-6):
        if v and st is None: st=i
        if (not v or i==len(a)-1) and st is not None:
            en=i if (v and i==len(a)-1) else i-1
            seg=a[st:en+1]; runs.append((float(seg.sum()),int(len(seg)),float(seg.max())))
            st=None
    if not runs: return 0.,0,0.
    # energy-dominant critical event
    return max(runs,key=lambda z:z[0])

def solve_ref(y):
    yd=DATA[DATA.t.dt.year==y]
    sc=m.Scenario('Electricity only',**COMMON)
    o=m.solve_year(yd,sc,return_timeseries=True)
    ts=o.pop('_timeseries'); return y,o,ts

def h2_eue(y, storage, power):
    yd=DATA[DATA.t.dt.year==y]
    sc=m.Scenario('threshold',**COMMON,hydrogen=True,h2_demand_gw=5.0,electrolyser_power_gw_each=10.0,
                  h2_storage_gwh=float(storage),fuelcell_power_gw_each=float(power)/3.0)
    return float(m.solve_year(yd,sc)['eue_gwh'])

def bisect_storage(y, hi=2500., tol=20.):
    if h2_eue(y,hi,15.)>1e-5: return np.nan
    lo=0.
    while hi-lo>tol:
        mid=(lo+hi)/2
        if h2_eue(y,mid,15.)<=1e-5: hi=mid
        else: lo=mid
    return hi/1000.

def bisect_power(y, storage=2500., hi=20., tol=.2):
    if h2_eue(y,storage,hi)>1e-5: return np.nan
    lo=0.
    while hi-lo>tol:
        mid=(lo+hi)/2
        if h2_eue(y,storage,mid)<=1e-5: hi=mid
        else: lo=mid
    return hi

def work(y, low_vre_thr):
    yy,o,ts=solve_ref(y)
    yd=DATA[DATA.t.dt.year==y]
    # aggregate VRE potential GW from installed capacities in the benchmark
    vre=np.zeros(len(yd))
    for _,(_,cap,col) in m.VRE.items(): vre += cap*yd[col].to_numpy(float)
    event_energy,event_dur,event_peak=max_event_stats(ts.shed_gw)
    out=dict(year=y,reference_eue_gwh=o['eue_gwh'],critical_event_energy_gwh=event_energy,
             critical_event_duration_h=event_dur,peak_shortfall_gw=event_peak,
             renewable_drought_duration_h=longest_true_run(vre<low_vre_thr))
    if o['eue_gwh']>1e-6:
        out['critical_storage_zero_shortage_twh']=bisect_storage(y)
        out['critical_reconversion_zero_shortage_gw']=bisect_power(y)
    else:
        out['critical_storage_zero_shortage_twh']=np.nan;out['critical_reconversion_zero_shortage_gw']=np.nan
    return out

def run_duration(y, duration_h):
    # Keep battery discharge power at the original 60 GW system total, vary energy duration.
    # Original benchmark energy = 300 GWh = 5 h at 60 GW, so scale = duration/5.
    scale=duration_h/5.0
    yd=DATA[DATA.t.dt.year==y]
    sc=m.Scenario(f'Battery {duration_h:g} h',**COMMON,battery_power_scale=1.0,battery_energy_scale=scale)
    o=m.solve_year(yd,sc); return dict(year=y,technology=f'Battery {duration_h:g} h',duration_h=duration_h,eue_gwh=o['eue_gwh'],lole_h=o['lole_h'])

def run_h2(y):
    yd=DATA[DATA.t.dt.year==y]
    sc=m.Scenario('Hydrogen storage + reconversion',**COMMON,hydrogen=True,h2_demand_gw=5.0,electrolyser_power_gw_each=10.0,h2_storage_gwh=1000.,fuelcell_power_gw_each=5.0)
    o=m.solve_year(yd,sc); return dict(year=y,technology='Hydrogen 1 TWh + 15 GW',duration_h=1000/15,eue_gwh=o['eue_gwh'],lole_h=o['lole_h'])

def main():
    init()
    calib=DATA[DATA.t.dt.year<=1998]
    vre=np.zeros(len(calib))
    for _,(_,cap,col) in m.VRE.items(): vre += cap*calib[col].to_numpy(float)
    low_vre_thr=float(np.percentile(vre,10))
    years=list(range(1999,2018)); rows=[]
    with ProcessPoolExecutor(max_workers=4,initializer=init) as ex:
        fut={ex.submit(work,y,low_vre_thr):y for y in years}
        for i,f in enumerate(as_completed(fut),1):
            rows.append(f.result()); print('scaling',i,'/',len(fut),flush=True)
    df=pd.DataFrame(rows).sort_values('year')
    outd=ROOT/'results'/'reduced';outd.mkdir(parents=True,exist_ok=True)
    df.to_csv(outd/'physical_scaling_weather_years.csv',index=False)
    z=df.dropna(subset=['critical_storage_zero_shortage_twh']).copy()
    metrics=[]
    for target,pred in [('critical_storage_zero_shortage_twh','critical_event_energy_gwh'),
                        ('critical_storage_zero_shortage_twh','critical_event_duration_h'),
                        ('critical_storage_zero_shortage_twh','renewable_drought_duration_h'),
                        ('critical_reconversion_zero_shortage_gw','peak_shortfall_gw'),
                        ('critical_reconversion_zero_shortage_gw','critical_event_energy_gwh')]:
        x=z[pred].to_numpy(float);y=z[target].to_numpy(float)
        sp=spearmanr(x,y);pe=pearsonr(x,y);lr=linregress(x,y)
        metrics.append(dict(target=target,predictor=pred,n=len(z),spearman_rho=sp.statistic,spearman_p=sp.pvalue,
                            pearson_r=pe.statistic,pearson_p=pe.pvalue,slope=lr.slope,intercept=lr.intercept,r_squared=lr.rvalue**2))
    pd.DataFrame(metrics).to_csv(outd/'physical_scaling_statistics.csv',index=False)
    # duration comparison
    jobs=[]
    with ProcessPoolExecutor(max_workers=4,initializer=init) as ex:
        for y in years:
            for d in [4.,24.,100.]: jobs.append(ex.submit(run_duration,y,d))
            jobs.append(ex.submit(run_h2,y))
        dr=[]
        for i,f in enumerate(as_completed(jobs),1):
            dr.append(f.result());
            if i%10==0: print('duration',i,'/',len(jobs),flush=True)
    dur=pd.DataFrame(dr).sort_values(['technology','year'])
    ref=pd.read_csv(outd/'mechanism_test_ensemble.csv'); ref=ref[ref.scenario=='Electricity only'][['year','eue_gwh']].rename(columns={'eue_gwh':'reference_eue_gwh'})
    dur=dur.merge(ref,on='year');dur['eue_avoided_gwh']=dur.reference_eue_gwh-dur.eue_gwh
    dur.to_csv(outd/'duration_storage_comparison.csv',index=False)
    summ=dur.groupby(['technology','duration_h']).agg(mean_eue_gwh=('eue_gwh','mean'),mean_eue_avoided_gwh=('eue_avoided_gwh','mean'),improved_years=('eue_avoided_gwh',lambda s:int((s>1e-6).sum())),zero_shortage_years=('eue_gwh',lambda s:int((s<=1e-6).sum()))).reset_index()
    summ.to_csv(outd/'duration_storage_summary.csv',index=False)
    print('\nSCALING\n',pd.DataFrame(metrics).to_string(index=False))
    print('\nDURATION\n',summ.to_string(index=False))
if __name__=='__main__': main()
