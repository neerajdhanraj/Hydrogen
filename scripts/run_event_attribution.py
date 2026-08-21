#!/usr/bin/env python3
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import importlib.util, sys, itertools, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('h2lp',ROOT/'scripts'/'reduced_model.py'); m=importlib.util.module_from_spec(spec);sys.modules['h2lp']=m;spec.loader.exec_module(m)
DATA_PATH=ROOT/'data'/'real'/'MERRA2_subset_demand_wind_solar.csv'
DATA=None; CLIM=None; ACT=None
WCOL=['wind_region2','wind_region5','wind_region6']; SCOL=['solar_region2','solar_region5','solar_region6']; DCOL=['demand_region2','demand_region4','demand_region5']

def init():
 global DATA,CLIM,ACT
 DATA=m.load_data(DATA_PATH)
 train=DATA[(DATA.t.dt.year>=1980)&(DATA.t.dt.year<=1998)].copy(); train['hoy']=train.groupby(train.t.dt.year).cumcount()
 CLIM=train.groupby('hoy')[WCOL+SCOL+DCOL].mean().reset_index(drop=True)
 ACT=DATA[DATA.t.dt.year==2006].reset_index(drop=True).copy()

def work(flags,h2):
 w,s,d=flags; x=ACT.copy()
 if not w: x[WCOL]=CLIM[WCOL].to_numpy()
 if not s: x[SCOL]=CLIM[SCOL].to_numpy()
 if not d: x[DCOL]=CLIM[DCOL].to_numpy()
 if h2:
  sc=m.Scenario('H2 + reconversion',transmission_scale=3.5,extra_firm_gw=5,hydrogen=True,h2_demand_gw=5,electrolyser_power_gw_each=10,h2_storage_gwh=1000,fuelcell_power_gw_each=5)
 else:
  sc=m.Scenario('Electricity only',transmission_scale=3.5,extra_firm_gw=5)
 o=m.solve_year(x,sc);o.update(W=w,S=s,D=d,system='H2 + reconversion' if h2 else 'Electricity only');return o

def shapley(df):
 # exact Shapley for 3 binary features relative 000; v(S)=EUE at combination S
 import math
 vals={(int(r.W),int(r.S),int(r.D)):r.eue_gwh for r in df.itertuples()}
 names=['wind','solar','demand']; out={}
 for i,name in enumerate(names):
  phi=0.0
  others=[j for j in range(3) if j!=i]
  for k in range(0,3):
   for subset in itertools.combinations(others,k):
    a=[0,0,0]
    for j in subset:a[j]=1
    b=a.copy();b[i]=1
    weight=math.factorial(k)*math.factorial(3-k-1)/math.factorial(3)
    phi+=weight*(vals[tuple(b)]-vals[tuple(a)])
  out[name]=phi
 return out

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['electricity','h2'],required=True);args=ap.parse_args()
 hmode=args.mode=='h2'; out=ROOT/'results'/'reduced'/f'observed_factorial_2006_{args.mode}.csv';jobs=[(f,hmode) for f in itertools.product([0,1],repeat=3)];rows=[];t0=time.time()
 with ProcessPoolExecutor(max_workers=4,initializer=init) as ex:
  fut={ex.submit(work,*j):j for j in jobs}
  for i,z in enumerate(as_completed(fut),1):
   rows.append(z.result());
   if i%4==0: print(i,'/',len(jobs),'in',round(time.time()-t0,1),'s',flush=True)
 df=pd.DataFrame(rows).sort_values(['system','W','S','D']);df.to_csv(out,index=False)
 for system,g in df.groupby('system'):
  sh=shapley(g); base=float(g.query('W==0 and S==0 and D==0').eue_gwh.iloc[0]); full=float(g.query('W==1 and S==1 and D==1').eue_gwh.iloc[0])
  print(system,'climatology',round(base,1),'actual',round(full,1),'delta',round(full-base,1),'Shapley', {k:round(v,1) for k,v in sh.items()},flush=True)
 print('saved',out)
