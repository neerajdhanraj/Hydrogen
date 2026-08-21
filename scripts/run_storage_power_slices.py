#!/usr/bin/env python3
"""Solve sparse, interpretable cross-weather H2 phase slices.

The full 2006 2-D phase map is retained. Across all 19 held-out years we solve
additional cells that identify whether the sign change survives weather-year
variation without requiring hundreds of redundant annual LPs:
  - storage slice at 15 GW reconversion: 0, 0.5, 1, 2 TWh;
  - power slice at 1 TWh storage: 0, 3, 9, 15 GW.
The 0 and central 1 TWh/15 GW cells are reused from the solved mechanism ensemble;
this runner generates only the missing cells requested by --mode.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse, importlib.util, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('h2lp',ROOT/'scripts'/'reduced_model.py')
m=importlib.util.module_from_spec(spec);sys.modules['h2lp']=m;spec.loader.exec_module(m)
DATA=None

def init(path):
    global DATA; DATA=m.load_data(Path(path))

def work(year,storage_gwh,power_total_gw):
    sc=m.Scenario(name=f'S{storage_gwh:g}_P{power_total_gw:g}',transmission_scale=3.5,extra_firm_gw=5.0,
                  hydrogen=True,h2_demand_gw=5.0,electrolyser_power_gw_each=10.0,
                  h2_storage_gwh=float(storage_gwh),fuelcell_power_gw_each=float(power_total_gw)/3.0)
    o=m.solve_year(DATA[DATA.t.dt.year==year],sc); o.update(year=year,h2_storage_gwh=storage_gwh,fuelcell_total_gw=power_total_gw); return o

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--mode',choices=['storage','power'],required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--years',nargs='*',type=int,default=list(range(1999,2018)));a=ap.parse_args()
    cells=[(500.0,15.0),(2000.0,15.0)] if a.mode=='storage' else [(1000.0,3.0),(1000.0,9.0)]
    jobs=[(y,s,p) for y in a.years for s,p in cells];rows=[];t0=time.time();out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(a.data,)) as ex:
        fut={ex.submit(work,*j):j for j in jobs}
        for i,z in enumerate(as_completed(fut),1):
            j=fut[z]
            try:r=z.result()
            except Exception as e:r={'year':j[0],'h2_storage_gwh':j[1],'fuelcell_total_gw':j[2],'success':False,'message':repr(e)}
            rows.append(r)
            if i%4==0 or i==len(jobs): print(f'{i}/{len(jobs)} in {time.time()-t0:.1f}s',flush=True)
    pd.DataFrame(rows).sort_values(['year','h2_storage_gwh','fuelcell_total_gw']).to_csv(out,index=False);print('saved',out)
if __name__=='__main__':main()
