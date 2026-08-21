#!/usr/bin/env python3
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse, importlib.util, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('h2lp',ROOT/'scripts'/'reduced_model.py'); m=importlib.util.module_from_spec(spec); sys.modules['h2lp']=m; spec.loader.exec_module(m)
DATA=None

def init(path):
 global DATA; DATA=m.load_data(Path(path))
def work(year,sd,fc):
 sc=m.Scenario(name=f'S{sd:g}_F{3*fc:g}',transmission_scale=3.5,extra_firm_gw=5.0,hydrogen=True,h2_demand_gw=5.0,electrolyser_power_gw_each=10.0,h2_storage_gwh=float(sd),fuelcell_power_gw_each=float(fc))
 out=m.solve_year(DATA[DATA.t.dt.year==year],sc); out.update(year=year,h2_storage_gwh=sd,fuelcell_total_gw=3*fc); return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--year',type=int,default=2006);ap.add_argument('--storage',nargs='+',type=float,required=True);ap.add_argument('--fc-each',nargs='+',type=float,default=[0,.5,1,2,3,5,8]);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
 jobs=[(args.year,s,f) for s in args.storage for f in args.fc_each]; rows=[]; t0=time.time()
 with ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(args.data,)) as ex:
  fut={ex.submit(work,*j):j for j in jobs}
  for i,z in enumerate(as_completed(fut),1):
   rows.append(z.result());
   if i%5==0 or i==len(jobs): print(i,'/',len(jobs),'in',round(time.time()-t0,1),'s',flush=True)
 pd.DataFrame(rows).sort_values(['h2_storage_gwh','fuelcell_total_gw']).to_csv(args.out,index=False); print('saved',args.out)
if __name__=='__main__':main()
