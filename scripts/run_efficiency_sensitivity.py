#!/usr/bin/env python3
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import importlib.util,sys,argparse,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('h2lp',ROOT/'scripts'/'reduced_model.py');m=importlib.util.module_from_spec(spec);sys.modules['h2lp']=m;spec.loader.exec_module(m)
DATA=None

def init(data):
 global DATA;DATA=m.load_data(Path(data))

def work(year):
 yd=DATA[DATA.t.dt.year==year]
 sc=m.Scenario('H2 + reconversion (2030 efficiency)',transmission_scale=3.5,extra_firm_gw=5,hydrogen=True,h2_demand_gw=5,electrolyser_power_gw_each=10,h2_storage_gwh=1000,fuelcell_power_gw_each=5,electrolyser_efficiency=0.6963,fuelcell_efficiency=0.4869)
 o=m.solve_year(yd,sc);o['year']=year;return o
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--years',nargs='+',type=int,required=True);ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
 rows=[];t=time.time()
 with ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(str(ROOT/'data/real/MERRA2_subset_demand_wind_solar.csv'),)) as ex:
  fs={ex.submit(work,y):y for y in args.years}
  for f in as_completed(fs): rows.append(f.result())
 out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).sort_values('year').to_csv(out,index=False);print('saved',out,'n',len(rows),'sec',round(time.time()-t,1))
