#!/usr/bin/env python3
"""Receding-horizon operational robustness for the held-out weather years.

The model sees only 3 or 7 days of held-out weather at each decision point and
commits the first 24 h. Storage states are carried forward. A seasonal terminal
reserve floor is learned solely from 1980-1998 climatology, so no weather beyond
the current forecast window leaks into operations.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse,importlib.util,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('h2lp',ROOT/'scripts'/'reduced_model.py');m=importlib.util.module_from_spec(spec);sys.modules['h2lp']=m;spec.loader.exec_module(m)
DATA=None

def init(path):
 global DATA;DATA=m.load_data(Path(path))
def scenario(system):
 if system=='electricity':return m.Scenario('Electricity only',transmission_scale=3.5,extra_firm_gw=5.0)
 return m.Scenario('H2 + reconversion',transmission_scale=3.5,extra_firm_gw=5.0,hydrogen=True,h2_demand_gw=5.0,electrolyser_power_gw_each=10.0,h2_storage_gwh=1000.0,fuelcell_power_gw_each=5.0)
def run_one(year,system,lookahead_h,commit_h=24):
 yd=DATA[DATA.t.dt.year==year].reset_index(drop=True);sc=scenario(system)
 pol=pd.read_csv(ROOT/'results'/'reduced'/f'calibration_climatology_reserve_policy_{system}.csv')
 batt={n:float(pol.loc[8759,f'battery_soc_r{n}_gwh_target']) for n in (2,5,6)};h2=float(pol.loc[8759,'h2_inventory_gwh_target']) if sc.hydrogen else None
 eue=0.;lole=0;peak=0.;fc=0.;el=0.;steps=0;records=[]
 for start in range(0,8760,commit_h):
  end=min(start+lookahead_h,8760);commit_end=min(start+commit_h,8760);block=yd.iloc[start:end];target_idx=end-1
  bterm={n:float(pol.loc[target_idx,f'battery_soc_r{n}_gwh_target']) for n in (2,5,6)};hterm=float(pol.loc[target_idx,'h2_inventory_gwh_target']) if sc.hydrogen else None
  o=m.solve_year(block,sc,return_timeseries=True,cyclic=False,initial_battery_soc_gwh=batt,terminal_battery_soc_min_gwh=bterm,initial_h2_soc_gwh=h2,terminal_h2_soc_min_gwh=hterm)
  if not o.get('success'): raise RuntimeError(f'window {start}:{end} failed: {o.get("message")}')
  ts=o['_timeseries'];ncommit=commit_end-start;c=ts.iloc[:ncommit];shed=c.shed_gw.to_numpy(float);eue+=float(shed.sum());lole+=int((shed>1e-6).sum());peak=max(peak,float(shed.max(initial=0)))
  batt={n:float(c[f'battery_soc_r{n}_gwh'].iloc[-1]) for n in (2,5,6)}
  if sc.hydrogen:
   h2=float(c.h2_inventory_gwh.iloc[-1]);fc+=float(c.fuelcell_gw.sum());el+=float(c.electrolysis_gw.sum())
  records.append((start,commit_end,float(shed.sum()),h2 if sc.hydrogen else np.nan));steps+=1
 return dict(year=year,system=system,lookahead_h=lookahead_h,commit_h=commit_h,steps=steps,eue_gwh=eue,lole_h=lole,max_shortfall_gw=peak,fuelcell_output_gwh=fc,electrolysis_gwh=el,final_h2_inventory_gwh=h2 if sc.hydrogen else np.nan)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--years',nargs='+',type=int,required=True);ap.add_argument('--lookahead',nargs='+',type=int,default=[72,168]);ap.add_argument('--commit-h',type=int,default=24);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();jobs=[(y,s,h,a.commit_h) for y in a.years for h in a.lookahead for s in ('electricity','h2')];rows=[];t0=time.time()
 with ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(a.data,)) as ex:
  fut={ex.submit(run_one,*j):j for j in jobs}
  for i,z in enumerate(as_completed(fut),1):
   j=fut[z]
   try:r=z.result()
   except Exception as e:r={'year':j[0],'system':j[1],'lookahead_h':j[2],'commit_h':j[3],'error':repr(e)}
   rows.append(r);print(i,'/',len(jobs),'in',round(time.time()-t0,1),'s',flush=True)
 pd.DataFrame(rows).sort_values(['year','lookahead_h','system']).to_csv(a.out,index=False);print('saved',a.out)
if __name__=='__main__':main()
