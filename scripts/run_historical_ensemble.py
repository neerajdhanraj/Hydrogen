#!/usr/bin/env python3
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse, json, importlib.util, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
LP_PATH=ROOT/'scripts'/'reduced_model.py'
spec=importlib.util.spec_from_file_location('h2lp',LP_PATH); h2lp=importlib.util.module_from_spec(spec); sys.modules['h2lp']=h2lp; spec.loader.exec_module(h2lp)
DATA=None

def init_worker(data_path):
    global DATA
    DATA=h2lp.load_data(Path(data_path))

def task(year, sc_dict):
    sc=h2lp.Scenario(**sc_dict); yd=DATA[DATA.t.dt.year==year]
    out=h2lp.solve_year(yd,sc)
    out['year']=year
    return out

def scenarios_set(name):
    S=h2lp.Scenario
    common=dict(transmission_scale=3.5,extra_firm_gw=5.0)
    if name=='mechanisms':
        return [
            S('Electricity only',**common),
            S('H2 rigid load',**common,hydrogen=True,h2_demand_gw=5.0,electrolyser_power_gw_each=10.0,h2_storage_gwh=0.0,fuelcell_power_gw_each=0.0),
            S('H2 flexible production',**common,hydrogen=True,h2_demand_gw=5.0,electrolyser_power_gw_each=10.0,h2_storage_gwh=1000.0,fuelcell_power_gw_each=0.0),
            S('H2 + reconversion',**common,hydrogen=True,h2_demand_gw=5.0,electrolyser_power_gw_each=10.0,h2_storage_gwh=1000.0,fuelcell_power_gw_each=5.0),
        ]
    if name=='mechanisms_h2':
        return scenarios_set('mechanisms')[1:]
    if name=='counterfactuals':
        return [
            S('Electricity only',**common),
            S('H2 + reconversion',**common,hydrogen=True,h2_demand_gw=5.0,electrolyser_power_gw_each=10.0,h2_storage_gwh=1000.0,fuelcell_power_gw_each=5.0),
            S('Battery expansion',transmission_scale=3.5,extra_firm_gw=5.0,battery_power_scale=2.0,battery_energy_scale=2.0),
            S('Transmission +20%',transmission_scale=4.2,extra_firm_gw=5.0),
            S('Firm capacity +5 GW',transmission_scale=3.5,extra_firm_gw=10.0),
            S('Demand response 5%',transmission_scale=3.5,extra_firm_gw=5.0,demand_response_fraction=0.05),
        ]
    if name=='counterfactuals_alt':
        return scenarios_set('counterfactuals')[2:]
    if name=='reference': return [S('Electricity only',**common)]
    raise ValueError(name)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--set',choices=['reference','mechanisms','mechanisms_h2','counterfactuals','counterfactuals_alt'],default='mechanisms'); ap.add_argument('--workers',type=int,default=2); ap.add_argument('--years',nargs='*',type=int)
    args=ap.parse_args(); args.out.parent.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(args.data,usecols=['t']); yrs=sorted(pd.to_datetime(d.t).dt.year.unique().tolist()) if not args.years else args.years
    scs=scenarios_set(args.set); jobs=[(y,asdict(sc)) for sc in scs for y in yrs]
    print(f'Running {len(jobs)} solves: {len(yrs)} years x {len(scs)} scenarios with {args.workers} workers',flush=True)
    rows=[]; t0=time.time()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(str(args.data),)) as ex:
        futs={ex.submit(task,y,sc):(y,sc['name']) for y,sc in jobs}
        done=0
        for fut in as_completed(futs):
            y,nm=futs[fut]
            try: o=fut.result()
            except Exception as e: o={'year':y,'scenario':nm,'success':False,'message':repr(e)}
            rows.append(o); done+=1
            if done%5==0 or done==len(jobs):
                pd.DataFrame(rows).sort_values(['scenario','year']).to_csv(args.out,index=False)
                print(f'{done}/{len(jobs)} completed in {time.time()-t0:.1f}s',flush=True)
    pd.DataFrame(rows).sort_values(['scenario','year']).to_csv(args.out,index=False)
    print('saved',args.out,flush=True)
if __name__=='__main__': main()
