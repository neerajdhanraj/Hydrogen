#!/usr/bin/env python3
"""Reproduce the saved 2006 hourly traces used in Figure 3."""
from pathlib import Path
import pandas as pd
from reduced_model import load_data, solve_year, Scenario

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'real'/'MERRA2_subset_demand_wind_solar.csv'
OUT=ROOT/'results'/'reduced'

def main():
    df=load_data(DATA)
    yd=df[df.t.dt.year==2006].reset_index(drop=True)
    common=dict(transmission_scale=3.5,extra_firm_gw=5.0)
    cases=[
        ('timeseries_2006_baseline.csv', Scenario('Electricity only',**common)),
        ('timeseries_2006_h2.csv', Scenario('H2 + reconversion',**common,hydrogen=True,h2_demand_gw=5.0,
            electrolyser_power_gw_each=10.0,h2_storage_gwh=1000.0,fuelcell_power_gw_each=5.0)),
    ]
    OUT.mkdir(parents=True,exist_ok=True)
    for fn,sc in cases:
        r=solve_year(yd,sc,return_timeseries=True)
        if not r.get('success'):
            raise RuntimeError(r)
        ts=r.pop('_timeseries').rename(columns={'shed_gw':'shortage_gw'})
        ts.to_csv(OUT/fn,index=False)
        print(fn, 'EUE_GWh=', round(r['eue_gwh'],6), 'LOLE_h=', r['lole_h'])

if __name__=='__main__':
    main()
