#!/usr/bin/env python3
"""Generate calibration-only seasonal reserve trajectories for limited-foresight tests."""
from pathlib import Path
import importlib.util,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('h2lp',ROOT/'scripts'/'reduced_model.py');m=importlib.util.module_from_spec(spec);sys.modules['h2lp']=m;spec.loader.exec_module(m)
d=m.load_data(ROOT/'data'/'real'/'MERRA2_subset_demand_wind_solar.csv')
train=d[(d.t.dt.year>=1980)&(d.t.dt.year<=1998)].copy();train['hoy']=train.groupby(train.t.dt.year).cumcount()
cols=[c for c in d.columns if c!='t'];clim=train.groupby('hoy')[cols].mean().reset_index(drop=True);clim.insert(0,'t',pd.date_range('2001-01-01',periods=8760,freq='h'))
scenarios=[
 ('electricity',m.Scenario('Calibration climatology electricity',transmission_scale=3.5,extra_firm_gw=5.0)),
 ('h2',m.Scenario('Calibration climatology H2',transmission_scale=3.5,extra_firm_gw=5.0,hydrogen=True,h2_demand_gw=5.0,electrolyser_power_gw_each=10.0,h2_storage_gwh=1000.0,fuelcell_power_gw_each=5.0)),
]
for tag,sc in scenarios:
 o=m.solve_year(clim,sc,return_timeseries=True);ts=o['_timeseries'];out=pd.DataFrame({'hoy':np.arange(8760)})
 for n in (2,5,6):out[f'battery_soc_r{n}_gwh_target']=ts[f'battery_soc_r{n}_gwh']
 if sc.hydrogen: out['h2_inventory_gwh_target']=ts.h2_inventory_gwh
 out['climatology_eue_gwh']=o['eue_gwh'];path=ROOT/'results'/'reduced'/f'calibration_climatology_reserve_policy_{tag}.csv';out.to_csv(path,index=False)
 print('saved',path,'EUE',o['eue_gwh'])
