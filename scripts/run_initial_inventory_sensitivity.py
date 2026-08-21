#!/usr/bin/env python3
"""Reproduce the 2006 fixed start/end hydrogen-inventory sensitivity."""
from pathlib import Path
import importlib.util
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reduced_model', ROOT/'scripts'/'reduced_model.py')
model = importlib.util.module_from_spec(spec)
sys.modules['reduced_model'] = model
spec.loader.exec_module(model)

data = model.load_data(ROOT/'data'/'real'/'MERRA2_subset_demand_wind_solar.csv')
year_data = data[data.t.dt.year == 2006]
rows = []
for fraction in (0.0, 0.5, 1.0):
    scenario = model.Scenario(
        'H2 + reconversion',
        transmission_scale=3.5,
        extra_firm_gw=5.0,
        hydrogen=True,
        h2_demand_gw=5.0,
        electrolyser_power_gw_each=10.0,
        h2_storage_gwh=1000.0,
        fuelcell_power_gw_each=5.0,
        h2_initial_soc_fraction=fraction,
    )
    result = model.solve_year(year_data, scenario)
    rows.append({
        'initial_fraction': fraction,
        'eue_gwh': result['eue_gwh'],
        'lole_h': result['lole_h'],
    })
out = ROOT/'results'/'reduced'/'h2_initial_inventory_2006_sensitivity.csv'
pd.DataFrame(rows).to_csv(out, index=False)
print(out)
