#!/usr/bin/env python3
"""Reproduce the 12-country direct-observation robustness analysis.

The central experiment uses six-hour chronology. The temporal-resolution check
uses three-hour chronology. Firm capacity is calibrated on 2015 only using the
rule reported in the Supplementary Information.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('direct_model', ROOT/'scripts'/'direct_observation_model.py')
m = importlib.util.module_from_spec(spec)
sys.modules['direct_model'] = m
spec.loader.exec_module(m)

PROFILES = ROOT/'data'/'external'/'opsd'/'external_profiles_2015_2019.csv.gz'
OSM = ROOT/'data'/'external'/'pypsa_eur_osm_v06'
OUT = ROOT/'results'/'direct_observation'
OUT.mkdir(parents=True, exist_ok=True)
CENTRAL_RES = 6
CHECK_RES = 3

prof, active, countries, edges = m.load_inputs(PROFILES, OSM)
base = m.Portfolio()
chosen, cal = m.calibrate_firm(prof, active, countries, edges, base, CENTRAL_RES)
cal.to_csv(OUT/'external_2015_calibration_grid_reproduced.csv', index=False)
portfolio = m.Portfolio(**{**asdict(base), 'firm_peak_fraction': chosen})
central_s = m.H2_STORAGE_FRACTIONS[2]
central_p = m.RECONVERSION_MEAN_LOAD_FRACTIONS[-1]

# Six-hour mechanism experiment.
rows = []
for year in range(2016, 2020):
    cases = [
        None,
        m.H2Case('h2_rigid', 0.0, 0.0),
        m.H2Case('h2_flexible', central_s, 0.0),
        m.H2Case('h2_central', central_s, central_p),
    ]
    for case in cases:
        rows.append(m.solve_year(prof, active, countries, edges, year, portfolio, case, CENTRAL_RES))
mech = pd.DataFrame(rows)
mech.to_csv(OUT/'external_mechanism_test_6h_reproduced.csv', index=False)
piv = mech.pivot(index='year', columns='scenario', values='eue_gwh')

# Six-hour storage and reconversion slices for stressed years.
annual_load_twh = float(mech.annual_load_twh.iloc[0])
mean_load_gw = annual_load_twh * 1000 / 8760
storage_fractions = [0.0, m.H2_STORAGE_FRACTIONS[1], central_s, 2*central_s]
power_fractions = [0.0, m.RECONVERSION_MEAN_LOAD_FRACTIONS[0], m.RECONVERSION_MEAN_LOAD_FRACTIONS[1], central_p]
threshold_rows = []
for year in (2016, 2017, 2019):
    ref = float(piv.loc[year, 'reference'])
    for sf in storage_fractions:
        case = m.H2Case('storage_slice', sf, central_p)
        o = m.solve_year(prof, active, countries, edges, year, portfolio, case, CENTRAL_RES)
        threshold_rows.append({'year':year,'slice':'storage','fraction':sf,'physical_value':sf*annual_load_twh,'unit':'TWh H2','eue_gwh':o['eue_gwh'],'reference_eue_gwh':ref,'eue_avoided_gwh':ref-o['eue_gwh']})
    for pf in power_fractions:
        case = m.H2Case('power_slice', central_s, pf)
        o = m.solve_year(prof, active, countries, edges, year, portfolio, case, CENTRAL_RES)
        threshold_rows.append({'year':year,'slice':'power','fraction':pf,'physical_value':pf*mean_load_gw,'unit':'GW electric','eue_gwh':o['eue_gwh'],'reference_eue_gwh':ref,'eue_avoided_gwh':ref-o['eue_gwh']})
pd.DataFrame(threshold_rows).to_csv(OUT/'external_threshold_refined_6h_reproduced.csv', index=False)

# Three-hour chronology check.
resolution_rows = []
for year in (2016, 2017, 2019):
    ref3 = m.solve_year(prof, active, countries, edges, year, portfolio, None, CHECK_RES)
    h23 = m.solve_year(prof, active, countries, edges, year, portfolio, m.H2Case('h2_central', central_s, central_p), CHECK_RES)
    ref6 = float(piv.loc[year, 'reference']); h26 = float(piv.loc[year, 'h2_central'])
    for res, rv, hv in ((3, ref3['eue_gwh'], h23['eue_gwh']), (6, ref6, h26)):
        resolution_rows.append({'year':year,'resolution_h':res,'reference_eue_gwh':rv,'h2_eue_gwh':hv,'eue_avoided_gwh':rv-hv,'fraction_avoided':(rv-hv)/rv if rv>0 else np.nan})
pd.DataFrame(resolution_rows).to_csv(OUT/'external_resolution_sensitivity_reproduced.csv', index=False)

# Six-hour structural robustness.
rob = []
for td in (0.20, 0.35, 0.50):
    p = m.Portfolio(**{**asdict(portfolio), 'transmission_derate': td})
    for year in (2016, 2017, 2019):
        r = m.solve_year(prof, active, countries, edges, year, p, None, CENTRAL_RES)
        h = m.solve_year(prof, active, countries, edges, year, p, m.H2Case('h2_central', central_s, central_p), CENTRAL_RES)
        rob.append({'test':'transmission_derate','value':td,'year':year,'reference_eue_gwh':r['eue_gwh'],'h2_eue_gwh':h['eue_gwh'],'fraction_avoided':(r['eue_gwh']-h['eue_gwh'])/r['eue_gwh'] if r['eue_gwh']>0 else np.nan})
for year in (2016, 2017, 2019):
    r = m.solve_year(prof, active, countries, edges, year, portfolio, None, CENTRAL_RES)
    h = m.solve_year(prof, active, countries, edges, year, portfolio, m.H2Case('h2_central_national', central_s, central_p, 'national'), CENTRAL_RES)
    rob.append({'test':'h2_geography','value':'national','year':year,'reference_eue_gwh':r['eue_gwh'],'h2_eue_gwh':h['eue_gwh'],'fraction_avoided':(r['eue_gwh']-h['eue_gwh'])/r['eue_gwh'] if r['eue_gwh']>0 else np.nan})
for vre in (0.60, 0.80):
    b = m.Portfolio(vre_energy_share_target=vre, transmission_derate=0.35)
    cf, _ = m.calibrate_firm(prof, active, countries, edges, b, CENTRAL_RES)
    p = m.Portfolio(**{**asdict(b), 'firm_peak_fraction': cf})
    for year in (2016, 2017, 2019):
        r = m.solve_year(prof, active, countries, edges, year, p, None, CENTRAL_RES)
        h = m.solve_year(prof, active, countries, edges, year, p, m.H2Case('h2_central', central_s, central_p), CENTRAL_RES)
        rob.append({'test':'vre_energy_share','value':vre,'year':year,'reference_eue_gwh':r['eue_gwh'],'h2_eue_gwh':h['eue_gwh'],'fraction_avoided':(r['eue_gwh']-h['eue_gwh'])/r['eue_gwh'] if r['eue_gwh']>0 else np.nan})
pd.DataFrame(rob).to_csv(OUT/'external_structural_robustness_6h_reproduced.csv', index=False)

metadata = {
    'active_countries': active,
    'network_countries': countries,
    'n_active_countries': len(active),
    'n_network_countries': len(countries),
    'n_cross_border_corridors': len(edges),
    'calibration_year': 2015,
    'test_years': [2016, 2017, 2018, 2019],
    'central_resolution_h': CENTRAL_RES,
    'resolution_check_h': CHECK_RES,
    'chosen_firm_peak_fraction': chosen,
}
(OUT/'external_validation_design_reproduced.json').write_text(json.dumps(metadata, indent=2))
print(f'Reproduced 12-country outputs in {OUT}')
