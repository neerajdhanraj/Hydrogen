#!/usr/bin/env python3
"""Check frozen numerical outputs against headline values reported in the manuscript."""
from pathlib import Path
import json
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT/'results'
failures=[]

def check(label, actual, expected, tol):
    if not math.isfinite(float(actual)) or abs(float(actual)-float(expected)) > tol:
        failures.append(f'{label}: {actual} != {expected} ± {tol}')

def check_equal(label, actual, expected):
    if actual != expected: failures.append(f'{label}: {actual} != {expected}')

m = pd.read_csv(R/'reduced'/'mechanism_statistical_summary.csv').set_index('scenario')
check('reference mean EUE', m.loc['Electricity only','mean_eue_gwh'], 197.8, 0.06)
check('reference median EUE', m.loc['Electricity only','median_eue_gwh'], 109.8, 0.06)
check('reference mean shortage hours', m.loc['Electricity only','mean_lole_h'], 13.47, 0.01)
check('rigid mean EUE', m.loc['H2 rigid load','mean_eue_gwh'], 709.7, 0.06)
check('rigid mean shortage hours', m.loc['H2 rigid load','mean_lole_h'], 40.63, 0.01)
check('central mean EUE', m.loc['H2 + reconversion','mean_eue_gwh'], 38.1, 0.06)
check('central mean shortage hours', m.loc['H2 + reconversion','mean_lole_h'], 1.95, 0.01)
check('central mean avoided EUE', m.loc['H2 + reconversion','mean_eue_avoided_gwh'], 159.7, 0.06)
check('central CI low', m.loc['H2 + reconversion','ci95_low_gwh'], 87.8, 0.06)
check('central CI high', m.loc['H2 + reconversion','ci95_high_gwh'], 236.0, 0.06)
check('Wilcoxon p', m.loc['H2 + reconversion','wilcoxon_one_sided_p'], 3.95e-4, 1e-6)

phase = pd.read_csv(R/'reduced'/'cross_weather_phase_summary.csv')
z=phase[(phase['slice']=='storage')&(phase['value']==0.5)].iloc[0]
check('0.5 TWh mean benefit', z.mean_eue_avoided_gwh, 35.6, 0.06)
check('0.5 TWh CI low', z.ci95_low_gwh, 5.7, 0.06)
check('0.5 TWh CI high', z.ci95_high_gwh, 67.4, 0.06)
check_equal('0.5 TWh improved years', int(z.improved_years), 9)
check_equal('0.5 TWh worse years', int(z.worse_years), 3)
z=phase[(phase['slice']=='power')&(phase['value']==3.0)].iloc[0]
check('3 GW mean benefit', z.mean_eue_avoided_gwh, 114.6, 0.06)

foresight=pd.read_csv(R/'reduced'/'operational_foresight_summary.csv').set_index('lookahead_h')
check('7-day aggregate avoided EUE', foresight.loc[168,'sum_eue_avoided_gwh'], 1839.7, 0.06)
check('7-day retained fraction', foresight.loc[168,'aggregate_benefit_retention_fraction']*100, 84.3, 0.06)

stats=pd.read_csv(R/'reduced'/'physical_scaling_statistics.csv')
def stat(target,predictor): return stats[(stats.target==target)&(stats.predictor==predictor)].iloc[0]
r=stat('minimum_tested_storage_zero_shortage_twh','annual_scarcity_energy_gwh')
check('storage-scarcity Spearman rho', r.spearman_rho, 0.887, 0.0006)
check('storage-scarcity p', r.spearman_p, 1.20e-4, 5e-7)
r=stat('reconversion_saturation_grid_gw','peak_shortfall_gw')
check('power-peak Spearman rho', r.spearman_rho, 0.753, 0.0006)
check('power-peak p', r.spearman_p, 4.73e-3, 5e-6)

dur=pd.read_csv(R/'reduced'/'duration_storage_summary.csv').set_index('technology')
check('4 h battery benefit', dur.loc['Battery +15 GW, 4 h','mean_eue_avoided_gwh'], 95.0, 0.06)
check('24 h battery benefit', dur.loc['Battery +15 GW, 24 h','mean_eue_avoided_gwh'], 498.3, 0.06)
check('100 h battery benefit', dur.loc['Battery +15 GW, 100 h','mean_eue_avoided_gwh'], 662.8, 0.06)
check('central H2 three-year benefit', dur.loc['Hydrogen 1 TWh + 15 GW','mean_eue_avoided_gwh'], 421.7, 0.06)

econ=pd.read_csv(R/'reduced'/'resilience_economic_boundary.csv')
anchor=econ.iloc[((econ.storage_capex_eur_per_kwh-6.0).abs() + (econ.reconversion_capex_eur_per_kw-500).abs()/100).argmin()]
check('economic boundary anchor EUR/MWh', anchor.breakeven_reliability_value_eur_per_mwh, 7245.6, 1.0)
cost=pd.read_csv(R/'reduced'/'technology_cost_screening.csv').set_index('option')
check('full-chain screen EUR/MWh', cost.loc['H2 full-chain sensitivity','cost_per_mwh_eue_avoided_eur'], 13667.6, 1.0)

all34=pd.read_csv(R/'all34'/'all34_mechanism_6h.csv')
def a34(y,s): return all34[(all34.year==y)&(all34.scenario==s)].iloc[0]
check('34-country 2016 reference', a34(2016,'reference').eue_gwh, 448.3, 0.06)
check('34-country 2016 rigid', a34(2016,'h2_rigid').eue_gwh, 719.7, 0.06)
check('34-country 2016 central', a34(2016,'h2_central').eue_gwh, 287.5, 0.06)
check('34-country 2017 rigid', a34(2017,'h2_rigid').eue_gwh, 20.5, 0.06)
check('34-country 2018 rigid', a34(2018,'h2_rigid').eue_gwh, 5.1, 0.06)

ext=pd.read_csv(R/'direct_observation'/'external_mechanism_test_6h.csv')
def ex(y,s): return ext[(ext.year==y)&(ext.scenario==s)].iloc[0]
for y,ref,rigid in [(2016,59.7,355.3),(2017,99.5,446.3),(2019,4.7,73.6)]:
    check(f'12-country {y} reference', ex(y,'reference').eue_gwh, ref, 0.06)
    check(f'12-country {y} rigid', ex(y,'h2_rigid').eue_gwh, rigid, 0.06)
    check(f'12-country {y} central', ex(y,'h2_central').eue_gwh, 0.0, 1e-6)
res=pd.read_csv(R/'direct_observation'/'external_resolution_sensitivity.csv')
r3=res[res.resolution_h==3]
check('12-country 3h aggregate reference', r3.reference_eue_gwh.sum(), 833.8, 0.06)
check('12-country 3h aggregate H2', r3.h2_eue_gwh.sum(), 4.0, 0.06)
check('12-country 3h fraction avoided', r3.eue_avoided_gwh.sum()/r3.reference_eue_gwh.sum()*100, 99.5, 0.06)

inv=pd.read_csv(R/'reduced'/'h2_initial_inventory_2006_sensitivity.csv')
for f in (0.0,0.5,1.0):
    row=inv[inv.initial_fraction==f].iloc[0]
    check(f'2006 inventory sensitivity {f}', row.eue_gwh, 633.7, 0.06)

if failures:
    print('VALIDATION: FAIL')
    for f in failures: print('-',f)
    raise SystemExit(1)
print('VALIDATION: PASS')
print('All checked frozen outputs agree with the rounded numerical values reported in the manuscript and Supplementary Information.')
