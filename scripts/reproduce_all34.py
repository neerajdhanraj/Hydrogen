#!/usr/bin/env python3
"""Reproduce the released 34-country six-hour result tables.

This script uses only files packaged in data/external/all34_opsd and
 data/external/pypsa_eur_osm_v06. It regenerates the five CSV tables used by
Figure 5, the Results section, Supplementary Note 7, and the claim-traceability
file.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import numpy as np
import importlib.util, sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('m34',ROOT/'scripts/all34_country_model.py')
m=importlib.util.module_from_spec(spec);sys.modules['m34']=m;spec.loader.exec_module(m)
PROFILES=ROOT/'data/external/all34_opsd/all34_profiles_2015_2018.csv.gz'
OSM=ROOT/'data/external/pypsa_eur_osm_v06'
OUT=ROOT/'results/all34';OUT.mkdir(parents=True,exist_ok=True)
RES=6

prof,active,countries,edges=m.load_inputs(PROFILES,OSM)
base=m.Portfolio()

# 2015 calibration grid, using the predeclared rule from the manuscript.
cal=[]
chosen=None
for f in np.round(np.arange(.35,.71,.05),2):
    p=m.Portfolio(**{**asdict(base),'firm_peak_fraction':float(f)})
    o=m.solve_year(prof,active,countries,edges,2015,p,None,RES)
    passes=(o['eue_fraction_load']<=0.0002 and o['lole_h']<=48)
    cal.append({'firm_peak_fraction':float(f),'eue_gwh':o['eue_gwh'],'lole_h':o['lole_h'],'passes_rule':passes})
    if passes and chosen is None: chosen=float(f)
if chosen is None: raise RuntimeError('No calibration point passed the fixed rule')
pd.DataFrame(cal).to_csv(OUT/'all34_calibration_2015_6h.csv',index=False)
portfolio=m.Portfolio(**{**asdict(base),'firm_peak_fraction':chosen})

annual_2016=m.solve_year(prof,active,countries,edges,2016,portfolio,None,RES)['annual_load_twh']*1000 # GWh
mean_load_gw=annual_2016*1000/(8760)  # TWh->GWh above then /h; retained only for reporting checks
# Fractions transferred from the reduced benchmark and used in the release.
central_sf=0.00076910
central_pf=0.10105

# Main mechanism table 2016-2018.
rows=[]
cases=[('reference',None),('h2_rigid',m.H2Case('h2_rigid',0.0,0.0)),('h2_flexible',m.H2Case('h2_flexible',central_sf,0.0)),('h2_central',m.H2Case('h2_central',central_sf,central_pf))]
for year in [2016,2017,2018]:
    for label,case in cases:
        o=m.solve_year(prof,active,countries,edges,year,portfolio,case,RES)
        rows.append({'year':year,'scenario':label,'eue_gwh':o['eue_gwh'],'lole_h':o['lole_h'],'max_shortfall_gw':o['max_shortfall_gw'],'resolution_h':RES,'annual_load_twh':o['annual_load_twh']})
pd.DataFrame(rows).to_csv(OUT/'all34_mechanism_6h.csv',index=False)

# Storage slice: 1x, 2x, 4x and 8x the transferred storage level at fixed central reconversion.
st=[]
for sf in [central_sf,2*central_sf,4*central_sf,8*central_sf]:
    case=m.H2Case('h2_storage_slice',sf,central_pf)
    o=m.solve_year(prof,active,countries,edges,2016,portfolio,case,RES)
    st.append({'h2_storage_twh':o['h2_storage_twh'],'h2_reconversion_gw':o['h2_reconversion_gw'],'eue_gwh':o['eue_gwh'],'lole_h':o['lole_h'],'year':2016,'resolution_h':RES})
pd.DataFrame(st).to_csv(OUT/'all34_storage_slice_2016_6h.csv',index=False)

# Power slice at 4x central storage.
pw=[]
for pf in [0.0,central_pf,1.5*central_pf,2.0*central_pf]:
    case=m.H2Case('h2_power_slice',4*central_sf,pf)
    o=m.solve_year(prof,active,countries,edges,2016,portfolio,case,RES)
    pw.append({'h2_storage_twh':o['h2_storage_twh'],'h2_reconversion_gw':o['h2_reconversion_gw'],'eue_gwh':o['eue_gwh'],'lole_h':o['lole_h'],'year':2016,'resolution_h':RES})
pd.DataFrame(pw).to_csv(OUT/'all34_power_slice_2016_6h.csv',index=False)

# Structural checks used in Supplementary Note 7.
rob=[]
for td in [.20,.35,.50]:
    p=m.Portfolio(**{**asdict(portfolio),'transmission_derate':td})
    for label,case in [('reference',None),('h2_central',m.H2Case('h2_central',central_sf,central_pf))]:
        o=m.solve_year(prof,active,countries,edges,2016,p,case,RES)
        rob.append({'test':f'transmission_{int(td*100)}pct','scenario':label,'eue_gwh':o['eue_gwh'],'year':2016,'resolution_h':RES})
# National H2 inventory.
for label,case in [('reference',None),('h2_central',m.H2Case('h2_central_national',central_sf,central_pf,'national'))]:
    o=m.solve_year(prof,active,countries,edges,2016,portfolio,case,RES)
    rob.append({'test':'national_h2','scenario':label,'eue_gwh':o['eue_gwh'],'year':2016,'resolution_h':RES})
# Kosovo removal.
prof_x=prof[prof.country!='XK'].copy(); active_x=[x for x in active if x!='XK']; countries_x=[x for x in countries if x!='XK']; edges_x=edges[(edges.a!='XK')&(edges.b!='XK')].copy().reset_index(drop=True)
for label,case in [('reference',None),('h2_rigid',m.H2Case('h2_rigid',0.0,0.0)),('h2_central',m.H2Case('h2_central',central_sf,central_pf))]:
    o=m.solve_year(prof_x,active_x,countries_x,edges_x,2016,portfolio,case,RES)
    rob.append({'test':'exclude_kosovo','scenario':label,'eue_gwh':o['eue_gwh'],'year':2016,'resolution_h':RES})
pd.DataFrame(rob).to_csv(OUT/'all34_structural_robustness_2016_6h.csv',index=False)
print(f'Regenerated all 34-country release tables in {OUT}')
