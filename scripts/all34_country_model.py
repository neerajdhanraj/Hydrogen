#!/usr/bin/env python3
"""European-scale 34-country replication of the hydrogen-resilience mechanism.

Electricity network: PyPSA-Eur OpenStreetMap v0.6 high-voltage branches
aggregated to physical country interfaces and restricted to the 34 active
countries used in the released replication.

Chronology: the released 2015-2018 country profiles assembled from Open Power
System Data. Direct national chronology is retained where available; documented
nearest-country same-year chronology proxies are used where a wind/solar series
is absent, while each target country's own load scale and planning capacities
are preserved.

The model is a fixed-capacity linear adequacy benchmark, not a reconstruction of
the historical European fleet. 2015 is used only for calibration and 2016-2018
are held out. Hydrogen scale is transferred from the reduced benchmark using
dimensionless fractions of electricity demand and mean load.

Units inside optimization: GW, GWh, hours, EUR/MWh.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse, json, math, time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
import networkx as nx

H2_DEMAND_ANNUAL_FRACTION=0.0336865
ELECTROLYSER_MEAN_LOAD_FRACTION=0.2021
H2_STORAGE_FRACTIONS=(0.0,0.00038455,0.00076910,0.00153820)
RECONVERSION_MEAN_LOAD_FRACTIONS=(0.0,0.02021,0.06063,0.10105)
ETA_EL=0.72
ETA_FC=0.55

@dataclass(frozen=True)
class Portfolio:
    vre_energy_share_target: float = 0.70
    firm_peak_fraction: float = 0.60
    battery_power_mean_fraction: float = 0.10
    battery_duration_h: float = 4.0
    battery_eta_charge: float = 0.95
    battery_eta_discharge: float = 0.95
    transmission_derate: float = 0.35
    voll_eur_mwh: float = 10000.0
    firm_mc_eur_mwh: float = 70.0
    battery_cycle_eur_mwh: float = 1.0

@dataclass(frozen=True)
class H2Case:
    label: str
    storage_fraction_annual_load: float | None = None
    reconversion_fraction_mean_load: float = 0.0
    spatial_mode: str = 'pooled' # pooled or national

class VI:
    def __init__(self,T,n_active,n_edges,h2:bool,spatial_mode='pooled'):
        self.T=T; self.blocks={}; self.n=0
        def add(name,n):
            s=slice(self.n,self.n+n); self.blocks[name]=s; self.n+=n; return s
        for name in ('firm','wind','solar','shed','bch','bdis','bsoc'):
            add(name,T*n_active)
        add('flow',T*n_edges)
        if h2:
            if spatial_mode=='pooled':
                add('el',T); add('fc',T); add('h2soc',T)
            elif spatial_mode=='national':
                add('el',T*n_active); add('fc',T*n_active); add('h2soc',T*n_active)
            else: raise ValueError(spatial_mode)
    def i2(self,name,t,j,width): return self.blocks[name].start+t*width+j
    def i1(self,name,t): return self.blocks[name].start+t


def load_inputs(profile_path:Path, osm_dir:Path):
    prof=pd.read_csv(profile_path,compression='gzip',parse_dates=['time'])
    active=sorted(prof.country.unique())
    buses=pd.read_csv(osm_dir/'buses.csv')
    edges=pd.read_csv(osm_dir/'country_interconnects.csv')
    # Final all-country validation is restricted exactly to the active current
    # PyPSA-Eur country set. No extra passive transit countries are permitted.
    countries=list(active)
    edges=edges[edges.a.isin(countries)&edges.b.isin(countries)&(edges.total_nominal_mw>0)].copy().reset_index(drop=True)
    G=nx.Graph(); G.add_nodes_from(countries); G.add_edges_from(edges[['a','b']].itertuples(index=False,name=None))
    if not nx.is_connected(G):
        raise ValueError(f'34-country OSM graph is disconnected: {[sorted(c) for c in nx.connected_components(G)]}')
    return prof,active,countries,edges


def year_arrays(prof,active,year,resolution_h=1):
    g=prof[prof.year==year].copy()
    pivot={}
    for col in ['load_mw_fixed_2015','wind_chronology_index','solar_chronology_index']:
        p=g.pivot(index='time',columns='country',values=col).reindex(columns=active)
        if resolution_h>1:
            # The preprocessing removes 29 February so every weather year has
            # exactly 8,760 samples. Calendar resampling would reinsert empty
            # bins in leap years; aggregate consecutive standardized hours
            # instead, preserving the intended 365-day chronology.
            if len(p) % resolution_h:
                raise ValueError(f'{year}: 8760-hour profile not divisible by {resolution_h}')
            a=p.to_numpy(float).reshape(-1,resolution_h,len(active)).mean(axis=1)
            p=pd.DataFrame(a,index=p.index[::resolution_h],columns=active)
        pivot[col]=p
    idx=pivot['load_mw_fixed_2015'].index
    if not all(x.index.equals(idx) for x in pivot.values()): raise RuntimeError('index mismatch')
    # Convert MW -> GW.
    return idx, pivot['load_mw_fixed_2015'].to_numpy(float)/1000.0, pivot['wind_chronology_index'].to_numpy(float), pivot['solar_chronology_index'].to_numpy(float)


def build_fixed_capacities(prof,active,portfolio:Portfolio):
    _,load15,w15,s15=year_arrays(prof,active,2015,1)
    means=load15.mean(axis=0); peaks=load15.max(axis=0)
    # Recover effective 2015 deployment proxy from the prepared profile by
    # comparing source normalization metadata is avoided here. Instead assign
    # technology capacity in proportion to each country's observed 2015 peak
    # generation index-weighted load scale, preserving country heterogeneity
    # through chronology while holding a common planning rule.
    # Base wind/solar power are tied to mean load. Solar/wind split is set by
    # observed 2015 potential-energy balance, not by a test year.
    mw=np.maximum(w15.mean(axis=0),1e-6); ms=np.maximum(s15.mean(axis=0),1e-6)
    # Equal potential-energy contribution from wind and solar at continental
    # scale provides a neutral high-VRE benchmark; country capacities scale with
    # 2015 mean demand and inverse own CF so each active country's potential
    # VRE energy is proportional to demand before the continental target factor.
    # This avoids encoding historical deployment choices into the stress test.
    wind_cap_raw=0.5*means/mw
    solar_cap_raw=0.5*means/ms
    base_energy=((w15*wind_cap_raw+s15*solar_cap_raw).sum()) # GWh because 1 h
    load_energy=load15.sum()
    scale=portfolio.vre_energy_share_target*load_energy/base_energy
    wind_cap=wind_cap_raw*scale; solar_cap=solar_cap_raw*scale
    firm_cap=portfolio.firm_peak_fraction*peaks
    batt_p=portfolio.battery_power_mean_fraction*means
    batt_e=batt_p*portfolio.battery_duration_h
    return {
        'mean_load_gw':means,'peak_load_gw':peaks,'wind_cap_gw':wind_cap,'solar_cap_gw':solar_cap,
        'firm_cap_gw':firm_cap,'battery_power_gw':batt_p,'battery_energy_gwh':batt_e,
        'vre_scale':float(scale),'load_energy_2015_gwh':float(load_energy),
    }


def solve_year(prof,active,countries,edges,year,portfolio:Portfolio,h2case:H2Case|None,resolution_h=1,return_ts=False,solver_method='highs'):
    idx,load,wind_idx,solar_idx=year_arrays(prof,active,year,resolution_h)
    T=len(idx); dt=float(resolution_h); A=len(active); N=len(countries); E=len(edges)
    caps=build_fixed_capacities(prof,active,portfolio)
    active_pos={c:i for i,c in enumerate(active)}; node_pos={c:i for i,c in enumerate(countries)}
    h2=h2case is not None and h2case.storage_fraction_annual_load is not None
    mode=h2case.spatial_mode if h2 else 'pooled'
    vi=VI(T,A,E,h2,mode)
    n=vi.n; c=np.zeros(n); lb=np.zeros(n); ub=np.full(n,np.inf)
    def set2(name,upper,cost=0.0,lower=0.0):
        sl=vi.blocks[name]; arr=np.asarray(upper,float)
        if arr.ndim==1: arr=np.tile(arr,(T,1))
        ub[sl]=arr.reshape(-1); lb[sl]=lower; c[sl]=cost*1000.0*dt
    set2('firm',caps['firm_cap_gw'],portfolio.firm_mc_eur_mwh)
    set2('wind',wind_idx*caps['wind_cap_gw'][None,:],-0.01)
    set2('solar',solar_idx*caps['solar_cap_gw'][None,:],-0.01)
    set2('shed',load,portfolio.voll_eur_mwh)
    set2('bch',caps['battery_power_gw'],portfolio.battery_cycle_eur_mwh/2)
    set2('bdis',caps['battery_power_gw'],portfolio.battery_cycle_eur_mwh/2)
    set2('bsoc',caps['battery_energy_gwh'],0)
    # OSM thermal-rating sums are not NTC. A predeclared uniform derate makes
    # the country-interface transport benchmark conservative; tested separately.
    flow_caps=edges.total_nominal_mw.to_numpy(float)/1000.0*portfolio.transmission_derate
    sl=vi.blocks['flow']; lb[sl]=np.tile(-flow_caps,T); ub[sl]=np.tile(flow_caps,T)

    annual_load_gwh=float(load.sum()*dt); mean_load_gw=annual_load_gwh/(T*dt)
    h2meta={}
    if h2:
        sf=float(h2case.storage_fraction_annual_load); pf=float(h2case.reconversion_fraction_mean_load)
        h2_energy=H2_DEMAND_ANNUAL_FRACTION*annual_load_gwh
        h2_demand_gw=h2_energy/(T*dt)
        el_cap=ELECTROLYSER_MEAN_LOAD_FRACTION*mean_load_gw
        store=h2case.storage_fraction_annual_load*annual_load_gwh
        fc_cap=pf*mean_load_gw # electric output capacity
        shares=(load.sum(axis=0)*dt)/annual_load_gwh
        if mode=='pooled':
            for name,up,cost in [('el',el_cap,1.0),('fc',fc_cap,5.0),('h2soc',store,0.0)]:
                s=vi.blocks[name]; ub[s]=up; c[s]=cost*1000.0*dt if name!='h2soc' else 0
        else:
            set2('el',el_cap*shares,1.0); set2('fc',fc_cap*shares,5.0); set2('h2soc',store*shares,0.0)
        h2meta={'h2_demand_twh':h2_energy/1000.0,'electrolyser_gw':el_cap,'h2_storage_twh':store/1000.0,'h2_reconversion_gw':fc_cap,'h2_spatial_mode':mode}

    rows=[]; cols=[]; vals=[]; rhs=[]; r=0
    def add(j,v): rows.append(r); cols.append(j); vals.append(v)
    # Incident edges by country.
    inc={cc:[] for cc in countries}
    for e,er in enumerate(edges.itertuples(index=False)):
        inc[er.a].append((e,-1.0)); inc[er.b].append((e,+1.0))
    # Nodal electricity balances.
    for t in range(T):
        for cc in countries:
            if cc in active_pos:
                a=active_pos[cc]
                for name in ('firm','wind','solar','shed','bdis'): add(vi.i2(name,t,a,A),1.0)
                add(vi.i2('bch',t,a,A),-1.0)
                if h2:
                    if mode=='pooled':
                        # pooled H2 conversions distributed by fixed load share
                        share=float((load[:,a].sum()*dt)/annual_load_gwh)
                        add(vi.i1('fc',t),share); add(vi.i1('el',t),-share)
                    else:
                        add(vi.i2('fc',t,a,A),1.0); add(vi.i2('el',t,a,A),-1.0)
                rr=float(load[t,a])
            else:
                rr=0.0
            for e,sign in inc[cc]: add(vi.i2('flow',t,e,E),sign)
            rhs.append(rr); r+=1
        # Battery dynamics are cyclic on each complete year.
        for a in range(A):
            add(vi.i2('bsoc',t,a,A),1.0)
            add(vi.i2('bsoc',T-1 if t==0 else t-1,a,A),-1.0)
            add(vi.i2('bch',t,a,A),-portfolio.battery_eta_charge*dt)
            add(vi.i2('bdis',t,a,A),dt/portfolio.battery_eta_discharge)
            rhs.append(0.0); r+=1
        if h2:
            if mode=='pooled':
                add(vi.i1('h2soc',t),1.0); add(vi.i1('h2soc',T-1 if t==0 else t-1),-1.0)
                add(vi.i1('el',t),-ETA_EL*dt); add(vi.i1('fc',t),dt/ETA_FC)
                rhs.append(-h2_demand_gw*dt); r+=1
            else:
                for a in range(A):
                    add(vi.i2('h2soc',t,a,A),1.0); add(vi.i2('h2soc',T-1 if t==0 else t-1,a,A),-1.0)
                    add(vi.i2('el',t,a,A),-ETA_EL*dt); add(vi.i2('fc',t,a,A),dt/ETA_FC)
                    rhs.append(-h2_demand_gw*float(shares[a])*dt); r+=1
    Aeq=coo_matrix((vals,(rows,cols)),shape=(r,n)).tocsr()
    st=time.time()
    res=linprog(c,A_eq=Aeq,b_eq=np.asarray(rhs),bounds=list(zip(lb,ub)),method=solver_method,options={'presolve':True})
    runtime=time.time()-st
    if not res.success:
        return {'year':year,'success':False,'message':res.message,'runtime_s':runtime}
    x=res.x
    shed=x[vi.blocks['shed']].reshape(T,A).sum(axis=1)
    eue=float(shed.sum()*dt); lole=float(np.count_nonzero(shed>1e-6)*dt)
    out={
        'year':year,'success':True,'scenario':'reference' if not h2 else h2case.label,
        'resolution_h':resolution_h,'eue_gwh':eue,'lole_h':lole,'max_shortfall_gw':float(shed.max()),
        'annual_load_twh':annual_load_gwh/1000.0,'eue_fraction_load':eue/annual_load_gwh,
        'runtime_s':runtime,'solver_method':solver_method,'n_variables':n,'n_equalities':r,
        **asdict(portfolio),**h2meta,
    }
    if return_ts:
        out['_timeseries']=pd.DataFrame({'time':idx,'shed_gw':shed})
    return out


def calibrate_firm(prof,active,countries,edges,base:Portfolio,resolution_h=3):
    """Calibrate firm capacity using 2015 only.

    The rule is fixed before any 2016--2019 result is inspected: choose the
    smallest tested firm-capacity fraction for which annual EUE is at most
    0.02% of annual electricity demand and shortage duration is at most 48 h.
    This intentionally leaves a rare but non-zero adequacy tail in calibration
    instead of tuning the model to zero shortage.
    """
    grid=np.round(np.arange(0.35,0.91,0.05),2)
    rows=[]; chosen=None
    for f in grid:
        p=Portfolio(**{**asdict(base),'firm_peak_fraction':float(f)})
        o=solve_year(prof,active,countries,edges,2015,p,None,resolution_h)
        rows.append(o); print('calibration',f,'EUE',round(o.get('eue_gwh',np.nan),2),'LOLE',o.get('lole_h'),flush=True)
        if o.get('success') and o.get('eue_fraction_load',np.inf)<=0.0002 and o.get('lole_h',np.inf)<=48:
            chosen=float(f)
            break
    tab=pd.DataFrame(rows)
    if chosen is None:
        chosen=float(tab.loc[tab.eue_gwh.idxmin(),'firm_peak_fraction'])
    return chosen,tab
