#!/usr/bin/env python3
"""Historical-weather six-region dispatch LP for hydrogen-resilience analysis.

Uses the public Hilbers et al. six-region test-system topology/capacities and the
MERRA2-derived hourly demand/wind/solar subset (1980-2017). The electricity
model is a fixed-capacity hourly linear dispatch benchmark. Hydrogen is added as
an explicit pooled carrier with geographically distributed electrolysis and
reconversion, allowing mechanism/phase experiments without hidden slack.

Units: power GW; energy GWh; variable costs EUR/MWh; annual costs returned EUR.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

NODES = (1,2,3,4,5,6)
LINES = ((1,2,30.0),(1,5,20.0),(1,6,10.0),(2,3,40.0),(3,4,30.0),(4,5,30.0),(5,6,10.0))
THERMAL = {
    'base1': (1,20.0,20.0), 'peak1': (1,25.0,100.0),
    'base3': (3,50.0,20.0), 'peak3': (3,20.0,100.0),
    'base6': (6,20.0,20.0), 'peak6': (6,20.0,100.0),
}
VRE = {
    'wind2': (2,40.0,'wind_region2'), 'solar2': (2,20.0,'solar_region2'),
    'wind5': (5,40.0,'wind_region5'), 'solar5': (5,30.0,'solar_region5'),
    'wind6': (6,30.0,'wind_region6'), 'solar6': (6,20.0,'solar_region6'),
}
BATT_NODES = (2,5,6)
DEMAND_COL = {2:'demand_region2',4:'demand_region4',5:'demand_region5'}

@dataclass(frozen=True)
class Scenario:
    name: str
    # Existing battery benchmark from published six-region operate configuration.
    battery_power_scale: float = 1.0
    battery_energy_scale: float = 1.0
    battery_eta_charge: float = 0.95
    battery_eta_discharge: float = 0.95
    # Transmission counterfactual.
    transmission_scale: float = 1.0
    # Firm low-carbon counterfactual: added equally at nodes 1,3,6.
    extra_firm_gw: float = 0.0
    # Demand response represented as interruptible electricity demand with a
    # low penalty; capped as a fraction of hourly demand. It is reported separately.
    demand_response_fraction: float = 0.0
    demand_response_cost_eur_mwh: float = 300.0
    # Hydrogen sector. A pooled H2 backbone connects electrolysers/fuel cells at 2,5,6.
    hydrogen: bool = False
    h2_demand_gw: float = 0.0
    electrolyser_power_gw_each: float = 20.0
    electrolyser_efficiency: float = 0.72
    fuelcell_power_gw_each: float = 0.0
    fuelcell_efficiency: float = 0.55
    h2_storage_gwh: float = 0.0
    # If set, fixes both start- and end-of-year H2 inventory to this fraction
    # of storage capacity. None retains the standard cyclic free-boundary case.
    h2_initial_soc_fraction: Optional[float] = None
    # Explicit H2 imports, disabled by default.
    h2_import_power_gw: float = 0.0
    h2_import_cost_eur_mwh: float = 120.0
    # Variable costs.
    voll_eur_mwh: float = 10000.0
    battery_cycle_cost_eur_mwh: float = 1.0
    fuelcell_vom_eur_mwh: float = 5.0
    electrolyser_vom_eur_mwh: float = 1.0
    extra_firm_vom_eur_mwh: float = 70.0

class VarIndex:
    def __init__(self, T:int, sc:Scenario):
        self.T=T; self.sc=sc; self.blocks={}; self.n=0
        def add(name,n):
            s=slice(self.n,self.n+n); self.blocks[name]=s; self.n+=n; return s
        for k in THERMAL: add(k,T)
        for k in VRE: add(k,T)   # dispatch, bounded by availability
        for n in BATT_NODES:
            add(f'bch{n}',T); add(f'bdis{n}',T); add(f'bsoc{n}',T)
        for a,b,_ in LINES: add(f'f{a}{b}',T)
        for n in DEMAND_COL: add(f'shed{n}',T)
        if sc.demand_response_fraction>0:
            for n in DEMAND_COL: add(f'dr{n}',T)
        if sc.extra_firm_gw>0:
            for n in (1,3,6): add(f'firm{n}',T)
        if sc.hydrogen:
            # One pooled H2 backbone. Conversion is distributed equally across R2/R5/R6
            # in the electricity balances to avoid artificial spatial arbitrage and LP degeneracy.
            add('el_total',T); add('fc_total',T); add('h2soc',T)
            if sc.h2_import_power_gw>0: add('h2imp',T)
    def idx(self,name,t): return self.blocks[name].start+t


def solve_year(
    year_df:pd.DataFrame, sc:Scenario, *, return_timeseries=False, cyclic=True,
    initial_battery_soc_gwh=None, terminal_battery_soc_min_gwh=None,
    initial_h2_soc_gwh=None, terminal_h2_soc_min_gwh=None
) -> Dict:
    """Solve an hourly dispatch horizon.

    The publication ensemble uses the default cyclic boundary on complete 8760-h
    weather years.  Non-cyclic boundaries are exposed for rolling-horizon
    robustness tests; in that mode the caller supplies inherited storage states
    and may impose a terminal reserve floor without leaking information from
    beyond the forecast horizon.
    """
    d=year_df.reset_index(drop=True)
    T=len(d)
    if T < 1:
        raise ValueError('Dispatch horizon must contain at least one hour')
    if cyclic and T != 8760:
        # Cyclic short-horizon solves are allowed only when explicitly useful,
        # but the default publication path remains a full non-leap year.
        pass
    vi=VarIndex(T,sc); nvar=vi.n
    c=np.zeros(nvar); lb=np.zeros(nvar); ub=np.full(nvar,np.inf)

    batt_cap = 100.0 * sc.battery_energy_scale
    def _state_map(value, default):
        if value is None:
            return {n: float(default) for n in BATT_NODES}
        if np.isscalar(value):
            return {n: float(value) for n in BATT_NODES}
        return {n: float(value[n]) for n in BATT_NODES}

    batt_initial = _state_map(initial_battery_soc_gwh, 0.5*batt_cap)
    batt_terminal_min = None if terminal_battery_soc_min_gwh is None else _state_map(terminal_battery_soc_min_gwh, 0.0)

    def set_block(name, upper, cost=0.0, lower=0.0):
        s=vi.blocks[name]; lb[s]=lower
        if np.isscalar(upper): ub[s]=upper
        else: ub[s]=np.asarray(upper,float)
        c[s]=cost*1000.0 # EUR/MWh -> EUR/GWh

    for k,(node,cap,mc) in THERMAL.items(): set_block(k,cap,mc)
    for k,(node,cap,col) in VRE.items(): set_block(k,cap*d[col].to_numpy(float),-0.01)
    for n in BATT_NODES:
        set_block(f'bch{n}',20.0*sc.battery_power_scale,sc.battery_cycle_cost_eur_mwh/2)
        set_block(f'bdis{n}',20.0*sc.battery_power_scale,sc.battery_cycle_cost_eur_mwh/2)
        set_block(f'bsoc{n}',100.0*sc.battery_energy_scale,0.0)
    for a,b,cap in LINES:
        s=vi.blocks[f'f{a}{b}']; lb[s]=-cap*sc.transmission_scale; ub[s]=cap*sc.transmission_scale
    for n,col in DEMAND_COL.items():
        set_block(f'shed{n}',d[col].to_numpy(float),sc.voll_eur_mwh)
    if sc.demand_response_fraction>0:
        for n,col in DEMAND_COL.items():
            set_block(f'dr{n}',sc.demand_response_fraction*d[col].to_numpy(float),sc.demand_response_cost_eur_mwh)
    if sc.extra_firm_gw>0:
        for n in (1,3,6): set_block(f'firm{n}',sc.extra_firm_gw/3.0,sc.extra_firm_vom_eur_mwh)
    if sc.hydrogen:
        set_block('el_total',3.0*sc.electrolyser_power_gw_each,sc.electrolyser_vom_eur_mwh)
        set_block('fc_total',3.0*sc.fuelcell_power_gw_each,sc.fuelcell_vom_eur_mwh)
        set_block('h2soc',sc.h2_storage_gwh,0.0)
        if sc.h2_import_power_gw>0: set_block('h2imp',sc.h2_import_power_gw,sc.h2_import_cost_eur_mwh)

    if batt_terminal_min is not None:
        for n in BATT_NODES:
            j=vi.idx(f'bsoc{n}',T-1)
            lb[j]=max(lb[j], min(batt_cap, batt_terminal_min[n]))
    if sc.hydrogen and terminal_h2_soc_min_gwh is not None:
        j=vi.idx('h2soc',T-1)
        lb[j]=max(lb[j], min(sc.h2_storage_gwh, float(terminal_h2_soc_min_gwh)))

    # Build sparse equalities: 6 nodal balances * T + 3 battery dynamics*T + H2*T.
    rows=[]; cols=[]; vals=[]; rhs_list=[]; r=0
    # helper
    def coef(name,t,v): rows.append(r); cols.append(vi.idx(name,t)); vals.append(v)
    line_lookup={(a,b):(cap) for a,b,cap in LINES}
    incident={n:[] for n in NODES}
    for a,b,cap in LINES:
        incident[a].append((f'f{a}{b}',-1.0)); incident[b].append((f'f{a}{b}',+1.0))
    thermal_by_node={n:[] for n in NODES}
    for k,(n,cap,mc) in THERMAL.items(): thermal_by_node[n].append(k)
    vre_by_node={n:[] for n in NODES}
    for k,(n,cap,col) in VRE.items(): vre_by_node[n].append(k)

    for t in range(T):
        for n in NODES:
            for k in thermal_by_node[n]: coef(k,t,1.0)
            for k in vre_by_node[n]: coef(k,t,1.0)
            for name,sign in incident[n]: coef(name,t,sign)
            if n in BATT_NODES:
                coef(f'bdis{n}',t,1.0); coef(f'bch{n}',t,-1.0)
            if sc.extra_firm_gw>0 and n in (1,3,6): coef(f'firm{n}',t,1.0)
            if sc.hydrogen and n in (2,5,6):
                coef('fc_total',t,1.0/3.0); coef('el_total',t,-1.0/3.0)
            rhs=0.0
            if n in DEMAND_COL:
                coef(f'shed{n}',t,1.0)
                if sc.demand_response_fraction>0: coef(f'dr{n}',t,1.0)
                rhs=float(d.at[t,DEMAND_COL[n]])
            rhs_list.append(rhs); r+=1
        # Battery storage dynamics. Annual publication runs are cyclic;
        # rolling-horizon runs inherit the committed state explicitly.
        for n in BATT_NODES:
            coef(f'bsoc{n}',t,1.0)
            if t == 0 and not cyclic:
                rhs_b = batt_initial[n]
            else:
                coef(f'bsoc{n}',T-1 if t==0 else t-1,-1.0)
                rhs_b = 0.0
            coef(f'bch{n}',t,-sc.battery_eta_charge)
            coef(f'bdis{n}',t,1.0/sc.battery_eta_discharge)
            rhs_list.append(rhs_b); r+=1
        if sc.hydrogen:
            coef('h2soc',t,1.0)
            fixed_h2_boundary = sc.h2_initial_soc_fraction is not None
            noncyclic_h2 = (not cyclic) or fixed_h2_boundary
            if t == 0 and noncyclic_h2:
                pass
            else:
                coef('h2soc',T-1 if t==0 else t-1,-1.0)
            coef('el_total',t,-sc.electrolyser_efficiency)
            coef('fc_total',t,1.0/sc.fuelcell_efficiency)
            if sc.h2_import_power_gw>0: coef('h2imp',t,-1.0)
            if t == 0 and noncyclic_h2:
                if fixed_h2_boundary:
                    initial = sc.h2_initial_soc_fraction * sc.h2_storage_gwh
                elif initial_h2_soc_gwh is None:
                    initial = 0.5 * sc.h2_storage_gwh
                else:
                    initial = float(initial_h2_soc_gwh)
                rhs_list.append(initial - sc.h2_demand_gw)
            else:
                rhs_list.append(-sc.h2_demand_gw)
            r+=1

    if sc.hydrogen and sc.h2_initial_soc_fraction is not None:
        if not 0.0 <= sc.h2_initial_soc_fraction <= 1.0:
            raise ValueError('h2_initial_soc_fraction must be between 0 and 1')
        # Prevent end-of-year depletion from subsidising the annual result.
        coef('h2soc',T-1,1.0)
        rhs_list.append(sc.h2_initial_soc_fraction * sc.h2_storage_gwh); r+=1

    Aeq=coo_matrix((vals,(rows,cols)),shape=(r,nvar)).tocsr()
    res=linprog(c,A_eq=Aeq,b_eq=np.asarray(rhs_list),bounds=list(zip(lb,ub)),method='highs-ds',options={'presolve':True})
    if not res.success:
        return {'scenario':sc.name,'success':False,'status':res.status,'message':res.message}
    x=res.x
    shed=np.zeros(T); dr=np.zeros(T)
    for n in DEMAND_COL: shed += x[vi.blocks[f'shed{n}']]
    if sc.demand_response_fraction>0:
        for n in DEMAND_COL: dr += x[vi.blocks[f'dr{n}']]
    # GWh (GW * 1h); TWh divide 1000.
    eue=float(shed.sum())
    lole=int(np.count_nonzero(shed>1e-6))
    max_short=float(shed.max())
    total_demand=float(sum(d[col].sum() for col in DEMAND_COL.values()))
    out={
        'scenario':sc.name,'success':True,'objective_eur':float(res.fun),
        'electricity_demand_twh':total_demand/1000.0,
        'eue_gwh':eue,'eue_twh':eue/1000.0,'lole_h':lole,
        'max_shortfall_gw':max_short,'dr_gwh':float(dr.sum()),
        'h2_demand_twh':sc.h2_demand_gw*T/1000.0 if sc.hydrogen else 0.0,
    }
    # Dispatch/use metrics
    out['thermal_twh']=sum(float(x[vi.blocks[k]].sum()) for k in THERMAL)/1000.0
    out['vre_twh']=sum(float(x[vi.blocks[k]].sum()) for k in VRE)/1000.0
    out['battery_discharge_twh']=sum(float(x[vi.blocks[f'bdis{n}']].sum()) for n in BATT_NODES)/1000.0
    if sc.hydrogen:
        out['electrolysis_twh_el']=float(x[vi.blocks['el_total']].sum())/1000.0
        out['fuelcell_twh_el']=float(x[vi.blocks['fc_total']].sum())/1000.0
        out['h2_inventory_max_twh']=float(x[vi.blocks['h2soc']].max())/1000.0
        out['h2_import_twh']=float(x[vi.blocks['h2imp']].sum())/1000.0 if sc.h2_import_power_gw>0 else 0.0
    if return_timeseries:
        ts=pd.DataFrame({'t':d['t'].to_numpy(),'shed_gw':shed,'dr_gw':dr})
        for n in DEMAND_COL: ts[f'demand_r{n}']=d[DEMAND_COL[n]].to_numpy()
        ts['vre_dispatch_gw']=sum(x[vi.blocks[k]] for k in VRE)
        ts['thermal_gw']=sum(x[vi.blocks[k]] for k in THERMAL)
        ts['battery_discharge_gw']=sum(x[vi.blocks[f'bdis{n}']] for n in BATT_NODES)
        ts['battery_charge_gw']=sum(x[vi.blocks[f'bch{n}']] for n in BATT_NODES)
        for n in BATT_NODES:
            ts[f'battery_soc_r{n}_gwh']=x[vi.blocks[f'bsoc{n}']]
            ts[f'battery_charge_r{n}_gw']=x[vi.blocks[f'bch{n}']]
            ts[f'battery_discharge_r{n}_gw']=x[vi.blocks[f'bdis{n}']]
        if sc.hydrogen:
            ts['electrolysis_gw']=x[vi.blocks['el_total']]
            ts['fuelcell_gw']=x[vi.blocks['fc_total']]
            ts['h2_inventory_gwh']=x[vi.blocks['h2soc']]
        out['_timeseries']=ts
    return out


def load_data(path:Path)->pd.DataFrame:
    df=pd.read_csv(path); df['t']=pd.to_datetime(df['t'])
    # Repository subset removes leap days and provides 8760 hours per year.
    counts=df.groupby(df.t.dt.year).size()
    if not (counts==8760).all(): raise ValueError(counts[counts!=8760])
    return df
