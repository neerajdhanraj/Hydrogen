#!/usr/bin/env python3
from pathlib import Path
import math, json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'results'/'reduced'
D.mkdir(parents=True, exist_ok=True)
SEED=20260814
rng=np.random.default_rng(SEED)

def boot_mean_ci(x, nboot=20000):
    x=np.asarray(x,float)
    b=rng.choice(x,(nboot,len(x)),replace=True).mean(1)
    return float(np.mean(x)), float(np.percentile(b,2.5)), float(np.percentile(b,97.5))

def paired_stats(df, baseline='Electricity only'):
    piv=df.pivot(index='year',columns='scenario',values='eue_gwh').sort_index()
    rows=[]
    base=piv[baseline]
    for sc in piv.columns:
        x=piv[sc]
        avoided=base-x
        mean,lo,hi=boot_mean_ci(avoided)
        nz=np.abs(avoided.to_numpy())>1e-9
        if sc==baseline:
            p=np.nan
        elif nz.any():
            try:
                # one-sided test for EUE avoided > 0; negative effects will yield p near 1
                p=float(wilcoxon(avoided, zero_method='pratt', alternative='greater', method='approx').pvalue)
            except Exception:
                p=np.nan
        else: p=np.nan
        rows.append(dict(scenario=sc,n=len(x),mean_eue_gwh=float(x.mean()),median_eue_gwh=float(x.median()),
                         p95_eue_gwh=float(np.percentile(x,95)),mean_lole_h=float(df[df.scenario==sc].lole_h.mean()),
                         median_lole_h=float(df[df.scenario==sc].lole_h.median()),max_eue_gwh=float(x.max()),
                         mean_eue_avoided_gwh=mean,ci95_low_gwh=lo,ci95_high_gwh=hi,
                         improved_years=int((avoided>1e-9).sum()),worse_years=int((avoided<-1e-9).sum()),neutral_years=int((np.abs(avoided)<=1e-9).sum()),
                         wilcoxon_one_sided_p=p))
    return pd.DataFrame(rows)

def crf(r,n): return r*(1+r)**n/((1+r)**n-1)

def annual_fixed(capex_eur, lifetime, fom_pct=0.0, r=0.07):
    return capex_eur*crf(r,lifetime)+capex_eur*fom_pct/100

mech=pd.read_csv(D/'mechanism_test_ensemble.csv')
cf=pd.read_csv(D/'counterfactual_test_ensemble.csv')
ref=pd.read_csv(D/'reference_ensemble.csv')
ms=paired_stats(mech); cs=paired_stats(cf)
# Use the same paired-bootstrap draw reported for the central 1 TWh / 15 GW
# configuration in the storage-power analysis, so repeated summary tables do
# not carry slightly different Monte Carlo confidence-interval endpoints.
phase_summary = D/'cross_weather_phase_summary.csv'
if phase_summary.exists():
    phase = pd.read_csv(phase_summary)
    central = phase[(phase['slice']=='storage') & (phase['value']==1.0)]
    if len(central)==1:
        idx = ms.index[ms.scenario=='H2 + reconversion'][0]
        ms.loc[idx,'ci95_low_gwh'] = float(central.ci95_low_gwh.iloc[0])
        ms.loc[idx,'ci95_high_gwh'] = float(central.ci95_high_gwh.iloc[0])
ms.to_csv(D/'mechanism_statistical_summary.csv',index=False)
cs.to_csv(D/'counterfactual_statistical_summary.csv',index=False)

# Training/test adequacy summary
ref['period']=np.where(ref.year<=1998,'Calibration 1980–1998','Held-out test 1999–2017')
rs=[]
for name,g in ref.groupby('period'):
    rs.append(dict(period=name,n_years=len(g),mean_eue_gwh=g.eue_gwh.mean(),median_eue_gwh=g.eue_gwh.median(),p95_eue_gwh=np.percentile(g.eue_gwh,95),
                   mean_lole_h=g.lole_h.mean(),median_lole_h=g.lole_h.median(),max_eue_gwh=g.eue_gwh.max(),max_lole_h=g.lole_h.max()))
pd.DataFrame(rs).to_csv(D/'reference_period_summary.csv',index=False)

# Shapley attribution from exact 2^3 factorial
import itertools

def shapley(df):
    vals={(int(r.W),int(r.S),int(r.D)):float(r.eue_gwh) for r in df.itertuples()}
    names=['Wind','Solar','Demand']; out=[]
    for i,name in enumerate(names):
        phi=0.0; others=[j for j in range(3) if j!=i]
        for k in range(3):
            for subset in itertools.combinations(others,k):
                a=[0,0,0]
                for j in subset:a[j]=1
                b=a.copy();b[i]=1
                wt=math.factorial(k)*math.factorial(2-k)/math.factorial(3)
                phi+=wt*(vals[tuple(b)]-vals[tuple(a)])
        out.append((name,phi))
    return out, vals[(0,0,0)], vals[(1,1,1)]
shrows=[]
for label,file in [('Electricity only','observed_factorial_2006_electricity.csv'),('H2 + reconversion','observed_factorial_2006_h2.csv')]:
    g=pd.read_csv(D/file); sh,b,f=shapley(g)
    for comp,v in sh: shrows.append(dict(system=label,component=comp,shapley_eue_gwh=v,climatology_eue_gwh=b,actual_2006_eue_gwh=f,total_delta_eue_gwh=f-b))
pd.DataFrame(shrows).to_csv(D/'observed_2006_shapley.csv',index=False)

# Event summary
b=pd.read_csv(D/'timeseries_2006_baseline.csv',parse_dates=['t']); h=pd.read_csv(D/'timeseries_2006_h2.csv',parse_dates=['t'])
start=pd.Timestamp('2006-01-29'); end=pd.Timestamp('2006-02-06')
bw=b[(b.t>=start)&(b.t<end)]; hw=h[(h.t>=start)&(h.t<end)]
# inspect likely columns robustly
summary={'window_start':str(start),'window_end_exclusive':str(end)}
for lab,g in [('baseline',bw),('h2',hw)]:
    for col in ['shortage_gw','fuelcell_gw','electrolysis_gw','h2_inventory_gwh']:
        if col in g: summary[f'{lab}_{col}_energy_or_extreme']=float(g[col].sum()) if col!='h2_inventory_gwh' else float(g[col].max())
# EUE if shortage_gw is hourly GW -> GWh sum
if 'shortage_gw' in bw: summary['baseline_window_eue_gwh']=float(bw.shortage_gw.sum())
if 'shortage_gw' in hw: summary['h2_window_eue_gwh']=float(hw.shortage_gw.sum())
if 'fuelcell_gw' in hw: summary['h2_window_fuelcell_output_gwh']=float(hw.fuelcell_gw.sum())
if 'electrolysis_gw' in hw: summary['h2_window_electrolysis_gwh']=float(hw.electrolysis_gw.sum())
if 'h2_inventory_gwh' in hw:
    summary['h2_window_inventory_min_gwh']=float(hw.h2_inventory_gwh.min());summary['h2_window_inventory_max_gwh']=float(hw.h2_inventory_gwh.max())
if 'shortage_gw' in bw:
    summary['baseline_peak_shortfall_gw']=float(bw.shortage_gw.max())
    summary['baseline_shortage_hours']=int((bw.shortage_gw>1e-6).sum())
if 'shortage_gw' in hw:
    summary['h2_peak_shortfall_gw']=float(hw.shortage_gw.max())
    summary['h2_shortage_hours']=int((hw.shortage_gw>1e-6).sum())
(D/'event_2006_summary.json').write_text(json.dumps(summary,indent=2))

# 2030 technology screening costs from PyPSA technology-data values retrieved 2026-08-14.
r=0.07
costrows=[]
# H2 marginal reconversion conditioned on existing H2 production + 1 TWh store
fc_gw=15; fc_capex=476.5391764*1e6*fc_gw # €/kW * 1e6 kW/GW
fc_ann=annual_fixed(fc_capex,30,0.5812,r)
# full chain sensitivity: 30GW charger +1TWh store+15GW FC
el_capex=436.5098856*1e6*30
store_capex=6.010*1e6*1000 # €/kWh * 1e6 kWh/GWh
full_h2=fc_ann+annual_fixed(el_capex,30,0.0,r)+annual_fixed(store_capex,30,0.43,r)
# Battery expansion +60GW inverter +300GWh storage
inv_capex=213.9279*1e6*60
bat_capex=189.861*1e6*300
bat_ann=annual_fixed(inv_capex,10,0.3375,r)+annual_fixed(bat_capex,25,0.0,r)
# Firm +5 GW OCGT proxy
ocgt_capex=581.3949*1e6*5
ocgt_ann=annual_fixed(ocgt_capex,25,1.7795,r)
# DR activation
cf_dr=cf[cf.scenario=='Demand response 5%']
dr_act_gwh=float(cf_dr.dr_gwh.mean()); dr_ann=dr_act_gwh*1000*300

avoid=dict(zip(cs.scenario,cs.mean_eue_avoided_gwh))
for sc,ann,note in [
 ('H2 + reconversion',fc_ann,'Incremental 15 GW H2-to-power module; assumes electrolysis and 1 TWh H2 storage already serve the H2 sector.'),
 ('H2 full-chain sensitivity',full_h2,'30 GW electrolysis + 1 TWh H2 storage + 15 GW H2-to-power; conservative full-chain screening.'),
 ('Battery expansion',bat_ann,'Incremental +60 GW inverter and +300 GWh battery energy capacity.'),
 ('Firm capacity +5 GW',ocgt_ann,'5 GW OCGT capital/FOM proxy; fuel and carbon costs excluded, so this is a lower-bound fixed-cost screen.'),
 ('Demand response 5%',dr_ann,'Activation cost at model penalty of €300/MWh; controlled curtailment, not an investment cost.'),
 ('Transmission +20%',np.nan,'Not costed because the six-region network has abstract line lengths and the expansion yields ~zero EUE benefit.')]:
    base_sc='H2 + reconversion' if sc=='H2 full-chain sensitivity' else sc
    av=float(avoid.get(base_sc,np.nan))
    costrows.append(dict(option=sc,annual_screening_cost_meur=ann/1e6 if np.isfinite(ann) else np.nan,mean_eue_avoided_gwh=av,
                         cost_per_mwh_eue_avoided_eur=(ann/(av*1000) if np.isfinite(ann) and av>1e-9 else np.nan),note=note))
pd.DataFrame(costrows).to_csv(D/'technology_cost_screening.csv',index=False)

# Compact headline JSON
h2=ms[ms.scenario=='H2 + reconversion'].iloc[0]; rigid=ms[ms.scenario=='H2 rigid load'].iloc[0]
headline={
 'held_out_years':'1999-2017','n_test_years':19,
 'reference_mean_eue_gwh':float(ms[ms.scenario=='Electricity only'].mean_eue_gwh.iloc[0]),
 'reference_median_eue_gwh':float(ms[ms.scenario=='Electricity only'].median_eue_gwh.iloc[0]),
 'h2_reconversion_mean_eue_gwh':float(h2.mean_eue_gwh),'h2_mean_eue_avoided_gwh':float(h2.mean_eue_avoided_gwh),
 'h2_eue_avoided_ci95_gwh':[float(h2.ci95_low_gwh),float(h2.ci95_high_gwh)],'h2_wilcoxon_p':float(h2.wilcoxon_one_sided_p),
 'rigid_h2_mean_eue_gwh':float(rigid.mean_eue_gwh),'rigid_h2_mean_eue_change_gwh':float(-rigid.mean_eue_avoided_gwh),
 'worst_test_year':int(ref[ref.year>=1999].sort_values('eue_gwh',ascending=False).year.iloc[0]),
 'worst_test_eue_gwh':float(ref[ref.year>=1999].eue_gwh.max())}
(D/'headline_results.json').write_text(json.dumps(headline,indent=2))
print(json.dumps(headline,indent=2))
print('\nCost screening:\n',pd.read_csv(D/'technology_cost_screening.csv').to_string(index=False))
