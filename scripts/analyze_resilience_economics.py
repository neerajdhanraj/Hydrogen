from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results'/'reduced'
# central reduced-model benefit, conditioned on hydrogen demand/electrolysis already existing
s=pd.read_csv(D/'mechanism_statistical_summary.csv'); avoided=float(s.loc[s.scenario=='H2 + reconversion','mean_eue_avoided_gwh'].iloc[0])
# economics sensitivity: storage and reconversion capex; 7% WACC; 30 y life; FOM included as fractions of capex
r=.07;n=30;crf=r*(1+r)**n/((1+r)**n-1)
store_costs=np.arange(1.,16.01,1.) # EUR/kWh_H2 energy inventory
reconv_costs=np.arange(250.,1000.1,50.) # EUR/kW electric output
rows=[]
for cs in store_costs:
 for cp in reconv_costs:
  store_cap=cs*1e9 # 1 TWh = 1e9 kWh
  power_cap=cp*15e6 # 15GW = 15e6 kW
  annual=store_cap*(crf+.0043)+power_cap*(crf+.005812)
  breakeven=annual/(avoided*1000.)
  rows.append(dict(storage_capex_eur_per_kwh=cs,reconversion_capex_eur_per_kw=cp,annualized_cost_meur=annual/1e6,breakeven_reliability_value_eur_per_mwh=breakeven))
pd.DataFrame(rows).to_csv(D/'resilience_economic_boundary.csv',index=False)
# anchor values from package cost assumptions
anchor_store=6.010; anchor_p=476.5391764
z=pd.DataFrame(rows); k=((z.storage_capex_eur_per_kwh-anchor_store).abs()+(z.reconversion_capex_eur_per_kw-anchor_p).abs()/50).idxmin()
print('avoided',avoided,'GWh/yr')
print('nearest anchor',z.loc[k].to_dict())
print('Reliability-equivalent firm comparison:', pd.read_csv(D/'counterfactual_statistical_summary.csv').query("scenario=='Firm capacity +5 GW'")[['mean_eue_avoided_gwh']].to_dict('records'))
