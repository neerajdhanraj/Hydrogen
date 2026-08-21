from __future__ import annotations
import os,sys,importlib.util
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('h2lp',ROOT/'scripts'/'reduced_model.py');m=importlib.util.module_from_spec(spec);sys.modules['h2lp']=m;spec.loader.exec_module(m)
DATA=None;PATH=ROOT/'data'/'real'/'MERRA2_subset_demand_wind_solar.csv';COMMON=dict(transmission_scale=3.5,extra_firm_gw=5.0)
def init():
 global DATA;DATA=m.load_data(PATH)
def work(y,name,dur):
 yd=DATA[DATA.t.dt.year==y]
 if name=='Hydrogen': sc=m.Scenario('Hydrogen 1 TWh + 15 GW',**COMMON,hydrogen=True,h2_demand_gw=5.,electrolyser_power_gw_each=10.,h2_storage_gwh=1000.,fuelcell_power_gw_each=5.)
 else:
  # Add a 15 GW storage block on top of the 60 GW / 300 GWh reference battery fleet.
  pscale=(60.0+15.0)/60.0
  escale=(300.0+15.0*dur)/300.0
  sc=m.Scenario(f'Battery +15 GW, {dur:g} h',**COMMON,battery_power_scale=pscale,battery_energy_scale=escale)
 o=m.solve_year(yd,sc);return {'year':y,'technology':sc.name,'duration_h':(1000/15 if name=='Hydrogen' else dur),'eue_gwh':o['eue_gwh'],'lole_h':o['lole_h']}
def main():
 years=[2006,2009,2002];jobs=[];rows=[]
 with ProcessPoolExecutor(max_workers=4,initializer=init) as ex:
  for y in years:
   for d in [4.,24.,100.]:jobs.append(ex.submit(work,y,'Battery',d))
   jobs.append(ex.submit(work,y,'Hydrogen',0))
  for i,f in enumerate(as_completed(jobs),1):rows.append(f.result());print(i,'/',len(jobs),flush=True)
 d=pd.DataFrame(rows);ref=pd.read_csv(ROOT/'results'/'reduced'/'mechanism_test_ensemble.csv');ref=ref[ref.scenario=='Electricity only'][['year','eue_gwh']].rename(columns={'eue_gwh':'reference_eue_gwh'})
 d=d.merge(ref,on='year');d['eue_avoided_gwh']=d.reference_eue_gwh-d.eue_gwh;d.to_csv(ROOT/'results'/'reduced'/'duration_storage_comparison.csv',index=False)
 s=d.groupby(['technology','duration_h']).agg(mean_eue_gwh=('eue_gwh','mean'),mean_eue_avoided_gwh=('eue_avoided_gwh','mean'),improved_years=('eue_avoided_gwh',lambda x:int((x>1e-6).sum())),zero_shortage_years=('eue_gwh',lambda x:int((x<=1e-6).sum()))).reset_index();s.to_csv(ROOT/'results'/'reduced'/'duration_storage_summary.csv',index=False);print(s.to_string(index=False))
if __name__=='__main__':main()
