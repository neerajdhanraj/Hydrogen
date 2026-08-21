#!/usr/bin/env python3
"""Regenerate Supplementary Figures 1-5 from released result tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from shapely import wkt

ROOT=Path(__file__).resolve().parents[1]
RED=ROOT/'results'/'reduced'; EXT=ROOT/'results'/'direct_observation'; OUT=ROOT/'figures'/'generated'; OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42})

def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/f'{name}.png',dpi=450,bbox_inches='tight')
    plt.close(fig)

# Supplementary Figure 1: 38-year reference adequacy.
r=pd.read_csv(RED/'reference_ensemble.csv').sort_values('year')
fig,ax=plt.subplots(figsize=(7.0,3.5))
ax.axvspan(1979.5,1998.5,alpha=.12); ax.axvspan(1998.5,2017.5,alpha=.10)
ax.plot(r.year,r.eue_gwh,marker='o',ms=3.2,lw=1.1)
row=r[r.year==2006].iloc[0]; ax.annotate(f"2006\n{row.eue_gwh/1000:.2f} TWh",xy=(2006,row.eue_gwh),xytext=(2008,row.eue_gwh+320),arrowprops=dict(arrowstyle='->',lw=.7))
ax.set_xlabel('Weather year'); ax.set_ylabel('Unserved energy (GWh yr$^{-1}$)'); ax.set_title('Reference-system adequacy across the 38-year record',loc='left',weight='bold'); ax.spines[['top','right']].set_visible(False)
save(fig,'figS1_reference_weather_years')

# Supplementary Figure 2: efficiency sensitivity.
e=pd.read_csv(RED/'h2_efficiency_sensitivity_2030.csv')
central=pd.read_csv(RED/'mechanism_test_ensemble.csv'); central=central[central.scenario=='H2 + reconversion'][['year','eue_gwh']].rename(columns={'eue_gwh':'central_eue_gwh'})
e=e.merge(central,on='year')
fig,ax=plt.subplots(figsize=(6.6,3.8)); ax.scatter(e.central_eue_gwh,e.eue_gwh,s=24)
mx=max(e.central_eue_gwh.max(),e.eue_gwh.max())*1.03; ax.plot([0,mx],[0,mx],ls='--',lw=.8)
ax.set_xlabel('Unserved energy, central efficiencies (GWh)'); ax.set_ylabel('Unserved energy, 2030 efficiencies (GWh)'); ax.set_title('Hydrogen resilience is robust to lower 2030 conversion efficiencies',loc='left',weight='bold'); ax.spines[['top','right']].set_visible(False)
save(fig,'figS2_h2_efficiency_sensitivity')

# Supplementary Figure 3: 2006 phase surface and Shapley attribution.
ph=pd.read_csv(RED/'h2_phase_2006.csv')
base=float(pd.read_csv(RED/'mechanism_test_ensemble.csv').query("year==2006 and scenario=='Electricity only'").eue_gwh.iloc[0])
piv=ph.pivot(index='h2_storage_gwh',columns='fuelcell_total_gw',values='eue_gwh').sort_index()
delta=piv-base
fig,axs=plt.subplots(1,2,figsize=(7.2,3.6),gridspec_kw={'width_ratios':[1.15,1]})
ax=axs[0]; norm=TwoSlopeNorm(vcenter=0,vmin=float(delta.min().min()),vmax=float(delta.max().max())); im=ax.imshow(delta.values,aspect='auto',origin='lower',norm=norm,cmap='RdBu_r')
ax.set_xticks(range(len(delta.columns)),[f'{x:g}' for x in delta.columns]); ax.set_yticks(range(len(delta.index)),[f'{x/1000:g}' for x in delta.index]); ax.set_xlabel('H$_2$-to-power capacity (GW)'); ax.set_ylabel('H$_2$ storage energy (TWh)'); ax.set_title('Detailed phase surface: 2006',loc='left',weight='bold'); fig.colorbar(im,ax=ax,label=r'$\Delta$ unserved energy (GWh)')
sh=pd.read_csv(RED/'observed_2006_shapley.csv'); comps=['Wind','Solar','Demand']; x=np.arange(3); w=.34
for i,(sys,label) in enumerate([('Electricity only','Electricity only'),('H2 + reconversion','H$_2$ + reconversion')]):
    z=sh[sh.system==sys].set_index('component').loc[comps]; axs[1].bar(x+(i-.5)*w,z.shapley_eue_gwh,w,label=label)
axs[1].set_xticks(x,comps); axs[1].set_ylabel('Shapley contribution to 2006 unserved energy (GWh)'); axs[1].set_title('Observed 2006 tail-risk attribution',loc='left',weight='bold'); axs[1].legend(frameon=False); axs[1].spines[['top','right']].set_visible(False)
fig.tight_layout(); save(fig,'figS3_2006_phase_and_attribution')

# Supplementary Figure 4: 12-country direct-observation validation.
mech=pd.read_csv(EXT/'external_mechanism_test_6h.csv'); thr=pd.read_csv(EXT/'external_threshold_refined_6h.csv'); res=pd.read_csv(EXT/'external_resolution_sensitivity.csv'); design=json.loads((EXT/'external_validation_design.json').read_text())
fig=plt.figure(figsize=(7.3,7.0)); gs=fig.add_gridspec(3,2,height_ratios=[1.25,1,1],hspace=.42,wspace=.32)
ax=fig.add_subplot(gs[0,:]); buses=pd.read_csv(ROOT/'data'/'external'/'pypsa_eur_osm_v06'/'buses.csv'); lines=pd.read_csv(ROOT/'data'/'external'/'pypsa_eur_osm_v06'/'lines.csv',quotechar="'")
for geom in lines.geometry:
    try:
        g=wkt.loads(geom); xx,yy=g.xy; ax.plot(xx,yy,lw=.12,alpha=.30)
    except Exception: pass
active=set(design['active_countries']); ba=buses[buses.country.isin(active)]; bp=buses[~buses.country.isin(active)]; ax.scatter(bp.x,bp.y,s=.4,alpha=.25); ax.scatter(ba.x,ba.y,s=1.2,alpha=.8); ax.set_xlim(-11,32); ax.set_ylim(34,62); ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False); ax.set_title('12-country direct-observation validation network',loc='left',weight='bold')
piv=mech.pivot(index='year',columns='scenario',values='eue_gwh'); ax=fig.add_subplot(gs[1,0]); x=np.arange(len(piv)); w=.2
for i,s in enumerate(['reference','h2_rigid','h2_flexible','h2_central']): ax.bar(x+(i-1.5)*w,piv[s],w,label=s.replace('h2_',''))
ax.set_xticks(x,piv.index); ax.set_ylabel('Unserved energy (GWh)'); ax.set_title('Independent mechanism replication',loc='left',weight='bold'); ax.legend(frameon=False,ncol=2); ax.spines[['top','right']].set_visible(False)
ax=fig.add_subplot(gs[1,1]); st=thr[thr.slice=='storage'];
for y in [2016,2017,2019]:
    z=st[st.year==y].sort_values('physical_value'); ax.plot(z.physical_value,z.eue_gwh,marker='o',ms=3,label=str(y))
ax.set_xlabel('Hydrogen storage (TWh)'); ax.set_ylabel('Unserved energy (GWh)'); ax.set_title('Storage threshold shifts externally',loc='left',weight='bold'); ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False)
ax=fig.add_subplot(gs[2,0]); pw=thr[thr.slice=='power'];
for y in [2016,2017,2019]:
    z=pw[pw.year==y].sort_values('physical_value'); ax.plot(z.physical_value,z.eue_gwh,marker='o',ms=3,label=str(y))
ax.set_xlabel('H$_2$-to-power capacity (GW)'); ax.set_ylabel('Unserved energy (GWh)'); ax.set_title('Reconversion remains power-constrained',loc='left',weight='bold'); ax.spines[['top','right']].set_visible(False)
ax=fig.add_subplot(gs[2,1]); rr=res.pivot(index='year',columns='resolution_h',values='fraction_avoided')*100; x=np.arange(len(rr)); ax.bar(x-.18,rr[6],.36,label='6 h'); ax.bar(x+.18,rr[3],.36,label='3 h'); ax.set_xticks(x,rr.index); ax.set_ylim(90,101); ax.set_ylabel('Reference EUE avoided (%)'); ax.set_title('Benefit survives finer chronology',loc='left',weight='bold'); ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False)
fig.tight_layout(); save(fig,'figS4_direct_observation_validation')

# Supplementary Figure 5: structural robustness of direct-observation validation.
rob=pd.read_csv(EXT/'external_structural_robustness_6h.csv'); order=[('transmission_derate','0.2'),('transmission_derate','0.35'),('transmission_derate','0.5'),('h2_geography','national'),('vre_energy_share','0.6'),('vre_energy_share','0.8')]; labels=['Transmission\n20% rating','Transmission\n35% rating','Transmission\n50% rating','National H$_2$\nstorage','VRE energy\n60%','VRE energy\n80%']; refs=[]; hs=[]
for test,val in order:
    z=rob[(rob.test==test)&(rob.value.astype(str)==val)]; refs.append(z.reference_eue_gwh.mean()); hs.append(z.h2_eue_gwh.mean())
fig,ax=plt.subplots(figsize=(7.0,3.6)); x=np.arange(len(order)); w=.36; ax.bar(x-w/2,refs,w,label='Electricity reference'); ax.bar(x+w/2,hs,w,label='H$_2$ + reconversion'); ax.set_xticks(x,labels); ax.set_ylabel('Mean EUE across 2016, 2017 and 2019 (GWh)'); ax.legend(frameon=False,ncol=2); ax.spines[['top','right']].set_visible(False); ax.set_title('Structural robustness of the 12-country validation',loc='left',weight='bold'); fig.tight_layout(); save(fig,'figS5_direct_observation_robustness')

print('Supplementary Figures 1-5 generated')
