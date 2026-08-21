"""Generate final Figure 6 from released numerical outputs."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'results' / 'reduced'
OUTS = [ROOT / 'figures' / 'generated']
for o in OUTS: o.mkdir(parents=True, exist_ok=True)

sc = pd.read_csv(D/'physical_scaling_weather_years.csv')
z = sc[sc.reference_eue_gwh > 1e-6].copy()
stats = pd.read_csv(D/'physical_scaling_statistics.csv')
dur = pd.read_csv(D/'duration_storage_summary.csv')
econ = pd.read_csv(D/'resilience_economic_boundary.csv')

C = {'green':'#009E73','purple':'#7A4E9D','grey':'#626D78','light':'#D9DEE3'}
plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':10.1,'axes.labelsize':8.8,
    'xtick.labelsize':7.8,'ytick.labelsize':7.8,'legend.fontsize':7.2,
    'axes.linewidth':0.75,'pdf.fonttype':42,'ps.fonttype':42
})

def panel(ax, letter):
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontweight='bold', fontsize=12.5, va='bottom')

def tidy(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', color=C['light'], lw=.6, ls='--', alpha=.8)
    ax.set_axisbelow(True)

fig = plt.figure(figsize=(7.25, 6.05))
gs = fig.add_gridspec(2,2,left=.08,right=.965,top=.965,bottom=.10,wspace=.35,hspace=.42)

ax=fig.add_subplot(gs[0,0]); panel(ax,'a'); tidy(ax)
ax.scatter(z.annual_scarcity_energy_gwh,z.minimum_tested_storage_zero_shortage_twh,s=32,color=C['green'],edgecolor='white',lw=.5,zorder=3)
coef=np.polyfit(z.annual_scarcity_energy_gwh,z.minimum_tested_storage_zero_shortage_twh,1)
xx=np.linspace(0,z.annual_scarcity_energy_gwh.max()*1.03,100); ax.plot(xx,np.polyval(coef,xx),color=C['grey'],lw=1.25)
r=stats[(stats.target=='minimum_tested_storage_zero_shortage_twh')&(stats.predictor=='annual_scarcity_energy_gwh')].iloc[0]
ax.text(.05,.94,f"Spearman $\\rho$ = {r.spearman_rho:.2f}\n$p$ = {r.spearman_p:.1e}",transform=ax.transAxes,va='top',fontsize=7.4)
ax.set_xlabel('Annual scarcity energy (GWh)'); ax.set_ylabel('Minimum tested H$_2$ storage\nfor zero shortage (TWh)')
ax.set_title('Storage requirement tracks scarcity energy',loc='left',weight='bold',pad=6)

ax=fig.add_subplot(gs[0,1]); panel(ax,'b'); tidy(ax)
ax.scatter(z.peak_shortfall_gw,z.reconversion_saturation_grid_gw,s=32,color=C['purple'],edgecolor='white',lw=.5,zorder=3)
coef=np.polyfit(z.peak_shortfall_gw,z.reconversion_saturation_grid_gw,1)
xx=np.linspace(0,z.peak_shortfall_gw.max()*1.05,100); ax.plot(xx,np.polyval(coef,xx),color=C['grey'],lw=1.25)
r=stats[(stats.target=='reconversion_saturation_grid_gw')&(stats.predictor=='peak_shortfall_gw')].iloc[0]
ax.text(.05,.94,f"Spearman $\\rho$ = {r.spearman_rho:.2f}\n$p$ = {r.spearman_p:.1e}",transform=ax.transAxes,va='top',fontsize=7.4)
ax.set_xlabel('Peak shortage power (GW)'); ax.set_ylabel('Reconversion capacity at\ntested benefit saturation (GW)')
ax.set_title('Discharge power tracks peak deficit',loc='left',weight='bold',pad=6)

ax=fig.add_subplot(gs[1,0]); panel(ax,'c'); tidy(ax)
order=['Battery +15 GW, 4 h','Battery +15 GW, 24 h','Hydrogen 1 TWh + 15 GW','Battery +15 GW, 100 h']
dd=dur.set_index('technology').loc[order].reset_index(); x=np.arange(4)
ax.bar(x,dd.mean_eue_avoided_gwh,color=['#A9CAE2','#5D9ED1',C['green'],'#385C9D'],width=.76)
ax.set_xticks(x,['4 h\nbattery','24 h\nbattery','H$_2$\n1 TWh store','100 h\nbattery'])
ax.set_ylabel('Mean EUE avoided (GWh)'); ax.set_title('Duration governs deep-event protection',loc='left',weight='bold',pad=6)
ax.text(.04,.94,'Three hardest held-out years',transform=ax.transAxes,va='top',color=C['grey'],fontsize=7.2)

ax=fig.add_subplot(gs[1,1]); panel(ax,'d')
p=econ.pivot(index='storage_capex_eur_per_kwh',columns='reconversion_capex_eur_per_kw',values='breakeven_reliability_value_eur_per_mwh')
X,Y=np.meshgrid(p.columns,p.index)
levels=[0,2500,5000,7500,10000,15000,25000]
cs=ax.contourf(X,Y,p.values,levels=levels,cmap='YlGnBu')
cb=fig.colorbar(cs,ax=ax,pad=.025,shrink=.92); cb.set_label('Break-even reliability value\n(EUR/MWh avoided EUE)',fontsize=7.0); cb.ax.tick_params(labelsize=6.5)
ax.scatter([476.539],[6.01],s=52,color='black',marker='*',zorder=5)
ax.annotate('2030 anchor\n~EUR7.2k/MWh',xy=(476.539,6.01),xytext=(620,3.25),arrowprops=dict(arrowstyle='->',lw=.8),fontsize=7.0)
ax.set_xlabel('Reconversion CAPEX (EUR/kW)'); ax.set_ylabel('H$_2$ storage CAPEX (EUR/kWh)')
ax.set_title('Economics is a reliability-value boundary',loc='left',weight='bold',pad=6)

for out in OUTS:
    fig.savefig(out/'fig6_design_principles_and_economics.pdf',bbox_inches='tight',pad_inches=.04)
    fig.savefig(out/'fig6_design_principles_and_economics.png',dpi=600,bbox_inches='tight',pad_inches=.04)
plt.close(fig)
print('Final Figure 6 generated')
