from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from shapely import wkt

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'results'/'reduced'; F=ROOT/'figures'/'generated'
F.mkdir(parents=True, exist_ok=True)

# Nature-style compact graphics. Exact numerical source data only.
plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':7.6,'axes.titlesize':8.5,'axes.labelsize':7.8,
    'xtick.labelsize':7.0,'ytick.labelsize':7.0,'legend.fontsize':6.8,'axes.linewidth':0.75,
    'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','figure.facecolor':'white'
})
C={'blue':'#0072B2','sky':'#56B4E9','green':'#009E73','orange':'#E69F00','red':'#D55E00','purple':'#7A4E9D','grey':'#626D78','black':'#202124','light':'#D9DEE3','light2':'#EEF1F4'}

def tidy(ax, grid=True):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(direction='out',length=3,width=.7)
    if grid:
        ax.grid(axis='y',color=C['light'],lw=.55,ls='--',alpha=.8,zorder=0)

def panel(ax,l):
    ax.text(-.12,1.06,l,transform=ax.transAxes,fontweight='bold',fontsize=11,va='bottom')

def save(fig,name):
    fig.savefig(F/f'{name}.pdf',bbox_inches='tight',pad_inches=.03)
    fig.savefig(F/f'{name}.png',dpi=600,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)

# ---------------- Figure 1 ----------------
fig=plt.figure(figsize=(7.15,3.55),layout='constrained')
gs=fig.add_gridspec(2,2,height_ratios=[.95,1.35],wspace=.20,hspace=.25)
ax=fig.add_subplot(gs[0,:]); ax.axis('off'); panel(ax,'a'); ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.text(.50,.88,'Historical demand, wind and solar',ha='center',weight='bold',fontsize=10.5)
ax.text(.50,.70,'MERRA-2-derived country-hourly series',ha='center',color=C['grey'],fontsize=8.3)
x0,xm,x1=.08,.50,.92; y=.32
ax.add_patch(Rectangle((x0,y-.07),xm-x0,.14,fc='#E4F1F8',ec='none'))
ax.add_patch(Rectangle((xm,y-.07),x1-xm,.14,fc='#E7F5EE',ec='none'))
ax.plot([x0,x1],[y,y],color=C['black'],lw=1.1)
for x in [x0,xm,x1]: ax.plot([x,x],[y-.14,y+.14],color=C['black'],lw=1.0)
ax.text((x0+xm)/2,.53,'Calibration period',ha='center',weight='bold',color=C['blue'],fontsize=8.8)
ax.text((xm+x1)/2,.53,'Held-out resilience test',ha='center',weight='bold',color=C['green'],fontsize=8.8)
ax.text((x0+xm)/2,.10,'1980-1998\n19 hourly weather years',ha='center',va='top',color=C['blue'],linespacing=1.3)
ax.text((xm+x1)/2,.10,'1999-2017\n19 unseen weather years',ha='center',va='top',color=C['green'],linespacing=1.3)

ax=fig.add_subplot(gs[1,0]); ax.axis('off'); panel(ax,'b'); ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.text(.5,.98,'Reduced-order European electricity benchmark',ha='center',va='top',weight='bold',fontsize=8.8)
pos={1:(.10,.44),2:(.36,.76),3:(.57,.57),4:(.82,.76),5:(.82,.23),6:(.40,.15)}
edges=[(1,2),(1,3),(1,5),(1,6),(2,3),(3,4),(4,5),(5,6)]
for a,b in edges: ax.plot([pos[a][0],pos[b][0]],[pos[a][1],pos[b][1]],color='#A6ADB5',lw=1.8,zorder=1)
labels={1:'R1\nfirm',2:'R2\nGermany',3:'R3\nfirm',4:'R4\nFrance',5:'R5\nUnited\nKingdom',6:'R6\nSpain'}
for n,(x,y0) in pos.items():
    ax.add_patch(Circle((x,y0),.073,fc='white',ec=C['black'],lw=.9,zorder=2))
    ax.text(x,y0,labels[n],ha='center',va='center',fontsize=6.5,weight='bold' if n in [2,4,5,6] else None,zorder=3,linespacing=1.05)
    if n in [2,5,6]: ax.scatter(x+.052,y0-.063,s=24,color=C['green'],edgecolor='white',lw=.4,zorder=4)
ax.scatter(.57,.29,s=28,color=C['green'],edgecolor='white',lw=.4)
ax.text(.62,.29,'H$_2$-coupled\nregions',va='center',fontsize=7.2,color=C['green'])

ax=fig.add_subplot(gs[1,1]); ax.axis('off'); panel(ax,'c'); ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.text(.5,.98,'Mechanism-first hydrogen experiment',ha='center',va='top',weight='bold',fontsize=8.8)
boxes=[(.02,.60,.21,.18,'Rigid H$_2$\nproduction',C['red']),(.27,.60,.21,.18,'Flexible\nelectrolysis',C['blue']),(.52,.60,.21,.18,'H$_2$\nstorage',C['green']),(.77,.60,.20,.18,'H$_2$ $\to$ power',C['purple'])]
for x,y0,w,h,t,c in boxes:
    ax.add_patch(FancyBboxPatch((x,y0),w,h,boxstyle='round,pad=.008,rounding_size=.012',fc='white',ec=c,lw=1.35))
    ax.text(x+w/2,y0+h/2,t,ha='center',va='center',color=c,weight='bold',fontsize=7.2,linespacing=1.05)
for i in range(3):
    x1=boxes[i][0]+boxes[i][2]; x2=boxes[i+1][0]
    ax.add_patch(FancyArrowPatch((x1+.006,.69),(x2-.006,.69),arrowstyle='-|>',mutation_scale=8,lw=.8,color=C['black']))
ax.text(.5,.45,'Three governing quantities',ha='center',weight='bold',fontsize=8.1)
for yy,lhs,rhs in [(.32,'weather deficit energy','storage requirement'),(.20,'weather deficit power','reconversion requirement'),(.08,'forecast horizon','accessible resilience')]:
    ax.text(.04,yy,lhs,va='center',fontsize=6.9)
    ax.add_patch(FancyArrowPatch((.42,yy),(.56,yy),arrowstyle='-|>',mutation_scale=8,lw=.75,color=C['grey']))
    ax.text(.60,yy,rhs,va='center',fontsize=6.9,weight='bold')
save(fig,'fig1_study_design_full')

# ---------------- Figure 2 ----------------
mech=pd.read_csv(D/'mechanism_test_ensemble.csv'); st=pd.read_csv(D/'mechanism_statistical_summary.csv')
order=['Electricity only','H2 rigid load','H2 flexible production','H2 + reconversion']
disp=['Electricity\nonly','Rigid\nH$_2$','Flexible\nH$_2$','H$_2$ +\nreconv.']; cols=[C['grey'],C['red'],C['blue'],C['green']]
rng=np.random.default_rng(20260814)
fig=plt.figure(figsize=(7.15,4.2),layout='constrained'); gs=fig.add_gridspec(1,3,width_ratios=[1.3,1.08,1.0],wspace=.18)
for j,(metric,ylabel,title,ylim) in enumerate([('eue_gwh','Annual unserved energy (GWh)','Annual shortage energy',(-40,2900)),('lole_h','Annual system shortage hours (h)','Shortage duration',(-2,120))]):
    ax=fig.add_subplot(gs[0,j]); panel(ax,chr(97+j)); tidy(ax,True)
    for i,(sc,col) in enumerate(zip(order,cols)):
        vals=mech.loc[mech.scenario==sc,metric].to_numpy(); q1,med,q3=np.percentile(vals,[25,50,75]); jit=rng.normal(0,.045,len(vals))
        ax.scatter(np.full(len(vals),i)+jit,vals,s=13,facecolor=col,edgecolor='white',lw=.35,alpha=.88,zorder=3)
        ax.plot([i-.15,i+.15],[med,med],color=C['black'],lw=1.2,zorder=4)
        ax.plot([i,i],[q1,q3],color=C['black'],lw=2.7,zorder=3)
    ax.set_xticks(range(4),disp); ax.set_ylabel(ylabel); ax.set_ylim(*ylim); ax.set_title(title,loc='left',weight='bold',pad=8)
    if j==0: ax.text(.02,.98,'n = 19 years',transform=ax.transAxes,va='top',color=C['grey'])
ax=fig.add_subplot(gs[0,2]); panel(ax,'c'); tidy(ax,False)
sub=st.set_index('scenario').loc[['H2 rigid load','H2 flexible production','H2 + reconversion']]
y=np.arange(3); mean=-sub.mean_eue_avoided_gwh.to_numpy(); lo=-sub.ci95_high_gwh.to_numpy(); hi=-sub.ci95_low_gwh.to_numpy()
for yi,m,l,h,col in zip(y,mean,lo,hi,[C['red'],C['blue'],C['green']]):
    ax.errorbar(m,yi,xerr=np.array([[m-l],[h-m]]),fmt='o',ms=6,color=col,ecolor=col,lw=1.5,capsize=3,mec='white',mew=.5,zorder=3)
    # Put value labels outside the confidence whiskers so text never collides with markers or intervals.
    xpos=(h+35) if m>=0 else (l-35)
    ax.text(xpos,yi,f'{m:+.0f}',ha='left' if m>=0 else 'right',va='center',fontsize=7.0,
            bbox=dict(fc='white',ec='none',alpha=.92,pad=.15),zorder=5)
ax.axvline(0,color=C['black'],lw=.9); ax.set_yticks(y,['Rigid H$_2$','Flexible H$_2$','H$_2$ +\nreconv.']); ax.invert_yaxis(); ax.set_xlim(-400,800)
ax.set_xlabel(r'$\Delta$ unserved energy (GWh)'); ax.set_title('Paired effect',loc='left',weight='bold',pad=8); ax.grid(axis='x',color=C['light'],lw=.55,ls='--')
save(fig,'fig2_hydrogen_mechanisms')

# ---------------- Figure 3 ----------------
raw=pd.read_csv(ROOT/'data/real/MERRA2_subset_demand_wind_solar.csv',parse_dates=['t'])
train=raw[(raw.t.dt.year>=1980)&(raw.t.dt.year<=1998)].copy(); train['hoy']=train.groupby(train.t.dt.year).cumcount()
wc=['demand_region2','demand_region4','demand_region5','wind_region2','wind_region5','wind_region6','solar_region2','solar_region5','solar_region6']; clim=train.groupby('hoy')[wc].mean().reset_index(drop=True)
act=raw[raw.t.dt.year==2006].reset_index(drop=True).copy();
for c in wc: act[c+'_clim']=clim[c].to_numpy()
act['demand']=act[['demand_region2','demand_region4','demand_region5']].sum(axis=1); act['demand_clim']=act[[c+'_clim' for c in ['demand_region2','demand_region4','demand_region5']]].sum(axis=1)
act['wind_cf']=(40*act.wind_region2+40*act.wind_region5+30*act.wind_region6)/110; act['wind_cf_clim']=(40*act.wind_region2_clim+40*act.wind_region5_clim+30*act.wind_region6_clim)/110
act['solar_cf']=(20*act.solar_region2+30*act.solar_region5+20*act.solar_region6)/70; act['solar_cf_clim']=(20*act.solar_region2_clim+30*act.solar_region5_clim+20*act.solar_region6_clim)/70
b=pd.read_csv(D/'timeseries_2006_baseline.csv',parse_dates=['t']); h=pd.read_csv(D/'timeseries_2006_h2.csv',parse_dates=['t'])
start=pd.Timestamp('2006-01-29'); end=pd.Timestamp('2006-02-06'); w=act[(act.t>=start)&(act.t<end)]; bw=b[(b.t>=start)&(b.t<end)]; hw=h[(h.t>=start)&(h.t<end)]
fig=plt.figure(figsize=(7.15,5.75),layout='constrained'); gs=fig.add_gridspec(4,1,height_ratios=[1,1,1,1.2],hspace=.05)
ax=fig.add_subplot(gs[0]); panel(ax,'a'); tidy(ax,True); ax.plot(w.t,w.demand,color=C['black'],lw=1.25,label='Observed 2006'); ax.plot(w.t,w.demand_clim,color=C['grey'],lw=1,ls='--',label='1980-1998 climatology'); ax.set_ylabel('Electricity\ndemand (GW)'); ax.text(.01,.94,'Demand stress',transform=ax.transAxes,weight='bold',va='top'); ax.legend(frameon=False,ncol=2,loc='upper right',bbox_to_anchor=(1,.995)); ax.tick_params(labelbottom=False)
ax=fig.add_subplot(gs[1]); panel(ax,'b'); tidy(ax,True); ax.plot(w.t,w.wind_cf,color=C['blue'],lw=1.2,label='Wind'); ax.plot(w.t,w.wind_cf_clim,color=C['sky'],lw=1,ls='--',label='Wind climatology'); ax.plot(w.t,w.solar_cf,color=C['orange'],lw=1.2,label='Solar'); ax.plot(w.t,w.solar_cf_clim,color='#F4C76A',lw=1,ls='--',label='Solar climatology'); ax.set_ylabel('Capacity factor'); ax.set_ylim(0,1); ax.text(.01,.94,'Renewable availability',transform=ax.transAxes,weight='bold',va='top'); ax.legend(frameon=False,ncol=2,loc='upper right',bbox_to_anchor=(1,.995)); ax.tick_params(labelbottom=False)
ax=fig.add_subplot(gs[2]); panel(ax,'c'); tidy(ax,True); ax.fill_between(bw.t,0,bw.shortage_gw,color=C['grey'],alpha=.28,label='Electricity only'); ax.plot(bw.t,bw.shortage_gw,color=C['grey'],lw=.8); ax.fill_between(hw.t,0,hw.shortage_gw,color=C['green'],alpha=.38,label='H$_2$ + reconversion'); ax.plot(hw.t,hw.shortage_gw,color=C['green'],lw=.9); ax.set_ylabel('Unserved\npower (GW)'); ax.set_ylim(0,46); ax.text(.01,.94,'System shortage',transform=ax.transAxes,weight='bold',va='top'); ax.legend(frameon=False,ncol=2,loc='upper right',bbox_to_anchor=(1,.995)); ax.tick_params(labelbottom=False)
ax=fig.add_subplot(gs[3]); panel(ax,'d'); tidy(ax,True); ax.fill_between(hw.t,0,-hw.electrolysis_gw,color=C['sky'],alpha=.48,label='Electrolysis'); ax.fill_between(hw.t,0,hw.fuelcell_gw,color=C['green'],alpha=.48,label='H$_2$-to-power'); ax.axhline(0,color=C['black'],lw=.65); ax.set_ylabel('H$_2$ conversion\npower (GW)'); ax.text(.01,.94,'Hydrogen response',transform=ax.transAxes,weight='bold',va='top'); ax2=ax.twinx(); ax2.plot(hw.t,hw.h2_inventory_gwh/1000,color=C['purple'],lw=1.3,label='H$_2$ inventory'); ax2.set_ylabel('H$_2$ inventory (TWh)',color=C['purple']); ax2.tick_params(axis='y',colors=C['purple']); ax2.spines['top'].set_visible(False); ax2.spines['right'].set_color(C['purple']); ln,lb=ax.get_legend_handles_labels(); ln2,lb2=ax2.get_legend_handles_labels(); ax.legend(ln+ln2,lb+lb2,frameon=False,ncol=3,loc='lower left',bbox_to_anchor=(.0,.01))
for a in [fig.axes[0],fig.axes[1],fig.axes[2],fig.axes[3]]: a.xaxis.set_major_locator(mdates.DayLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
save(fig,'fig3_observed_2006_event')

# ---------------- Figure 4 ----------------
storage_phase=pd.read_csv(D/'cross_weather_storage_phase.csv'); power_phase=pd.read_csv(D/'cross_weather_power_phase.csv'); op=pd.read_csv(D/'operational_foresight_detail.csv')
yorder=(storage_phase[['year','reference_eue_gwh']].drop_duplicates().sort_values(['reference_eue_gwh','year'],ascending=[False,True]).year.tolist()); svals=sorted(storage_phase.storage_twh.unique())
mat=(storage_phase.assign(delta=lambda x:x.eue_gwh-x.reference_eue_gwh).pivot(index='year',columns='storage_twh',values='delta').loc[yorder,svals].to_numpy())
fig=plt.figure(figsize=(7.15,5.35),layout='constrained'); gs=fig.add_gridspec(2,2,width_ratios=[1.08,1],wspace=.18,hspace=.20)
ax=fig.add_subplot(gs[0,0]); panel(ax,'a'); norm=TwoSlopeNorm(vmin=-1200,vcenter=0,vmax=1500); im=ax.imshow(mat,origin='upper',aspect='auto',cmap='RdBu_r',norm=norm); ax.set_xticks(range(len(svals)),[f'{x:g}' for x in svals]); ax.set_yticks(range(len(yorder)),[str(y) for y in yorder]); ax.set_xlabel('H$_2$ storage energy (TWh)'); ax.set_ylabel('Weather year'); ax.set_title('Weather-year sign map at 15 GW',loc='left',weight='bold',pad=6)
for y0 in [2001,2006,2017]:
    if y0 in yorder: ax.scatter([1],[yorder.index(y0)],s=34,facecolors='none',edgecolors=C['black'],lw=.9)
cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.02); cb.set_label(r'$\Delta$EUE (GWh yr$^{-1}$)',fontsize=7.2); cb.ax.tick_params(labelsize=6.5)

ax=fig.add_subplot(gs[0,1]); panel(ax,'b'); tidy(ax,True); agg=[]
for s,g in storage_phase.groupby('storage_twh'):
    vals=g.reference_eue_gwh-g.eue_gwh; agg.append((s,vals.mean(),np.percentile(vals,2.5),np.percentile(vals,97.5)))
a=pd.DataFrame(agg,columns=['x','m','lo','hi']).sort_values('x'); ax.errorbar(a.x,a.m,yerr=[a.m-a.lo,a.hi-a.m],fmt='o-',color=C['green'],mec='white',mew=.5,lw=1.3,capsize=2.5); ax.axhline(0,color=C['grey'],lw=.8,ls='--'); ax.set_xlabel('H$_2$ storage energy (TWh)'); ax.set_ylabel('Mean EUE avoided (GWh yr$^{-1}$)'); ax.set_title('Storage crosses the ensemble zero',loc='left',weight='bold',pad=6); ax.text(.04,.96,'0.5 TWh: 9 improve, 3 worsen\n1 TWh: 12 improve, 0 worsen\n2 TWh: zero shortage in all 19 years',transform=ax.transAxes,va='top',fontsize=6.7,bbox=dict(fc='white',ec='none',alpha=.86,pad=1.5))

ax=fig.add_subplot(gs[1,0]); panel(ax,'c'); tidy(ax,True); agg=[]
for p,g in power_phase.groupby('reconversion_gw'):
    vals=g.reference_eue_gwh-g.eue_gwh; agg.append((p,vals.mean(),np.percentile(vals,2.5),np.percentile(vals,97.5)))
a=pd.DataFrame(agg,columns=['x','m','lo','hi']).sort_values('x'); ax.errorbar(a.x,a.m,yerr=[a.m-a.lo,a.hi-a.m],fmt='o-',color=C['purple'],mec='white',mew=.5,lw=1.3,capsize=2.5); ax.axhline(0,color=C['grey'],lw=.8,ls='--'); ax.set_xlabel('H$_2$-to-power capacity (GW), 1 TWh store'); ax.set_ylabel('Mean EUE avoided (GWh yr$^{-1}$)'); ax.set_title('Reconversion power controls benefit',loc='left',weight='bold',pad=6); ax.text(.04,.95,'3 GW improves all 12 stressed years.\nBenefit reaches the tested plateau by 9 GW.',transform=ax.transAxes,va='top',fontsize=6.7,bbox=dict(fc='white',ec='none',alpha=.88,pad=1.5))

ax=fig.add_subplot(gs[1,1]); panel(ax,'d'); tidy(ax,True); sel=[2006,2009,2002,2000,2012,2003]; styles=[('Annual perfect foresight',C['grey'],'o'),('3-day look-ahead',C['red'],'x'),('7-day look-ahead',C['green'],'o')]
# infer columns robustly
# Annual perfect-foresight benefit is stored alongside each rolling run; use one copy per year.
pf=(op.groupby('year',as_index=False)['annual_pf_eue_avoided_gwh'].first().set_index('year').reindex(sel))
ax.plot(range(len(sel)),pf['annual_pf_eue_avoided_gwh'],marker='o',color=C['grey'],lw=1.2,ms=4.2,label='Annual perfect foresight')
for hours,label,col,mk in [(72,'3-day look-ahead',C['red'],'x'),(168,'7-day look-ahead',C['green'],'o')]:
    z=op[op.lookahead_h==hours].set_index('year').reindex(sel)
    ax.plot(range(len(sel)),z['rolling_eue_avoided_gwh'],marker=mk,color=col,lw=1.2,ms=4.2,label=label)
ax.axhline(0,color=C['grey'],lw=.8,ls='--'); ax.set_xticks(range(len(sel)),[str(y) for y in sel],rotation=35,ha='right'); ax.set_ylabel('EUE avoided (GWh)'); ax.set_title('Benefit under limited foresight',loc='left',weight='bold',pad=6); ax.legend(frameon=False,loc='lower right',fontsize=6.4)
save(fig,'fig4_cross_weather_thresholds_and_foresight')

print('Figures 1-4 generated')
