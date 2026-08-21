"""Generate final Figure 5 from released all-34-country outputs.

Panel a uses the packaged cartographic rendering asset
figure so the country-background presentation is reproduced exactly. The
underlying OSM-derived buses, branches and country-interface data are released
under data/external/pypsa_eur_osm_v06 and are the source of the graph itself.
Panels b-e are regenerated directly from the released numerical outputs.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "all34"
OUTS = [ROOT / 'figures' / 'generated']
for out in OUTS:
    out.mkdir(parents=True, exist_ok=True)

mech = pd.read_csv(D / "all34_mechanism_6h.csv")
stor = pd.read_csv(D / "all34_storage_slice_2016_6h.csv")
powr = pd.read_csv(D / "all34_power_slice_2016_6h.csv")
map_img = plt.imread(ROOT / "assets" / "fig5_country_map_panel.png")

COL = {
    "reference": "0.45",
    "rigid": "#d97706",
    "flexible": "#5277a8",
    "central": "#168a87",
    "power": "#6f58a8",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.7,
    "axes.titlesize": 10.2,
    "axes.labelsize": 8.7,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.4,
    "axes.linewidth": 0.75,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig = plt.figure(figsize=(7.25, 9.25))
gs = GridSpec(
    3, 2, figure=fig,
    height_ratios=[1.72, 1.0, 1.0],
    hspace=0.38, wspace=0.30,
    left=0.075, right=0.985, top=0.97, bottom=0.155,
)

def panel_letter(ax, letter, x=-0.12, y=1.035):
    ax.text(x, y, letter, transform=ax.transAxes, fontweight="bold", fontsize=12.5,
            va="bottom", ha="left")

def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.6, linestyle="--", alpha=0.8)
    ax.set_axisbelow(True)

# a - final country-background rendering, given extra vertical space to reduce stretching
ax = fig.add_subplot(gs[0, :])
panel_letter(ax, "a", x=-0.055, y=1.01)
ax.imshow(map_img, aspect="auto", interpolation="lanczos")
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_title("34-country European country-interface graph", loc="left", fontweight="bold", pad=7)
ax.text(
    0.005, -0.035,
    "34 active countries  •  74 OSM-derived country corridors  •  no extra passive transit countries",
    transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color="0.25"
)

# b
ax = fig.add_subplot(gs[1, 0]); panel_letter(ax, "b"); clean(ax)
y = mech[mech.year == 2016].set_index("scenario").eue_gwh
keys = ["reference", "h2_rigid", "h2_flexible", "h2_central"]
labels = ["Electricity\nreference", "Rigid H$_2$", "Flexible H$_2$", "H$_2$ +\nreconversion"]
cols = [COL["reference"], COL["rigid"], COL["flexible"], COL["central"]]
vals = [float(y[k]) for k in keys]
bars = ax.bar(np.arange(4), vals, width=0.62, color=cols)
ax.set_ylabel("Unserved energy (GWh)")
ax.set_xticks(range(4), labels)
ax.set_ylim(0, 820)  # explicit headroom keeps the 720 annotation clear of the border
ax.set_title("2016 mechanism replication", loc="left", fontweight="bold", pad=6)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 16, f"{v:.0f}", ha="center", va="bottom", fontsize=7.8)

# c
ax = fig.add_subplot(gs[1, 1]); panel_letter(ax, "c"); clean(ax)
yyrs = [2017, 2018]
x = np.arange(len(yyrs))
width = 0.19
keys = ["reference", "h2_rigid", "h2_flexible", "h2_central"]
for i, (k, c) in enumerate(zip(keys, cols)):
    vv = [float(mech[(mech.year == yr) & (mech.scenario == k)].eue_gwh.iloc[0]) for yr in yyrs]
    ax.bar(x + (i - 1.5)*width, vv, width, color=c)
ax.set_xticks(x, [str(y) for y in yyrs])
ax.set_ylabel("Unserved energy (GWh)")
ax.set_ylim(0, 23.5)
ax.set_title("Rigid demand creates shortage in otherwise adequate years", loc="left", fontweight="bold", fontsize=9.5, pad=6)

# d
ax = fig.add_subplot(gs[2, 0]); panel_letter(ax, "d"); clean(ax)
ax.plot(stor.h2_storage_twh, stor.eue_gwh, marker="o", ms=5.5, lw=1.9, color=COL["central"])
ref = 448.2716898427933
ax.axhline(ref, color="0.5", ls="--", lw=1.0)
ax.text(0.97, 0.94, "Electricity reference", transform=ax.transAxes, ha="right", va="top", fontsize=7.2)
ax.set_xlabel("Hydrogen storage (TWh)")
ax.set_ylabel("2016 unserved energy (GWh)")
ax.set_ylim(0, 470)
ax.set_title("Energy capacity reduces the scarcity tail", loc="left", fontweight="bold", pad=6)
ax.annotate("Power-limited plateau", xy=(15.0, 87.7), xytext=(12.0, 200),
            arrowprops=dict(arrowstyle="->", lw=0.8, color="0.35"), fontsize=7.2, color="0.2")

# e
ax = fig.add_subplot(gs[2, 1]); panel_letter(ax, "e"); clean(ax)
ax.plot(powr.h2_reconversion_gw, powr.eue_gwh, marker="o", ms=5.5, lw=1.9, color=COL["power"])
ax.axhline(ref, color="0.5", ls="--", lw=1.0)
ax.text(0.97, 0.94, "Electricity reference", transform=ax.transAxes, ha="right", va="top", fontsize=7.2)
ax.set_xlabel("H$_2$-to-power capacity (GW)\n(9.65 TWh store)")
ax.set_ylabel("2016 unserved energy (GWh)")
ax.set_ylim(0, 500)
ax.set_title("Reconversion power unlocks stored energy", loc="left", fontweight="bold", pad=6)
ax.text(54.3, 30, "18.6", ha="center", va="bottom", fontsize=7.6)
ax.text(72.4, 12, "0", ha="center", va="bottom", fontsize=7.6)

# consolidated scenario legend
handles = [
    Patch(facecolor=COL["reference"], label="Electricity reference"),
    Patch(facecolor=COL["rigid"], label="Rigid H$_2$"),
    Patch(facecolor=COL["flexible"], label="Flexible H$_2$"),
    Patch(facecolor=COL["central"], label="H$_2$ + reconversion"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.53, 0.035),
           handlelength=1.5, columnspacing=2.4)

for out in OUTS:
    fig.savefig(out / "fig5_all34_validation.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out / "fig5_all34_validation.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print("Final Figure 5 generated")
