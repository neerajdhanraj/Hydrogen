# Hydrogen-enabled power-system resilience: reproducibility code

This repository contains the study-specific code, analysis-ready input data, and frozen numerical outputs for **“Energy-power-operation boundaries for hydrogen-enabled power-system resilience.”**

The release is limited to computational material that supports results reported in the manuscript or Supplementary Information. Development utilities, superseded scripts, caches, draft files, and unrelated analyses are excluded.

## Repository structure

- `scripts/` — modelling, scenario, sensitivity, statistical-analysis, validation, and figure scripts.
- `data/real/` — 38-year MERRA-2-derived demand, wind, and solar data used by the reduced benchmark.
- `data/external/all34_opsd/` — 34-country 2015–2018 chronology and source audit used by the European-scale replication.
- `data/external/opsd/` — direct-observation chronology used by the 12-country robustness test.
- `data/external/pypsa_eur_osm_v06/` — PyPSA-Eur/OpenStreetMap-derived network tables used in the country-interface analyses.
- `data/technology/` — technology assumptions used in the efficiency and economic sensitivities.
- `results/` — frozen numerical outputs used for the reported values and figures.
- `figures/reference/` — reference figure images corresponding to the submitted manuscript.
- `figures/generated/` — created when figure scripts are run.

## Manuscript-to-code map

| Reported analysis | Primary code |
|---|---|
| Reduced historical-weather dispatch | `reduced_model.py`, `run_historical_ensemble.py` |
| Rigid, flexible, and reconversion hydrogen mechanisms | `run_historical_ensemble.py` |
| 2006 storage–power surface | `run_storage_power_surface.py` |
| Observed wind/solar/demand attribution | `run_event_attribution.py` |
| Conversion-efficiency sensitivity | `run_efficiency_sensitivity.py` |
| Fixed start/end hydrogen inventory sensitivity | `run_initial_inventory_sensitivity.py` |
| Cross-weather storage and reconversion thresholds | `run_storage_power_slices.py`, `analyze_storage_power_slices.py` |
| Limited operational foresight | `build_reserve_policy.py`, `run_limited_foresight.py`, `analyze_limited_foresight.py` |
| Severe 2006 event time series | `export_2006_event_timeseries.py` |
| Statistical summaries | `analyze_reduced_study.py` |
| Storage-duration comparison and physical scaling | `run_duration_comparison.py`, `analyze_physical_scaling.py` |
| Reliability-value screening | `analyze_resilience_economics.py` |
| 34-country European replication | `all34_country_model.py`, `reproduce_all34.py` |
| 12-country direct-observation validation | `direct_observation_model.py`, `reproduce_direct_observation.py` |
| Main figures | `make_figures_1_to_4.py`, `make_figure_5.py`, `make_figure_6.py` |
| Numerical manuscript cross-check | `validate_reported_results.py` |

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce the reduced benchmark

```bash
bash scripts/run_reduced_study.sh
```

This workflow solves the fixed reference ensemble, hydrogen mechanism cases, 2006 event analyses, cross-weather storage and power slices, efficiency and inventory sensitivities, limited-foresight dispatch, duration comparison, physical-scaling analysis, and economic screening.

## Reproduce the 34-country replication

```bash
python scripts/reproduce_all34.py
```

The model uses the packaged OPSD chronology and PyPSA-Eur/OpenStreetMap-derived country-interface network. It is a fixed-capacity country-interface adequacy experiment, not a native nodal PyPSA-Eur capacity-expansion solve.

## Reproduce the 12-country direct-observation validation

```bash
python scripts/reproduce_direct_observation.py
```

The central validation uses six-hour chronology; the temporal-resolution robustness check uses three-hour chronology. Only countries with complete directly observed load, wind, and solar chronology are active energy nodes in this experiment; the remaining network countries are passive transit nodes.

## Regenerate figures

```bash
python scripts/reproduce_figures.py
```

The plotting workflow uses the released numerical result tables. Reference PNGs are retained separately for visual comparison.

## Validate reported numerical results

```bash
python scripts/validate_reported_results.py
```

This checks headline manuscript and Supplementary Information values against the frozen result tables, including the mechanism results, storage/power thresholds, limited foresight, physical scaling, duration comparison, economic screen, 34-country replication, and 12-country validation.

## Data provenance and licensing

Upstream data sources and redistribution notes are described in `data/external/SOURCE_AND_LICENSE_NOTES.md`, `data/real/PROVENANCE.md`, and `LICENSE_NOTICE.md`. Third-party data retain their upstream terms and attribution requirements.
