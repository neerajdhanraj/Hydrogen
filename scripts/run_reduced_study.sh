#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATA="data/real/MERRA2_subset_demand_wind_solar.csv"
mkdir -p results/reduced

python scripts/run_historical_ensemble.py --data "$DATA" --out results/reduced/reference_ensemble.csv --set reference --workers 4 --years $(seq 1980 2017)
python scripts/run_historical_ensemble.py --data "$DATA" --out results/reduced/mechanism_test_ensemble.csv --set mechanisms --workers 4 --years $(seq 1999 2017)
python scripts/run_historical_ensemble.py --data "$DATA" --out results/reduced/counterfactual_test_ensemble.csv --set counterfactuals --workers 4 --years $(seq 1999 2017)
python scripts/run_storage_power_surface.py --data "$DATA" --out results/reduced/h2_phase_2006.csv --year 2006 --storage 0 100 500 1000 2000 4000 --fc-each 0 1 3 5 8 --workers 4
python scripts/run_event_attribution.py --mode electricity
python scripts/run_event_attribution.py --mode h2
python scripts/run_efficiency_sensitivity.py --years $(seq 1999 2017) --out results/reduced/h2_efficiency_sensitivity_2030.csv --workers 4
python scripts/run_initial_inventory_sensitivity.py
python scripts/run_storage_power_slices.py --data "$DATA" --out results/reduced/cross_weather_storage_slice_new.csv --mode storage --workers 4 --years $(seq 1999 2017)
python scripts/run_storage_power_slices.py --data "$DATA" --out results/reduced/cross_weather_power_slice_new.csv --mode power --workers 4 --years $(seq 1999 2017)
python scripts/analyze_storage_power_slices.py
python scripts/build_reserve_policy.py
python scripts/run_limited_foresight.py --data "$DATA" --out results/reduced/limited_foresight_top6_stress.csv --years 2006 2009 2002 2000 2012 2003 --lookahead 72 168 --commit-h 24 --workers 4
python scripts/analyze_limited_foresight.py
python scripts/export_2006_event_timeseries.py
python scripts/analyze_reduced_study.py
python scripts/run_duration_comparison.py
python scripts/analyze_physical_scaling.py
python scripts/analyze_resilience_economics.py
