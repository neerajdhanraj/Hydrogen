# Historical weather-energy input provenance

File used by the v3 mechanistic study:

`MERRA2_subset_demand_wind_solar.csv`

SHA-256:

`f8f667495d3d4b075d99e098ce89f0ca9f7851d1f2a79a4c73b91b032de8a8b7`

The series are a reproducible subset of the public MERRA-2-derived European country-aggregate demand, wind-power and solar-power dataset documented as:

Bloomfield, H., Brayshaw, D. & Charlton-Perez, A. **MERRA2 derived time series of European country-aggregate electricity demand, wind power generation and solar power generation.** University of Reading Dataset (2020). DOI: `10.17864/1947.239`.

The reduced six-region mapping/topology follows the open Hilbers renewable test power-system benchmark associated with:

Hilbers, A. P., Brayshaw, D. J. & Gandy, A. **Importance subsampling: improving power system planning under climate-based uncertainty.** *Applied Energy* 251, 113114 (2019). DOI: `10.1016/j.apenergy.2019.04.110`.

The local file contains 38 complete non-leap 8,760-hour years (1980-2017). Columns used by the study are German, French and United Kingdom demand; German, United Kingdom and Spanish wind; and German, United Kingdom and Spanish solar. Leap days are absent in the benchmark time series, allowing each historical year to be solved as an equal-length 8,760-hour dispatch problem.

The v3 package does not claim that this reduced subset is equivalent to a high-resolution ERA5/PyPSA-Eur dataset. It is used to establish the mechanistic sign/threshold result under real multi-decadal weather variability; high-resolution ERA5/PyPSA-Eur replication is the recommended external-validity gate before final journal submission.
