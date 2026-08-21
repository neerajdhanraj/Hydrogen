# External data source and license notes

## PyPSA-Eur OpenStreetMap high-voltage network

Source record: Xiong, B., Fioriti, D., Neumann, F., Riepin, I. & Brown, T., *Prebuilt Electricity Network for PyPSA-Eur based on OpenStreetMap Data*, version 0.6, Zenodo DOI 10.5281/zenodo.14144752.

The processed PyPSA-Eur OSM dataset is distributed under the Open Data Commons Open Database License (ODbL 1.0), with underlying OpenStreetMap data subject to ODbL attribution/share-alike requirements. Retain attribution to OpenStreetMap contributors and the PyPSA-Eur dataset when redistributing or adapting these files.

Packaged source-derived files used by v5:
- `pypsa_eur_osm_v06/buses.csv`
- `pypsa_eur_osm_v06/lines.csv`
- `pypsa_eur_osm_v06/links.csv`
- `pypsa_eur_osm_v06/country_interconnects.csv` (derived in this study)

## Open Power System Data time series

Source: Open Power System Data, *Data Package Time series*, version 2020-10-06, DOI 10.25832/time_series/2020-10-06. The package documents hourly load, wind and solar data for European countries, based in this version on TSO and ENTSO-E Transparency sources.

The v5 package stores only the derived, filtered chronology used by the study:
- `opsd/external_profiles_2015_2019.csv.gz`
- `opsd/PROVENANCE.json`

When reusing these data, retain the OPSD attribution and consult the original data package/legal notes for source-specific terms that may apply to primary ENTSO-E/TSO data.
