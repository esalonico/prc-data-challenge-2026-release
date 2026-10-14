# NOTICE

## Code

Copyright (C) 2026 the `unbelievable-snowflake` team, PRC Data Challenge 2026.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License in
[`LICENSE`](LICENSE) for more details.

The GPL covers the source code in this repository. It does not cover the third-party data described below, which
keeps its own licence.

## Data used by the final predictions

The repository contains no challenge data. External data appears only as snapshots (the data files as downloaded)
in the public artifact bucket (listed in [`manifests/artifacts.sha256`](manifests/artifacts.sha256)), with the licences below.
Inputs and their sources: [`docs/DATA.md`](docs/DATA.md).

### PRC Data Challenge 2026 files (organisers)

Training, ranking and submission-template parquet files distributed by the challenge organisers to registered
teams. They are not redistributed here. `scripts/data/download_data.py` fetches them with a team's own credentials.

### ADSB.lol historical archive: ODbL 1.0

**Contains data from ADSB.lol (https://adsb.lol), licensed under ODbL 1.0.**

- Source: the `globe_history` daily releases, <https://github.com/adsblol/globe_history_2025> and
  <https://github.com/adsblol/globe_history_2026>; documentation <https://www.adsb.lol/docs/open-data/historical/>.
  The database is licensed under the Open Database License 1.0 (<https://opendatacommons.org/licenses/odbl/1-0/>).
- Use: aircraft positions within 10 km of the ten challenge airports, filtered on the fly (`scripts/data/adsb/`).
  They feed the ADS-B off-block features and travel-time map (`scripts/features/build_adsb_features.py`,
  `scripts/features/build_adsb_coverage.py`), the trajectory leg (`scripts/legs/traj/`) and the unmatched-row sensor.
- **Share-alike.** We publish no database derived from ADSB.lol data: no filtered position files, no stand
  centroids, no travel-time maps, no feature tables. The code in this repository is the complete method that turns
  the ADSB.lol database into those derived databases (ODbL section 4.6(b)). Our predictions and trained models are
  Produced Works (ODbL section 4.3) and carry this notice.

### OpenStreetMap: ODbL 1.0

**© OpenStreetMap contributors.** Data available under the Open Database License,
<https://www.openstreetmap.org/copyright>.

- Source: Overpass API extracts of `aeroway=parking_position` / `aeroway=gate` nodes
  (`scripts/data/download_osm_stands.py`) and of the aeroway surface network with runway centrelines
  (`scripts/data/download_osm_surface.py`) for the ten challenge airports.
- Use: label-free stand positions for the ADS-B gate features and the trajectory leg's airport frame; runway
  centrelines for the travel-time map.
- The unmodified extracts are in the artifact bucket under ODbL 1.0; no OSM-derived database is published.

### Iowa Environmental Mesonet (IEM) ASOS/METAR archive

METAR observations courtesy of the **Iowa Environmental Mesonet, Iowa State University**,
<https://mesonet.agron.iastate.edu/request/download.phtml>. IEM describes its site materials as public domain.
Disclaimer: <https://mesonet.agron.iastate.edu/disclaimer.php>.

- Use: origin-airport weather at take-off (`scripts/data/download_metar.py`, `src/prc_challenge/weather.py`).

### OurAirports

Airport and runway reference data from **OurAirports** (public domain), <https://ourairports.com/data/>, files
`airports.csv` and `runways.csv` from <https://davidmegginson.github.io/ourairports-data/>.

- Use: airport reference points for the ADS-B spatial filter; runway thresholds for the ADS-B surface features and
  the trajectory leg's airport frame.

Software dependencies and their licences: [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).
