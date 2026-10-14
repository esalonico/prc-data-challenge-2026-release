# Taxi-out time prediction: PRC Data Challenge 2026

Team **`unbelievable-snowflake`**. This repository holds the code that builds our final submission to the
[PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/): the taxi-out time (off-block to take-off,
in seconds) of every departure from ten large European airports in January, February, June and July 2026
(670,790 rows, `final_ranking.parquet`), scored by RMSE.

|                           |                                                            |
| ------------------------- | ---------------------------------------------------------- |
| Final file                | `F2FP`                                                   |
| Built by                  | `pipeline/run_all.sh` from the stored models             |
| Method                    | [`docs/METHOD.md`](docs/METHOD.md)                        |
| Rebuild and verification  | [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md)            |
| Inputs and their licences | [`docs/DATA.md`](docs/DATA.md), [`NOTICE.md`](NOTICE.md) |
| Licence                   | GNU GPL v3 or later ([`LICENSE`](LICENSE))                |

## The system in one paragraph

Departures are routed by whether the organisers' file links them to a Network Manager (NM) flight record.
**Matched departures** (98.6% of rows) get a cell-weighted blend of 14 models trained on one feature
matrix: 3 CatBoost and 4 LightGBM models that differ in their target (taxi time, or taxi time counted from
the NM actual, estimated or planned off-block time), 3 MLPs, 2 networks with periodic numeric embeddings, a
GRU over neighbouring movements, and a network that reads the flight's own ADS-B ground track. The blend weights
depend on the airport group and the ADS-B coverage of the row. A tail model then adjusts rows likely to be very
long. **Unmatched departures** use a mixture of a schedule-copy model and a specialist expert, corrected by ADS-B
off-block estimates where the aircraft was seen. Predictions are projected into the window allowed by the NM
planned off-block time and clipped at zero. A rule for LIRF departures without an NM record then decides between
the schedule copy and a real taxi time.

## Quick start

```bash
git clone https://github.com/esalonico/prc-data-challenge-2026 && cd prc-data-challenge-2026
uv sync                                                  # Python 3.13, versions pinned in uv.lock
uv run python scripts/data/download_data.py              # organiser files -> data/raw/ (team credentials)
uv run python scripts/release/fetch_artifacts.py         # stored models + open-data snapshots, SHA-256 checked
pipeline/run_all.sh                                      # -> $SERVE_WS/artifacts/experiments/Final-serve4m/F2FP.parquet
```

`run_all.sh` also needs the 2025 feature caches and the ADS-B store. They are keyed by the challenge's row IDs or
derived from ADS-B positions, so they are not distributed: [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) gives
the commands that rebuild them from the organiser files and the public ADS-B archive.

## Repository layout

```text
pipeline/            the final build, one numbered step per file (run_all.sh runs them all)
  steps/             Python helpers used only by the steps
scripts/
  data/              downloads: organiser files, METAR, OSM, OurAirports; adsb/ = ADS-B archive ingest
  features/          feature blocks: AOBT clocks, stand sequence, neighbours, planned taxi, ADS-B
  legs/              model legs: s4_allyear (matrices, CatBoost/LightGBM), MLP, nnx/ (PLR, GRU), traj/
  compose/           blend weights, blend, tail model, unmatched chain, final composition
  release/           artifact publish/fetch and the release checks
src/prc_challenge/   library code used by the scripts
configs/             validation windows, model parameters; final/ = blend weights, LIRF rule
manifests/           organiser-file and artifact checksums, run manifests of the stored models, final shas
docs/                METHOD, REPRODUCTION, DATA
```

## Credits and licence

Copyright (C) 2026 the `unbelievable-snowflake` team.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See [`LICENSE`](LICENSE).

Third-party data keeps its own licence ([`NOTICE.md`](NOTICE.md)): *Contains data from ADSB.lol, licensed under
ODbL 1.0* and *© OpenStreetMap contributors*. Dependencies: [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).
The matching-window observation (section 3.4 of METHOD) was first reported in a public competitor repository.
