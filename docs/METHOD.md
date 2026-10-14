# Method

## 1. Task and metric

Predict `TAXITIME_SEC_mvt`, the time from off-block (`BLOCK_TIME_UTC_mvt`) to take-off (`MVT_TIME_UTC_mvt`), for
every departure from ten airports (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM). Training data: all
2025 movements with labels. Metric: RMSE over all rows.

| Ranking | Months of 2026                |    Rows | Where it is scored                   |
| ------- | ----------------------------- | ------: | ------------------------------------ |
| public  | January, July                 | 344,841 | public board during the challenge    |
| final   | January, February, June, July | 670,790 | by the organisers after the deadline |

The organisers blank only the departure block time and taxi time in the ranking files. Everything else is supplied,
including the Network Manager (NM) clocks of departures that the file links to an NM flight record ("matched").

Two facts shape the solution:

1. **The label is administrative.** About 1.4% of departures carry no NM flight record ("unmatched"). Their block
   time is often a copy of the schedule or a day-shifted stamp, which gives taxi "times" of hours. These rows hold a
   large share of the squared error.
2. **Retrospective information is allowed.** Matched rows come with the NM actual off-block time (`AOBT_3_flt`)
   and the planned times (`LOBT_flt`, `EOBT_1_flt`, `IOBT_flt`), and open ADS-B archives cover the ranking months.
   The model is a post-operations reconstruction, not a pre-departure forecast.

## 2. Validation

- **Development view.** One fit on the 2025 "D" months {2, 3, 4, 5, 6, 8, 9, 10}, scored on January and July 2025
  together, the calendar analogue of the public ranking. Pooled RMSE is sqrt(total SSE / total N); no row is ever
  dropped. Internal selection uses blocked D folds {2, 8}, {3, 9}, {4, 10}, {5, 6}, with a D-fold out-of-fold bank of
  every leg.
- **Fresh gate.** Once January/July 2025 had been reused for about ten sequential decisions, the November/December
  2025 labels were opened as the gate for new decisions, scored only with models fitted on D months. Each read was
  counted.
- **Four-month scorecard and transfer gate** (final phase). Candidates are scored on 2025 January, February, June
  and July, the months of the final ranking. A candidate passes only if its gain on the body of the distribution is
  positive under a day bootstrap, under importance weights that mimic 2026, and on both month pairs (Jan/Jul and
  Feb/Jun). Rules for the LIRF mechanism were checked by leave-one-month-out on 2025 months 1–10.
- **Pre-registration.** Every arm has an acceptance rule written before its evaluation data is read. Every public
  upload had a pre-registered centre and a keep/revert rule.
- **Leakage guards.** No departure block or taxi time of any flight is a feature. Features are built per calendar
  month, so no context crosses the February–June gap of the ranking. References that depend on labels are
  cross-fitted. Before any all-year refit, a parity screen compares the distribution of every column between 2025
  training rows and 2026 ranking rows.

## 3. The system

### 3.1 Features of matched departures

One matrix of about 265 columns, built month by month:

- planning and calendar fields, origin METAR at take-off, an unimpeded-taxi reference by airport, stand and runway;
- temporal context: previous and next movements at the airport and on the runway;
- the airport's completed-arrival state and expected stand and runway occupancy;
- stand sequence (the stand's previous occupant, turnaround timing), neighbour context and planned taxi, built
  from other flights' clock fields only;
- NM clocks: offsets between take-off and AOBT, LOBT and EOBT;
- ADS-B (ADSB.lol): per departure, the last position at the stand (matched to label-free stand centroids and OSM
  stand positions), first-seen time and place, confirmation flags, measured surface load, and a learned
  travel-time map (stand × 50 m cell × heading → typical time since off-block), with explicit coverage flags.

### 3.2 Legs and blend

Fourteen legs, all fitted once on the full year 2025 and stored:

| Legs | Model                                                                                    | Targets / inputs                           |
| ---- | ---------------------------------------------------------------------------------------- | ------------------------------------------ |
| 3    | CatBoost, 7,994 trees                                                                    | taxi time; taxi time from AOBT; from EOBT  |
| 4    | LightGBM, learning rate 0.01, fixed rounds                                               | taxi time; from AOBT; from EOBT; from LOBT |
| 3    | MLP with categorical embeddings                                                          | the same matrix                            |
| 2    | MLP with periodic numeric embeddings ("PLR"), ResNet trunk                               | taxi time; from EOBT                       |
| 1    | GRU over 16 + 16 airport and 8 + 8 runway neighbours (label-free fields), PLR embeddings | taxi time                                  |
| 1    | 1-D convolutional ResNet over the flight's own ADS-B ground track                        | taxi time                                  |

The blend weights depend on the cell airport group × ADS-B coverage class. In each cell they are a simplex fit in
member-error space on the D-fold out-of-fold bank (labels ≤ 2,700 s), shrunk 50% toward a global vector that is
itself shrunk 50% toward equal weights; cells with fewer than 20,000 rows use the global vector
(`configs/final/blend_weights_d.json`). Two small groups of matched rows take the CatBoost prediction instead of the
blend: E17 (rows whose NM record is from another day) and E18 (LIRF rows with an extra-day stamp).

### 3.3 Tail model (T1)

On matched rows outside E17/E18, a calibrated probability that the taxi is long (`p_long`) and a long-taxi expert
(`mu_long`), both all-year boosters on a contiguous matrix, move the blend: pred + 0.35 · p_long · (mu_long − pred).

### 3.4 Unmatched departures and the matching window

A mixture `p · schedule + (1 − p) · expert`. `p` is a calibrated probability that the block time is a schedule
copy, computed from the month-isolated lineage of the flight. The expert is a CatBoost model trained on genuine
(non-copy) unmatched rows, capped at schedule + 3,606 s. On top of it:

- an ADS-B sensor expert where the aircraft is observed (ceiling 3,606 s);
- the travel-time map where the sensor did not fire (weight 0.85);
- LIRF arms: a stand-level taxi model, and month-regime corrections of the copy share from the ranking file's own
  arrival rows;
- the same flight's destination clocks where they are known downstream.

Matched predictions are then projected into `LOBT_flt ± 3,606 s` and everything is clipped at 0. The window is the
one within which the file's movement-to-NM matching appears to operate (an observation first reported in a public
competitor repository). No matched 2025 label violates it, and nothing in the rule is fitted.

### 3.5 LIRF label mechanism

At LIRF, a departure without an NM record has one of three labels: ordinary (`BLOCK` is the real off-block), copy
(`BLOCK = SCHED`, so y = b = take-off − schedule) or, rarely, a date bug (`BLOCK` = the schedule's local date + the
real off-block clock time). From the 2025 files only:

- With delay b between 3 h and 6 h and no midnight crossing, the copy share is seasonal (0.43–0.60 in June–October,
  0.67–1.0 in January–May). It also depends on the callsign format: ordinary taxis almost always carry a clean ICAO
  prefix + number, while garbled forms mark the schedule-fallback path.
- Special operators (charter, cargo, business, rare or unknown prefixes; a frozen list of 244 regular ICAO prefixes
  defines the rest) are schedule copies in 34 of 34 such 2025 rows, and in 0.947 of the rows 1–3 h late.

The rule (`pipeline/steps/lirf_rule.py`, parameters in `configs/final/lirf_rule_params.json`, fitted on 2025 only):
special-operator rows with 1 h ≤ b < 6 h get pred = b; the other rows with 3 h ≤ b < 6 h are shifted per season ×
callsign format (summer: clean −3,758 s, other −1,506 s; other months: clean −200 s, other −383 s). Leave-one-month-out
on 2025 months 1–10: −1.36 s for the 3–6 h part, a further −0.07 s for the 1–3 h part.

## 4. Pipeline

`pipeline/` rebuilds the final file from the stored models in about 1.5 hours on an 8-core laptop (steps and checks
in [`REPRODUCTION.md`](REPRODUCTION.md)). Nothing is refitted: the steps build the four months' features, predict the
fourteen legs, blend, apply T1, the unmatched chain and the window (F1) and the LIRF rule (F2F).

## 5. Results

Public scores on January and July 2026 (RMSE, seconds). The local numbers are in the decision entries of each stage.

| Upload | File | What changed                                                          | Public RMSE |        Kept        |
| ------ | ---- | --------------------------------------------------------------------- | ----------: | :----------------: |
| v31    |      | three searched neural legs                                            |    244.8018 |        yes        |
| v32    |      | ADS-B travel-time map block on matched rows                           |    244.3690 |        yes        |
| v33    |      | the map for unmatched EHAM rows                                       |    242.5401 |        yes        |
| v34    |      | LIRF copy probability recalibrated toward the Nov/Dec regime          |    242.9631 |         no         |
| v35    |      | LIRF matched copy mixture removed                                     |    241.2706 |        yes        |
| v36    |      | ADS-B trajectory leg                                                  |    240.5487 |        yes        |
| v37    |      | updated arms for unmatched rows                                       |    240.7476 |         no         |
| v38    |      | cell-weighted blend (method d)                                        |    237.5802 |        yes        |
| v39    |      | GBM legs on the month-isolated matrix                                 |    237.5665 |        yes        |
| v40    |      | all legs month-isolated (parity fix)                                  |    237.6000 |        yes        |
| v41    |      | CatBoost at 7,994 trees + a LIRF 24 h hedge                           |    241.8351 | no (hedge dropped) |
| v42    | F0   | four-month serve path from the stored models; CatBoost at 7,994 trees |    237.8783 |        yes        |
| v43    | F1   | LightGBM legs at learning rate 0.01                                   |    237.8600 |        yes        |
| v44    | F2   | LIRF rule, parts A (date bug) and B (3–6 h, season)                  |    237.5150 |       split       |
| v45    | F2B  | part B only                                                           |    236.4358 |        yes        |
| v46    | F2C  | part B by season × NM-gap crowding                                   |    236.8877 |         no         |
| v47    | F2D  | part B by season × callsign format                                   |    236.2816 |        yes        |
| v48    | F2E  | special operators get the schedule copy                               |    235.6516 |        yes        |
| v49    | F2F  | the same for special operators 1–3 h late                            |    235.6236 |        yes        |
| v50    | F2EP | F2E                                                                   |    211.9163 |        yes        |
| —     | F2FP | F2F: **the final file**                                        |     211.885 |         —         |

F2FP was not uploaded. Its public score follows from v48, v49 and v50 because its two changes against F2E touch
disjoint rows: sqrt(211.9163² + 235.6236² − 235.6516²) = 211.885 (±0.0002 from the rounding of the scores).

The two largest modelling steps came from new information (NM clocks and ADS-B), not from model choice. ADS-B gains
on the public board were about twice the local ones, because ADS-B coverage is higher in 2026 than in the 2025 rows
used for validation.

## 6. What did not work

All with pre-registered rules.

- **LIRF date-bug rule (part A).** Split from part B by v44 and v45: +1.08 s on the public board. Most of the
  rows it moved were schedule copies. Dropped.
- **LIRF part B by NM-gap crowding.** −1.005 s by leave-one-month-out, +0.45 s on the public board (v46). Dropped.
- **Cross-fitted residual corrector on the final base.** Body −0.72 s on the four-month scorecard, worse in 8 of 10
  months: it learns month-specific stand, runway and operator patterns that do not transfer.
- **Recalibrating the LIRF copy probability** toward the November/December regime: +0.42 s on the public board
  (v34). Reverted.
- **Copy-probability hedge** (shrinking P(copy) toward its cell mean). Every shrinkage is worse in all ten months, so
  P(copy) is not over-confident across months.
- **Airframe turnaround features** in the unmatched expert. Four-month body −0.14 s, Jan/Jul +0.45 s worse.
- **Departure block copied from an arrival's in-block**: 17 same-stand matches in 2.1 million rows. **Arrival-copy
  episodes**: no signal. **Disruption-day corrections**: no 2025 bias on high taxi-in days.
- **Cell bias corrections** (airport × season, weekday, load, hour, airline, stand, runway × hour, aircraft type):
  all worse out of sample.
- **Copy-probability model for LIRF with stand area, night schedule and callsign format**: −0.12 to −0.14 s, better in
  only 5–6 of 10 months.
- **Per-airport LightGBM as a 15th leg**: failed its first gate.
- **GPU CatBoost refit at learning rate 0.015**: the ungated GPU recipe barely learns on this matrix. Stopped.
- **Other external data**: ERA5/ERA5-Land, NASA POWER, EUROCONTROL same-day airport statistics, destination METAR,
  RVR and trend remarks, altimeter, BTS, GFS, strike and disruption calendars. Aggregate day- or hour-level regime
  features act as time identifiers and do not transfer.
- **Modelling**: robust losses (Huber, MAE, quantile, log), sample reweighting, seed bagging once LightGBM was in the
  blend, extra GBM target variants beyond four, neighbour NM-clock blocks, ADS-B coverage-shift augmentation,
  residual targets relative to the ADS-B map.

## 7. Limitations

- February and June 2026 were never scored on any board during the challenge; the four-month scorecard is the only
  evidence for them.
- ADS-B coverage swings by airport and month (LTFM has almost none; LFPG partly; LIRF in July 2026), and those rows
  keep much larger errors.
- The ranking files are re-dithered between the public and the final release, so the serve path recomputes every
  feature for the final file; the January/July rows of F0 differ from the public ones by about 14 s rms per row.
- The release is verified on the serve path: the stored models rebuild the final file byte for byte. The training
  code of every stored model is included with its run manifest, but was not rerun for this release.
- January/July 2025 was reused for sequential decisions and the public board for whole-file confirmations. Treat
  the local numbers as optimistic by a few tenths of a second per decision.
