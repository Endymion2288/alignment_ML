# Nov-2022 Metrology Provenance and Calypso Station-ry Contract V1

Workbook 67 / 2026-09-02. Entry 66 froze
`cad_survey_nov22_reproduces_slide_C_dx_but_lacks_validated_ry_mapping_measurement_covariance_and_cross_year_iov`.
This stage does not squeeze another `ry` from the existing dump. It hunts
independent evidence that can lift the three ingest gates, and it writes
the mathematical contract of `/Tracker/Align/Stations` `ry` from Calypso
source plus a geometry-only finite-difference test.

It does not retrain V2/V3/Transformer, change the frozen pairwise/route
policy, restack 2024 r0022 collision-like tracks, invent a cosine cut,
enter a full-module identifiability map, run Newton, write geometry, or
emit an alignment payload. Kabsch `-7.75 mrad`, layer-`x` slope,
`STEREOANGLE`, `LAYERPITCH`, `/Tracker/Align/Planes` rotations, and
existing conditions constants are not used as station `ry`. Population
scatter is not a measurement covariance.

## Frozen contract

- Real data remains `residual_dq_monitoring_only`.
- `geometry_write_allowed=false`.
- Frozen V2 checkpoint SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`.
- Entry-61 unlock grid is not retuned: `σ(ry)≲20 mrad` or
  `σ(C_dx)≲0.315 mm`; leftover after a tight `ry` is
  `σ(C_dx)≈0.345 mm`; useful precision remains `0.5 mrad / 0.080 mm`.
- A 2022 survey is not a 2024/2025 station rigid-body correction
  without mechanical-stability evidence.
- `C_dx` remains a common-static *candidate* until opening/thermal
  evidence. Station rigid motion stays IOV-specific.

## Decision

`nov22_raw_covariance_and_iov_unresolved_station_ry_is_global_left_multiply_about_faser_origin`

| blocker | status | exact required evidence |
| --- | --- | --- |
| raw Nov-2022 measurement covariance | **unresolved** | original wafer/sensor table with per-point or fit covariance, instrument, date, operator, repeats, SHA256 |
| Calypso `/Tracker/Align/Stations` `ry` software contract | **resolved** | global left-multiply `r′ = g·r` about the FASER origin; `∂x/∂ry = +z`, `∂z/∂ry = −x` |
| survey estimator → Stations `ry` mapping | **unresolved** | a survey rigid rotation about the FASER origin, not the sensor centroid / station GeoModel origin / support beam |
| IOV | **unresolved, 2022-only** | named mechanical/opening/closing/thermal state, plus independent 2022→target-year stability if used in 2024/2025 |

Official slots stay `feasibility_only` / `unavailable`. Fisher is not
entered. No `measured` slot is filled.

## Blocker 1 — raw metrology provenance (unresolved)

Read-only inventory, no physics inferred from filenames:

| location | result |
| --- | --- |
| this repository | only `docs/cad_survey_nov22.txt` and the two adjacent PDFs |
| Calypso tree | no `Delta FASER` / `cad_survey_nov22` / `04Nov2022` producing script |
| `/eos/experiment/faser` depth 2 | no `survey/` / `metrology/` / `cad/` / `geo/` trees; README (30 Apr 2024) says reconstruction lives under `data0` |
| CVMFS `poolcond` | yearly `FASER-0X_YYYY_Align.pool.root` files only; reconstruction conditions, not survey |
| `/eos/home-f/fcadoux`, `/eos/user/f/fcadoux`, `/eos/home-d/dcasper`, `/eos/user/d/dcasper` | exist, mode `drwx------`, `PermissionError`; not readable from this account |
| `/eos/user/c/casper`, `/eos/home-c/casper` | do not exist |

`docs/04Nov2022_Survey.pdf` (D. Casper, survey data from Franck/Cadoux,
2022-11-04) states:

- Cadoux CAD with the support beam in the nominal location (translation TBD).
- Wafer IDs assigned manually.
- GeoModelTest dumped as-built sensor positions, **no alignment applied**.
- Stations 1–3 front/pigtail averages fix the three unknown translations.
- IFT `z` is the only shift they “may want to account for in the nominal geometry”.
- Stations 1–3 are “consistent with 350 µrad horizontal misalignment of the support beam”.

That PDF does not name a source-data filename, an instrument, an
operator, a repeat count, or a covariance. Literature UNIGE CMM
precision from NIMA 1034 (2022) 166825 / arXiv:2112.01116 is
plane-assembly metrology (~5 µm in-plane, 10–15 µm out-of-plane) and
**must not** be reused as Nov-2022 covariance of this dump.

No machine-readable ingestion layer was built, because that would
invent instrument/operator/uncertainty fields that are not in the file.
Quoted means of `cad_survey_nov22` remain the entry-66 parser output.

## Blocker 2 — Calypso station-`ry` contract (software: resolved; survey mapping: unresolved)

### Source path

1. `TrackerAlignDBTool::stationAlignment` writes
   `[dx, dy, dz, rx, ry, rz]` mm/rad as
   `T(dx,dy,dz) * Rz(rz) * Ry(ry) * Rx(rx)`
   (`TrackerAlignDBTool.cxx`).
2. `/Tracker/Align/Stations` is registered
   `addChannel(..., 3, TrackerDD::global)` in `SCT_DetectorFactory.cxx`.
   `dirkey` inverts the labels (its “level 1” = Stations); the folders
   are the same.
3. `SiDetectorManager::setAlignableTransformGlobalDelta` conjugates
   `c = T.inverse() * g * T` with `T = child fullPhysVol getDefAbsoluteTransform`.
   Therefore `T*c = g*T` and a global point transforms as **`r′ = g·r`**.
4. `FaserSCT_AlignCondAlg` applies the container into
   `SCTAlignmentStore`. `FaserSCT_DetectorElementCondAlg` copies
   `SiDetectorElement` objects bound to that store into
   `SCT_DetectorElementCollection`. `SiDetectorElement::center()` reads
   `transformHit()` from the store.
5. `GeoModelTestAlg` currently dumps
   `SCT_DetectorManager::getDetectorElementCollection()`, the
   **unaligned** DetStore collection. A live GeoModelTest `±ry` dump
   would reprint as-built positions and cannot validate Stations `ry`.
   A later aligned dump must read `SCT_DetectorElementCollection`.

FASERNU-04 `SCTTopLevel-02` (HVS 107793): SCT parent `z = 1237.4 mm`,
Interface `POSZ = −3097.55 mm`, so the station-0 GeoModel origin is
`z = −1860.15 mm`. That origin is **not** the rotation pivot of a
global `g`.

Planes (level 2) use a different conjugation,
`Translate(0,0, element z) * alignment * inverse`, so Planes `ry` is
not Stations `ry`.

### Finite-difference result (software sensitivity only)

Point cloud: 24 IFT side=0 implied GeoModel nominals from
`cad_survey_nov22` (`corrected − offset`). Injected `ry = ±0.0001 rad`.
This is not an alignment solve and is not a geometry candidate.

| estimator | RMS(numeric − model) mm/rad |
| --- | ---: |
| `r′ = Ry(ry)·r` about FASER origin `(0,0,0)` | `3.1e-6` |
| rotation about sensor centroid | `1860.15` |
| rotation about station GeoModel origin `(0,0,−1860.15)` | `1860.15` |
| rotation about Casper Delta FASER / support beam | `1228.69` |

Analytic Jacobian at `ry=0`: `∂x/∂ry = +z`, `∂y/∂ry = 0`,
`∂z/∂ry = −x`. IFT mean `z = −1860.15 mm`, so
`⟨∂x/∂ry⟩ = −1860.15 mm/rad`. Numeric central difference matches
the exact `sinc(ry)` factor of finite `Ry`.

The sensor centroid and the station GeoModel origin are almost the
same point for IFT (`z ≈ −1860.15 mm`, `x,y ≈ 0`). Both are
**different** from Stations `ry`: a rotation about either of those
pivots has `⟨∂x/∂ry⟩ ≈ 0` at the IFT, while Stations `ry` translates
the whole IFT by `z·ry`.

Workbook 17 physical-refit smoke (`+10 mrad` IFT, truth-matched track
state) gave `Δx = −18.5981 mm` ⇒ `Δx/ry = −1859.81 mm/rad`. Same
sign and magnitude as IFT mean `z`. That is a track-state response,
not a sensor-center dump, and is recorded only as independent
physical-chain consistency. It is not a survey prior.

### What this does *not* do

It does not map Kabsch `ry = −7.75 mrad` or `dz`-vs-`x`
`ry = −7.70 mrad` onto Stations `ry`. Those estimators rotate about
the sensor centroid / mean `x`, which the finite difference just
rejected. The software contract is unique; the survey mapping is not.

## Blocker 3 — IOV (unresolved, 2022-only)

Talk date 2022-11-04. Neither the PDF nor `cad_survey_nov22` records
opening, closing, thermal, or mechanical state. CERN backbone survey
and UNIGE plane CMM predate Nov-22 and do not prove 2022→2024/2025
station rigid stability. Yearly Align POOL files are reconstruction
conditions, not survey.

- Station rigid motion remains IOV-specific.
- `C_dx` remains a common-static candidate; it is **not** upgraded.
- 2022 cannot be used as a 2024/2025 station rigid correction.

Required evidence to use the survey in 2024/2025:

1. named mechanical / installation / opening / closing / thermal state
   of the Nov-2022 survey;
2. an independent stability measurement covering 2022 to the target year;
3. an explicit statement of which conditions tag the survey may constrain.

## Official slots

| slot | availability | value | sigma |
| --- | --- | ---: | --- |
| `ift_C_dx` | `feasibility_only` | `+0.2541169 mm` | `null` |
| `ift_l0_minus_l2_dx` | `feasibility_only` | `+0.5082339 mm` | `null` |
| `ift_station0_ry` | `unavailable` | `null` | `null` |

The three ingest gates of entry 62 are not simultaneously satisfied,
so Fisher is not entered. Entry-61 rank / sigma / scan grids stay
frozen and unused.

## Unresolved assumptions (recorded, not guessed)

- Native 2021 CAD → Calypso axis signs remain unresolved and are not
  used.
- Cadoux/Casper EOS homes exist but are unreadable from this account;
  they were not treated as empty of data.
- A live aligned sensor-center dump was not run, because GeoModelTest
  dumps the unaligned manager collection. The contract is validated
  from source plus `T*c = g*T` numerics. A later dump must read
  `SCT_DetectorElementCollection`.
- Implied nominals in `cad_survey_nov22` are Casper’s GeoModelTest
  as-built reprint, used here as the FD point cloud, not as a
  measurement.

## Minimum request

1. Original Nov-2022 wafer/sensor table: identifier, measured
   `(x,y,z)` or offset from a named nominal, 3×3 (or diagonal)
   covariance, instrument, date, operator, repeats, SHA256. Do not
   send population standard deviations as `sigma`.
2. A survey rigid rotation about the FASER origin, identified as
   `/Tracker/Align/Stations` `ry`. Do not send Kabsch, `dz`-vs-`x`,
   `LAYERPITCH`, `STEREOANGLE`, or Planes `ry`.
3. IOV / mechanical-stability statement as listed above.

## Reports

`outputs/nov22_metrology_provenance_station_ry_contract_v1/`

Source SHA256 `1ea2e62d5340bc6242cdd887954cd92761d3472cf736e1831a5094ff83c1c8f6`.
Config SHA256 `446d87eb322b9d2b60ae5bda7c87ee361662e354845b79bc4aa752553d3c3008`.
HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`.
