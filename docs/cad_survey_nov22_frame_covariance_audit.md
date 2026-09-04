# cad_survey_nov22 Frame and Covariance Audit V1

Workbook 66 / 2026-09-02. Entry 65 left the survey-prior interface ready
but without a validated Calypso global `ry` mapping or a real measurement
covariance. This stage ingests `docs/cad_survey_nov22.txt` as immutable
evidence and asks one question: does the file supply an independent,
frame-correct external measurement with a defensible uncertainty?

It does not retrain V2/V3/Transformer, change the frozen pairwise/route
policy, restack 2024 r0022 collision-like tracks, invent a cosine cut,
enter a full-module identifiability map, run Newton, write geometry, or
emit an alignment payload.

## Frozen contract

- Real data remains `residual_dq_monitoring_only`.
- `geometry_write_allowed=false`.
- Frozen V2 checkpoint SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`.
- Entry-61 unlock grid is not retuned: `σ(ry)≲20 mrad` or
  `σ(C_dx)≲0.315 mm`; leftover after a tight `ry` is
  `σ(C_dx)≈0.345 mm`; useful precision remains `0.5 mrad / 0.080 mm`.
- Population scatter is not a Gaussian prior `sigma`.
- Existing `/Tracker/Align` constants are reconstruction alignment
  state, not an independent survey.
- A 2022 survey is not a 2024/2025 station rigid-body correction
  without mechanical-stability evidence.

## Source

`docs/cad_survey_nov22.txt` is stored unmodified.

| field | value |
| --- | --- |
| SHA256 | `1ea2e62d5340bc6242cdd887954cd92761d3472cf736e1831a5094ff83c1c8f6` |
| size | 21284 bytes |
| sensors | 192 unique `(station, layer, phi_module, eta_module, side)` |
| Delta FASER | `(0.006016, 268.964, 1228.694) mm` |

The producing script of this dump is not in Calypso or this repository.
The 2022-11-04 Casper slides state that GeoModelTest dumped as-built
sensor positions with no alignment applied, FASER identifiers were
attached to wafers, and Stations 1–3 front/pigtail averages fixed the
support-beam translation. This file is that comparison output, not the
raw instrument table.

## Identifier tuple

The five-index tuple is `FaserSCT_ID::wafer_id(station, layer,
phi_module, eta_module, side)`. Meanings are taken from Calypso source,
not from plots:

| index | name | meaning | evidence |
| ---: | --- | --- | --- |
| 0 | `station` | 0=Interface/IFT, 1=Upstream, 2=Central, 3=Downstream | `FaserSCT_ID.h`, `IdDictInterface.xml`, `SCT_DetectorFactory.cxx` |
| 1 | `layer` | three planes per station, upstream to downstream | `FaserSCT_ID.h`, `SCT_Station.cxx`, `NUMLAYERS=3` |
| 2 | `phi_module` | vertical row: Bottom=0 … Top=3 | `IdDictInterface.xml`, `SCT_Frame.cxx` |
| 3 | `eta_module` | Starboard=−1, Port=+1, facing downstream | `FaserSCT_ID.h`, `IdDictInterface.xml` |
| 4 | `side` | 0=Upper/pigtail/front, 1=Lower | `IdDictInterface.xml`, `SCTBRLMODULE SIDEUPPER=0` |

File station/layer summaries use `side=0` only, matching the 2022
slides (CAD wafer z spacing is not the GeoModel 0.6 mm gap).

## Parser regression

Per-sensor lines are printed to 3 decimals. Station and layer summaries
keep higher precision. Exact reproduction is the quoted summary block.
Recomputation from printed sensors is only a rounding-consistency check
(`atol=5e-4 mm`).

Quoted IFT layer means give

```text
C_dx = (x_L0 − x_L2) / 2 = +0.25411693479855924 mm
Δx_L0−L2 = +0.5082338695971185 mm
```

These match workbook 64 (`+0.2541169 mm`, `+0.5082339 mm`). Station 0
quoted mean matches the slide `(0.727, −0.982, −27.772) mm` to the
quoted precision. Layers 0/1/2 average exactly to Station 0, so they
are IFT. The 3-decimal sensor reprint yields `C_dx≈0.2540625 mm` and
must not be used for the entry-64 regression.

File `Sigma` is closer to the population standard deviation (`ddof=0`)
of side=0 offsets than to the sample standard deviation. It is
sensor-to-sensor geometry scatter, not a measurement error.

## Frame inventory

This dump is already in the 2022 survey-adjusted FASER frame:

- File header: all four stations are shifted by Delta FASER to
  approximate the FASER origin.
- Casper 2022-11-04: the three translations come from matching average
  nominal and average survey positions of front sensors in Stations
  1–3; IFT is not used for that translation.
- Axis order in the file: `x` horizontal, `y` vertical, `z` beam.

The native 2021 CAD → Calypso rotation remains
`unresolved_assumption_not_required_for_this_file_C_dx`. No sign is
chosen because it “looks close”.

Calypso station constants remain `T(dx,dy,dz)*Rz(rz)*Ry(ry)*Rx(rx)`
in millimetres/radians, extracted as `ry=asin(R_xz)`,
`rx=atan2(-R_yz,R_zz)`, `rz=atan2(-R_xy,R_xx)`.

## Geometry diagnostics (not priors)

Using side=0 sensors only:

- Station translations are the quoted station means.
- IFT relative displacements follow from quoted layer means.
- A Kabsch rigid map of IFT sensors produces `ry=-7.7537 mrad`
  (`rx=+4.085 mrad`, `rz=+3.048 mrad`, RMS `0.219 mm`). That fit
  absorbs the same L0/L2 `x` contrast as `C_dx`, so it is not an
  independent `ry`. Its translation
  `(-13.70, -8.58, -27.84) mm` is also not the quoted station mean
  `(0.727, -0.982, -27.772) mm`.
- `δz ≈ −ry (x−x0)` is independent of the layer-`x` slope and gives
  `ry=-7.6956 mrad`, with per-layer values
  `-7.831 / -7.683 / -7.573 mrad` (span `0.258 mrad`).

None of these numbers is Calypso `/Tracker/Align/Stations` `ry`. The
mapping is forbidden until the rotation origin, the unique
CAD/FASER→Calypso convention, a measurement covariance, and an IOV are
proven. 2021 CAD normal tilt, 2022 layer-`x` slope, module stereo,
`LAYERPITCH`, and `/Tracker/Align/Planes` rotations are not mapped to
station `ry`.

## Covariance and IOV

No per-point uncertainty, repeat-survey uncertainty, instrument
precision, or fit covariance is present in the file or the adjacent
PDFs. Population `Sigma` is diagnostic only and is not written into a
Gaussian prior. `constructed_measurement_covariance` stays `null`.

The talk date is 2022-11-04. Station rigid motion is IOV-specific by
default. `C_dx` remains a common-static candidate pending
opening/thermal/metrology evidence. Residuals, corrections, and
constants are not averaged across 2022/2023/2024/2025 or across
conditions tags.

## Official constraint slots

The three ingest gates of entry 62 are not simultaneously satisfied:

1. parameter mapping validated for Calypso station `ry` — **no**
2. measurement covariance with independent provenance — **no**
3. year/conditions IOV identified for 2024/2025 use — **no**

Therefore the official slots stay

| slot | availability | value | sigma |
| --- | --- | --- | --- |
| `ift_C_dx` | `feasibility_only` | `+0.2541169 mm` | `null` |
| `ift_l0_minus_l2_dx` | `feasibility_only` | `+0.5082339 mm` | `null` |
| `ift_station0_ry` | `unavailable` | `null` | `null` |

No `measured` slot is filled. The Fisher combiner is not entered. No
Newton step is run.

## Decision

`cad_survey_nov22_reproduces_slide_C_dx_but_lacks_validated_ry_mapping_measurement_covariance_and_cross_year_iov`

The file is the per-sensor source of the 2022 slide `C_dx` central
value in the survey-adjusted FASER frame. That is not enough to write
an alignment prior or a geometry candidate. A reproducible negative
plus an explicit data request is the success criterion of this stage.

## Minimum request to the alignment / hardware / survey teams

1. Per-point or rigid-body-fit covariance of the Nov 2022 wafer table:
   sensor identifier, measured global `(x,y,z)` or offset from a named
   nominal, 3×3 (or at least diagonal) measurement covariance in that
   frame, instrument, date, operator, repeat count. Do not send
   station/module population standard deviations as `sigma`.
2. Unique FASER/Calypso station-`ry` definition for this table:
   rotation origin (station GeoModel origin vs sensor centroid vs
   support beam), axis after `T*Rz*Ry*Rx`, and a quantified IFT
   in-plane yaw if that is the intended object. Do not send
   `LAYERPITCH` slope, `STEREOANGLE`, or `/Tracker/Align/Planes` `ry`.
3. IOV statement: which year/conditions tag the 2022 survey may
   constrain; mechanical-stability evidence if it is to be used in
   2024/2025; whether `C_dx` is claimed common-static after
   opening/thermal checks.

## Reports

`outputs/cad_survey_nov22_frame_covariance_audit_v1/`

Workbook 67 continues from this freeze:
[Nov-2022 metrology provenance and Calypso station-ry contract](nov22_metrology_provenance_station_ry_contract.md).
The software Stations-`ry` contract is now unique (global left-multiply
about the FASER origin). The raw covariance and 2024/2025 IOV remain
unresolved, so official slots stay `feasibility_only` / `unavailable`.
