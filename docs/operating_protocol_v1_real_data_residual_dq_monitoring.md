# Operating Protocol V1 real-data residual/DQ monitoring

Workbook 52 / 2026-08-23. Entry 51 showed that `{dy,rx,rz}` is
Fisher-identifiable and nearly orthogonal to frozen `A`, but the
self-nulling correction does not transfer between 14973 and 14974.
This stage freezes that conclusion as a production-like DQ protocol.

No new Station calibration mode is constructed. C_dx Mode is not
started. No finite-difference probe is generated, no Newton step is
run, and no payload is written. Monitoring uses the frozen mode-0
backbone, V2 checkpoint, route policy, threshold `0.001`,
`unmatched_penalty=-1.0`, `physical_edge_deduplicated` observation,
and the current official geometry only.

## Frozen invariants

- Propagation: mode-0
- Association: frozen V2 (`0c85a001…766a27`)
- Observation: `anchor_selected_field_edge` + `physical_edge_deduplicated`
- Geometry: official `FASERNU-04` / `OFLCOND-FASER-06` / `r0022`
- Calibration reference: 14973 + 14974 only
- `geometry_write_allowed=false`
- `station_calibration_mode_available=false`
- `cdx_mode_allowed=false`

Forbidden: official conditions writes; any self-nulling correction;
C_dx Mode; joint Newton; new layer/module DoF; opening the sealed
test; treating residual reduction as success; converting an
`alignment_drift_candidate` into a geometry update.

## Residual observables

| Channel | Residual | Role |
| --- | --- | --- |
| `dy` | `residual_y_mm` | Isolation DQ observable |
| `rx` | `residual_ty` | Isolation DQ observable |
| `dx` | `residual_x_mm` | Cross-level-sensitive; reported only |
| `ry` | `residual_tx` | Cross-level-sensitive; reported only |
| `rz` | none | No dedicated 4-vector channel; not fabricated |
| `dz` | none | Never a track-driven observable |

Robust standardized shift is
`(run median − reference median) / (1.4826 × MAD)` against the pooled
14973/14974 field-edge residuals. It is a DQ number, not a correction.

## Alarm criteria

Pre-registered. Do not retune from the monitoring sample.

1. `selected_routes == 0` and `n_all_pairs_candidates ≥ 50` →
   `association_or_reconstruction_degradation`
2. `0 < selected_routes < 10` →
   `insufficient_statistics_for_alignment_dq` (not an alignment anomaly)
3. Selected routes exist but one event owns more than 50% of them →
   `association_or_reconstruction_degradation`
4. Isolation `|robust z| > 5` versus the 14973/14974 reference →
   `detector_condition_change`
5. Otherwise `nominal_monitoring`

A repeatable same-sign isolation `|robust z| ≥ 3` on at least two
non-reference, non-insufficient runs is tagged
`alignment_drift_candidate` only. It is never inverted into a station
payload.

## First monitoring corpus

The five existing full-segment current-geometry associations are the
first long-term scan. Neighboring 2024 r0022 runs
(14971, 14972, 14980, 14981, 14985, 14989, 15007) are expansion
candidates only. They may enter after residual-blind occupancy,
current-geometry Athena, and frozen V2 association. Occupancy windows
remain frozen for 14973–14977 and are not re-picked here.

| run | role | selected | 2/3/4-st | max event share | `dy` z | `rx` z | status |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| 14973 | calibration_reference | 121 | 38/83/0 | 0.008 | −0.00 | −0.02 | nominal |
| 14974 | calibration_reference | 109 | 34/73/2 | 0.009 | 0.00 | 0.03 | nominal |
| 14975 | monitoring | 156 | 60/95/1 | 0.006 | −0.13 | 0.01 | nominal |
| 14976 | monitoring | 63 | 18/45/0 | 0.016 | −0.07 | 0.03 | nominal |
| 14977 | monitoring | 2 | 0/2/0 | 0.50 | −0.06 | −0.14 | insufficient |

14975 and 14976 stay inside the calibration-reference band. The
Theil–Sen slopes versus run number are `dy ≈ −0.12 mm/run` and
`rx ≈ −1.9×10⁻⁵ /run`, small against the reference robust scales
(`dy` 6.71 mm, `rx` 0.00524). There is no repeatable long-term
`dy/rx` drift in this corpus. 14977 is
`insufficient_statistics_for_alignment_dq`, not anomalous alignment.

## Unique decision

**`real_data_residual_dq_monitoring_only`**

- `geometry_write_allowed=false`
- `station_calibration_mode_available=false`
- `cdx_mode_allowed=false`
- Do not extract any alignment payload from the current self-nulling
  residuals

Geometry calibration may be reopened only if one of these external
prerequisites is met:

1. Independent survey or external station constraints fix `dx`, `ry`,
   and `dz`, **and** an independent demonstration shows `C_dx` inside
   the 1.5–1.7 µm isolation budget; **or**
2. A new independent real-data topology or track sample empirically
   demonstrates a cross-run transferable subspace.

Until then, current real data support residual/DQ monitoring only.

## Outputs

Under `outputs/operating_protocol_v1_real_data_residual_dq_monitoring_v1/`:

- `run_level_dq_report.json`
- `time_stability_report.json`
- `alignment_drift_candidate_report.json`
- `operating_protocol_monitoring_decision.json`
