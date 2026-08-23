# Operating Protocol V1 real-data Station Mode self-nulling dry-run (full segment)

Workbook 49 / 2026-08-22. This stage starts **calibration-only Station Mode
self-nulling** on the full remaining occupancy segments. It is still transfer
and data-quality validation. It does **not** write the official conditions
database.

## Updated conclusion from the scaling study

The empty selected graph on the frozen 100-event windows was
**statistics-limited**, not a frozen-V2 association failure. All-pairs
candidate graphs were non-empty on every real-data window. Selected routes
appear stably at `n1000`, `n10000`, and the full remaining segment.

Frozen V2, route policy, score threshold `0.001`, and
`unmatched_penalty=-1.0` are unchanged.

## Frozen invariants

- Canonical propagation: mode-0
- Association: frozen V2
  (`outputs/mc24_v3_expanded_trainval_v2_bce_control_v1`, SHA256
  `0c85a001…766a27`)
- Observation: `anchor_selected_field_edge` + `physical_edge_deduplicated`
- Station Mode: 5-DoF + `dz=5 mm` survey prior; written `dz` stays 0
- Capture criteria, mode-validity contract, frozen leakage operator `A`
  and the 10% stability gate
- Blind roles: `14973/14974` calibration, `14975/14976` holdout, `14977`
  held-out DQ (never used for selection)

Forbidden: retrain V2; retune thresholds or unmatched penalty; joint
station+`C_dx` Newton; new layer/module DoF; C_dx Mode; treating residual
reduction as correct alignment; official conditions writes.

## DQ versus alignment correctness

| Product | Role |
| --- | --- |
| Route completeness fraction, edge reuse, event concentration, run-to-run overlap | Data quality only |
| Selected-route station residuals and leave-one-out closure | Data quality only |
| Implied `|C_dx|` from frozen `A` and a future Newton `dx/ry` | Contamination diagnostic, **not** a `C_dx` measurement |
| Residual drop after self-nulling | Never alignment success |

`geometry_write_allowed` stays false until the mode-validity contract, the
frozen A 10% gate, and Station Mode capture criteria all pass **and**
independent `C_dx` evidence exists. That last condition is not met on real
data in this stage.

## Full-segment selected-route DQ

Calibration selected graphs are non-empty and not single-event dominated:

| run | role | selected routes | complete 4-station | all-pairs |
| ---: | --- | ---: | ---: | ---: |
| 14973 | calibration | 121 | 0 | 70560 |
| 14974 | calibration | 109 | 2 | 52626 |
| 14975 | holdout | 156 | 1 | 95288 |
| 14976 | holdout | 63 | 0 | 34786 |
| 14977 | held-out DQ | 2 | 0 | 1329 |

The real-data stand-in for the requested route-quality audit is
`truth_free_complete_route_fraction` (MC labels are absent). Complete
four-station routes remain rare. That is recorded as DQ, not as a reason
to retune V2.

## Station Mode capture plan

Only calibration runs receive current + 12 axial finite-difference probes.
Current-geometry reconstruction is reused from the scaling study. Holdout
and 14977 stay current-only and do not estimate geometry.

Condor cluster **1000840** (2 jobs, `nextweek`, 12000 MB, eossubmit).

## Outputs

Under `outputs/operating_protocol_v1_real_data_station_mode_self_nulling_fullscale_v1/`:

- `route_dq_fullscale_report.json`
- `cdx_leakage_diagnostic_report.json`
- `station_mode_self_nulling_report.json`
- `operating_protocol_next_decision.json`

## Self-nulling Newton (calibration only)

Condor **1000840** finished return 0 (14974 at 15:58, 14973 at 16:37).
Identity / field-candidate samples were materialized for the 12 FD probes.
Self-nulling Gauss-Newton used the **existing** full-segment current V2
selected routes; V2 was not rerun or retuned.

| run | dx mm | ry mrad | implied `C_dx` from dx (µm) | implied `C_dx` from ry (µm) |
| ---: | ---: | ---: | ---: | ---: |
| 14973 | −1.30 | 20.0 | −22 | 628 |
| 14974 | +6.57 | 33.5 | +111 | 1049 |

Run-to-run `dx` spread is 7.87 mm. Both implied `|C_dx|` values exceed the
frozen 1.5–1.7 µm operating band. `dz` proposals (658 mm / 1250 mm) are
unphysical; the selected graph is still almost entirely short routes, so
the 6-DoF station solve is not identified. These numbers are **DQ /
contamination diagnostics**, not a geometry to write and not a `C_dx`
measurement.

Frozen MC `A` 10% stability already passed at registration (rel. spread
7.1% < 10%). Real data cannot remeasure `A` without dedicated `C_dx`
variation.

## Unique next decision

`cross_level_contaminated`

- `geometry_write_allowed=false`
- C_dx Mode blocked
- official conditions unchanged
- residual reduction is not alignment success
