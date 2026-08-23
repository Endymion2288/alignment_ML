# Operating Protocol V1 real-data dry-run (transfer / DQ)

Workbook 48 / 2026-08-21. This is still transfer and data-quality
validation. It does **not** write the official conditions database.

## Frozen invariants

Unchanged from Operating Protocol V1 / large-statistics MC transfer:

- Canonical propagation: mode-0
- Association: frozen V2 checkpoint, calibration, and route policy
  (`outputs/mc24_v3_expanded_trainval_v2_bce_control_v1`, ungated)
- Observation: `anchor_selected_field_edge` + `physical_edge_deduplicated`
- Station Mode: 5-DoF + `dz=5 mm` survey prior; written `dz` stays 0
- IFT-Internal Mode: 1-D `C_dx` only, payload `L0=+C_dx`, `L1=0`, `L2=-C_dx`
- Capture criteria, mode-validity contract, frozen leakage operator `A`
  and the 10% stability gate
- Sealed test: never opened
- Real chain: `/Tracker/Align → SCT_ClusterContainer → SegmentFitRefit →
  SegmentsRefit → NtupleDumper → Acts(mode 0)`

Forbidden: retrain; re-register capture/A/thresholds; joint station+`C_dx`
Newton; same-stage iteration; new layer/module DoF; treating residual
reduction as correct alignment.

## Workbook 03 provenance gate

The 2022 data0 IFT PHYS/xAOD re-export remains blocked: event IDs `1...100`
did not match the existing PHYS sequence, and local segments were empty.

This dry-run admits **2024 r0022 reconstructed xAOD** with persisted
`SCT_ClusterContainer`, `SegmentFit`, and `Segments`. Geometry tag
`FASERNU-04`, conditions tag `OFLCOND-FASER-06`, reconstruction tag
`r0022`. Legacy PHYS is provenance auxiliary only.

Selected time blocks (first `00000` xAOD segment, 100 events):

| run | role | geometry estimation |
| ---: | --- | --- |
| 14973 | calibration | yes |
| 14974 | calibration | yes |
| 14975 | holdout | no; evaluate frozen candidate only |
| 14976 | holdout | no |
| 14977 | held_out_dq | never used in geometry estimation |

The split is frozen in
`configs/operating_protocol_v1_real_data_dryrun.yaml` **before** residuals
are inspected. MC truth and MC calibration labels are forbidden.

The first 100 events of each `00000` segment were empty of SCT clusters
under official current geometry. Do **not** rerun the 13-point alignment
scan on that empty window. A residual-blind occupancy preflight scanned
r0022 segments in filename order (official geometry, no alignment overlay,
no V2, no residual branches) and froze windows with
`configs/operating_protocol_v1_real_data_occupancy_window_rule.yaml`:
in `skip_events=0,100,200,...` order, the first 100-event contiguous
window that meets the pre-registered minima (16 events with clusters, 16
with current-geometry SegmentFit tracklets, 32 tracklets, 8 four-station
events). Frozen windows (occupancy counts only):

| run | role | segment | skip_events |
| ---: | --- | --- | ---: |
| 14973 | calibration | 00007 | 49500 |
| 14974 | calibration | 00005 | 74400 |
| 14975 | holdout | 00005 | 21600 |
| 14976 | holdout | 00004 | 95500 |
| 14977 | held_out_dq | 00005 | 135900 |

Provenance:
`outputs/operating_protocol_v1_real_data_occupancy_preflight_v1/frozen_windows.json`.
Empty ROOT with `n_tracklets=0` is not success. Minima and V2 were not
lowered or retuned.

## Station Mode dry-run (phase 1)

`C_dx` is declared fixed by the **current official geometry / external
alignment** (`cdx_fixed_by=external_geometry`). True `C_dx` is unknown, so
every block also reports a cross-level contamination diagnostic from:

- unbiased residual pattern
- run-to-run parameter stability
- frozen `A_dx` / `A_ry`

If the implied station `dx` pattern corresponds to `|C_dx| > 1.5–1.7 µm`,
or the contract precondition cannot be proven,
`geometry_write_allowed=false` and only diagnostics are kept.

The estimator is a **self-nulling** Gauss-Newton linearized at the official
additional-payload origin (current + axial finite-difference probes). There
is no known MC residual target. Residual drop is recorded as DQ only.

## Dedicated `C_dx` Mode (phase 2, later)

Runs only on blocks that already passed Station Mode contract, station
parameter stability, and frozen framework capture/DQ. The station
six-vector is frozen to that accepted geometry. Only `C_dx` floats.
Statistical error and the frozen station→`C_dx` systematic (`B` RSS
0.715 µm) are reported separately. The `C_dx` result is **not** fed back
into a same-stage station solve.

## Blind transfer checks

- Only calibration blocks produce a candidate geometry.
- Holdout blocks evaluate a frozen geometry: unbiased residual, route
  stability, parameter drift.
- Run 14977 is held-out DQ and never enters geometry estimation.

## Station Mode physical wave-1 (2026-08-21)

Holdout / held-out DQ scans are `current_geometry_only` (one official-geometry
point). Calibration scans keep current + 12 axial finite-difference probes.
NtupleDumper was rebuilt with `ExportTrackletPropagationAllPairs`.

Condor **1000432** failed (`Calypso_SET_UP` leak). **1000433** wrote payloads
then failed because the sqlite file had only `OFLP200` while data jobs read
`CONDBR3`. Retry **1000435** clones both instances but used the empty first
100 events of `00000`. Wave-1 was regenerated after occupancy windows froze:
`outputs/operating_protocol_v1_real_data_station_mode_physical_frozen_windows_v2/`,
Condor **1000607**. Cluster **1000604** wrote empty ntuples because NtupleDumper
defaults require a CKF long track and `stableBeams()`; these r0022 xAODs have
SegmentFit occupancy but `stableBeams()` false and no IFT-inclusive
`CKFTrackCollection`. Real-data alignment export now uses `--NoTrackFilt
--no_stable` and points those handles at `SegmentFitRefit`. Official conditions
are not written.

## Outputs

Under `outputs/operating_protocol_v1_real_data_dryrun_v1/`:

- `real_data_station_mode_report.json`
- `real_data_cdx_mode_report.json`
- `operating_protocol_decision.json`

Each block is labelled exactly one of: `valid_for_station_mode`,
`valid_for_cdx_mode`, `cross_level_contaminated`, `dq_failed`,
`geometry_write_candidate`, `insufficient_real_data_occupancy`.

Conditions-writing rehearsal is allowed only after both modes are stable
on their own preconditions, held-out DQ does not worsen, parameters do not
drift anomalously, and the contract has no reject indicators. This item
does not modify the official conditions DB.
