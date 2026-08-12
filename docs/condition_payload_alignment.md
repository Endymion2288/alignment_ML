# Station Alignment Conditions Payload and Coordinate-Level Closure

## Status Update: Physical Refit Is Now Available

The original coordinate-level study remains a useful controlled surrogate, but
it is no longer the only geometry response path. `--refit-segments` now reads
the persistent `SCT_ClusterContainer`, rebuilds `SegmentFitRefit` and
`SegmentsRefit` under the active SCT/ACTS conditions, and exports new
field-aware propagation records. The MC24 station-3 `+1 mm` physical closure
has been completed with 20 refit tracklets and 27 truth-matched pairs. See
[Displaced-geometry segment refit](displaced_geometry_refit.md) for the data
contract, commands, raw audits, and result.

## What Was Validated

V1 needs a station-level `delta_x, delta_y` convention that is anchored in
Calypso conditions rather than chosen only by a Python residual formula. The
workspace `TrackerAlignDBTool` now accepts an alignment constant
`station:<id>` with six global components
`[dx_mm, dy_mm, dz_mm, rx_rad, ry_rad, rz_rad]`; V1 accepts only `dx_mm` and
`dy_mm`. The payload writer emits `/Tracker/Align` as a SQLite plus POOL
payload, an XML catalog, and a JSON manifest recording the convention.

A controlled payload with `station:3 = [1, 0, 0, 0, 0, 0]` was written and
given to the MC24 exporter. The job log confirms all required condition-chain
steps: SQLite override of `/Tracker/Align`, creation of `SCTAlignmentStore`,
creation of `FaserActsAlignment`, and successful event processing.

## Historical Exporter-Only Control

Before adding the refit chain, the condition payload reached the SCT and ACTS
condition algorithms while the exporter still read persisted local segment
`TrackParameters` from xAOD. The states did not change, as expected; that
historical negative control is retained in
`outputs/mc24_muon_fasernu_5events_physical_station3_dx1mm/condition_runtime_audit.json`.

It establishes that merely rerunning an exporter is not a deformed-geometry
reconstruction. The new refit path resolves this boundary at the persisted
cluster layer, without raw-hit reconstruction.

## V1 Coordinate-Level Surrogate

`apply_station_alignment_payload.py` reads the verified manifest and makes a
new canonical ROOT file with

```text
x' = x + dx_station
y' = y + dy_station
```

It intentionally leaves slopes and covariance unchanged. This is a
translation-only coordinate-level surrogate, not a reconstruction rerun. To
avoid an inconsistent shifted source state, the closure and scan use only
truth-matched propagation pairs whose source is the zero-offset reference IFT
(station 0).

For the retained station-3 `+1 mm` payload, 15 reference-source pairs pass the
truth-match requirement (`>= 0.99`). The observed coordinate increment agrees
with the manifest to `8.9e-16 mm`; the reference-fixed weighted solver recovers
station 3 as `[+1.0, 0.0] mm` with normal-matrix rank 6.

## Coordinate-Level Capture Scan

The payload-calibrated scan uses mode-1 MC truth q/p propagation records,
truth-fixed association, 100 random offset directions per magnitude, three
refinement iterations, a chi-square gate of 25, and `0.01 mm` recovery
tolerance. It has 15 reference-source pairs.

| Translation per movable station [mm] | Capture fraction | Mean active-pair fraction |
| ---: | ---: | ---: |
| 0, 0.1, 0.5, 1, 2, 5 | 1.00 | 1.00 |
| 10 | 0.39 | 0.624 |
| 20 | 0.05 | 0.214 |
| 50 | 0.01 | 0.082 |

This is the current capture range of the **coordinate-level, reference-source
surrogate only**. It must not be interpreted as the physical FASER detector
capture range. A physical scan requires a local segment refit from displaced
hits and propagation from the corresponding displaced source surface.

## Reproduction

Create a new payload directory; all commands refuse to overwrite artifacts.

```bash
cd /eos/home-x/xcheng/FASER
source alignment_ML/scripts/setup_environment.sh calypso

python alignment_ML/scripts/write_station_alignment_payload.py \
  --output-dir alignment_ML/outputs/example_station3_dx1mm/payload \
  --offset 3:1.0:0.0
```

The resulting SQLite and catalog can be supplied to the workspace exporter
with `--tracker-align-sqlite` and `--tracker-align-pool-catalog`. For the V1
surrogate, switch to the ML environment and apply the manifest to an existing
canonical file:

```bash
source alignment_ML/scripts/setup_environment.sh ml
cd alignment_ML

python scripts/apply_station_alignment_payload.py \
  --input outputs/mc24_muon_fasernu_5events_fieldaware_v4_truthcontrol/tracklets.root \
  --output outputs/example_station3_dx1mm/coordinate_injected_tracklets.root \
  --payload-manifest outputs/example_station3_dx1mm/payload/alignment_payload.json
```

Use `run_payload_alignment_closure.py` and
`run_payload_coordinate_capture_scan.py` only after reviewing their JSON
metadata fields `deformed_geometry_repropagation=false` and
`reference_source_only=true`.

## Next Physical Scan

The required geometry-level refit is now implemented and validated. The next
remaining task is a physical capture-range scan: produce independent dx/dy
payloads, rerun the same cluster-to-segment refit chain for each, and apply the
truth-fixed closure to every result. Do not reinterpret the historical
coordinate-level capture table as that detector-level scan.
