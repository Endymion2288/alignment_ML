# Displaced-Geometry Segment Refit

## Decision

The persistent xAOD `SCT_ClusterContainer` is sufficient for the V1
station-level `dx, dy` displaced-geometry segment refit. It preserves the
local measurement, local covariance, cluster/strip identifiers, and RDO
identifier list needed by `SegmentFitAlg`. The workflow does **not** need to
return to RDO for a rigid station translation.

This conclusion is both source-audited and runtime-validated on the MC24 100
GeV FASERnu muon control file. It applies to rebuilding local segments from
existing clusters, not to a study that changes clustering, digitization, or
charge calibration.

## Why the Cluster Layer Is Sufficient

The input `CollectionTree` persists `SCT_ClusterContainer` as
`Tracker::FaserSCT_ClusterContainer_p3`. Its p3 converter stores and restores:

- local position `m_localPos`;
- the 2-by-2 local covariance entries `m_mat00`, `m_mat01`, and `m_mat11`;
- cluster identifier and compressed `m_rdoList` strip identifiers;
- width and time-bin information.

During readback,
`FaserSCT_ClusterContainerCnv_p3` obtains the current
`SCT_DetectorElementCollection` and constructs each transient cluster with
that detector element. Consequently `cluster.globalPosition()` is evaluated
with the active SCT alignment conditions rather than a persisted global
position. `SegmentFitAlg` consumes `SCT_ClusterContainer`, uses the cluster
global position and local covariance, and writes a new `TrackCollection`.
`GhostBusters` then derives a new segment collection from that refit output.

The relevant source locations are:

- `TrackerEventTPCnv/src/TrackerPrepRawData/FaserSCT_ClusterCnv_p3.cxx`;
- `TrackerEventTPCnv/src/FaserSCT_ClusterContainerCnv_p3.cxx`;
- `TrackerSegmentFit/src/SegmentFitAlg.h` and `SegmentFitAlg.cxx`;
- `FaserActsKalmanFilter/src/GhostBusters.h`.

## Reproducible Refit Chain

`faser_ntuple_maker.py` now provides `--refit-segments`. It deliberately uses
new StoreGate keys `SegmentFitRefit` and `SegmentsRefit`, preserving the
persisted nominal `SegmentFit`/`Segments` collections for comparison.

```bash
cd /eos/home-x/xcheng/FASER
source alignment_ML/scripts/setup_environment.sh calypso

faser_ntuple_maker.py INPUT-xAOD.root --isMC --useIFT --nevents 5 \
  --export-tracklets --export-tracklet-propagation --refit-segments \
  --outfile alignment_ML/outputs/example_refit_nominal/enhanced_tracklets.root

faser_ntuple_maker.py INPUT-xAOD.root --isMC --useIFT --nevents 5 \
  --export-tracklets --export-tracklet-propagation --refit-segments \
  --tracker-align-sqlite PAYLOAD/tracker_alignment.sqlite \
  --tracker-align-pool-catalog PAYLOAD/PoolFileCatalog.xml \
  --outfile alignment_ML/outputs/example_refit_displaced/enhanced_tracklets.root
```

Each job must show the following in its log before its result is accepted:

- SQLite override of `/Tracker/Align` for the displaced job;
- a new `SCTAlignmentStore`;
- a new `FaserActsAlignment`;
- `SegmentFitAlg` processing the requested events;
- successful execution.

Convert both enhanced files with `convert_ntuple_tracklets.py` and
`convert_ntuple_tracklet_propagations.py`, then run:

```bash
source alignment_ML/scripts/setup_environment.sh ml
cd alignment_ML

python scripts/audit_refit_geometry_response.py \
  --nominal-tracklets NOMINAL/tracklets.root \
  --nominal-propagations NOMINAL/propagations.root \
  --displaced-tracklets DISPLACED/tracklets.root \
  --displaced-propagations DISPLACED/propagations.root \
  --payload-manifest PAYLOAD/alignment_payload.json \
  --output-dir outputs/example_refit_response

python scripts/audit_propagation_mode_components.py \
  --tracklets NOMINAL/tracklets.root --propagations NOMINAL/propagations.root \
  --output-dir outputs/example_mode_audit

python scripts/run_refit_alignment_closure.py \
  --nominal-tracklets NOMINAL/tracklets.root \
  --nominal-propagations NOMINAL/propagations.root \
  --displaced-tracklets DISPLACED/tracklets.root \
  --displaced-propagations DISPLACED/propagations.root \
  --payload-manifest PAYLOAD/alignment_payload.json \
  --output-dir outputs/example_refit_closure --reference-station 0
```

All three commands refuse to overwrite their output artifacts.

## Station-3 `+1 mm` Validation

The retained control uses a global payload
`station:3 = [1, 0, 0, 0, 0, 0]` and five MC24 events.

| Check | Result |
| --- | --- |
| Refit tracklets | 20, four stations in every event |
| Truth-matched propagation pairs | 27 per q/p mode at truth-match fraction >= 0.99 |
| Mean S3 refit `delta x` | `1.000000000000069 mm` |
| Largest S3 position-response error | `5.5e-13 mm` |
| Largest pair residual-response error | `5.5e-13 mm` |
| Largest combined-covariance element change | `5.7e-14` |
| Largest slope residual change | `2.9e-14` |

The observed behavior is exactly the expected rigid translation: only S3
global `x` changes, pairs ending at S3 receive `delta r_x = +1 mm`, and a
translation leaves slopes and covariance invariant up to floating-point
roundoff. Raw records are retained in
`outputs/mc24_muon_fasernu_5events_segment_refit_geometry_response/`.

The truth-fixed, all-station weighted solve uses mode 0 and station 0 as the
reference. It has rank 6 and recovers

```text
S0 = [0, 0] mm
S1 = [5.8e-15, 4.7e-13] mm
S2 = [6.8e-14, 4.8e-13] mm
S3 = [1.000000000000064, 4.8e-13] mm
```

The largest movable-station error is `4.9e-13 mm`. This is a physical
displaced-geometry refit closure, not the earlier coordinate-level surrogate.

## Propagation Mode 0 Versus Mode 1

`audit_propagation_mode_components.py` writes one raw CSV per mode with
`rx_mm`, `ry_mm`, `rtx`, `rty`, propagated/target/combined sigmas, combined
covariance determinant and log-determinant, pulls, and 4D chi-square. It also
writes a same-pair decomposition:

```text
chi2_1 - chi2_0 = residual_effect_at_S0 + covariance_effect_after_r1
```

For the 27 nominal refit pairs:

| Quantity | Result |
| --- | ---: |
| Mean chi2, mode 0 | 202.36 |
| Mean chi2, mode 1 | 553.95 |
| Mean residual effect at fixed mode-0 covariance | 0.0880 |
| Mean covariance effect after switching the residual | 351.49 |
| Mean `log det(S_1) - log det(S_0)` | -14.205 |

Thus the large mode-1 chi-square increase is caused overwhelmingly by a much
narrower propagated/combined covariance, not degraded raw residuals. The raw
mode-0/mode-1 comparison is in
`outputs/mc24_muon_fasernu_5events_segment_refit_mode_audit_nominal/`; the
same conclusion remains true for the displaced output.

`SegmentFitAlg` currently assigns every local segment `q/p = 1e-5 / MeV` and
variance `5e-6 / MeV^2` in its source. The refit q/p audit confirms these
identical values for all 20 tracklets and only 11/19 truth-labelled signs
match. Native local q/p is therefore not a usable measurement. Mode 1 is an
MC truth-q/p diagnostic only; mode 0 is retained for the operational V1
alignment closure.

## Scope and Next Scan

No RDO rerun is needed for this V1 refit. Return to RDO only when the study
requires re-clustering under changed electronics/calibration conditions,
digitization changes, or a geometry effect that invalidates the saved cluster
measurement itself.

The `+1 mm` closure establishes the physical response path but is not yet a
detector capture-range measurement. A physical scan must create independent
payloads over several `dx, dy` magnitudes, rerun this same refit chain for each
payload, and apply the fixed-truth closure to each result. The statistical
uncertainty of a difference between two refits of the same hits is not modeled
by the nominal covariance used as the deterministic WLS weight here.
