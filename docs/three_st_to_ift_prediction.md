# Independent 3ST→IFT Prediction Chain (Yasu Stage 1)

Workbook 118.  After Stage 0 locked `CKFTrackCollectionWithoutIFT` as the
official S1+S2+S3 CKF, this task materializes an independent prediction
from that native 5D state and 5×5 to IFT.

It does **not** refit, does not call `KalmanFitterTool.fit`, does not use
WB109 LTO (which keeps IFT), and does not start from SegmentFit dummy
`q/p`.  IFT measurements come from `SCT_ClusterContainer` and must not
enter the prediction.

## Chain

```
S1+S2+S3 hits
  → official CKFTrackCollectionWithoutIFT (already persisted)
  → native (loc1, loc2, phi, theta, q/p) + full 5×5
  → ACTS conversion copied from the WB98 helper
  → FaserActsExtrapolationTool, field-aware, backward
  → IFT station plane z = −1860.15 mm (WB87/WB98 value)
  → independent FaserSCT_Cluster loc0 residual on IFT wafers
```

WB98 only propagated **forward** from the four-station `front()` (IFT)
to S1/S2/S3 and skipped `targetZ <= sourceZ`.  This chain must go the
other way.  `AllowBackwardToIft` is required.

Official prediction is Model 1 (MS/Eloss on).  Model 0 is diagnostic.
The residual is

```
r = loc0_cluster − loc0_predicted
```

`loc0_cluster` is `FaserSCT_Cluster.localPosition` `Trk::locX`, the same
observable as `ClusterLocalDumpAlg`.  `insideBounds` is recorded and is
**not** required for residual availability.  BoundaryCheck / approach
scans are Stage 6, not this stage.

## Failure classes (kept, not deleted)

- `no_without_ift_candidate` — selection loss; not a warning count
- `ift_measurement_leak` — IFT on WithoutIFT MOT or TSOS
- `acts_start_unavailable`
- `acts_s1_to_ift_propagation_failed` — magnet-crossing plane hop
- `independent_ift_measurement_absent` / `no_ift_cluster_in_event`
- `ift_prediction_available_residual_unavailable`
- `surface_not_in_identifier_map` / `propagate_surface_failed`

ACTS warning counts are **not** reconstruction loss.

## Allowed decisions

- `three_st_to_ift_prediction_established`
- `three_st_to_ift_prediction_not_established`

PASS authorizes Stage 2 (MC q/p calibration) only.  Residual
conditionals open only if both splits have at least one independent
IFT residual.  This does **not** claim q/p is unbiased, does not claim
a `q/p ↔ R_y/d_x` weak mode, and does not reopen B14M / B15 /
Measurement Model V2 / alignment.

## Official run

`yasu_s1_three_st_to_ift_prediction_20260908T131812Z_b30ae4b2`

```
decision = three_st_to_ift_prediction_established
mc_qp_calibration_authorized = true
residual_conditional_authorized = true
```

20-event construction / validation smoke.  Backward IFT-plane Model 1
succeeded on 20/20 and 19/19 WithoutIFT tracks.  IFT leak = 0.  One
validation event has no WithoutIFT candidate (selection loss, already
seen in Stage 0).  Wafer-surface hops that return ACTS
`PropagatorError:3` (`StepCountLimitReached`) or `SurfaceError:1`
(`GlobalPositionNotOnSurface`) are residual-level failures, not track
deletion.  Those ERROR counts are not reconstruction loss.

All-cluster residuals mix unmatched IFT clusters.  Residuals tagged
`reconstruction_associated` (unique 4ST TSOS IFT identifiers) are a
diagnostic family only; Stage 3 must preregister which family is
official.  Their means are **not** a Yasu-bias claim.
