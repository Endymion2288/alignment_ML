# Leave-Target-Out Prediction Contract (Stage B / Task B10)

Workbook 106.  WB105 left A+B mixed: the official Cin is not a
leave-target-out prediction, and it is already overwide before
propagation.  This task asks whether a statistically independent
prediction state can be established after excluding the current
target-station measurements.

The task does not invent `Cov(pred,target)`, call
`KalmanFitterTool.fit` a leave-target-out export, drop
`100043/37`, reopen raw CKF, or enter Measurement Model V2.

## Official result

```
verdict = NOT_ESTABLISHED
decision = leave_target_out_prediction_contract_not_established
primary_case = leave_target_out_prediction_contract_not_established
next_step = independent_leave_target_out_helper_not_kalmanfittertool_fit
measurement_model_v2_authorized = false
independence_proven = false
lto_states_materialized = false
```

Official run `sbb10_leave_target_out_20260906T211854Z_951d7052`.
Input remains the WB103 contracted sample: 1989 rows / 1974 pairs.

## Calypso interface catalog

Pinned Calypso `40892527e9c65409afd2378a2abfc25ddbddac03`,
Athena 24.0.41, ACTS 32.0.2.

| Interface | Exists | Station-level LTO 5×5 at source |
| --- | --- | --- |
| `KalmanFitterTool.fit` | yes | **no** |
| `getUnbiasedResidual(cluster_z)` | yes | **no** |
| `getUnbiasedResidual(IFT cluster list)` | yes | **no** |
| dedicated perigee / leave-one-station-out export | **no** | **no** |

`KalmanFitterTool.fit` rebuilds every `measurementsOnTrack` hit,
sets `FaserActsOutlierFinder.cluster_z = -1e6`, and therefore marks
IFT hits with `z < -100 mm` as outliers.  That is leave-**source**-out
(station 0), not leave-**target**-out (stations 1/2/3).  The seed
covariance is scaled by `SeedCovarianceScale=100` and then ×10.

`getUnbiasedResidual(cluster_z)` only special-cases
`cluster_z < -10000`.  A real target-station z (47.4 / 1237.4 /
2427.4) is not excluded.  The IFT cluster-list overload **appends**
the supplied clusters and then still adds every track measurement.

## Dump inventory

The official WB98 dump has no
`leave_target_out_state` / `leave_target_out_covariance` fields.
`Cin` is `Trk::Track.trackParameters().front()` of
`CKFTrackCollection`.

## Target independence

Target station is in the same CKF tracklet set for **100%** of
official pairs.  Prediction and target measurement are not
independent.  `Cov(pred,target)` was not estimated.

Acceptance is independence of the prediction state, not a better
χ².  That proof is absent, so the contract is not established.

A later independent helper in `alignment_ML` must **not** wrap
`KalmanFitterTool.fit` as-is.  That wrapper cannot include station 0
and exclude a target station at the same time.
