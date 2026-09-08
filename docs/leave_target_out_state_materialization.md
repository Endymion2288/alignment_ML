# Independent Leave-Target-Out State Materialization (Stage B / Task B13)

Workbook 109.  WB108 blocked V4 arms C/D because no independent
leave-target-out state existed.  This task materializes that state
on the frozen WB103 contracted sample.

The helper lives only in `alignment_ML`.  It does **not** call
`KalmanFitterTool.fit`.  It reads the same frozen xAOD
`CKFTrackCollection`, removes every measurement attributed to the
current target station, refits the remaining hits with an inline
ACTS Kalman filter (IFT kept, no outlier finder, MS/Eloss on), and
exports a source-surface state plus a full 5×5.

Acceptance is the LTO contract, not covariance closure.
`100043/37` is retained.  Truth q/p, dummy q/p, empirical
`Cov(pred,target)`, Cin rescale, Q tuning, and Measurement Model V2
are forbidden.

## Required export

```
event/source identity
source station
target station
used station IDs
used measurement IDs/counts
excluded target measurement IDs/counts
fit success / failure reason
reference/source surface
native track parameters
5×5 covariance
signed q/p
state/covariance units and frame
material / field / geometry / conditions hashes
```

`target_station_measurements_used = 0` must be machine-checkable
on every row.  Failures are kept.  No new selection from fit
results.

## Six-point contract

1. Target exclusion is proven.
2. Every successful fit has a source/reference-surface state + 5×5.
3. The covariance is not the WB107 `front()` Cin, not the official
   seed covariance, and not an empirical construction.
4. q/p is the fit result, not dummy or truth.
5. Construction / validation success and failure reasons are
   reported on the full WB103 denominator.
6. Closure information is not used to select states.

Official PASS tokens:

```
decision = leave_target_out_state_materialization_established
independence_proven = true
lto_states_materialized = true
b14_authorized = true
```

This does **not** set `transport_covariance_validated` or authorize
Measurement Model V2.  Only a PASS here authorizes Task B14.  B15
remains closed.
