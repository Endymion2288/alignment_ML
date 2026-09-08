# CKF Fit Covariance Semantics (Stage B / Task B11)

Workbook 107.  Explains why WB105 Cin is already overwide before
propagation (`λ = 0.027 / 0.054 / 0.153 / 11.35`, pencil `6.80`).
Evidence is the pinned Calypso / ACTS path, not class-name guesswork.
Cin is not rescaled, clipped, re-diagonalized, or empirically
calibrated.

## Official result

```
verdict = DIAGNOSED
decision = ckf_fit_covariance_semantics_audited
primary_case = official_cin_is_global_kf_refit_front_state
secondary_cases = official_refit_marks_ift_as_outlier,
                  seed_covariance_inflated_before_refit,
                  target_stations_used_in_refit
suitable_independent_propagation_seed = false
measurement_model_v2_authorized = false
cin_modified = false
```

Official run `sbb11_ckf_covariance_semantics_20260906T211908Z_596ac546`.

## Production chain

1. **CKF2** combinatorial Kalman filter, MS and energy loss on,
   `Acts::GainMatrixSmoother` connected.  Optional fitted parameters
   at the seed-tool target plane (`addFittedParamsToTrack=true`).
2. **CreateTrkTrackTool** persists measurement TSOS as **smoothed**,
   outlier TSOS as **filtered**, hole TSOS as **predicted**.  If
   fitted params are supplied they are inserted at the front as a
   Hole.
3. **KalmanFitterTool.fit** refits the `Trk::Track` into
   `CKFTrackCollection` when the refit succeeds.  It uses every
   `measurementsOnTrack` hit, marks IFT (`z < -100 mm`) as outliers,
   inflates the seed covariance (`×100` on the diagonal, then `×10`),
   and calls `createTrack` **without** fitted params.  Hit covariance
   is hardcoded to `0.08²/12`.
4. **CkfActsTransportDumpAlg** exports
   `track->trackParameters()->front()` as Cin.

After a successful official refit, `front()` is the most upstream
measurement surface.  For IFT that surface is an **outlier**, so the
persisted parameters are **filtered / predicted**, not a measurement
update, and not a leave-target-out state.  Stations 1–3 remain in
the refit as smoothed measurements.

`trackParameters().front()` is therefore **not** a suitable
independent propagation seed.

## Cin shape (diagnostic, unchanged)

663 unique contracted tracks, truth used only as a source-surface
diagnostic:

| λ(C_emp, Cin) | Pencil |
| --- | --- |
| 0.027 / 0.054 / 0.153 / 11.35 | 6.80 on `x` |

Overcoverage is present **before** propagation.  This task does not
repair Cin.

Per-TSOS filter/smoother flags are **unavailable** on the current
jsonl / ntuple.  A later provenance-only dump is allowed; Cin
recalibration is not.
