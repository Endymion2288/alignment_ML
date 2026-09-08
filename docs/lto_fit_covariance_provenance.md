# LTO Fit Covariance Provenance (Stage B / Task B14R)

Workbook 111.  WB110 showed that independent LTO states exist but
the exported 5×5 fails the uncertainty contract (undercoverage).
This task asks what that 5×5 actually is.

It does **not** try to pass closure.  It does **not** enter B15.

## Questions

1. Is `n_measurements_in_fit` the ACTS measurement-flag count, or
   just a dump alias for input hits?
2. Does the exported covariance follow the seed scale?
3. Which ACTS track state is exported?
4. Which 5D directions are actually constrained after a target is
   removed?
5. Are the WB110 pull RMS values typical or tail-dominated?
6. Why does `100043/37` keep official q/p and seed-scale σ(q/p)?

## Cases

- A: exported covariance is seed or predicted
- B: remaining measurements do not constrain the 5D state
- C: fitted covariance is seed-independent but empirically too small
- D: typical covariance is fine; a few fits dominate the pulls
- E: mixed

Seed scales `0.1 / 1 / 10` are pre-registered sensitivity smokes.
The production scale stays 1.  No scale is chosen from truth or χ².

## Field definitions

`n_measurements_in_fit` is `fittedTrack.nMeasurements()`, i.e. ACTS
`calculateTrackQuantities` counting `TrackStateFlag::MeasurementFlag`.
It is not the input hit count and not `used_measurement_count`.

The exported 5×5 is `KalmanFitter` `fittedParameters` at the source
station plane (`KalmanFitterTargetSurfaceStrategy::first`).  That is
the smoothed first measurement after transport to the reference
surface, not `KalmanFitterTool.fit`.

`ndof`, filtered/smoothed/outlier counts, seed covariance, and the
first/last predicted/filtered/smoothed matrices exist only on the
B14R smoke schema.  Frozen WB109 dumps keep the B13 schema.

## Decision tokens

The task may only land in:

```
lto_exported_covariance_is_seed_or_predicted_state
lto_fit_information_insufficient
lto_fitter_covariance_semantics_mismatch
tail_dominated
mixed_or_inconclusive
```

None of these authorize B15, V4 C/D, or Measurement Model V2.

## Official tokens

The official B14R run landed on mixed B+D:

```
decision = mixed_or_inconclusive
lto_cin_contract_established = false
b15_authorized = false
```

The exported 5×5 is the Kalman fitted state at the source plane,
not a raw seed or predicted matrix.  Typical fit means are nearly
seed-independent, but the 5×5 widths are not.  y-pull RMS is
tail-dominated.  Do not pick a seed scale from the smoke.
