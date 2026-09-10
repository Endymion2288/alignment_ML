# Profile Basin Topology and Flat-Direction Termination Diagnosis (Task B14M-S)

Workbook 130. Incoming freeze is **WB129**. This book does **not** repair
the optimizer and does **not** submit 1989.

```
WB128 Jacobian contract PASS
        ↓
WB129 B14M restart invariance FAIL
      37/T1, 37/T2, 86/T1
        ↓
WB130 DO NOT FIX YET
      audit flat-direction stationarity
      + frozen line-search replay
      + fixed-alpha nuisance cross-start
      + profile continuation
        ↓
decision = flat_direction_termination_not_stationary
        ↓
NEXT book: repair termination / globalization only
        ↓
rerun the original four restarts
        ↓
ONLY IF PASS: restart_invariance_established
        ↓
full-sample preflight
```

Official run: `sbb14ms_profile_basin_20260909T214204Z_45f7bda5`.

High-χ² WB129 endpoints are not stationary in the supported / profile
subspace. Rank deficiency is not a successful termination. Fixed-α
nuisance starts collapse to one χ²; forward / backward continuation
coincide. This is not a certified pair of likelihood basins.

`min(R0,R1,R2,R3)` is not a fix. Held-out target measurements were
inaccessible while basins were classified.

```
restart_invariance_established = false
full_sample_authorized = false
b15_authorized = false
```
