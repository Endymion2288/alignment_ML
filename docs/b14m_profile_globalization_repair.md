# Profile Globalization Repair and Stationarity Recontract (Task B14M-T)

Workbook 131. Incoming freeze is **WB130**. This book repairs **numerical
globalization and termination only**. It does **not** submit 1989.

```
WB128 Jacobian PASS
        ↓
WB129 restart invariance FAIL          (historical; not a validity certificate)
        ↓
WB130 no multibasin / no hysteresis
      high-χ² endpoints non-stationary
      flat_direction termination invalid
        ↓
WB131 explicit nuisance profiling
      + range-space trust-region
      + stationarity-aware termination
      recontract ALL 12 identities × 4 restarts
```

Statistical objective remains raw `chi2`. An LM multiplier, if used inside
the trust-region subproblem, is algorithmic globalization only:

```
prior_introduced = false
ridge_added = false
statistical_model_unchanged = true
```

Trust-region constants were preregistered in
`configs/b14m_profile_globalization_repair_v1.yaml` before any new 37/86
result. They must not be retuned from WB131 output.

A PASS (Case A) authorizes WB132 full-sample preflight only.
`full_sample_authorized` stays false.
