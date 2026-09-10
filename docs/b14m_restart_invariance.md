# Profile-Likelihood Smoke Reopen and Restart Invariance (Stage B / Task B14M-R)

Workbook 129. Incoming freeze is **WB128**. The Jacobian contract is
already closed. This book only reopens the login-scale B14M smoke.

```
WB127 mean function PASS
        ↓
WB128 independent derivative reference PASS
      Jacobian contract PASS
      B14M reopened
        ↓
WB129 profile likelihood smoke
      + four pre-registered restarts
      + objective / prediction invariance
        ↓
ONLY IF PASS: restart_invariance_established = true
        ↓
WB130 1989 preflight
```

The statistical model is unchanged:

```
chi2(theta) = Σ r_i(theta)^T R_i^{-1} r_i(theta)
chi2_prof(alpha) = min_nu chi2(alpha, nu)
R_i = (0.08 mm)^2 / 12
```

No prior, no ridge, no truth q/p, no target measurement in the fit.
q/p stays an explicit nuisance. The numerical Jacobian is the inherited
WB123 repaired tangent plus WB127 certified mean. That is not a new
model.

Restart tolerances are recovered from frozen WB114/WB115. They were not
set after seeing this run.

## Official result

Official run: `sbb14mr_restart_invariance_20260909T183700Z_f1c280a7`

```
decision = profile_optimizer_restart_sensitive
verdict  = FAIL
b14m_smoke_passed = false
restart_invariance_established = false
restart_invariance_authorized = false
jacobian_contract_established = true
full_sample_authorized = false
```

```
config SHA      4ce8379b4cab239b5124cafea4f17ae8805a2b676eb7b506c597ae6e38d68890
decision SHA    67aa191def9863d39dec6643bf117516aa95039d6a90e0e38a09ac5b161059ea
helper SHA      686d980fbf72a872af74b4cb9b4a0bdbbd608e615a787dcc55b4986027c6ec4d
```

Dump root: `outputs/leave_target_out_dump_v1/b14m_reopen_smoke/`

Controls `100043/0` and `100043/1` are restart-invariant on the
profiled objective and on predictions. Their nuisance parameters are
not unique (Case C). `100043/37` converges with χ² ~ 1e4–1e5 and stays
in the sample; T1/T2 are restart-sensitive, T3 is invariant.
`100048/86` T1 is a profile-level multimodality / globalization
failure, not a Jacobian failure. T2/T3 are invariant. Target leakage =
0. 1989 was not submitted.

The next book must remain on B14M-R. It must not start WB130 or submit
1989.
