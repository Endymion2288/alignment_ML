# Workbook 88 — convergence-metric repair for zero and numerical-floor chi2

日期： 2026-09-30
分支： `4station`
HEAD at inspection： `eea34212265e95060f1facf9bb744e104e5e2958`
任务： 只检验 Workbook 87 第 10 节预注册的收敛判据修复。不重跑 Calypso，不改求解器，不重判 WB86。

```text
WB86 remains FAIL.
This workbook does not requalify WB86.
ml_alignment_eval_authorized = false
alignment_oracle_qualified_for_physical_FASER = false
```

---

## A. Frozen protocol

Workbook 87 section 10 is the authoritative protocol. It required

```text
scaled_update_norm < 1e-3
and (validation_relative_change < 1e-3
     or chi2 == 0
     or abs(chi2) is below a preregistered absolute floor)
```

for `CONSECUTIVE_REQUIRED = 2` iterations, with `MAX_ITERATIONS = 10` unchanged.
It named the absolute floor and did not assign its number. That number was
frozen before any WB88 replay, as the solver constant that already exists:

```text
ABSOLUTE_CHI2_FLOOR = DEFAULT_DAMPING = 1e-8
```

It was not fit to a trajectory.

The predicate lives in `alignment/wb88_convergence_policy.py`. It does not call
the solver. For a recorded step with current chi2 `c` and previous chi2 `p`:

```text
iteration 1, p missing:
    metric = null
    regime = no_previous

abs(c) <= 1e-8 or abs(p) <= 1e-8:
    metric = abs(c)
    regime = absolute_floor

otherwise:
    metric = |c - p| / |p|
    regime = relative
```

A step passes when `scaled_update_norm < 1e-3` and `metric` is defined and
`metric < 1e-3`. Two consecutive passes stop. Exact `c = p = 0` gives
`metric = 0`, so it does not stay null. A large chi2 keeps the historical
relative ratio. The scaled-update gate is still required; a small chi2 alone
does not stop.

Preregistration, hashed before replay:

```text
c0fc1426aacc39744d761ec7a7a72fb9f7f74485a5d6aaa0808c8e060e87d7f3
  outputs/mc24_four_station_wb88_convergence_policy_v1/wb88_preregistration.json
```

Evaluation is a replay of the 13 WB87 trajectories from cluster `1160511`.
No Calypso job was submitted. WB86 code and WB86 outputs were not modified.
The historical WB86 predicate was recomputed beside the new one. On these 13
events the recomputed WB86 stop agrees with the recorded WB87 reason.

A large wrong chart means `solver_scaled_truth_error >= 1e-3`, using the same
`PARAMETER_SCALES` and the same `1e-3` already frozen for the update gate.

---

## B. Exact-zero controls

Events 0 and 510.

At iterations 2 and 3 both have

```text
chi2 = 0
previous_chi2 = 0
historical_relative_change = null
objective_stability_metric = 0
regime = absolute_floor
scaled_update_norm = 0
solver_scaled_truth_error = 0
```

WB86 consecutive count stays 0. WB88 reaches consecutive count 2 at iteration 3.
The chart at that iteration is the WB87 terminal chart, which is the truth.

Event 510 still has `n_dropped = 3` on every iteration. The stopping predicate
does not look at rank. A later qualification that keeps the frozen analyzable
rule (`n_dropped == 0`) would still exclude event 510. That exclusion is the
closed rank hypothesis, not a failure of this stopping rule.

---

## C. Numerical-floor controls

Events 105, 151, and 358. After the first physical step the chi2 is far below
`1e-8`, and the historical ratio of two floor values stays above `1e-3`.

```text
105  k=2  chi2 = 5.96e-19   historical ratio = 1.00    metric = 5.96e-19
     k=3  chi2 = 7.19e-21   historical ratio = 0.988   metric = 7.19e-21
     WB88 stops at iteration 3
     solver-scaled truth error = 2.50e-12

151  k=2  chi2 = 7.42e-19   historical ratio = 1.00    metric = 7.42e-19
     k=3  chi2 = 2.90e-21   historical ratio = 0.996   metric = 2.90e-21
     WB88 stops at iteration 3
     solver-scaled truth error = 9.14e-14

358  k=2  chi2 = 1.17e-08   historical ratio = 1.00    metric = 1.00
     k=3  chi2 = 1.07e-20   historical ratio = 1.00    metric = 1.07e-20
     k=4  chi2 = 3.15e-20   historical ratio = 1.95    metric = 3.15e-20
     WB88 stops at iteration 4
     solver-scaled truth error = 4.00e-6
```

Event 358 iteration 2 is still regime `relative`: `1.17e-8` is just above the
floor, the ratio is 1, and the gate does not pass. Iterations 3 and 4 are on
the floor. The metric there is the absolute chi2, not the ratio. The `4e-6`
scaled truth error is the already established `dz ≈ -2e-5 mm` left-SE(3)
coupling. This workbook does not change that chart.

---

## D. Pathological negative controls

The floor clause does not create a new stop on any large-error event.

```text
239   no WB88 stop. Iteration 3 had a small update and a relative metric of
      3.5e-6, so the consecutive count became 1. Iteration 4 jumps S1 rz to
      -840 mrad (scaled update 91) and the count returns to 0.
      Terminal scaled truth error = 1.72.
247   no WB88 stop. Terminal scaled truth error = 0.598.
      Terminal S1 rz = +33.19 mrad, S3 rz = +12.93 mrad.
2285  no WB88 stop. Terminal scaled truth error = 0.455.
      Terminal S3 rz = -26.82 mrad.
1586  no WB88 stop. Terminal scaled truth error = 30.7.
      Terminal S3 dx = -153.25 mm, S3 ry = +65.38 mrad.
```

Events 657 and 1104 do stop, at the same iteration as WB86, in regime
`relative`. The floor clause is not what stops them.

```text
657   stop at iteration 9, inherited from WB86
      S1 rz = -0.500 mrad (truth -0.500)
      S3 rz = -62.609 mrad (truth +0.500)
      solver-scaled truth error = 1.052
      projected a_hat = -43.918, a_true = 0.707
      rho = 1.000
      n_dropped = 0

1104  stop at iteration 7, inherited from WB86
      S1 rz = +0.0297 mrad (truth -0.500)
      S3 rz = +0.500 mrad (truth +0.500)
      solver-scaled truth error = 8.83e-3
      projected estimate unavailable because n_dropped = 2
      rho = 1099
      n_dropped = 2
```

These two remain the rotation / rank pathology. The new predicate does not
repair them and does not newly accept them.

---

## E. Historical versus new stopping

| event | WB86 status | WB88 status | WB88 stop | scaled truth error |
| --- | --- | --- | --- | --- |
| 0 | max_iterations | converged | 3 | 0 |
| 510 | max_iterations | converged | 3 | 0 |
| 105 | max_iterations | converged | 3 | 2.50e-12 |
| 151 | max_iterations | converged | 3 | 9.14e-14 |
| 358 | max_iterations | converged | 4 | 4.00e-6 |
| 239 | max_iterations | max_iterations | — | 1.72 at iteration 10 |
| 247 | max_iterations | max_iterations | — | 0.598 at iteration 10 |
| 657 | converged | converged | 9 | 1.052 |
| 1104 | converged | converged | 7 | 8.83e-3 |
| 2285 | max_iterations | max_iterations | — | 0.455 at iteration 10 |
| 1586 | max_iterations | max_iterations | — | 30.7 at iteration 10 |
| 489 | execution refusal | execution refusal | — | — |
| 1745 | execution refusal | execution refusal | — | — |

Events 489 and 1745 stay `reproduced_missing_hit`. The stopping rule never
sees a solver step for them.

---

## F. False-convergence audit

Stops created by the absolute-floor clause while the solver-scaled truth error
is at least `1e-3`:

```text
none
```

Stops that WB88 inherits from WB86, at a chart that is still large:

```text
657    scaled truth error 1.052    regime relative
1104   scaled truth error 8.83e-3  regime relative
```

Both were already WB86 convergences. WB88 does not add a large-error stop.

---

## G. Conclusion

```text
convergence_metric_repair_supported
```

On this frozen 13-event replay the repaired predicate recognizes the exact-zero
fixed points and the numerical-floor fixed points, keeps the scaled-update
gate, and does not turn a wrong large chart into a new convergence. It also
does not remove the two historical WB86 stops that land on a wrong rotation
chart.

Summary sha256:

```text
3a2b023a504d07ccadd2ab21ff6f5c3b1cd9f710f7e92f9100e49bea20bc23b1
  outputs/mc24_four_station_wb88_convergence_policy_v1/wb88_summary.json
```

---

## 明确没做的事

- 没有重跑 WB86，没有改 WB86 JSON，没有把 WB86 改成 PASS
- 没有重跑 Calypso
- 没有改 `solve_common_track`、秩阈值、有限差分步长、left-SE(3)、association、corpus
- 没有放宽 `1e-3`，没有提高 `MAX_ITERATIONS`
- 没有引入阻尼、trust region、backtracking、IRLS
- 没有修旋转 chart，没有改 rank handling
- 没有打开 Final Blind，没有启动 ML-vs-truth

下一件被允许、但本 workbook 不执行的事，是单独预注册一次大样本物理验证，
使用这个已经冻结的停止规则，并使用新的输出目录。657 与 1104 的旋转/秩病理
在那次验证之前仍然分开，不由停止规则吸收。
