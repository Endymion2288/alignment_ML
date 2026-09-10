# 正式路径导数残余契约（Stage B / Task B14V）

Workbook 121。WB120 已经建立正式 Mode-B `h_i(θ)` 的同路径 `rk_free`
Jacobian，并证明函数值与正式 predicted loc0 一致。本任务只分类冻结
`h, h/2, h/4, h/8` 阶梯上剩下的两处矛盾。

```
source bound
  → 无边界 navigator/stepper（nullopt cov，BoundaryCheck false）
  → 终点 free state
  → 支撑平面相交
  → 局部 loc0
  → residual = m_loc0 - loc0
```

不改 Jacobian 实现，不重调 Gauss–Newton，不改生产 `stepTolerance`，
不按结果挑最好 FD 步长或积分精度，不放宽冻结 5% gate，不把正式
sequential likelihood 换成 direct-from-source，不用任何 ACTS Jacobian
替换正式 likelihood，不加 prior / ridge，不删 37/86，不用 truth q/p，
不以修 5D Cin 为目标，不进 B14M / B15 / Measurement Model V2。禁止
dummy-cov bounded `transportJacobian`。不重跑 Athena；只使用冻结的
WB120 dumps。

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(θ) = Σ r_i(θ)^T R_i^{-1} r_i(θ)
R_i = (0.08 mm)² / 12
r_i = m_loc0 - predicted_loc0_on_supporting_plane
```

预注册 consistency envelope 是 `||J_h/4 - J_h/8||`。它只用来判断
analytic-vs-FD 是否落在 FD 自身不确定度内，**不**修改 5% gate，也
**不**挑选某一档 FD。

## 允许的判决

- `official_path_derivative_contract_established`
- `fd_reference_not_precise_enough_for_derivative_certification`
- `small_column_relative_metric_pathology`
- `official_path_derivative_still_inconsistent`
- `mixed_or_inconclusive`

`jacobian_contract_established` 与 `b14m_reopen_authorized` 要求第一类
**并且** control 0/1/37 PASS **并且** 86 PASS。pathology 不得自动放宽
gate。即便 PASS，下一步也只是 B14M smoke restart invariance，不是 1989。

Official run `sbb14v_official_residual_20260908T194821Z_c5b463f5`：

```
decision = mixed_or_inconclusive
smoke_gate_passed = true
jacobian_contract_established = false
b14m_reopen_authorized = false
free_state_jacobian_available = true
official_loc0_matches_diagnostic = true
control_fd_converged = true
control_five_percent_pass = false
focus_fd_converged = false
segment_composition_closed = true
step_level_D_persisted_in_acts = false
five_percent_gate_unchanged = true
active_residual_kinds =
  official_path_derivative_still_inconsistent,
  fd_reference_not_precise_enough_for_derivative_certification
next_step = keep_residual_diagnosis_without_shrinking_fd
```

## 本跑看到的事实

`100043/0` 与 `100043/37` 五列 × 三个 target 仍然通过冻结 5%。

`100043/1` 的 `loc1`（target 1/2）**不是**小列 relative-gate 病态。
FD 阶梯已经收敛（最后一对 rel ≈ 7e-4）。analytic 在每个 hit 上都是
严格的立体投影 `±0.02`；正式 FD 在前六个 hit 上也是这个值，随后多出
0.001–0.002 的传输耦合。`|J_a-J_FD|` 大约是 last-pair envelope 的 90
倍，所以 6.16% / 5.64% 是真实的局部导数差。把它乘上正式 0.01 mm loc1
步长后，残差效应只有 strip sigma 的 0.2%，但这不能放宽 gate。

`100048/86` 是 `fd_reference_not_precise_enough`。FD 四档不收敛。
analytic 是已经在 control 上认证过的同路径 `rk_free` 列。analytic 与
FD 的偏差与 FD 自身层间差同量级，因此禁止继续缩 FD 步长。下一步需要
独立于该阶梯的 derivative reference。

事件 1 与 86 的 hop 级 composition 相对误差为 0：
`jRkFreeAcc * boundToFree(start) = bound_to_free_rk_product`，
`jacTransport * jacToGlobal * jacobian = acts_composed`。正式 loc0 与
诊断 loc0 差仍为 0。ACTS 32.0.2 EigenStepper 在
`GenericDefaultExtension::transportMatrix` 里计算每步 `D`，当场乘进
`jacTransport`，不保存一步级 D，也没有更高精度 variational state。
允许的读出仍是 WB120 已经转储的 hop 级积。

这仍然不是 B14M PASS，不授权 restart invariance，不授权 1989。
