# 独立导数参考与缺失传输耦合（Stage B / Task B14W）

Workbook 122。WB121 冻结了两类不同残余：control `100043/1` 的 `loc1`
是相对已收敛 FD 的真实 analytic 差；`100048/86` 没有可认证的 FD
reference。本任务把这两个问题分开：定位 control 1 多出来的
0.001–0.002 loc1 耦合，并尝试用已经冻结的 `h, h/2, h/4, h/8` 为 86
建立独立 derivative reference。

```
source bound
  → bound→free
  → RK variational transport
  → material interaction / curvilinear reset（只看 mean Jacobian）
  → surface crossing
  → supporting-plane intersection
  → free→bound continuation
  → next-hop source
```

不改正式 `rk_free` Jacobian，不重调 Gauss–Newton，不改生产
`stepTolerance`，不增加 FD rung，不缩小 FD 步长，不按结果挑最好 FD
步长或积分精度，不放宽冻结 5% gate，不把正式 sequential likelihood
换成 direct-from-source，不用任何 ACTS Jacobian 替换正式 likelihood，
不加 prior / ridge，不删 37/86，不用 truth q/p，不以修 5D Cin 为目标，
不进 B14M / B15 / Measurement Model V2。禁止 dummy-cov bounded
`transportJacobian`。不重跑 Athena；只使用冻结的 WB120 dumps。

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(θ) = Σ r_i(θ)^T R_i^{-1} r_i(θ)
R_i = (0.08 mm)² / 12
r_i = m_loc0 - predicted_loc0_on_supporting_plane
```

残差效应小不等于契约通过。control 1 的额外 loc1 耦合乘上正式
0.01 mm 步长后仍只有 strip sigma 的约 0.2%；冻结 5% gate 仍然 FAIL。

## 允许的判决

- `missing_deterministic_transport_coupling_identified`
- `segment_variational_derivative_inconsistent`
- `derivative_composition_or_reparameterization_incomplete`
- `independent_reference_established_for_focus`
- `independent_reference_not_established`
- `mixed_or_inconclusive`

`jacobian_contract_established` 与 `b14m_reopen_authorized` 要求
control 0/1/37 PASS、86 有独立 reference、正式路径 analytic 与该
reference 一致、无 branch switching、target leakage = 0、且 5% gate
未改。找到缺失耦合只授权后续修 derivative implementation，本任务
内部不修。

正式跑 `sbb14w_independent_reference_20260908T204204Z_d305d4b7`：

```
decision = missing_deterministic_transport_coupling_identified
smoke_gate_passed = true
jacobian_contract_established = false
b14m_reopen_authorized = false
control1_station0_loc1_agrees = true
rk_magnet_hop_pos_to_dir_is_zero = true
material_reset_is_highest_priority_mechanism = false
focus_independent_reference_established = false
five_percent_gate_unchanged = true
small_physical_effect_does_not_pass_contract = true
next_step = repair_missing_pos_to_dir_variational_coupling_then_recontract
```

## 本跑看到的事实

`100043/1` 的 station 0 loc1：FD 与 `rk_free` 都是立体投影 `±0.02`
（ΔJ ~ 1e-13）。第一个 `|ΔJ| > 5e-5` 永远是 measurement 6，也就是
第一条长磁 hop。该 hop 的 free-transport 积有 `∂dir/∂pos = 0`，
last-reset `jacTransport` 同样为零，continuation 的 loc1 列是
`(-0.040, 0.9992, 0, 0, 0, 0)`。后续 0-step hits 继承 FD 多出来的
0.001–0.002 耦合。因此额外灵敏度进不了 `rk_free` chain：loc1 作为
横向位置从未通过磁场耦合进方向。

磁 hop 上确实有 material reset，但不是最高优先级机制：终点 SCT 面
没有 surface material，而且完整 RK 积里 `∂dir/∂pos` 已经是 0。未把
covariance transport 或 process noise 混进 mean-state derivative。

station 0 段与冻结全链 FD 闭合。没有新的 hop-start Athena FD。
0-step 继承把泄漏定位到磁 hop 的 variational 漏项，而不是已转储
矩阵的 composition（WB121 相对误差已是 0）。

`100048/86` 仍然没有独立 reference。冻结四档上的 Richardson 为
`not_applicable`（振荡 / 变号）。局部奇对称多项式只作诊断，不替换
正式 FD。在 control 1 修好之前禁止用 ACTS variational 自我认证。
逐 hit：station 0 稳定；第一个不稳定 hit 是剩下的第一条长 hop，随后
下游 hits 同向漂移。禁止继续缩小全局 FD 步长。

这仍然不是 B14M PASS，不授权 restart invariance，不授权 1989。
