# 正式支撑平面传输 Jacobian 契约（Stage B / Task B14U）

Workbook 120。WB119 冻结 `analytic_chain_or_chart_contract_broken`：
dummy-cov 绑到有限面的 `Propagator::Result::transportJacobian` 是正式
`h_i(θ)` 的**错误 Jacobian**。本任务只问一件事：ACTS 32.0.2 能不能在
正式 Mode-B 路径上给出 free-state / stepping variational Jacobian。

```
source bound
  → 无边界 navigator/stepper（nullopt cov，BoundaryCheck false）
  → 终点 free state
  → 支撑平面相交
  → 局部 loc0
  → residual = m_loc0 - loc0
```

不改 WB114 统计模型，不重调 Gauss–Newton，不改生产 `stepTolerance`，
不按结果挑最好 FD 步长或积分精度，不把正式 sequential likelihood
换成 direct-from-source，不用任何 ACTS Jacobian 替换正式 likelihood，
不加 prior / ridge，不删 37/86，不用 truth q/p，不以修 5D Cin
为目标，不进 B14M / B15 / Measurement Model V2。禁止把 dummy-cov
bounded `transportJacobian` 当作正式导数。不手写未经验证的磁场模型。

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(θ) = Σ r_i(θ)^T R_i^{-1} r_i(θ)
R_i = (0.08 mm)² / 12
r_i = m_loc0 - predicted_loc0_on_supporting_plane
```

正式 `h_i` 仍是一条确定性支撑平面顺序轨迹。生产
`PropagatorPlainOptions::stepTolerance` 仍是 ACTS 32.0.2 默认 `1e-4`。
FD 阶梯固定为 WB117 的 `h, h/2, h/4, h/8`。

正式 `nullopt` hop 上 EigenStepper 的 `jacTransport` 保持单位阵，因为
只有带起始协方差时才会打开 `covTransport`。允许的诊断是：在**同一条**
无边界支撑平面 hop 上打开 stepper 已有的这个开关，读出它已经维护的
`jacTransport` 段，并在 `MaterialInteractor` 把它们重置成 curvilinear
之前把 RK 段乘起来。这不是 dummy-cov 传到有限面。

转储的正式路径链是：

```
source bound
  → hop 起点 boundToFree
  → RK free-transport D 矩阵之积
  → 支撑平面相交 / 局部 loc0
  → residual = m_loc0 - loc0
```

`acts_composed = jacTransport * jacToGlobal * jacobian` 只作对照，它含
正式 `h_i` 没有的物质面 curvilinear 重置。主家族是 `rk_free_chain`。

## 允许的判决

- `official_supporting_plane_jacobian_established`
- `acts_free_state_jacobian_unavailable`
- `official_path_jacobian_inconsistent_with_fd`

`jacobian_contract_established` 与 `b14m_reopen_authorized` 要求第一类
**并且** control 0/1/37 PASS **并且** 86 PASS，无 branch switching，
泄漏为 0，统计模型不变。即便如此，下一步也只是 B14M smoke restart
invariance，不是 1989 行。

Official run `sbb14u_official_jacobian_20260908T184910Z_d5148ffc`：

```
decision = official_path_jacobian_inconsistent_with_fd
smoke_gate_passed = true
jacobian_contract_established = false
b14m_reopen_authorized = false
free_state_jacobian_available = true
official_loc0_matches_diagnostic = true
control_fd_converged = true
control_official_path_agrees_fd = false
focus_fd_converged = false
focus_official_path_agrees_fd = false
next_step = keep_official_path_jacobian_diagnosis
```

这不是 `acts_free_state_jacobian_unavailable`。正式路径现在有同路径
free-state Jacobian，函数值与正式 predicted loc0 差为 0，长 hop 的
`q/p` 列与正式 FD 同量级。冻结 5% 列契约仍未过：control `100043/1`
的 `loc1`（target 1/2），以及 focus 86 的多数列。

## 本跑看到的事实

正式 `nullopt` hop 保持 `covTransport = false`、`jacTransport = I`。
诊断 dummy 协方差只是 variational 开关。诊断支撑平面 loc0 与正式
Mode-B loc0 在全部 12 行的每一个 hop 上差为 0。包括 37/86 长磁 hop
在内，所有 hop 都有 free-state Jacobian（`chain_complete = true`）。

主家族 `rk_free` 对冻结 FD：

- `100043/0` 与 `100043/37`：5 列 × 3 个 target 全部过（FD 最后一对
  rel ≪ 0.05，`rk_free` vs FD rel ≤ 0.002）。
- `100043/1`：`loc0/phi/theta/q/p` 过；`loc1` 在 target 1/2 上相对误差
  0.0616 / 0.0564，未过 5%；target 3 的 `loc1` 过（0.027）。该列本身
  很小（norm ≈ 0.085）。
- `100048/86`：FD 阶梯仍不收敛。`rk_free` 的 `q/p` norm 是 O(10³)，
  与正式 FD 同量级；WB119 dummy-cov 绑面 `q/p` 列只有 O(10⁻³)–O(10⁻¹)。
  逐列 5% 仍不过。

对照家族 `acts_composed` 在 control 上仍与 FD 不一致（典型 rel
0.9–28）。那就是 WB119 的对象：物质面 curvilinear 协方差传输不是
正式 `h_i`。

能量损失均值更新不在 RK `D` 积里。control 的 `q/p` 列仍在 10⁻³
量级与 FD 一致，所以缺 `d(EL)/dθ` 不是 control 失败原因。

这仍然不是 B14M PASS，不授权 restart invariance，不授权 1989。
