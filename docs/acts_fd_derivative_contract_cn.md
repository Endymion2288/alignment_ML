# ACTS Transport Jacobian 与固定多尺度 FD 导数契约（Stage B / Task B14L）

Workbook 119。WB118 冻结 `mixed_or_inconclusive`，并留下
`acts_transport_jacobian_available = true`、
`analytic_vs_fd_not_yet_contracted = true`。本任务只问一件事：
ACTS 32.0.2 已经提供的 `Result::transportJacobian`，能不能给冻结的
正式 source→measurement 映射 `h_i(θ)` 建立稳定 derivative contract。

不改 WB114 统计模型，不重调 Gauss–Newton，不改生产 `stepTolerance`，
不按结果挑最好 FD 步长或积分精度，不把正式 sequential likelihood
换成 direct-from-source，不用 ACTS Jacobian 替换正式 likelihood，
不加 prior / ridge，不删 37/86，不用 truth q/p，不以修 5D Cin
为目标，不进 B14M / B15 / Measurement Model V2。

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
FD 阶梯固定为 WB117/WB118 的 `h, h/2, h/4, h/8`。`1e-5 / 1e-6`
只是 WB118 已预注册的 falsification，不得挑最好 tolerance。

不能拿 ACTS 一个 5×5 直接和 residual FD 列比。转储的链是：

```
source bound
  → ACTS bound-to-bound transportJacobian
  → ACTS 终点 boundToFree
  → 支撑平面相交 / 局部 chart
  → predicted loc0
  → residual = m_loc0 - loc0
```

Continuation 是 `transform_free_to_bound(intersection, time=0, freeDir, q/p)`，
不是 ACTS bound 终点本身。契约列只用前 5 个 bound 参数。ACTS q/p
与 FD `h(q/p)=1e-6` 都是 `1/GeV`。

## 允许的判决

- `finite_difference_not_reliable_for_focus_transport`
- `acts_transport_jacobian_not_numerically_converged`
- `analytic_chain_or_chart_contract_broken`
- `mixed_or_inconclusive`

`jacobian_contract_established` 与 `b14m_reopen_authorized` 只能在
control 0/1/37 PASS、86 derivative contract PASS、无 branch switching、
target leakage=0、统计模型未改时打开。即便如此，下一步也只是
B14M smoke restart invariance，不是 1989 全样本。

正式 run `sbb14l_derivative_contract_20260908T180541Z_33bda85a`：

```
decision = analytic_chain_or_chart_contract_broken
smoke_gate_passed = true
jacobian_contract_established = false
b14m_reopen_authorized = false
restart_invariance_authorized = false
full_sample_authorized = false
control_fd_converged = true
control_acts_chain_agrees_fd = false
focus_acts_chain_stable = true
focus_fd_converged = false
acts_end_loc0_matches_official = true
chain_complete = false
branch_switching = false
target_exclusion_holds = true
statistical_model_unchanged = true
next_step = diagnose_jacobian_chain_chart_units_or_projection
```

WB117/WB118 不是物理不可微结论。本 run 也不允许把 86 的 FD 失败
写成 `finite_difference_not_reliable_for_focus_transport`：同一条
ACTS-chain 在已知 FD PASS 的 control 0/1/37 上已经和冻结 FD 不一致。

## 本 run 结论

从 source 出发、官方 0 step 的几何 hop，在 `loc0 / loc1 / phi / theta`
上与正式 FD 一致到 1e-9–1e-13。这说明残差符号（`J_r = -dh/dθ`）、
前 5 列、q/p 单位、支撑平面 loc0 chart，以及
`J_proj * J_b2f * J_acts` 都对。ACTS 已经落在面上时，
`projection_composed` 与 `acts_bound` 相差约 1e-16。

第一个真正积分的官方 hop（`steps ≥ 1`）就已经不一致，包括
continuation 仍是单位阵的 source 第一跳。后面的 0-step 立体 hop
会把这段积分解的 Jacobian 继承下去。

成功 hop 上 ACTS 终点 loc0 仍等于官方支撑平面 loc0。函数值在 `θ0`
一致；dummy covariance 的 bound-to-surface `transportJacobian` 不是
正式 `h_i` 的导数。约 3 m 磁 hop 之后，正式 FD 的 `q/p` 列是
O(10²–10³)，ACTS-chain 的 `q/p` 列仍是 O(10⁻³)–O(10⁻¹)。

Dummy-cov 传播到面，不是正式无边界 `SurfaceReached` + 投影。37 的
长 hop 打到 4000-step 上限，后续 hop 报 `position not on surface`。
86 也有同类失败。正式 Mode B 仍然求值得出。

86 上能写出的 ACTS-chain 列在预注册 `1e-4 / 1e-5 / 1e-6` 之间稳定
（相对误差约 1e-5）。稳定不等于和正式 `h_i` 建立了 derivative contract。

这仍然不是 B14M PASS，不授权 restart invariance，不授权 1989 行。
