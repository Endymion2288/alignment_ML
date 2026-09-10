# 支撑平面 Jacobian 连续性契约（Stage B / Task B14J）

Workbook 117。WB116 已经用支撑平面把 `h_i(θ)` 在 smoke
0/1/37/86 的全部 surviving measurements 上变成可求值。剩下的
blocker 是 86 的 Jacobian spot-check。本任务审计该映射在局部是否
连续、是否有可认证的 Jacobian。不改 WB114 统计模型，不重调
Gauss–Newton，不按结果挑最好 FD 步长，不把正式 sequential
likelihood 换成 direct-from-source，不加 prior / ridge，不删
37/86，不用 truth q/p，不以修 5D Cin 为目标，不进 B14M / B15 /
Measurement Model V2。

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(θ) = Σ r_i(θ)^T R_i^{-1} r_i(θ)
R_i = (0.08 mm)² / 12
```

正式 `h_i` 仍是一条确定性的支撑平面顺序轨迹。direct-from-source
只作预注册对照。

## 固定 FD 阶梯

每个正式列都用预注册中心步长

```
h(loc0) = 0.01 mm
h(loc1) = 0.01 mm
h(phi)  = 1e-5
h(theta)= 1e-5
h(q/p)  = 1e-6 /GeV
```

和固定阶梯 `h, h/2, h/4, h/8`。审计记录全部四档，不把误差最小的
那一档当成正式 Jacobian。

每个扰动记录 predicted loc0/loc1、χ²、传播步数、path length、
几何/表面序列、inside-bounds、continuation-state 构造路径、
free-to-bound fallback、最终方向。

## 允许的判决

- `supporting_plane_jacobian_continuity_established`
- `finite_difference_step_not_in_asymptotic_region`
- `acts_navigation_material_branch_switching`
- `supporting_plane_continuation_state_discontinuous`
- `source_to_measurement_transport_not_smooth`
- `mixed_or_inconclusive`

若 direct PASS 而 sequential FAIL，根因优先落在支撑平面后的
continuation / sequential hop。若两者都 FAIL，对象仍是
source→measurement map。direct 数值更好也不能替换正式 likelihood。

正式 run `sbb14j_jacobian_continuity_20260908T150238Z_a30b79e2`：

```
decision = source_to_measurement_transport_not_smooth
smoke_gate_passed = false
jacobian_contract_established = false
b14m_reopen_authorized = false
restart_invariance_authorized = false
```

0/1/37 在完整阶梯上收敛，符号和 surface/path 一致。86 的
sequential 与 direct 都在 `loc1 / phi / q/p` 上失败；更小步长让
相对误差变差，而不是变好。没有几何/continuation 分支切换，也没有
free-to-bound fallback。正式 sequential likelihood 未改。

这不是 B14M PASS，也不授权 restart invariance 或 1989 行全样本。
