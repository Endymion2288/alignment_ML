# 86 共用网格影子段参考（Stage B / Task B14Z）

Workbook 125。进线冻结是 **WB124**，不是 WB123。WB123 已经修好
场梯度 tangent，并让 `100043/0,1,37` 通过。WB124 的自适应 DOPRI5
独立 FD 失败，因为 `+δ/−δ` 各自 adaptive stepping 后走了不同
accepted-step 序列，而且真空 mean 与生产段最多还有约 0.044 mm
函数值差。本任务**不再回调**那份 DOPRI5，**不增加也不缩小**
track-state FD rung，**不修改** `FieldGradientDefaultExtension`。
唯一目标是为 `100048/86` 的 required hops `1/6, 2/11, 3/11` 建立
与生产 mean **同一物理合同**的 common-grid shadow。

```
WB124:
自适应独立 FD 失败
（accepted-step 分支噪声 + 真空 mean 差）
        ↓
WB125:
冻结共用网格影子
同一物理 mean contract
（磁场 + 表面能量损失 + supporting plane；
不含 process noise）
        ↓
只有 C 自身收敛、网格 refinement 稳定、
mean contract 成立，并且 A≈C 在 5% 内
覆盖 1/6、2/11、3/11，同时 control 仍 PASS
        ↓
jacobian_contract_established
        ↓
只重开 B14M smoke
```

Required hops 冻结。禁止把 target 2/3 已经收敛的 earlier hit 6
偷换成 required reference：

```
target 1: hit 6
target 2: hit 11
target 3: hit 11
```

正式 `h_i(θ)` 一字不动。生产 `stepTolerance` 保持 `1e-4`。冻结
FD ladder 保持 `h, h/2, h/4, h/8`。5% gate 不放宽。不增加 rung，
不缩小 track-state FD，不回调 WB124 DOPRI5，不加 prior / ridge，
不用 truth q/p，不调 Q / Cin，不重开 B14M，不进 B15 / V4 /
Measurement Model V2，不提交 1989。禁止 analytic 自我认证。禁止
把真空 ODE 自动称为 production-map reference。禁止因为 target 3
production loc1 已经 3.53% PASS 就单独放行。

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(θ) = Σ r_i(θ)^T R_i^{-1} r_i(θ)
R_i = (0.08 mm)² / 12
r_i = m_loc0 - predicted_loc0_on_supporting_plane
```

## 允许的判决

- `focus_common_grid_reference_established`
- `shadow_mean_contract_not_established`
- `field_map_interpolation_nonsmoothness`
- `material_map_discontinuity`
- `supporting_plane_terminal_event_sensitivity`
- `shadow_integrator_resolution_unresolved`
- `repaired_variational_focus_inconsistent`
- `mixed_or_inconclusive`

认证不是单独的 `A≈C`。`C` 必须在冻结 track FD ladder 上自身收敛，
在预注册的 shadow-grid refinement 下稳定，mean contract 成立，并且
与 `A` 在 5% 内吻合 `loc1/phi/q/p`。生产 `B` 可以继续因 adaptive
stepping 而不收敛。

若名义影子 mean 连函数值都不能闭合，则直接判
`shadow_mean_contract_not_established`，不能谈 derivative。

## 影子积分器（只作诊断）

`alignment/leave_target_out_dump/CommonGridShadowIntegrator.hpp`

```
method: classical_rk4_common_grid
不是 EigenStepper / GenericDefaultExtension /
    FieldGradientDefaultExtension / WB124 DOPRI5
场：正式 FASERMagneticFieldWrapper::getField
能损：冻结表面 slab 上的 Acts::computeEnergyLossMean
多次散射 / process noise / 协方差：不进入 mean
```

网格只由 **nominal 未扰动 hop** 生成。所有 `±h…±h/8` 臂强制复用
同一冻结节点序列、同一物质划分、同一 RK4 stages。网格尺寸和
mean-closure 门限在看见任何 86 Jacobian 之前预注册：

```
coarse_max_step_mm   20
nominal_max_step_mm  10    # 一个 FASER 磁体格子尺度
fine_max_step_mm      5
mean_loc0_abs_mm      1e-3
mean_path_abs_mm      1e-2
mean_pos_abs_mm       1e-2
mean_dir_abs          1e-6
mean_qop_rel          1e-4
grid_deriv_rel_max    0.05
grid_mean_loc0_abs_mm 1e-3
```

refinement 只作用于影子积分网格，不作用于 track FD。coarse /
nominal / fine 三套都预先注册。

## 数值冻结（WB125）

Official run：`sbb14z_focus86_common_grid_shadow_20260909T130344Z_2aa503bd`

```
decision = shadow_mean_contract_not_established
control_0_pass = true
control_1_pass = true
control_37_pass = true
mean_path_unchanged = true
target_exclusion_holds = true       # 12 行，0 leaked
focus_independent_reference_established = false
jacobian_contract_established = false
b14m_reopen_authorized = false
next_step = keep_86_common_grid_shadow_mean_contract
```

Required hops 的生产 mean 实际包含磁场、supporting-plane 终止、
以及确定性表面能量损失（`n_surface = 5, 4, 1`）。体积物质只记录、
不进入 ACTS mean。process noise 不进入 mean。path length 与生产
完全一致。common-grid 的 `branch_identity_same` 全部为 true，
WB124 的 accepted-step 分支噪声已被消掉。

名义影子 **没有** 通过预注册 mean contract：T1/6 loc0 残差
0.015 mm，T2/11 0.00131 mm，T3/11 的方向和 q/p 残差未过 dir /
qop 门。函数值未闭合，因此导数不得认证。不得按 Jacobian 符合度
回调 mean 门限或网格。不得再回去调 track FD 或 WB124 DOPRI5。

Config SHA `547d727517659b2f5e8c20a14f282982fdfb71e0d4d34483b6ce640eba952dcc`。
Helper SHA `3c1f1d41a79b2ff179173088d82057bcc31362822481f63c1e81610699763844`。
影子积分器 SHA `be1e086040adc4e2c1b45732ceb21a0b49422731843655a322ffe3d049d8dce5`。
Decision SHA `15fbffbd06ef6747d3adecec9cdb6e64a0be9c9986bd08703e8e9c553f49d5c2`。
场梯度扩展 SHA 未改
`ca5e4f0ef1a7a09edd1c24099e7ce689c4f0511e2d4afab3a160ebd2d6ff03d8`。
