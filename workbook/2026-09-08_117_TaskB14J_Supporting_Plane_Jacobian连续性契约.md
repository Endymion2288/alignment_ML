# Workbook 117: Task B14J Supporting-Plane Jacobian 连续性契约

日期：2026-09-08
状态：**完成 / FAIL** —— 以 GitHub `master` `55cf982302a3c62c57b74f368d5e3ba7723fd33a` 为 checkpoint，在**不改变** WB114 measurement likelihood、不重调 Gauss–Newton、不按结果挑最好 FD 步长、不把正式 sequential likelihood 换成 direct-from-source、不加 prior / ridge、不删 `100043/37` 或 `100048/86`、不用 truth q/p、不以修 5D Cin 为目标的前提下，只对预注册 smoke `100043/0,1,37` 与 `100048/86` 做支撑平面 Jacobian 连续性审计。未进 B14M 全样本，未做 restart invariance，未进 B15，未进 V4 C/D，未进 Measurement Model V2。未覆盖 WB109 / WB114 / WB115 / WB116 dumps。**未提交 1989 行 Condor**。

**最终判定：`FAIL`**

- `decision = source_to_measurement_transport_not_smooth`
- `primary_case = source_to_measurement_transport_not_smooth`
- `smoke_gate_passed = false`
- `jacobian_contract_established = false`（0/1/37 PASS，86 FAIL）
- `b14m_reopen_authorized = false`
- `restart_invariance_authorized = false`
- `full_sample_authorized = false`
- `target_exclusion_holds = true`（24 行，0 leaked）
- `prior_introduced = false`
- `ridge_added = false`
- `statistical_model_unchanged = true`
- `profile_math_rewritten = false`
- `measurement_update_in_evaluator = false`
- `do_not_select_best_step = true`
- `do_not_switch_to_direct = true`
- `five_d_cin_not_the_objective = true`
- `b15_authorized = false`
- `measurement_model_v2_authorized = false`
- `do_not_force_5d_lto_covariance = true`
- `focus_identity_retained = true`
- `physical_nonidentifiability_not_claimed = true`

这仍然不是 B14M PASS。下一步不是重开 optimizer，也不是修 5D Cin，而是继续诊断 **86 附近 source→measurement map `h_i(θ)` 本身不光滑**。

## 起始状态

HEAD：`55cf982302a3c62c57b74f368d5e3ba7723fd33a`  
`Task B0–B14T: dump ACTS/LTO transport and keep profile likelihood fail-closed.`

工作树干净，与 `origin/master` 同步。未回到 `32c904...` 工作流。

WB109 PASS：`leave_target_out_state_materialization_established`  
Decision SHA：`af7ccd055c2ff3d34f92429e37be19a89ed90a9b0db11ec6770c161c79b6e933`

WB114 FAIL：`profile_likelihood_numerically_unstable`  
Decision SHA：`82304f5af943f97adfe069e02c8bacdd7491cca72e6ecc58ea82cc470428a18f`

WB115 FAIL：`profile_transport_contract_broken`  
Decision SHA：`dcf6ce865d0c1788e44d0a5a37588c7df3e4b90d31f1c31f67090155a9436c69`

WB116 FAIL：`mixed_or_inconclusive`  
Official run：`sbb14t_profile_transport_20260907T212851Z_f00dad92`  
Decision SHA：`3793b0e1a6355717fba567b58fddd406d23503ba312e3d5422a90b82fba6d720`  
Config SHA：`71add375d1c9f426ac557884516cffd17a09cf415953ea6f9a309a54526842eb`

WB116 已让 0/1/37/86 的 surviving measurements 全部可求值。唯一 blocker 是 `100048/86` 的 Jacobian spot-check（主要 `loc1 / phi / q/p`）。`b14m_reopen_authorized = false`。

冻结 likelihood 不得改写：

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(θ) = Σ r_i(θ)^T R_i^{-1} r_i(θ)
chi2_prof(α) = min_ν chi2(α, ν)
R_i = (0.08 mm)² / 12
```

WB103 冻结 denominator 保持：raw 2680 / ineligible 691 / contracted 1989 / official pairs 1974。

## 1. 本任务唯一问题

证明 measurement prediction map `h_i(θ)` 在 86 附近具有可认证的局部连续性和 Jacobian。

这不是调统计模型，不是让 χ² 变小，不是修 5D Cin，也不是为 37/86 找 physics result。

## 2. 固定阶梯，禁止挑最好步长

预注册中心步长保持 WB115/WB116：

```
h(loc0) = 0.01 mm
h(loc1) = 0.01 mm
h(phi)  = 1e-5
h(theta)= 1e-5
h(q/p)  = 1e-6 /GeV
```

每个参数固定记录 `h, h/2, h/4, h/8`。合同用整条阶梯判断收敛：最后一对（h/4 vs h/8）相对误差 ≤ 0.05，符号一致，surface/path 分支一致。**不**把误差最小的那一档写成正式 Jacobian，也**不**把官方步长改成 h/8。

每个扰动记录：predicted loc0/loc1、χ²、propagation steps、path length、geometry/surface sequence、inside-bounds、continuation-state 构造路径、free-to-bound fallback、final direction。

## 3. Sequential vs direct（预注册对照）

| 对照 | 含义 |
| --- | --- |
| direct PASS 且 sequential FAIL | 根因优先：支撑平面后 continuation state / sequential transport |
| 两者都 FAIL | 继续查 source→measurement map |
| direct 数值更好 | **不得**替换正式 likelihood |

正式对象仍是一条确定性顺序轨迹上的 `h_i(θ)`。

## 4. 实现

Config：`configs/supporting_plane_jacobian_continuity_v1.yaml`  
SHA：`d141d7e1ab0fd0c556362cb68a35e2904a04660022766814c05b368b94f21a72`

Helper 新开关 `EnableJacobianContinuity` 默认关闭。WB109 / WB114 / WB115 / WB116 路径保持可复现。B14J smoke 只写正式 Mode B + direct 对照，跳过 Mode A、20000-step overlay 和 Kalman replay。

Helper SHA：`a0266aca30bcbe4101547ebe9e2a80d0d642ca5ef4bb26a2166f478e2781c3f7`

测试：`tests/test_supporting_plane_jacobian_continuity.py` 7 passed（login，无 Athena）。

## 5. Smoke（login only：0 / 1 / 37 / 86）

新 root：`outputs/leave_target_out_dump_v1/b14j_smoke`  
未覆盖 WB109 official dump（仍 1951622 bytes）  
未覆盖 WB114 smoke（仍 256783 bytes）  
未覆盖 WB115 smoke（仍 1167777 bytes，SHA `ac34173927eb3047ec79e9abf481f8f67497d7583101c3837495554f955e99b9`）  
未覆盖 WB116 smoke（仍 3209591 bytes，SHA `ca71c527a5c42f10b4006b46577a7252e19691794eea3d7efb3ca2ccbd30d2e2`）

| dump | bytes | SHA |
| --- | --- | --- |
| `b14j_smoke/mc24_100043_00400_00499/ckf_leave_target_out_jacobian_continuity.jsonl` | 17195553 | `2ed001092b1a840a345309b8d55936c6ae4a8c7048e38be781e082bfc9c5c513` |
| `b14j_smoke/mc24_100048_00000_00049/ckf_leave_target_out_jacobian_continuity.jsonl` | 5803192 | `74e0f08ef6118ddb798d6fb004b4b299f4a296038688f5e5e1f61c4a324742f8` |

18 + 6 行：每个 (event, target) 有正式 Mode B（含 sequential+direct 四档阶梯）和一条 direct 对照行。

## 6. Events 0 / 1 / 37

正式 sequential 与 direct：三个 target 的五列全部 `ladder_converged`。  
最后一对相对误差在 1e-6–0.045，符号一致，surface/path 分支一致，无 free-to-bound fallback。

0/1 的 χ² 与 WB116 正式 Mode B 一致（例如 0/t1 = 108.514727）。37 仍可求值，χ² 仍约 2.9e5–5.7e5。Jacobian 连续性在 37 上成立，说明 Class M 的支撑平面跨越本身不是本任务的 blocker。

## 7. Class S：`100048/86`

三个 target 全部可求值。χ² 仍约 5.3e5–6.0e5。无 free-to-bound fallback。destination geometry / continuation construction / geometry-surface sequence 在 ±h 与相邻档之间保持一致。n_steps 只有自适应步进的小数变化。

固定阶梯（sequential，相对误差 h→h/2、h/2→h/4、h/4→h/8）：

| target | loc0 | loc1 | phi | theta | q/p |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.030 / 0.253 / 0.191 | 0.586 / 0.747 / **6.04**（符号翻） | 4.62 / 0.987 / **1.36**（符号翻） | 0.079 / 0.026 / 0.311 | 0.430 / 0.123 / **1.32**（符号翻） |
| 2 | 0.103 / 0.275 / 0.852 | 0.029 / 0.645 / 0.054 | 3.41 / 0.556 / 1.65 | 0.082 / 0.004 / 0.272 | 0.435 / 0.542 / **2.20**（符号翻） |
| 3 | 0.020 / 0.044 / 0.114 | 0.057 / 0.661 / 0.084 | 0.366 / 0.156 / 0.578 | 0.035 / 0.014 / 0.106 | 0.040 / 0.200 / **2.96**（符号翻） |

Direct-from-source 同样 FAIL，且 target 1 的 `loc1 / phi / q/p` 与 sequential 几乎同号、同量级。`n_direct_pass_sequential_fail = 0`，`n_both_fail = 3`。

因此：

- 不是 `finite_difference_step_not_in_asymptotic_region`：更小步长让误差变差，不是单调进入渐近区。
- 不是 `acts_navigation_material_branch_switching`：几何/表面序列与 continuation 构造路径稳定。
- 不是 `supporting_plane_continuation_state_discontinuous`：direct 没有单独 PASS。
- 是 `source_to_measurement_transport_not_smooth`：source→measurement map 在 86 的官方线性化点附近对 `loc1 / phi / q/p` 不可认证。

未因为 direct 数值不同而替换正式 likelihood。未挑 h/8。

## 8. Smoke gate

| 检查 | 结果 |
| --- | --- |
| 0/1/37/86 全部 target nominal 可求值 | PASS |
| 四档阶梯已记录 | PASS |
| 未挑最好步长 | PASS |
| sequential vs direct 已记录 | PASS |
| 0/1/37 Jacobian 收敛 + 符号 + 分支 | PASS |
| 含 86 的 Jacobian contract | **FAIL** |
| target leakage | 0 |
| prior / ridge / measurement update | 无 |
| 统计模型 | 未改 |

`smoke_gate_passed = false`  
`jacobian_contract_established = false`  
`b14m_reopen_authorized = false`  
`restart_invariance_authorized = false`

未重做 `nominal / loc1+1mm / phi+1e-3 / q/p×1.1`。合同要求先建立 Jacobian，才允许重开 B14M smoke。

## 9. 明确不做的事

- 不加 prior / ridge / 人为 χ² penalty
- 不按结果挑最好 FD 步长
- 不把正式 likelihood 换成 direct-from-source
- 不用 truth q/p，不删 37/86，不固定 q/p
- 不在 likelihood 内做 measurement update
- 不以修 5D Cin 为目标
- 不恢复 WB109 / WB107 / full-track CKF Cin
- 不提交 1989 行 Condor
- 不进 B14M smoke、B15、V4 C/D、Measurement Model V2、alignment、ML

## 10. 官方产物

Official run：`sbb14j_jacobian_continuity_20260908T150238Z_a30b79e2`  
Decision SHA：`6a82482ac06e944c8af638099d693738e4c8d7712c1571cabde77e07039b8bf7`  
Config SHA：`d141d7e1ab0fd0c556362cb68a35e2904a04660022766814c05b368b94f21a72`  
Helper SHA：`a0266aca30bcbe4101547ebe9e2a80d0d642ca5ef4bb26a2166f478e2781c3f7`  
Git HEAD：`55cf982302a3c62c57b74f368d5e3ba7723fd33a`

| 产物 | SHA |
| --- | --- |
| `jacobian_continuity_ladder.json` | `6c39c6d1542ce46c868c34b53ae526fb83963ec99b0a77d65f629c0d31ff85c4` |
| `sequential_vs_direct_jacobian.json` | `7fa2d804d31c2234fc843db205fd8dce03e92560b0f5ddc7f9b73983119d6fb9` |
| `supporting_plane_jacobian_continuity_decision.json` | `6a82482ac06e944c8af638099d693738e4c8d7712c1571cabde77e07039b8bf7` |
| `inherited_stage.json` | `e1b5d13a7a87d7e861d3721cce0a537e06b5f6b58b5fd9b40377b7861eb4ff16` |
| `COMPLETE.json` | `a4392b6afb3048eaa6b2e07b4c5280704da52f48cf11b3978e31742ae742e6f6` |

## 11. 下一步

```
WB116 / B14T:
  正式 Mode B = 支撑平面 + 平面图 loc0
  0/1/37/86 可求值
  86 Jacobian FAIL
  b14m_reopen_authorized = false
        ↓
WB117 / B14J:
  固定阶梯 h, h/2, h/4, h/8
  0/1/37：收敛 + 符号 + 分支 PASS
  86：sequential 与 direct 都 FAIL
  不是 FD 步长、不是 navigation branch、不是 continuation 单独问题
  decision = source_to_measurement_transport_not_smooth
  jacobian_contract_established = false
  b14m_reopen_authorized = false
        ↓
下一任务必须继续查 86 附近的 source→measurement map
  （不重调 GN，不加 prior，不删 86，不换 direct）
        ↓
jacobian_contract_established = true
        ↓
才重新打开 B14M smoke，先做 restart invariance
        ↓
仍然不能直接提交 1989 全样本
```

当前对象仍然是：证明 `h_i(θ)` 在 86 附近是定义良好且局部可微的函数。统计模型冻结。不重开 Gauss–Newton。不以修 5D Cin 为目标。
