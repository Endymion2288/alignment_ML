# Workbook 116: Task B14T Standalone Measurement Transport 合同修复

日期：2026-09-07
状态：**完成 / FAIL** —— 在**不改变** WB114 measurement likelihood、不重调 Gauss–Newton、不加 prior / ridge、不删 `100043/37` 或 `100048/86`、不用 truth q/p 的前提下，把 Class M（长磁铁跨越）和 Class S（近邻 stereo 面）分开，对照钉死的 ACTS 32.0.2 Kalman 导航语义，把正式 evaluator 改成支撑平面 / 平面图 loc0。未进 B14M 全样本，未进 B15，未进 V4 C/D，未进 Measurement Model V2。未覆盖 WB109 / WB114 / WB115 dumps。**未提交 1989 行 Condor**。

**最终判定：`FAIL`**

- `decision = mixed_or_inconclusive`
- `primary_case = mixed_or_inconclusive`
- `smoke_gate_passed = false`
- `b14m_reopen_authorized = false`
- `full_sample_authorized = false`
- `jacobian_contract_established = false`（0/1/37 PASS，86 FAIL）
- `target_exclusion_holds = true`
- `prior_introduced = false`
- `ridge_added = false`
- `statistical_model_unchanged = true`
- `profile_math_rewritten = false`
- `measurement_update_in_evaluator = false`
- `b15_authorized = false`
- `measurement_model_v2_authorized = false`
- `do_not_force_5d_lto_covariance = true`
- `focus_identity_retained = true`
- `physical_nonidentifiability_not_claimed = true`

这仍然不是 B14M PASS，也不是物理不可识别。下一步不是重开 B14M smoke，而是先处理 **86 已可求值之后的 Jacobian spot-check**。

## 起始状态

HEAD：`32c9044a8543845aaa0767d8089ba6fd731f31a9`

工作树从 WB98 起一直未提交。本任务只做增量，未 `git reset --hard`、未 `git clean`、未 checkout 覆盖。WB98–WB115 修改全部保留。

WB109 PASS：`leave_target_out_state_materialization_established`  
Decision SHA：`af7ccd055c2ff3d34f92429e37be19a89ed90a9b0db11ec6770c161c79b6e933`

WB114 FAIL：`profile_likelihood_numerically_unstable`  
Decision SHA：`82304f5af943f97adfe069e02c8bacdd7491cca72e6ecc58ea82cc470428a18f`

WB115 FAIL：`profile_transport_contract_broken`  
Official run：`sbb14n_profile_numerics_20260907T150627Z_d77ddddc`  
Decision SHA：`dcf6ce865d0c1788e44d0a5a37588c7df3e4b90d31f1c31f67090155a9436c69`  
Config SHA：`b338874546276e6fd2c4793c4ad02ccd9c8d95bed9e693b0d9659d6a04f5b57e`

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

让 standalone profile evaluator 对 WB109 surviving measurements 使用与真实 ACTS/FASER tracking 几何和导航语义一致的 transport contract，使 **nominal** likelihood 能对所有 surviving measurement surfaces 完整求值。

这不是调统计模型，不是为了让 χ² 变小，不是修 37/86 的 physics result。

## 2. Class M 与 Class S 必须分开

| 类 | 焦点 | WB115 表现 | 机制 |
| --- | --- | --- | --- |
| Class M `long_magnet_crossing_hop` | `100043/37` | IFT → 第一张下游 hit，`Propagation reached maximum number of steps` | 把本应由 navigator 穿过中间几何/材料面的路径，收成一次 bounded `SurfaceReached(next wafer)` 长 hop |
| Class S `local_stereo_surface_intersection` | `100048/86` | 磁铁 hop 成功；`1234.97 → 1235.86` mm；`Global to local transformation failed: position not on surface` | 已到支撑平面，但落在有限 sensor bounds 外 |

禁止把它们合并成一个 `propagation failed`。

## 3. ACTS Kalman 真实 transport（钉死源码，不是猜 API 名）

ACTS 32.0.2 / Athena 24.0.41：

- `KalmanFitter` **不**按测量列表直接指定 next surface。它等 `navigator.currentSurface`。测量不必预先排序。
- 第一次 actor 调用时，把每个 measurement `GeometryIdentifier` 插入为 **external surface**。源码注释原文：*We will try to hit those surface by ignoring boundary checks.*
- `Navigator` 对 external surface 使用 `BoundaryCheck(false)`；target volume 初始化也是 `BoundaryCheck(false)`。
- `propagate(start, target, options)` 仍然走 tracking-geometry navigator，并追加 `SurfaceReached` + `PathLimitReached`。
- `SurfaceReached` **默认** `BoundaryCheck(true)`。打不中有限 wafer 时 intersection 找不到，stepper 继续走，直到 `StepCountLimitReached`。这就是 Class M 的 Mode A 机制。
- 成功 abort 后 `makeResult` 调用 `stepper.boundState(target)`。自由坐标离平面超过 `s_onSurfaceTolerance = 1e-4` mm 时，`globalToLocal` 失败，报 *Global to local transformation failed: position not on surface.* 这就是 Class S 的 Mode A 机制。
- Kalman **会**在测量面上做 update，然后从 filtered state 继续。独立 likelihood **不能**复制这一步，否则冻结的 `chi2(θ)` 会变成有状态 filter 目标。
- `PlaneSurface` 的 loc0/loc1 是平面图。`insideBounds` 是另一件 active-rectangle 成员测试。residual `r = m_loc0 − predicted_loc0` 需要的是平面图坐标。

正式 Mode B 因此把 `SurfaceReached.boundaryCheck` 设为 `false`，与 Kalman external-surface 瞄准对齐。这不是把 tolerance 调到 86 通过。

## 4. 实现（只修 transport）

Config：`configs/profile_transport_contract_v1.yaml`  
SHA：`71add375d1c9f426ac557884516cffd17a09cf415953ea6f9a309a54526842eb`

Helper 仍是独立 `CkfLeaveTargetOutDumpAlg`。新开关默认关闭，WB109 / WB114 / WB115 路径保持可复现。

正式 evaluator（Mode B）：

- 仍按物理 z 顺序逐 measurement hop
- **不做** Kalman measurement update
- hop 目标是支撑平面（`BoundaryCheck(false)`）
- loc0 来自平面图；`inside_active_bounds` 只记录
- 正式 `maxSteps = 4000`（与 WB115 相同）
- 10× `s_onSurfaceTolerance` 只用于 `free→bound` 续传诊断，不是生产 hunting
- Mode A（bounded `SurfaceReached`）、direct-from-source、20000-step overlay 只作对照行
- 20000 步 **不是**正式修复；正式行仍写 `profile_max_steps = 4000`

Helper SHA：`7d75ab858a6eab7b5c06dc6a15dcc834bd1d1d9c8b1d64aff4dc806a84f4d10f`

测试：`tests/test_profile_transport_contract.py` 6 passed（login，无 Athena）。

## 5. Smoke（login only：0 / 1 / 37 / 86）

新 root：`outputs/leave_target_out_dump_v1/b14t_smoke`  
未覆盖 WB109 official dump（仍 1951622 bytes）  
未覆盖 WB114 smoke（仍 256783 bytes）  
未覆盖 WB115 smoke（仍 1167777 bytes，SHA `ac34173927eb3047ec79e9abf481f8f67497d7583101c3837495554f955e99b9`）

| dump | bytes | SHA |
| --- | --- | --- |
| `b14t_smoke/mc24_100043_00400_00499/ckf_leave_target_out_profile_transport.jsonl` | 3209591 | `ca71c527a5c42f10b4006b46577a7252e19691794eea3d7efb3ca2ccbd30d2e2` |
| `b14t_smoke/mc24_100048_00000_00049/ckf_leave_target_out_profile_transport.jsonl` | 1052114 | `d272c8ba670bb6e9b9d950079e5c0866ce4682c7ed5b3914815ac9f7c881f8a2` |

45 + 15 evaluate-only 行：每个 (event, target) 有正式 Mode B、Mode A、direct-from-source、diagnostic 20000、以及一次 WB109 LTO fitted replay。

## 6. Event 0/1

正式 Mode B：三个 target 全部到达，Jacobian 全部 PASS。

与冻结 WB115 的 predicted loc0 最大差 `< 1e-10` mm，χ² 差在数值噪声内。预注册容差 0.05 mm。`wb115_prediction_consistency = true`。

Sequential vs direct-from-source：6/6 组 loc0 差 `< 0.05` mm。未因为数值更好而切换；正式对象仍是一条确定性轨迹上的 `h_i(θ)`。

## 7. Class M：`100043/37`

单位合同（与 Kalman/LTO helper 同一 `seedFromOfficialMean`，未用 truth q/p）：

```
Athena q/p = -2.416e-6 /MeV
ACTS  q/p = -0.002416 /GeV
|p|       = 413847 MeV
charge    = −1
φ         = -0.743
θ         = 0.002795
particle  = muon
```

这不是低动量绕圈。Mode A 仍在第一段下游 hop 死于 maxSteps：

```
target 1:  z ≈ -1826.22 → 1207.45 mm
target 2/3: z ≈ -1826.22 → 17.45 mm
```

正式 Mode B 用 51–68 步、path ≈ 1844 / 3034 mm 到达支撑平面。最终位置在平面上（`z` 对齐到 1e-8 mm），但 `inside_active_bounds = false`（例如 y ≈ −70 mm）。checkpoint 分类是 **monotonic downstream progress**，不是 magnet 边界停滞，也不是 looping / tiny-step。

诊断 20000 步不是 PASS 条件。原合同需要很多步，是因为 bounded `SurfaceReached` 打不中有限 wafer，stepper 在找不到 intersection 后一直走到上限。改成支撑平面后，正式 4000 步已经足够。

正式 Mode B：三个 target 的全部 surviving hits 都可求值（16/16、16/16、15/15）。  
Jacobian：三个 target 全部 PASS。

χ² 很大（约 2.9e5–5.7e5）。这记录为 reconstruction / 轨迹与 active wafer 不对齐，**不是**本任务的删除或 truth-p 替换理由。37 保留。

## 8. Class S：`100048/86`

Mode A 磁铁 hop 仍然成功。失败在近邻 stereo：

```
surface A: z = 1234.97 mm, side 0, center (62.0, 26.87, 1234.97)
surface B: z = 1235.855 mm, side 1, center (62.0, 26.87, 1235.855)
dz ≈ 0.885 mm
bounds: type 6, loc0 ∈ [-30.72, 30.72], loc1 ∈ [-63.045, 63.045]
```

Mode B 交叉：

```
distance_to_plane = 0
predicted loc0 ≈ 30.739 mm
predicted loc1 ≈ 27.643 mm
inside_active_bounds = false
bounded_intersection_on_surface = false
```

答案：**已经到了支撑平面，但 loc0 超出 active 矩形约 19 μm。**  
因此 Mode A 的 `globalToLocal` 失败。这与“没到平面”是两种机制。

正式 Mode B：三个 target 的全部 surviving 投影都有定义（17/17、16/16、17/17）。依据是 ACTS Kalman external-surface `BoundaryCheck(false)` + 平面图 loc0，不是关掉 boundary check 来硬过 86。

Jacobian：三个 target 全部 FAIL（loc1 / φ / q/p 相对误差可到 O(1)）。B14T.11 要求：一旦 86 可求值，就必须保持 `jacobian_contract_established = true` 才能重开 optimizer。因此 **不能**重开 B14M。

## 9. Smoke gate

| 检查 | 结果 |
| --- | --- |
| events 0/1 全部 target nominal | PASS |
| 与 WB115 预测一致 | PASS |
| 0/1 Jacobian | PASS |
| 37 全部 surviving 可求值 | PASS |
| 86 全部投影有定义 | PASS |
| 含 86 的 Jacobian contract | **FAIL** |
| target leakage | 0 |
| prior / ridge / measurement update | 无 |
| 任意加大 maxSteps 当作正式修复 | 否 |
| 统计模型 | 未改 |

`smoke_gate_passed = false`  
`b14m_reopen_authorized = false`

## 10. 为何是 mixed，不是 A/STATE/PROJECTION/NAVIGATION

- 不是 `standalone_measurement_transport_contract_established`：86 的 Jacobian 未重建，不能重开 optimizer。
- 不是 `reconstructed_state_not_transportable_to_surviving_measurements`：单位/符号/导航/步长合同下，37 已经到达全部 surviving **平面**。
- 不是 `measurement_surface_projection_contract_not_established`：86 的 loc0 已由支撑平面图合法定义。
- 不是 `acts_navigation_transport_contract_not_established`：正式 Mode B 不再把 37 卡在磁铁长 hop。
- 不是 `physical_nonidentifiability`：B14T 禁止判这个。

因此：`mixed_or_inconclusive`。  
Class M 导航合同已修；Class S 投影合同已有源码依据；挡住 B14M 的是 **可求值之后的 86 Jacobian**。

## 11. 明确不做的事

- 不加 prior / ridge / 人为 χ² penalty
- 不把 20000 maxSteps 当作正式修复
- 不按 residual/truth 增大 tolerance
- 不用 truth q/p，不删 37/86，不固定 q/p
- 不在 likelihood 内做 measurement update
- 不恢复 WB109 / WB107 / full-track CKF Cin
- 不提交 1989 行 Condor
- 不进 B14M smoke、B15、V4 C/D、Measurement Model V2、alignment、ML

## 12. 官方产物

Official run：`sbb14t_profile_transport_20260907T212851Z_f00dad92`  
Decision SHA：`3793b0e1a6355717fba567b58fddd406d23503ba312e3d5422a90b82fba6d720`  
Config SHA：`71add375d1c9f426ac557884516cffd17a09cf415953ea6f9a309a54526842eb`  
Helper SHA：`7d75ab858a6eab7b5c06dc6a15dcc834bd1d1d9c8b1d64aff4dc806a84f4d10f`

| 产物 | SHA |
| --- | --- |
| `transport_failure_taxonomy.json` | `536faae3e5b5516f41388711cc242e8d16e437f8f26eaf3a7bfe485b720d2c9f` |
| `acts_kalman_transport_semantics.json` | `dc4f8b04f3ea6e404c67f54c8d2f7235b86338a4659009c7df3624c4180c05b4` |
| `magnet_crossing_transport_audit.json` | `2b848d68afbbdedc7849b97a71a5707709a101c40944514b00b95558b4582b3f` |
| `transport_state_unit_contract.json` | `fbd4cffa64550a7caa0fc47d2bcabc5b602392e6867bdb9e05a8a8b6175e1379` |
| `magnet_stepper_pathology.json` | `eabc4f94f3c7c3e930e798264e22c318ef86d2e708263fb03fc2e56245ac42be` |
| `stereo_surface_geometry_audit.json` | `7d30df24016548e75a91d9b137dec609812095322e7ecd0c3fed933e31988740` |
| `measurement_surface_projection_contract.json` | `a2c7bed67fc75b8c2349ca203b904ced7dd4793897c4e1c404f23185e91af961` |
| `standalone_likelihood_transport_contract.json` | `0c04b3da68a1bd0aaf92c894b68791356ee1fa1a14eda392ee96bf2aa4f48db1` |
| `sequential_vs_direct_prediction.json` | `92648cab5e277e11f56c544bdab7ac82c13be91de98163bdad3205f7a70bf4b4` |
| `transport_jacobian_spotcheck.json` | `8c72735eaeba431ec2b8c18cf1c4930746ba8243553c3af1e2e2958e128c51e1` |
| `profile_transport_contract_decision.json` | `3793b0e1a6355717fba567b58fddd406d23503ba312e3d5422a90b82fba6d720` |
| `inherited_stage.json` | `4f005f491ab48c8f5cf6cce572e5368c637283bf6ed4251851261171fd89bfef` |
| `COMPLETE.json` | `5a93fda3adc0890b89cd52ce9bbffde35b306bd536d125ca852a60cb40c1a740` |

## 13. 下一步

```
WB115:
  0/1 evaluator + Jacobian + GN 已过
  37 Class M / 86 Class S 不可求值
        ↓
WB116 / B14T:
  正式 Mode B = 支撑平面 + 平面图 loc0，无 measurement update
  37: 全部 surviving 平面可求值；Mode A 仍 maxSteps
  86: 全部投影有定义；机制 = 已在平面、越出 active bounds ~19 μm
  0/1 与 WB115 一致
  86 Jacobian FAIL
  decision = mixed_or_inconclusive
  b14m_reopen_authorized = false
        ↓
下一任务必须先重建 86 的 Jacobian contract
  （不重调 GN，不加 prior，不删 86）
        ↓
jacobian_contract_established = true
        ↓
才重新打开 B14M smoke
        ↓
再判断 profile observable identifiable
或 physical nonidentifiability
```

当前对象仍然是：先让 `h_i(θ)` 在真实几何/场下对所有 surviving measurements 成为定义良好的函数，并且在这些点上的 `∂r/∂θ` 仍然可信。统计模型冻结。不重开 Gauss–Newton。
