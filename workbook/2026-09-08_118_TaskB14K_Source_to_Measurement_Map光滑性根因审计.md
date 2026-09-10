# Workbook 118: Task B14K Source-to-Measurement Map 光滑性根因审计

日期：2026-09-08
状态：**完成 / FAIL** —— 以 GitHub `master` `55cf982302a3c62c57b74f368d5e3ba7723fd33a` 为 checkpoint，以 **WB117** 为最新冻结。在**不改变** WB114 measurement likelihood、不重调 Gauss–Newton、不按结果挑最好 FD 步长或 `stepTolerance`、不把正式 sequential likelihood 换成 direct-from-source、不加 prior / ridge、不删 `100043/37` 或 `100048/86`、不用 truth q/p、不以修 5D Cin 为目标的前提下，只对预注册 focus `100048/86` 与同 source 对照 `100048/44` 做 source→measurement 映射逐级根因审计。未重开 B14M，未做 restart invariance，未进 B15，未进 V4 C/D，未进 Measurement Model V2。未覆盖 WB109 / WB114 / WB115 / WB116 / WB117 dumps。**未提交 1989 行 Condor**。

**最终判定：`FAIL`**

- `decision = mixed_or_inconclusive`
- `primary_case = mixed_or_inconclusive`
- `smoke_gate_passed = true`
- `jacobian_contract_established = false`
- `b14m_reopen_authorized = false`
- `restart_invariance_authorized = false`
- `full_sample_authorized = false`
- `transport_deterministic = true`
- `projection_map_smooth = true`
- `integration_improves_but_not_converged = true`
- `acts_transport_jacobian_available = true`
- `acts_inconsistent_with_fd = false`
- `analytic_vs_fd_not_yet_contracted = true`
- `wb117_not_treated_as_physical_nonsmoothness = true`
- `target_exclusion_holds = true`（6 行，0 leaked）
- `prior_introduced = false`
- `ridge_added = false`
- `statistical_model_unchanged = true`
- `profile_math_rewritten = false`
- `do_not_select_best_step = true`
- `do_not_select_best_tolerance = true`
- `do_not_switch_to_direct = true`
- `five_d_cin_not_the_objective = true`
- `b15_authorized = false`
- `measurement_model_v2_authorized = false`
- `next_step = open_wb119_analytic_vs_fd_derivative_contract`

这仍然不是 B14M PASS。WB117 的 `source_to_measurement_transport_not_smooth` **不是**物理不可微结论。下一步是 WB119：在 0/1/37/86 上比较 ACTS / 解析 Jacobian 与固定有限差分阶梯。只有 derivative contract PASS，才允许重新打开 B14M smoke。

更早的误分类 run `sbb14k_map_smoothness_20260908T163048Z_6e7351eb`（`acts_transport_jacobian_inconsistent_with_fd`）已作废：当时把“ACTS Jacobian 存在”当成 inconsistency，没有做契约化比较。

## 起始状态

HEAD：`55cf982302a3c62c57b74f368d5e3ba7723fd33a`  
`Task B0–B14T: dump ACTS/LTO transport and keep profile likelihood fail-closed.`

未回到 `32c904...` 工作流。

WB109 PASS：`leave_target_out_state_materialization_established`  
Decision SHA：`af7ccd055c2ff3d34f92429e37be19a89ed90a9b0db11ec6770c161c79b6e933`

WB114 FAIL：`profile_likelihood_numerically_unstable`  
Decision SHA：`82304f5af943f97adfe069e02c8bacdd7491cca72e6ecc58ea82cc470428a18f`

WB115 FAIL：`profile_transport_contract_broken`  
Decision SHA：`dcf6ce865d0c1788e44d0a5a37588c7df3e4b90d31f1c31f67090155a9436c69`

WB116 FAIL：`mixed_or_inconclusive`  
Decision SHA：`3793b0e1a6355717fba567b58fddd406d23503ba312e3d5422a90b82fba6d720`

WB117 FAIL：`source_to_measurement_transport_not_smooth`  
Official run：`sbb14j_jacobian_continuity_20260908T150238Z_a30b79e2`  
Decision SHA：`6a82482ac06e944c8af638099d693738e4c8d7712c1571cabde77e07039b8bf7`  
Config SHA：`d141d7e1ab0fd0c556362cb68a35e2904a04660022766814c05b368b94f21a72`

WB117 已排除：FD 步长单纯过大、navigation / geometry 分支切换、sequential continuation 独有不连续、direct-from-source 能解决。0/1/37 PASS；86 的 sequential 与 direct 都 FAIL。冻结：

```
decision = source_to_measurement_transport_not_smooth
jacobian_contract_established = false
b14m_reopen_authorized = false
restart_invariance_authorized = false
full_sample_authorized = false
```

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

为什么 `100048/86` 附近的 `h_i(θ)` 对 `loc1 / phi / q/p` 无法形成稳定 Jacobian？

不要再证明它 FAIL。要定位**是哪一个连续数学操作**出现不连续或低于数值分辨率。

## 2. 预注册对照与固定阶梯

对照 `100048/44` 在看任何 Jacobian 之前，只用 WB109 native state（target 1）定义：

```
same source = mc24_100048_00000_00049
score = |Δq/p|/|q/p_86| + |Δθ|/0.01 + |Δloc0|/10 mm + |Δloc1|/10 mm
selected = 100048/44
score ≈ 7.498
```

未用 truth，未用 Jacobian。86 在该 source 内运动学孤立：最近事件仍有相对 q/p ~ 1、Δθ ~ 0.043。

固定中心步长保持 WB115–WB117：

```
h(loc0) = 0.01 mm
h(loc1) = 0.01 mm
h(phi)  = 1e-5
h(theta)= 1e-5
h(q/p)  = 1e-6 /GeV
```

每列固定记录 `h, h/2, h/4, h/8`。收敛合同：最后一对相对误差 ≤ 0.05 且符号一致。**不**挑最好步长。

积分精度只作 falsification，预注册三档，字段是 ACTS 32.0.2 官方 `PropagatorPlainOptions::stepTolerance`：

```
nominal        1e-4
tighter        1e-5
tighter_again  1e-6
```

**不**按 Jacobian 好看程度挑选生产 tolerance。正式生产仍是 `1e-4`。

## 3. 实现

Config：`configs/source_to_measurement_map_smoothness_v1.yaml`  
SHA：`b0277401ae2a36bdeb453a72902dc7fb7908cc0d8b5c1278bf8db1253b97560e`

Helper 新开关 `EnableMapSmoothness` 默认关闭。`stepTolerance` 沿官方 API 传入，默认 `1e-4`，WB109–WB117 路径保持可复现。B14K smoke 只写正式 Mode B，跳过 Mode A、direct、Kalman replay 和 diagnostic 20000。

Helper SHA：`078a45ac774971032fd1e4997cb04d2285b66f9fe6b80000f47f580ea0612b33`

测试：`tests/test_source_to_measurement_map_smoothness.py` 与 WB117 测试合计 12 passed（login，无 Athena）。分类器禁止把“ACTS Jacobian 存在”写成 inconsistency；`source_to_measurement_map_genuinely_nonsmooth` 在积分仍改善时禁止使用。

## 4. Smoke（login only：86 + 44）

新 root：`outputs/leave_target_out_dump_v1/b14k_smoke`  
未覆盖 WB109 official dump（仍 1951622 bytes）  
未覆盖 WB114 smoke（仍 256783 bytes）  
未覆盖 WB115 smoke（仍 1167777 bytes）  
未覆盖 WB116 smoke（仍 3209591 bytes）  
未覆盖 WB117 smoke（仍 17195553 / 5803192 bytes，SHA 不变）

| dump | bytes | SHA |
| --- | --- | --- |
| `b14k_smoke/mc24_100048_00000_00049/ckf_leave_target_out_map_smoothness.jsonl` | 4371700 | `a5612a96520baaf2675187292e65a5ef72ac740211b2d561a317521f35a7a810` |

6 行正式 Mode B：`100048/44` 与 `100048/86`，target 1/2/3。Athena 报告 `FPE OVERFLOWs: 6`，6 行仍完整写出。

## 5. 逐项排除

### 5.1 Repeatability — PASS

同一 `θ` 重复求值 ≥ 3 次，不是扰动。86 与 44 的三个 target：

```
max predicted-loc0 L2 delta = 0
max |Δχ²| = 0
steps_identical = true
transport_nondeterministic = false
```

同一输入本身稳定复现。可以谈 Jacobian，不是 nondeterministic transport。

### 5.2 Projection-only — PASS，不是病态相交

冻结最终 free position/direction，不再 propagate，只对

```
plane intersection → local transform → loc0
```

做有限差分。86 与 44 全部 hop：`projection_map_smooth = true`，`near_parallel = false`。

| 事件 | min \|n·d\| | max incidence (rad) |
| --- | --- | --- |
| 44 | 0.99999494 | 0.00318 |
| 86 | 0.99985966 | 0.01675 |

因此：

```
projection_map_smooth = true
transport_map_problem = true
```

不是 `supporting_plane_projection_ill_conditioned`。

### 5.3 FD signal vs transport noise — 不是低于噪声

重复噪声为 0。官方 `h` 的 `||h(θ+δ)-h(θ-δ)||`（loc0 L2）在 86 上为：

| target | loc1 | phi | q/p |
| --- | --- | --- | --- |
| 1 | 0.0132 | 0.00458 | 0.00787 |
| 2 | 0.0169 | 0.00809 | 0.00564 |
| 3 | 0.00302 | 0.00298 | 0.00344 |

全部 `n_above_noise = 12`，`official_h_signal_at_or_below_noise = false`。禁止再靠无限减小 FD 步长找答案。不是 `jacobian_below_transport_resolution`。

### 5.4 无量纲尺度 — 不是明显非线性

86：`q/p_nominal = -0.0118408 /GeV`，`h/|q/p| = 8.445e-5`。  
`h(loc1)/1 mm = 0.01`，`h(phi)/1 rad = 1e-5`。  
regime = `unclassified`。未据此挑选生产步长。

44：`q/p = +0.001160`，`h/|q/p| = 8.62e-4`。86 的 |q/p| 大约是 44 的 10 倍、符号相反；θ 为 0.01507 vs 0.00318。

### 5.5 Stagewise — 第一个不再收敛的操作

86 hop 0 的 `loc1 / phi`：**全部 stage 收敛**（相对误差 1e-9–1e-12）。source bound 与 bound-to-free 不是问题。

第一个失败（按 hop 顺序，不再把后续 hop 的 continuation `source_bound` 误报成源参数失败）：

| 86 target | loc1 | phi | q/p |
| --- | --- | --- | --- |
| 1 | hop 6 `free_near_plane` | hop 6 `free_near_plane` | hop 0 `plane_intersection`（last-pair 0.061） |
| 2 | hop 11 `predicted_loc0` | hop 11 `free_near_plane` | hop 0 `plane_intersection` |
| 3 | hop 11 `predicted_loc0` | hop 11 `free_near_plane` | hop 0 `plane_intersection` |

对应几何（正式 sequential，target 1）：

```
hop 0–5  IFT，z ≈ -1886 → -1822 mm，短 hop
hop 6    IFT → station 2，z -1822 → 1203 mm，path 3026 mm，129 steps
```

target 2/3 的 hop 6 是 IFT → station 1（path 1836 mm，84 steps），`loc1` 仍过关；phi/`predicted_loc0` 在 hop 11 的下一长 hop 失稳（t2：2316 mm / 77 steps；t3：1126 mm / 39 steps）。

hop 6 上 `loc1` 的 `free_near_plane` last-pair ≈ 0.080，而 `predicted_loc0` ≈ 2.07：立体投影把已经开始失稳的 transport FD 放大成残差列。

对照 44：`loc1 / phi` 全部 hop 收敛。`q/p` hop 0 `plane_intersection` 同样略超 0.05，但**正式残差 Jacobian 仍 PASS**。该 hop-0 `q/p` 现象是第一跳 loc0 对 `q/p` 的响应对 0.05 切过大严，不是 44 坏掉，也不能单独写成 86 的根因。

44 同样有千米级磁 hop（t1 hop 4：3065 mm / 95 steps）且 Jacobian 收敛。86 的独有条件不是“存在长 hop”，而是更大的 `|q/p|`、额外 IFT hop、以及这些长 hop 上 FD 不再收敛。

### 5.6 Propagator accuracy — 改善但未完成

正式列 last-pair 相对误差（`loc1 / phi / q/p` 平均）：

| 事件 / target | 1e-4 | 1e-5 | 1e-6 | 官方阶梯 |
| --- | --- | --- | --- | --- |
| 44 / 1 | 3.0e-4 | 1.9e-4 | 1.2e-2 | PASS |
| 44 / 2 | 2.2e-4 | 2.0e-4 | 1.1e-2 | PASS |
| 44 / 3 | 1.6e-4 | 1.6e-4 | 8.1e-3 | PASS |
| 86 / 1 | 2.91 | 0.94 | 0.25 | FAIL；`loc1` 仅在 1e-6 PASS |
| 86 / 2 | 1.30 | 0.87 | 0.98 | FAIL，非单调 |
| 86 / 3 | 1.21 | 0.61 | 0.21 | FAIL；`loc1` 仅在 1e-6 PASS |

86 官方 `1e-4` last-pair 与 WB117 同量级（t1：`loc1` 6.04，`phi` 1.36，`q/p` 1.32）。收紧积分**有帮助，但预注册最紧一档仍不能让 `phi / q/p` 过关**。因此：

- 不是 `transport_integration_resolution_insufficient`（该案要求最后一档整梯收敛）
- 也**不得**把 `1e-6` 写成生产设置
- 因为积分仍改善，**不得**写成 `source_to_measurement_map_genuinely_nonsmooth`

### 5.7 ACTS derivative inventory — 可用，但未契约化

ACTS 32.0.2 `Propagator::Result::transportJacobian` 在附加 dummy covariance 时存在。正式路径仍不运输协方差。complex-step / autodiff 需要改 ACTS，本任务不做。平面相交本身有解析导数；冻结投影已验证光滑。

ACTS 终点 `loc0` 与官方支撑平面 `loc0` 一致：44 为 12/12；86 为 16/16、15/15、15/15。`acts_inconsistent_with_fd = false`，因为还没有把 ACTS bound-to-bound Jacobian 链式对照正式残差列。那是 WB119，不是替换正式 likelihood。

### 5.8 Focus vs control

| | 86 | 44 |
| --- | --- | --- |
| q/p | −0.011841 /GeV | +0.001160 /GeV |
| θ | 0.01507 | 0.00318 |
| χ² (t1/t2/t3) | 6.01e5 / 5.33e5 / 5.74e5 | 420 / 381 / 553 |
| 确定性 | PASS | PASS |
| 投影 | PASS | PASS |
| 官方 Jacobian | FAIL | PASS |
| 选择依据 | — | WB109 运动学，无 Jacobian / truth |

86 独有的是更大的 `|q/p|`、更弯的 θ、更高的 χ²，以及长磁 hop 上 FD 失稳。对照证明同一 source、同一 helper、同一官方 `stepTolerance` 下 Jacobian **可以**收敛。

## 6. 为什么是 `mixed_or_inconclusive`

| 候选 | 为何不用 |
| --- | --- |
| `transport_integration_resolution_insufficient` | 趋势改善，但最紧预注册档未整梯收敛；禁止挑最好 tolerance |
| `supporting_plane_projection_ill_conditioned` | 投影-only 光滑，`n·d ≈ ±1` |
| `parameterization_scale_not_resolved` | 官方 h 相对 \|q/p\| 为 8.4e-5，FD 信号高于零噪声 |
| `acts_transport_jacobian_inconsistent_with_fd` | 尚未做公平列对列契约；终点 loc0 已一致 |
| `source_to_measurement_map_genuinely_nonsmooth` | 确定性通过，但积分仍改善，排除未完成 |

剩余事实：86 的正式 source→measurement FD 仍不收敛；第一个失稳操作是后续长磁传播，不是源参数图或冻结平面投影；ACTS 已能给出 transport Jacobian。诚实分类是 `mixed_or_inconclusive`。下一步是建立 derivative contract，而不是宣布物理不可微。

## 7. Smoke gate

| 检查 | 结果 |
| --- | --- |
| 86 与 44 三个 target 可求值 | PASS |
| 八份诊断产物已记录 | PASS |
| 未挑最好步长 / tolerance | PASS |
| target leakage | 0 |
| prior / ridge / measurement update | 无 |
| 统计模型 | 未改 |
| Jacobian contract | **仍未建立** |

`smoke_gate_passed = true` 只表示诊断材料齐全，**不**表示 Jacobian 通过。  
`jacobian_contract_established = false`  
`b14m_reopen_authorized = false`  
`restart_invariance_authorized = false`

## 8. 明确不做的事

- 不加 prior / ridge / 人为 χ² penalty
- 不按结果挑最好 FD 步长或 `stepTolerance`
- 不把正式 likelihood 换成 direct-from-source
- 不用 truth q/p，不删 37/86，不固定 q/p
- 不在 likelihood 内做 measurement update
- 不以修 5D Cin 为目标
- 不恢复 WB109 / WB107 / full-track CKF Cin
- 不提交 1989 行 Condor
- 不进 B14M smoke、B15、V4 C/D、Measurement Model V2、alignment、ML
- 不把 ACTS Jacobian 直接替换正式 likelihood

## 9. 官方产物

Official run：`sbb14k_map_smoothness_20260908T163845Z_2cbd5d0e`  
Decision SHA：`e22b5d9ce93106b779e968202b84e036c86dc9033639fe4bf833d8245752895d`  
Config SHA：`b0277401ae2a36bdeb453a72902dc7fb7908cc0d8b5c1278bf8db1253b97560e`  
Helper SHA：`078a45ac774971032fd1e4997cb04d2285b66f9fe6b80000f47f580ea0612b33`  
Git HEAD：`55cf982302a3c62c57b74f368d5e3ba7723fd33a`

| 产物 | SHA |
| --- | --- |
| `source_to_measurement_stagewise_jacobian.json` | `4a72dae7f399d881fc1817beb15c908db48b2074fec46b707fa0da67cc733db0` |
| `transport_repeatability.json` | `d7237a78795d324585818621a10b4bfb993b459ff8f2c4de6a1fde647e7eea27` |
| `transport_resolution_vs_fd_signal.json` | `a3909d5c86f17127999c8a4a41ae57bbb53936db0f59d51f2a3698865675702b` |
| `propagator_accuracy_convergence.json` | `07bf430c2a0744bd7c2f62cc3d8a6cc4656ee6e835ba923e48239403995ec2f4` |
| `supporting_plane_projection_smoothness.json` | `b782a23ee9acf0e8a6d6b377f7ca436c484d32b70ca5f40a1adf3a21484636bf` |
| `fd_scale_dimensionless_audit.json` | `30a84ef8c13932607456e69f8b57fb808bd2ab7ee4ca26b0e9ad7437bad08137` |
| `acts_derivative_capability_inventory.json` | `337c870b097619cc8dc5c91726a70a1d9fd969578adb66f7e98d24ae96b6c801` |
| `focus_vs_control_transport_smoothness.json` | `91cf1fef2cb806b6341baaf986cdd43548358da0db7c4fb96099a8557a6a9914` |
| `source_to_measurement_map_smoothness_decision.json` | `e22b5d9ce93106b779e968202b84e036c86dc9033639fe4bf833d8245752895d` |
| `inherited_stage.json` | `e0dd9df784525635871a452f4e9de0675d79cd64823e0ccd2ac0360a9c3d095e` |
| `COMPLETE.json` | `bcbcd7027f1029a7cddc261aa3c68890504c38f1fe48c2896773e94e92808f9c` |

## 10. 下一步

```
WB117 / B14J:
  86 source→measurement Jacobian 不收敛
  decision = source_to_measurement_transport_not_smooth
  不是物理最终结论
        ↓
WB118 / B14K:
  确定性 PASS
  投影-only PASS
  FD 信号高于零噪声
  对照 44 PASS
  86 官方 Jacobian 仍 FAIL
  第一失稳操作 = 后续长磁传播（t1 hop 6；t2/t3 hop 11）
  收紧积分有帮助但未完成
  ACTS transportJacobian 可用，尚未 vs FD 契约化
  decision = mixed_or_inconclusive
  jacobian_contract_established = false
  b14m_reopen_authorized = false
        ↓
WB119 / analytic-vs-FD derivative contract
  只比较 ACTS / 解析 Jacobian 与固定多尺度 FD
  范围：0 / 1 / 37 / 86
  不替换正式 likelihood
        ↓
derivative contract PASS
        ↓
jacobian_contract_established = true
        ↓
才重新打开 B14M smoke
        ↓
再做 restart invariance
        ↓
仍然不能直接提交 1989 全样本
```

当前最重要的判断：WB117 看到的“不光滑”**首先是长磁传播上有限差分相对积分噪声/分辨率不再稳定**，不是源参数图、不是冻结平面投影、也还不是已经证明的物理不可微。下一步用 ACTS 已有的 transport Jacobian 做契约，而不是再减小 FD 步长，也不是重开 B14M。
