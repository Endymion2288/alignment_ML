# Workbook 120: Task B14U Official Supporting-Plane Transport Jacobian Contract

日期：2026-09-08
状态：**完成 / FAIL** —— 以 GitHub `master` `55cf982302a3c62c57b74f368d5e3ba7723fd33a` 为 checkpoint，以 **WB119** 为最新冻结。在**不改变** WB114 measurement likelihood、不重调 Gauss–Newton、不改生产 `stepTolerance=1e-4`、不按结果挑最好 FD 步长或积分精度、不把正式 sequential likelihood 换成 direct-from-source、不用任何 ACTS Jacobian 替换正式 likelihood、不加 prior / ridge、不删 `100043/37` 或 `100048/86`、不用 truth q/p、不以修 5D Cin 为目标的前提下，只对预注册 control `100043/0,1,37` 与 focus `100048/86` 建立正式 Mode-B 支撑平面路径的同路径 Jacobian，并与冻结 `h,h/2,h/4,h/8` FD 列逐列比较。未重开 B14M，未做 restart invariance，未进 B15，未进 V4 C/D，未进 Measurement Model V2。未覆盖 WB109 / WB114 / WB115 / WB116 / WB117 / WB118 / WB119 dumps。**未提交 1989 行 Condor**。禁止把 dummy-cov bounded `transportJacobian` 当作正式导数。

**最终判定：`FAIL`**

- `decision = official_path_jacobian_inconsistent_with_fd`
- `primary_case = official_path_jacobian_inconsistent_with_fd`
- `smoke_gate_passed = true`
- `jacobian_contract_established = false`
- `b14m_reopen_authorized = false`
- `restart_invariance_authorized = false`
- `full_sample_authorized = false`
- `free_state_jacobian_available = true`
- `official_loc0_matches_diagnostic = true`
- `control_fd_converged = true`
- `control_official_path_agrees_fd = false`
- `focus_fd_converged = false`
- `focus_official_path_agrees_fd = false`
- `branch_switching = false`
- `target_exclusion_holds = true`（12 行，0 leaked）
- `prior_introduced = false`
- `ridge_added = false`
- `statistical_model_unchanged = true`
- `do_not_use_dummy_cov_bounded_transportJacobian = true`
- `b15_authorized = false`
- `measurement_model_v2_authorized = false`
- `next_step = keep_official_path_jacobian_diagnosis`

这仍然不是 B14M PASS。WB119 已证明“拿错了 Jacobian”。本任务给正式无边界支撑平面路径建立起同路径 free-state Jacobian，函数值与正式 Mode-B predicted loc0 差为 0，control 0/37 的五列与冻结 FD 一致，包括长 hop 的 `q/p`。契约仍未成立，因为 control `100043/1` 的 `loc1` 在 target 1/2 上超过冻结 5%，且 86 的 FD 阶梯仍不收敛。

不允许把本跑判成 `acts_free_state_jacobian_unavailable`：正式 `nullopt` 路径本身不填 `jacTransport`，但诊断接口在同一条无边界 hop 上读出了 stepper 已维护的 RK variational 段，12 行 `chain_complete = true`。也不允许把 86 单独说成“FD 不可靠因此契约已立”：control 1 的 `loc1` 已经和冻结 FD 不一致。不允许改生产 `stepTolerance`、减小 FD 步长、或把 ACTS 矩阵写进正式 likelihood。

## 起始状态

HEAD：`55cf982302a3c62c57b74f368d5e3ba7723fd33a`

WB119 FAIL：`analytic_chain_or_chart_contract_broken`  
Official run：`sbb14l_derivative_contract_20260908T180541Z_33bda85a`  
Decision SHA：`2388e8a8655eb1dc46e2b0207d7064f5689f9c69035f1f14b3b42bdd3ca7bf19`  
Config SHA：`da58a62fc0378976bce0401977b3388015d62e2964daa716d47c2ff31322fc86`  
Helper SHA：`8edc0d52378263b08ccefa8d87b419992e945507451d9238dc1173aa1828866c`

```
decision = analytic_chain_or_chart_contract_broken
jacobian_contract_established = false
b14m_reopen_authorized = false
control_fd_converged = true
control_acts_chain_agrees_fd = false
acts_end_loc0_matches_official = true
next_step = diagnose_jacobian_chain_chart_units_or_projection
```

WB119 已排除：0-step chart / 单位 / 残差符号 / 投影链错误。破裂点是 dummy-cov **bound-to-surface** `transportJacobian` 不是正式无边界 `h_i` 的导数；control 0/1/37 上已经不一致，所以不能归咎于 86 的特殊 FD。

冻结 likelihood 不得改写：

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(θ) = Σ r_i(θ)^T R_i^{-1} r_i(θ)
R_i = (0.08 mm)² / 12
```

WB103 冻结 denominator 保持：raw 2680 / ineligible 691 / contracted 1989 / official pairs 1974。

## 1. 本任务唯一问题

给正式 Mode-B 路径本身建立 Jacobian：

```
source bound → 无边界 navigator/stepper → final free state
  → supporting-plane intersection → local loc0 → residual
```

不得经过“为了拿 Jacobian 而附加 covariance 并绑定到有限 target surface”的旁路。

范围严格是预注册 `100043/0,1,37` 与 `100048/86`。不是 44。不是 1989。

## 2. ACTS 32.0.2 源码结论

EigenStepper（AthenaExternals 24.0.41 / ACTS 32.0.2）：

- `State` 在起始参数带 covariance 时置 `covTransport = true`，并填 `jacToGlobal = surface.boundToFreeJacobian`
- 每步接受后：`if (covTransport) jacTransport = D * jacTransport`，`D` 来自 `GenericDefaultExtension::transportMatrix`（ATL-SOFT-PUB-2009-002）
- 正式路径 `nullopt`：`covTransport = false`，`jacTransport` 保持 `I`
- `Propagator::Result::transportJacobian` 只在 `makeResult` 里对 **target surface** 做 `boundState` 时写入，那是 WB119 的旁路
- `MaterialInteractor` 在 `covTransport` 时调用 `transportCovarianceToCurvilinear`，把 `jacTransport` 重置并把 6×6 `jacobian` 折成 curvilinear。这不是正式 `h_i` 的均值路径

因此：正式路径**不直接暴露** free-state Jacobian；允许的诊断是打开 stepper 已有的 `covTransport` 开关，在同一条无边界 `SurfaceReached(BoundaryCheck=false) + projectSupportingPlane` hop 上收集 RK `D` 段之积。不手写磁场解析模型，不退回 dummy-cov bounded `transportJacobian`。

## 3. 实现

Config：`configs/official_supporting_plane_jacobian_v1.yaml`  
SHA：`7aeaa12fbb09d0b2ae7b26976b0cbb3978ae748e110380ede539cafe75e42b00`

Helper 新开关 `EnableOfficialSupportingPlaneJacobian` 默认关闭。打开后只写正式 Mode B。官方生产路径仍用 `nullopt`。诊断 dummy 1e-8 协方差**只**用来打开 `covTransport`，传播仍是无边界支撑平面。

每 hop 记录：

```
source chart / surface identity / 单位
official nullopt: covTransport, jacTransport==I
diagnostic: jacTransport, jacToGlobal, accumulated bound jacobian, RK D 积
supporting-plane intersection Jacobian (8×8)
projection d loc0 / d free (1×8)
continuation = f2b * intersection * rk_free * b2f_start
J_residual = - d loc0 / d θ_source   （前 5 列，q/p = 1/GeV）
```

两条残差列家族：

- `rk_free_chain`：正式路径主比较
- `acts_composed_chain`：对照（含物质面 curvilinear 重置）

Helper SHA：`bea305738d479f335fd64b1f7a92e0d9daf66a395873d17dda18bfa2e19927c3`

测试：`tests/test_official_supporting_plane_jacobian.py` 5 passed（login，无 Athena）。分类器只允许三类判决；禁止 prior / ridge / 换 likelihood / 挑最好步长 / dummy-cov bounded Jacobian / 重开 B14M / 提交 1989。

## 4. Smoke（login only：0/1/37 + 86）

新 root：`outputs/leave_target_out_dump_v1/b14u_smoke`  
未覆盖 WB109 official dump（仍 1951622 bytes）  
未覆盖 WB117 smoke（仍 17195553 bytes）  
未覆盖 WB118 smoke（仍 4371700 bytes，SHA `a5612a96520baaf2675187292e65a5ef72ac740211b2d561a317521f35a7a810`）  
未覆盖 WB119 smoke（SHA `e8c49c744cea08e8a9f07fbc48417d5cb2a042d8a0c9908f5a3b55e779407ba8` / `df67bdf16711779b8befccd83238fb1e7d11b539448f8057861e2b34912dc947`）

| dump | bytes | lines | SHA |
| --- | --- | --- | --- |
| `b14u_smoke/mc24_100043_00400_00499/ckf_leave_target_out_official_jacobian.jsonl` | 2241304 | 9 | `1c2daf7fda4e585d62624d1705934c3d621a69d560881f3d417d8a96bb1343ab` |
| `b14u_smoke/mc24_100048_00000_00049/ckf_leave_target_out_official_jacobian.jsonl` | 732933 | 3 | `831a2a3fc9ff6e0aa519895d642a5de576c42a3a947700fab48df58bb798a19d` |

12 行正式 Mode B：`100043/0,1,37` 与 `100048/86`，各 target 1/2/3。Athena 报告 100043 `FPE OVERFLOWs: 9`、100048 `FPE OVERFLOWs: 3`，12 行仍完整写出。

## 5. 逐项结果

### 5.1 正式路径 variational 状态 — 可获得

全部 12 行：`official_path_jacTransport_is_identity = true`（正式 `nullopt` 不填 Jacobian，与源码一致）。诊断 hop `diagnostic_cov_transport = true`，`free_state_jacobian_available = true`，`chain_complete = true`。包括 37/86 的长磁 hop。这不是 `acts_free_state_jacobian_unavailable`。

### 5.2 函数值与正式 Mode-B loc0 — PASS

全部 hop `official_minus_diagnostic_loc0 = 0`。`official_loc0_matches_diagnostic = true`。诊断打开 `covTransport` **没有**改变正式均值轨迹。

### 5.3 Control FD 阶梯 — PASS

`100043/0,1,37` 全部 5 列在冻结 `h…h/8` 上收敛。与 WB117/WB119 一致，不是重新挑选步长。

### 5.4 Control `rk_free` vs FD

| 事件 | loc0 | loc1 | phi | theta | q/p |
| --- | --- | --- | --- | --- | --- |
| 0 t1/t2/t3 | ≤1.4e-4 | ≤4.1e-3 | ≤1.0e-4 | ≤1.2e-4 | ≤1.5e-3 |
| 1 t1 | 2.5e-3 | **0.0616 FAIL** | 1.6e-3 | 0.013 | 1.0e-3 |
| 1 t2 | 2.4e-3 | **0.0564 FAIL** | 1.4e-3 | 0.014 | 9.5e-4 |
| 1 t3 | 1.0e-3 | 0.027 PASS | 1.6e-3 | 6.4e-3 | 1.3e-3 |
| 37 t1/t2/t3 | ≤9.3e-5 | ≤7.1e-4 | ≤2.0e-4 | ≤4.5e-5 | ≤1.3e-3 |

0 与 37 五列全过，包括 ~3 m 磁 hop 的 `q/p`（FD / `rk_free` norm 都是 O(10³)）。1 的 `loc1` 列本身很小（norm ≈ 0.085），相对误差略超冻结 5%。**不**因此放宽阈值或改 FD 步长。`control_official_path_agrees_fd = false`。

### 5.5 对照家族 `acts_composed` — 仍不是正式导数

control 上 `acts_composed` vs FD 相对误差典型 0.9–28。这就是 WB119 的 dummy-cov / 物质面 curvilinear 对象。主比较必须用 `rk_free`。

### 5.6 Focus 86 — FD 仍不收敛，契约不过

target 1 最后一对相对误差约：

| 参数 | last-pair rel | FD norm | rk_free norm | rk vs FD rel |
| --- | --- | --- | --- | --- |
| loc0 | 0.191 | 4.16 | 4.12 | 0.013 |
| loc1 | 6.04 | 0.660 | 0.082 | 0.976 |
| phi | 1.36 | 229 | 154 | 0.335 |
| theta | 0.311 | 7872 | 6650 | 0.177 |
| q/p | 1.32 | 3933 | 2803 | 0.291 |

与 WB119 的关键差别：`q/p` 列现在是 O(10³) 对 O(10³)，不再是 O(10⁻³) 对 O(10³)。同路径 Jacobian 已对上量级，但冻结 5% 列契约与 FD 阶梯收敛都未过。**不**据此减小 FD 步长，也**不**把 86 判成 FD 不可靠从而重开 B14M。

### 5.7 Continuation / 泄漏

Continuation 全部是 `transform_free_to_bound`，无 `plane_chart_fallback`。`branch_switching = false`。target leakage = 0。

### 5.8 不允许的读法

1. **不是** `acts_free_state_jacobian_unavailable`。Jacobian 已从正式无边界路径读出。
2. **不是** `official_supporting_plane_jacobian_established`。control 1 的 `loc1` 与 86 未过。
3. **不是**物理不可微，也不是该改生产 `stepTolerance` 或减小 FD 步长。
4. **不是**可以把 `acts_composed` 或 WB119 `transportJacobian` 写进正式 likelihood。

## 6. 正式审计

Official run：`sbb14u_official_jacobian_20260908T184910Z_d5148ffc`

| 文件 | SHA |
| --- | --- |
| `official_supporting_plane_jacobian_decision.json` | `ad63a600d3b2ec4c10744f73719e58f87ba87cbf7d77852d292c4a3997421a01` |
| `official_path_jacobian_vs_fd_ladder.json` | `63945ee437951f668e23340477591527b32b6744e1a9d543b056d24c2e6a9800` |
| `official_path_hop_decomposition.json` | `6352006c7c7edaf48009776662f25a3caa83e0c0274e7b862f48e21ba7c55476` |
| config | `7aeaa12fbb09d0b2ae7b26976b0cbb3978ae748e110380ede539cafe75e42b00` |
| helper | `bea305738d479f335fd64b1f7a92e0d9daf66a395873d17dda18bfa2e19927c3` |

`smoke_gate_passed = true` 只表示 0/1/37/86 的诊断材料齐全，**不**表示 Jacobian 契约通过。

## 7. 冻结

```
decision = official_path_jacobian_inconsistent_with_fd
jacobian_contract_established = false
b14m_reopen_authorized = false
restart_invariance_authorized = false
full_sample_authorized = false
b15_authorized = false
measurement_model_v2_authorized = false
free_state_jacobian_available = true
official_loc0_matches_diagnostic = true
do_not_replace_official_likelihood = true
do_not_use_dummy_cov_bounded_transportJacobian = true
do_not_select_best_step = true
do_not_select_best_tolerance = true
statistical_model_unchanged = true
```

主线仍然是：

```
WB119 拿错了 Jacobian
  → WB120 给正式 supporting-plane likelihood 建立同路径 derivative
  → derivative PASS          ← 本任务未达到（control 1 loc1 + 86）
  → reopen B14M smoke
  → restart invariance
  → only then consider full-sample profile campaign
```

下一步仍停在正式路径 Jacobian / FD 契约，**不**重开 B14M，**不**提交 1989。允许继续诊断 control 1 的小 `loc1` 列为何略超 5%，以及 86 在同路径 Jacobian 已对上 `q/p` 量级之后为何 FD 阶梯仍不收敛。不允许换一个更好看的 FD、改生产 `stepTolerance`、或把 ACTS 矩阵直接写进正式 likelihood。
