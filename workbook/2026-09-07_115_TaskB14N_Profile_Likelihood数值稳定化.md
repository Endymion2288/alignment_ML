# Workbook 115: Task B14N Profile Likelihood 数值稳定化与证伪

日期：2026-09-07
状态：**完成 / FAIL** —— 在**不改变** WB114 measurement likelihood、参数内容、target exclusion 和物理统计模型的前提下，建立可审计的逐 hit evaluator、z 序顺序传播、固定数值缩放、有限差分 Jacobian、Gauss–Newton + backtracking，并在预注册 smoke 上做 restart invariance。未引入 prior。未把 ridge 当作统计信息。未用人为 χ² penalty 代替传播失败。未恢复 5D Cin。未删 `100043/37` 或 `100048/86`。未改 WB103 contract。未进 B15，未进 V4 C/D，未进 Measurement Model V2。未覆盖 WB109 / WB114 dumps。**未提交全样本 HTCondor**（smoke gate 未过）。

**最终判定：`FAIL`**

- `decision = profile_transport_contract_broken`
- `primary_case = profile_transport_contract_broken`
- `active_mechanisms = B`
- `smoke_gate_passed = false`
- `full_sample_authorized = false`
- `synthetic_profile_passed = true`
- `acts_numerics_materialized = true`（仅 smoke）
- `target_exclusion_holds = true`（72 行，0 leaked）
- `prior_introduced = false`
- `ridge_added = false`
- `statistical_model_unchanged = true`
- `profile_math_rewritten = false`
- `b15_authorized = false`
- `measurement_model_v2_authorized = false`
- `do_not_force_5d_lto_covariance = true`
- `focus_identity_retained = true`
- `physical_nonidentifiability_not_claimed_from_propagation_failure = true`

这仍然不是 Measurement Model V2，也还不是物理不可识别。下一步是修 standalone measurement transport evaluator，不是重跑 B14M 物理/不确定度，也不是进 B15。

## 起始状态

HEAD：`32c9044a8543845aaa0767d8089ba6fd731f31a9`

工作树从 WB98 起一直未提交。本任务只做增量，未 `git reset --hard`、未 `git clean`、未 checkout 覆盖。

WB109 PASS：`leave_target_out_state_materialization_established`  
Decision SHA：`af7ccd055c2ff3d34f92429e37be19a89ed90a9b0db11ec6770c161c79b6e933`

WB110 FAIL：`lto_input_covariance_shape_not_validated`  
Decision SHA：`eba85c1835e90f85a1ca0aec4ee896cc1b611214f9213c7602e160cbd291857b`

WB111 DIAGNOSED：`mixed_or_inconclusive`  
Decision SHA：`c9d35002671327dd401e1ca67bd17cb7dacfadf85efcb53e15af6d94dfc2a644`

WB112 DIAGNOSED：`reduced_measurement_supported_state_with_weak_nuisance`  
Decision SHA：`015087449a030232cfd088458b394ae82c409bae9bc26827fc2c944ec8c80dd8`

WB113 FAIL：`target_independent_prior_not_available`  
Decision SHA：`a01bb5490c346241f8282f0eb617a088b85aff535a4a72bfd2a44ff4b0931ecd`

WB114 FAIL：`profile_likelihood_numerically_unstable`  
Decision SHA：`82304f5af943f97adfe069e02c8bacdd7491cca72e6ecc58ea82cc470428a18f`  
Config SHA：`de8876cd4b63c06f4e3fb7224f9eff7346cdb4e503e41b38b896c76ad3fb92b6`

冻结方向分类（不得改写）：

```
x      = measurement_dominated
y      = weakly_measured
tx/φ   = prior_dominated
ty     = measurement_dominated
q/p    = weakly_measured / explicit nuisance
five_d_state_physically_supported = false
```

WB103 冻结 denominator 保持：raw 2680 / ineligible 691 / contracted 1989 / official pairs 1974。

WB114 已冻结的 likelihood 不得改写：

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2_prof(alpha) = min_nu chi2(alpha, nu)
R_i = (0.08 mm)² / 12
```

合成 joint-NLS / explicit profile / Schur 仍然一致。本任务不重写 profile 数学。

## 1. 本任务唯一问题

同一个 measurement likelihood 在真实 ACTS 几何/磁场下，能否被稳定、可复现、与初值无关地数值求解？

如果不能，失败是在：

```
propagation
→ surface ordering
→ parameterization / scaling
→ Jacobian
→ optimizer globalization
```

中的哪一层。传播失败和 Hessian rank 3/5 **不得**直接写成物理不可识别。

## 2. 实现（不改统计模型）

Config：`configs/profile_likelihood_numerics_v1.yaml`  
SHA：`b338874546276e6fd2c4793c4ad02ccd9c8d95bed9e693b0d9659d6a04f5b57e`

Helper 仍是独立 `CkfLeaveTargetOutDumpAlg`。默认关闭，WB109 / WB114 路径保持可复现。B14N 打开：

- `EnableProfileNumerics`
- 按物理 z 顺序逐面 hop（先 downstream，再 upstream）
- 去掉 WB114 的 `maxStepSize = 10 m`，与 Kalman 一样用自适应步长
- 每 hop 写审计：station / surface / 初态 / q/p / 方向 / path length / steps / status / predicted loc0 / residual
- 固定数值尺度 `s = (1 mm, 1 mm, 1 mrad, 1 mrad, 1e-3 /GeV)`，不是 prior
- Gauss–Newton + backtracking；传播失败的 trial 只拒绝、缩短步长，不加 χ² penalty
- 预注册收敛：`converged | flat_direction | propagation_blocked | jacobian_invalid | max_iterations`
- smoke 上做中心差分 Jacobian + step-halving
- 从 WB114 official seed mean 和一次内部 Kalman fitted mean 做 **evaluate-only** replay

Helper SHA：`6ea29f09091454f586cebbbb44e5e4fe696aac341bf2d6c9bef6186cf7b51a8f`

测试：`tests/test_profile_likelihood_numerics.py` 6 passed（login，无 Athena）。

## 3. Smoke（login only）

新 root：`outputs/leave_target_out_dump_v1/b14n_smoke`  
未覆盖 WB109 official dump（仍 1951622 bytes）  
未覆盖 WB114 smoke（仍 256783 bytes）

| source | events | lines | SHA |
| --- | --- | --- | --- |
| `mc24_100043_00400_00499` | 0,1,37 | 54 | `ac34173927eb3047ec79e9abf481f8f67497d7583101c3837495554f955e99b9` |
| `mc24_100048_00000_00049` | 86 | 18 | `e5b4151fd1882bc4539e42d59cd025603a5f9b65cca625db2f301beb949758ad` |

每个 identity × 3 targets 写出：seed evaluate、Kalman fitted evaluate、以及 4 个 WB114 restart 的优化行。

## 4. B14N.1 / B14N.9：逐 measurement 传播

### 100043/0 和 100043/1

Nominal measurement likelihood **可以对全部 surviving hits 求值**。

### 100043/37

第一问的答案是：**不能**。

- 16 个 surviving hits（target 1；target 2/3 为 16/15）
- station 0 的 5 个 hop 成功
- 第一个失败面是离开 IFT 后的下一站 measurement surface  
  target 1：`z = -1826.22 → 1207.45`（station 2，因为 station 1 已被 exclude）  
  target 2/3：`z = -1826.22 → 17.45`（station 1）
- abort：`Propagation reached the configured maximum number of steps`
- 失败 Result 不暴露 step 计数，因此不能再只记 “step limit 10000”
- 后续 hop 仍从失败前的 IFT 状态出发，全部失败
- 内部 Kalman replay：`n_fit = 2`，`χ² ≈ 0`，**同样不能**对全部 surviving surfaces 求值

37/86 问题的分层回答：

```
37 = A + D
A. nominal state 本身就不能走完全部 surviving surfaces
D. 失败信息是 stepper / path limit，不是 optimizer trial
不是 C：z 序正确，没有回到上游
不是 B：evaluate-only / nominal 就已经失败
```

Kalman `n_fit=2` 说明 Kalman **从未**对这 16 个 hits 建立同等合同。独立 propagator 直接 hop IFT → station 1/2，中间隔着谱仪磁铁和未被测量的敏感面。这是 transport 合同问题。

### 100048/86

Nominal 仍然不能对全部 surviving hits 求值，但机制不同。

- 17/16/17 hits
- **磁铁跨越成功**：`z = -1822.25 → 1203.47`，128 steps，path ≈ 3026 mm
- 第一个失败面是 station 2 layer 1 的 stereo partner：  
  `1234.97 → 1235.86`（0.885 mm）  
  `Global to local transformation failed: position not on surface`
- 同一站后续 layer 2、以及 station 3 多数 hop 仍成功（16/17）
- Kalman `n_fit = 4`，`χ² = 37.64`，全部 surfaces 仍未达到

```
86 = A + C
A. nominal 不能对全部 surviving hits 求值
C. 近邻 wafer navigation / surface intersection，不是 step-limit
y 灾难是否仍在 measurement-level objective 里：当前还不能问，因为 χ² 仍未对全部 hits 定义
```

二者均保留。

## 5. B14N.2 / B14N.3：Kalman vs profile，surface 顺序

Surface ordering：12 个 seed evaluate 行全部 z-sorted；`returns_upstream_after_downstream = 0`。  
**不是** TSOS/container 乱序，也不是传播到下游再强制回上游。

两边都用 ACTS，但合同不自动相同：

| | Kalman | B14N profile |
| --- | --- | --- |
| hop | fitter 内部 surface sequence + update | 无 update 的逐 measurement hop |
| maxSteps | 10000 | 4000 / hop |
| maxStepSize | unlimited adaptive | 现已对齐 unlimited adaptive（WB114 曾是 10 m） |
| material | MS + EL | MS + EL |
| reference | `first` | source-plane seed / fitted |
| 成功定义 | 可用 outlier，不必到达每一个 used hit | **全部** surviving hits 都必须到达才有 χ² |

“都用了 ACTS”不是同一合同。37 上 Kalman 用 2 个 measurement 就返回；profile 要求 16 面都到达。

## 6. B14N.4 / B14N.5：缩放与 Jacobian

固定单位尺度，不是 prior，不是 truth / Hessian / seed campaign。  
`χ²(θ(z))` 由构造不变。合成不变性通过。

Jacobian（只在能求值的 0/1 evaluate 行上，12 份报告）：

| 参数 | median relative error | sign |
| --- | --- | --- |
| loc0 | 1.7e-6 | 一致 |
| loc1 | 6.6e-5 | 一致 |
| phi | 4.0e-4 | 一致 |
| theta | 1.0e-5 | 一致 |
| q/p | 9.3e-5 | 一致 |

`jacobian_contract_established = true`（在可求值的 smoke 上）。  
因此 event 0/1 的 optimizer 结果**可以**按数值算法解释，不能把 37/86 的失败解释成 derivative 坏了。

## 7. B14N.6–8：globalization 与 restart v2

Event 0/1 × target 1/2/3 × 4 restart：**24/24 成功**。

WB114 的 `loc1+1 mm` under-convergence **已经消失**。预注册等价容差下：

- `restart_invariance_holds = true`
- `loc1_restart_equivalent = true`

允许 nuisance 不同。终止原因多为 `flat_direction`。rank 常见 4/5 或 5/5。这**不是**失败：profile 里 nuisance null direction 是允许的。

没有 hidden ridge / prior / χ² penalty。

## 8. Smoke gate

```
events_0_1_all_restarts     true
loc1_restart_equivalent     true
jacobian_pass               true
no_ridge                    true
target_exclusion            true
nominal_propagation_pass    false   # 37/86 不能对全部 hits 求值
focus_not_step_limit_only   false
```

gate **FAIL**。按合同 **不提交** 1989 行全样本 HTCondor。

## 9. 正式产物

Official run：`sbb14n_profile_numerics_20260907T150627Z_d77ddddc`  
Output root：`outputs/profile_likelihood_numerics_v1`（新 run，不覆盖 WB109/WB114）

| 文件 | SHA |
| --- | --- |
| `profile_measurement_propagation_audit.json` | `0e8e268786e2127b9b6c84ec71f202537db64bcc6a3c511764f965a42d5fc81b` |
| `kalman_vs_profile_transport_contract.json` | `f6c8c4a8cda21f28b460f860c35d21dad09a89c42d32c95d484853bfbdc00922` |
| `profile_surface_ordering_audit.json` | `1822cbf073430075a935d13a7f0cb7ab30e4a8369ce78954f80225921ef51e7f` |
| `profile_parameter_scaling_contract.json` | `f2b2dabe11c3fb262237c7dde0ad0c507766ebf66ae7a81676c81c0dca7a0e4f` |
| `profile_jacobian_validation.json` | `04ba4bd0d9951b0fb25648e89239f19350f10136712390643b569af6b712c54b` |
| `profile_optimizer_trace.json` | `7666c5f5b745177c2e9c20854d0d2d302a1ef976a31a947269a4aa35f74a6d1a` |
| `profile_restart_invariance_v2.json` | `c05e10428d60efd04df3aaa89bb10e24ed8fbd068b234b4b85ca7e01c490a6f9` |
| `focus_identity_profile_numerics.json` | `ba02b6809d587d9692c89bab29fdbb0d679660f4b46657e8db23520ccd57f248` |
| `profile_numerical_contract_decision.json` | `dcf6ce865d0c1788e44d0a5a37588c7df3e4b90d31f1c31f67090155a9436c69` |
| `inherited_stage.json` | `820f10db52f6ec34eda5c676efc1021ccbb8c293e11ba92ab329140aae32a8d2` |
| `COMPLETE.json` | `3751db18589ef11d8494c53d509155b962a0afa8b511c91fda0f04dc951d0eef` |

## 10. 分类为何是 B 而不是 D/E

WB114 Case D（裸 Gauss–Newton under-convergence）在 event 0/1 上已经被 globalization + scaling **证伪为已修复**。  
当前挡住 smoke gate 的是 **standalone measurement propagation**：37 过不了磁铁 hop，86 在近邻 wafer 上 `not on surface`。Kalman 也没有对同一套 surviving surfaces 给出可对齐的全到达合同。

这是 Case B：`profile_transport_contract_broken`。

不是 Case C：可求值事件上 Jacobian 通过。  
不是 Case D：0/1 的 restart / loc1 已回到等价 profile minimum。  
不是 Case E：evaluator 还没对 37/86 定义 χ²，不能把传播失败升级成物理不可识别。  
不是 Case A：smoke gate 未过。

## 11. 明确不做的事

- 不把 37/86 的传播失败写成 physical nonidentifiability
- 不把 rank 4/5 或 `flat_direction` 写成失败
- 不加 prior / ridge / 人为 χ² penalty
- 不改 R_i，不恢复 WB109/WB107 Cin
- 不删 37/86，不固定 q/p
- 不提交 1989 行 Condor
- 不进 B15 / V4 C/D / Measurement Model V2 / alignment / ML

## 12. 下一步

```
WB114: measurement likelihood math OK, synthetic OK, 裸 GN 不稳定
        ↓
WB115 / B14N: Case B
  event 0/1: evaluator + Jacobian + backtracking + loc1 restart 已通过
  37: IFT→下一站 magnet hop 在独立 SurfaceReached 下失败
  86: 磁铁 hop 成功，近邻 stereo wafer 未上表面
        ↓
下一任务必须先修 transport evaluator
  （导航 / 中间敏感面 / 磁铁跨越 / stereo partner）
        ↓
37/86 nominal χ² 能对全部 surviving hits 求值之后
        ↓
才能重新打开 B14M 物理 / 不确定度
        ↓
之后才讨论是否还需要 B15
```

当前对象仍然是：**先证明同一个 measurement likelihood 能在真实 ACTS 几何/场下被完整求值。** 统计模型冻结。
