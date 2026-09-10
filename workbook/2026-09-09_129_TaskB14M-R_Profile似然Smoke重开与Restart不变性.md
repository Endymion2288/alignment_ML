# Workbook 129: Task B14M-R Profile 似然 Smoke 重开与 Restart 不变性

日期：2026-09-09
状态：**完成 / FAIL** —— 以 GitHub `master` `55cf982302a3c62c57b74f368d5e3ba7723fd33a` 为 checkpoint，以 **WB128** 为最新冻结。在**不改** WB114 measurement-level likelihood、不加 prior / ridge、不固定或删除 q/p、不用 truth q/p、不把 target measurement 读进 fit / line search / stopping / restart 选择、不用 full-track CKF Cin、不用 WB109 empirical covariance、不改 `FieldGradientDefaultExtension`、不回退 WB119 / WB120 / WB124 / WB125 Jacobian、不删 `100043/37` 或 `100048/86`、不换成 44、不重调 Gauss–Newton、不在看完结果后新设 tolerance 的前提下，只重开 login-scale B14M smoke，并执行四个预注册 restart。未提交 1989。未进 B15 / WB130 / V4 C/D / Measurement Model V2 / alignment / ML。未覆盖 WB109 / WB114–WB128 / 旧 `b14m_smoke` dumps。

**最终判定：`FAIL` / Case `profile_optimizer_restart_sensitive`**

- `decision = profile_optimizer_restart_sensitive`
- `primary_case = profile_optimizer_restart_sensitive`
- `verdict = FAIL`
- `b14m_smoke_passed = false`
- `restart_invariance_established = false`
- `restart_invariance_authorized = false`
- `jacobian_contract_established = true`（继承 WB128，hash 匹配）
- `shadow_mean_contract_established = true`
- `focus_independent_reference_established = true`
- `b14m_reopen_authorized = true`
- `full_sample_authorized = false`
- `target_exclusion_holds = true`（60 行，0 leaked）
- `prior_introduced = false`
- `ridge_added = false`
- `statistical_model_unchanged = true`
- `optimizer_contract_not_recoverable = false`
- `b15_authorized = false`
- `next_step = remain_on_b14m_r`

这是 **profile / optimizer restart-sensitivity FAIL**，不是 Jacobian 未认证，也不是 1989 授权。下一本不能自动进入 WB130，更不能提交 Condor。

## 起始状态

HEAD：`55cf982302a3c62c57b74f368d5e3ba7723fd33a`

WB128 PASS：`certified_mean_independent_reference_established`  
`jacobian_contract_established = true`，`b14m_reopen_authorized = true`，`restart_invariance_authorized = false`，`full_sample_authorized = false`。

## 冻结统计模型（一字不改）

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
r_i(theta) = m_i - h_i(theta)
chi2(theta) = Σ_i r_i^T R_i^{-1} r_i
chi2_prof(alpha) = min_nu chi2(alpha, nu)
R_i = (0.08 mm)^2 / 12
```

q/p 仍是 explicit nuisance。`q/p × 1.1` 只是 R3 restart seed，不是 prior。

## 已认证导数实现

数值 Jacobian = **WB123 repaired production tangent + WB127 certified mean**。

这是 derivative implementation，不是新的统计模型。均值路径仍是 official supporting-plane sequential，`computeEnergyLossBethe`。

## 从 WB114 / WB115 恢复的 optimizer contract

没有重新发明数字。从冻结 YAML 读出：

```
WB115 profile_optimizer:
  max_iterations = 50
  line_search = 1.0 … 0.0078125
  relative_chi2_decrease = 1e-8
  gradient_norm_z = 1e-6
  step_norm_z = 1e-8
  pinv_relative = 1e-8

WB114 profile_seed_invariance:
  chi2_rel_tolerance = 0.01
  prediction_abs_tolerance_mm = 0.1
  supported_abs_tolerance.loc0_mm = 0.05
  supported_abs_tolerance.theta = 0.0001
```

`rank(H) < 5` 不是自动 FAIL。damping ≠ prior ≠ statistical information。

## Smoke 范围

```
100043/0, 1, 37
100048/86
× target 1/2/3
× R0 nominal / R1 loc1+1mm / R2 phi+1e-3 / R3 q/p×1.1
```

0 / 1 = normal controls。37 = reconstruction-consistency control。86 = historical numerical focus。48 个 optimize 行全部 `transport_task = B14MR`，Jacobian 全部 `wb123_repaired_production_tangent`，终止全部 `flat_direction`（有效终止），全部 finite，无传播失败。

## 12 个 identity 的结果

| identity | obj invariant | pred invariant | branch | ν unique | note |
|---|---|---|---|---|---|
| 0 × T1/T2/T3 | yes（Δχ²_rel ≤ 8.5e-4） | yes（Δloc0 ≤ 1.5e-5 mm） | yes | no | Case C |
| 1 × T1/T2/T3 | yes（Δχ²_rel ≤ 2e-8） | yes（Δloc0 ≤ 2e-7 mm） | yes | no | Case C |
| 37 × T1 | **no**（0.088） | **no**（surviving meas；target Δloc0=0.019 mm） | yes | no | 大 χ² ~ 7e4，收敛但 multimodal |
| 37 × T2 | **no**（0.507） | **no**（target Δloc0=0.491 mm） | yes | no | R3 落到另一解 |
| 37 × T3 | yes（0.0022） | yes（0.003 mm） | yes | no | 大 χ² ~ 5.5e4 但 invariant |
| 86 × T1 | **no**（0.111） | yes（0.033 mm） | yes | no | R1 找到更低 χ² |
| 86 × T2/T3 | yes（~1e-9） | yes（≤ 3e-5 mm） | yes | no | Case C |

WB114 相对 χ² gate 是 1%。37/T1、37/T2、86/T1 超过该冻结 gate。没有在看完结果后改 tolerance。

## 对 37 的三个问题

```
Does optimizer converge?          yes（四 restart 均有效终止、finite、无传播失败）
Is objective restart-invariant?   no on T1/T2；yes on T3
Are predictions restart-invariant? no on T1/T2；yes on T3
```

大 χ²（O(1e4)–O(1e5)）本身不是删除理由。37 仍保留。T3 说明“大 χ² 但 restart invariant”是有意义的结果；T1/T2 才是当前 profile contract 的问题。

## 对 86 的分类

WB128 已关闭 86 derivative reference。本任务不能再写 “Jacobian not validated”。

86/T1 是 **profile-level** 失败：

```
profile_objective_multimodality
optimizer_globalization_failure
```

R1（loc1+1 mm）把 χ² 从 3568 降到 3177，held-out target loc0 仍在 0.1 mm 内。T2/T3 对四个 restart 数值上重合。不是 86 historical FD instability。

## Schur 局部一致性

在 `100043/0` 与 `100048/86` 的 nominal final neighborhood 比较 joint NLS / explicit ν / Schur。Hessian 常见 `rank 3–4`，`H_nn` rank 2。完整 5D pinv 步长可以沿 null 方向不同；**profiled α 步长一致**。这不是 Case F。没有因此更换 optimizer。

## Held-out

Stage A：fit / profile 不读 target measurement。  
Stage B：四 restart 冻结后才记 `held_out_residual`。该 residual 不反馈 optimizer、不选 restart、不改 gate。

## 正式产物

Official run：`sbb14mr_restart_invariance_20260909T183700Z_f1c280a7`

```
config SHA      4ce8379b4cab239b5124cafea4f17ae8805a2b676eb7b506c597ae6e38d68890
decision SHA    67aa191def9863d39dec6643bf117516aa95039d6a90e0e38a09ac5b161059ea
helper SHA      686d980fbf72a872af74b4cb9b4a0bdbbd608e615a787dcc55b4986027c6ec4d
dump 100043     74aa02deae3b0042edd9b85506c5d95f51798153a663dad83e9ce58eb206276c
dump 100048     9a64c70a2a2c2df1ff2c90b353584d831380304bf5e3b5729081c998e2c9586c
mean header SHA 8f9d162e59f590770baba1f0c393a7f10861a85495dd7bfbac676d8b59bd15da
field-grad SHA  ca5e4f0ef1a7a09edd1c24099e7ce689c4f0511e2d4afab3a160ebd2d6ff03d8
```

Dump：`outputs/leave_target_out_dump_v1/b14m_reopen_smoke/`  
（不覆盖 `b14m_smoke` / WB114–WB128）

Artifacts：

```
b14m_nominal_profile_trace.json
b14m_restart_profile_trace.json
restart_objective_invariance.json
restart_prediction_invariance.json
restart_parameter_nonuniqueness.json
restart_transport_branch_invariance.json
profile_schur_consistency.json
target_exclusion_audit.json
held_out_prediction_diagnostic.json
b14m_restart_invariance_decision.json
inherited_stage.json
COMPLETE.json
```

`tests/test_b14m_restart_invariance.py`：6 passed。

## 继承

```
WB127: shadow_mean_contract_established = true
       production_eloss_quantity = computeEnergyLossBethe
WB128: focus_independent_reference_established = true
       jacobian_contract_established = true
       b14m_reopen_authorized = true
WB103: contracted denominator = 1989
current: full_sample_authorized = false
```

SHA 匹配。`inherited_contract_mismatch` 未触发。

## 下一步

不要提交 1989。不要进入 WB130 preflight。

当前缺口是 **同一 smooth objective 上，37/T1、37/T2、86/T1 的四个预注册 restart 没有回到同一 profiled χ²（以及 37/T1–T2 的预测）**。弱 nuisance 不唯一在 0/1 和 86/T2–T3 上已经与稳定预测共存，那一部分是 Case C，不是本 FAIL 的原因。
