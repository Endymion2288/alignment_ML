# Workbook 130: Task B14M-S Profile Basin 拓扑与 Flat-Direction 终止诊断

日期：2026-09-09
状态：**诊断完成 / 未授权修复** —— 以 GitHub `master` `55cf982302a3c62c57b74f368d5e3ba7723fd33a` 为 checkpoint，以 **WB129** 为最新冻结。当前继续停留在 **B14M-R 主线**。这不是 full-sample preflight。

本任务**没有修改** optimizer / termination / line-search / scaling / `pinv_relative`。没有增加随机 restart。没有用 held-out target 选择 basin。没有把 `min(R0..R3)` 当作修复。未提交 1989。未进 B15 / V4 C/D / Measurement Model V2 / alignment / ML。未覆盖 WB109 / WB114–WB129 / 旧 `b14m_smoke` dumps。`FieldGradientDefaultExtension.hpp` SHA 未变。

## Official run

```
run_id   = sbb14ms_profile_basin_20260909T214204Z_45f7bda5
decision = flat_direction_termination_not_stationary
verdict  = DIAGNOSIS
reason   = high_chi2_endpoints_not_stationary
```

Decision SHA：`cb8d178ea67870ddf04bf6babd2957f5be6c29cb0709d31b35696ac535904d91`  
Config SHA：`7678eaa12cde9942cb7314959432a6f9249e2b3890217e797411fc61dfb1b042`  
FieldGradient SHA（冻结）：`ca5e4f0ef1a7a09edd1c24099e7ce689c4f0511e2d4afab3a160ebd2d6ff03d8`

HTCondor（不是 1989）：cluster `9289446`，2 个作业。

```
part_evt0_37.jsonl  sha256=777907ccebf544c5829b2442fac682e04e1f6c4b43d5ed2ef5ec62a977507db1
part_evt86.jsonl    sha256=6226afac52fa748b8252b5c68c150cba958e7a2d423fc1957ffa7398a92a14a0
```

## 起始冻结（未改写）

```
jacobian_contract_established = true
shadow_mean_contract_established = true
focus_independent_reference_established = true
b14m_reopen_authorized = true

b14m_smoke_passed = false
restart_invariance_established = false
full_sample_authorized = false
```

WB129 official run：`sbb14mr_restart_invariance_20260909T183700Z_f1c280a7`  
`decision = profile_optimizer_restart_sensitive`

失败 identity 只剩：

```
100043/37 × target 1
100043/37 × target 2
100048/86 × target 1
```

已通过 identity 不得重新解释为失败。尤其 PASS case 已证明：nuisance 可以不唯一，只要 profiled objective 和 physical prediction 保持不变。

## 冻结统计模型与 optimizer（未改）

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(theta) = Σ r_i^T R_i^{-1} r_i
chi2_prof(alpha) = min_nu chi2(alpha, nu)
R_i = (0.08 mm)^2 / 12

max_iterations = 50
line_search = 1.0 … 0.0078125
relative_chi2_decrease = 1e-8
gradient_norm_z = 1e-6
step_norm_z = 1e-8
pinv_relative = 1e-8
```

禁止项全部保持：prior / ridge / truth q/p / fixed q/p / deleted q/p / target-in-optimizer / full-track CKF Cin / WB109 empirical covariance / new measurement update。

## 核心判据

> 先确认高 χ² endpoint 到底是不是 stationary point。只有两个 endpoint 都真正 stationary，才有资格把它们称为两个 likelihood basin。
>
> rank deficiency 本身不允许触发“成功终止”。只有 supported / range 子空间已经没有可解析下降方向，才可以把 `flat_direction` 当作合法 profile termination。

WB129 全部 48 个 optimize run 的 `termination = flat_direction`。C++ 把“8 个 damp 全拒”标成 `flat_direction` 并直接停，**即使 `||g_R||` 和 `||H⁺g||` 都很大**：

```
if (!accepted)
  termination = (gradNorm < 1e-6 || stepNorm < 1e-8) ? "converged" : "flat_direction";
  break;
```

这不是 stationary test。

## 三个失败 identity：高 χ² endpoint 都不是 minimum

C++ 对每个 WB129 final 重算 `g`、`H`（optimizer-scaled）、SVD（`pinv_relative = 1e-8`）。

| identity | restart | χ² | rank | \|\|g_R\|\| | \|\|H⁺g\|\| | pred. Δχ²（幅度） | 冻结 LS 严格下降 |
|---|---|---:|---:|---:|---:|---:|---|
| 37/T1 | R0 | 75722 | 4 | 1.15e5 | 1.30 | 7.57e4 | 无 |
| 37/T1 | R1 | 75709 | 4 | 1.15e5 | 1.44 | 7.57e4 | 无 |
| 37/T1 | R2 | 75503 | 4 | 1.15e5 | 1.30 | 7.55e4 | 无 |
| 37/T1 | R3 | 69059 | 4 | 1.10e5 | 1.35 | 6.90e4 | 无 |
| 37/T2 | R0 | 110909 | 4 | 1.27e5 | 1.00 | 1.11e5 | 无 |
| 37/T2 | R1 | 110926 | 4 | 1.27e5 | 1.00 | 1.11e5 | 无 |
| 37/T2 | R2 | 110670 | 4 | 1.27e5 | 1.00 | 1.11e5 | 无 |
| 37/T2 | R3 | 54686 | 4 | 8.87e4 | 1.29 | 5.47e4 | 无 |
| 86/T1 | R0 | 3568 | **5** | 1.49e4 | 268 | 981 | 无 |
| 86/T1 | R1 | 3177 | **5** | 1.00e4 | 267 | 589 | 无 |
| 86/T1 | R2 | 3575 | **5** | 1.50e4 | 267 | 987 | 无 |
| 86/T1 | R3 | 3430 | **5** | 1.34e4 | 268 | 842 | 无 |

`||g_R|| ≫ gradient_norm_z`，`-H⁺g` 给出明确下降。不得称为另一物理 minimum。

86/T1 Hessian **满秩 5**。这不是 null-direction 物理非唯一性。

冻结 8-λ replay（1 … 1/128）在这些 5D endpoint **没有**找到 `chi2_new < chi2_final`。因此本任务**不**把主因写成 `flat_direction_termination_logic_broken`（那需要原合同 λ 已经能降 χ²）。主因仍是：

```
flat_direction_termination_not_stationary
```

GN 步太大，冻结 line-search 全拒，optimizer 却把它标成成功终止。本任务不降低 λ 下限去寻找下降。

`||g_α_prof||` 在失败 endpoint 上同样很大（37：native / z 都非零；86/T1：`||g_α_prof_z|| ~ 800`）。它们不是合法 profile stationary point。

## Fixed-α nuisance cross-start：Case C 被排除

四个 restart 只扰动 ν seed。对每个失败 identity 做 4 α × 4 ν = 16，**同一冻结 nuisance optimizer**。

| identity | 固定 α 后 4 个 ν seed 的 χ² | 相对 spread |
|---|---|---:|
| 37/T1 | 全部 → **96.1456** | ~1e-10 |
| 37/T2 | R0/R1/R2 α → 98.24；R3 α → 97.47 | < 1% |
| 86/T1 | 2836 … 2859（按 α 分簇，同 α 内重合） | ~1e-8 同 α 内 |

同一 α 下不同 ν seed **收敛到同一个 χ²**。

```
nuisance_profile_multibasin = false
```

差异不是“固定 α 下存在多个 nuisance minimum”。

更强的事实：固定 WB129 的 α 只优化 ν，χ² 从 7.5e4 / 1.1e5 / 3.5e3 落到 ~96 / ~98 / ~2840。五个参数的 joint run 停在高 χ²，是因为它在非 stationary 点把 `flat_direction` 当成了终止，而不是因为那里有第二个 profile minimum。

## Profile continuation：Case D / E 被排除

按 χ² 只选最高 / 最低 valid endpoint（不读 target）。λ = 0, 0.05, …, 1.00；A→B 与 B→A；ν warm-start。

| identity | A / B | fwd vs bwd | 形状 |
|---|---|---|---|
| 37/T1 | R0=75722 / R3=69059 | 每点 χ² 重合 ~96.15 | 平坦、唯一 |
| 37/T2 | R1=110926 / R3=54686 | 每点重合 ~98.2→97.2→97.5 | 单谷、唯一 |
| 86/T1 | R2=3575 / R1=3177 | 每点重合 2858→2836 | 单调、唯一 |

```
profile_objective_multimodality_established = false
profile_hysteresis_established = false
```

两条 branch 在同一 α 上得到同一 χ²。较高 χ² 的 5D endpoint 本身不是 stationary。没有共存的合法 profile basin。

Stage A 的 174 条 continuation / cross-start 行：`predicted_target_loc0 = null`。held-out 未读。

## 分开诊断（原因同类，表现不同）

### 37 / T1

Δχ²_rel ≈ 8.8%。surviving-measurement prediction 已 FAIL；target loc0 只差 0.019 mm。四个 5D endpoint 都不是 stationary。固定 α 后 ν profile 唯一且 χ²~96。不能因为 target 看起来稳就通过。

### 37 / T2

最强 failure（Δχ²_rel ≈ 50.7%）。R0/R1/R2 在 WB129 的 nIter=1 就停。R3 也不是 stationary。q/p×1.1 **没有**进入真正不同的 nuisance basin：固定 R3 的 α 后，四个 ν seed 都到 χ²=97.47。不得因此固定 q/p。

### 86 / T1

R0 χ²≈3568，R1≈3177。WB128 Jacobian PASS 已冻结，禁止再回到 Jacobian。满秩 Hessian + 巨大 GN step（~267）+ 冻结 LS 全拒 = 终止 / globalization 失败。固定 α 后 ν profile 唯一（~2840）。不是两个已证明的 profile minima。

## 负对照（不得改写 WB129 PASS）

| identity | χ² | \|\|H⁺g\|\| | pred. Δχ² | 冻结 LS | 本任务 |
|---|---:|---:|---:|---|---|
| 0/T1 | ~13.64 不变 | 1e-8 … 3e-7 | ~1e-10 | 无下降 | Case F |
| 86/T2 | 2888.46 完全重合 | 3e-7 … 1.6e-5 | ~1e-10 | 无下降 | Case F |
| 37/T3 | ~5.52e4 不变 | ~0.99 | ~5.5e4 | 无下降 | Case F |

0/T1 与 86/T2：ν 不唯一，但 objective / prediction 不变，supported 方向无可用下降。这是合法的 null-direction 非唯一。

37/T3 同样非 stationary（四个 seed 同样卡住），但 **restart-invariant**。它保持 WB129 PASS。当前诊断不会把“正常的 null-direction 非唯一”错判成失败；也不会因为 37/T3 也有 termination 标签就把 PASS 改写成 FAIL。

## 分类（只允许的主结果）

```
三个 failure:
  37/T1 → Case A  flat_direction_termination_not_stationary
  37/T2 → Case A  flat_direction_termination_not_stationary
  86/T1 → Case A  flat_direction_termination_not_stationary

三个 control:
  0/T1  → Case F  profile_nuisance_nonidentifiability_prediction_stable
  37/T3 → Case F  （restart-invariant stall；不重开 WB129 PASS）
  86/T2 → Case F
```

排除：

```
Case C  nuisance multibasin          — 同 α 下 ν seed 重合
Case D  chi2_prof(α) multibasin      — continuation 唯一
Case E  hysteresis                   — fwd = bwd
Case F  作为 failure 主判            — 禁止；χ² / prediction 并不 invariant
```

Case B（globalization）是下一本可以一起修的数值现象：joint 5D 走不到固定 α 后 ν-only 已经能到的更低 χ²。但按判断顺序，**高 χ² endpoint 不是 stationary，所以本任务主判是 Case A**。不得在两个非 stationary 点之间谈“两个 likelihood basin”。

## 本任务明确没有做的事

```
do_not_change_optimizer = true
do_not_change_termination_rule = true
do_not_change_line_search = true
do_not_take_min_of_four_restarts_as_fix = true
do_not_use_held_out_to_select_basin = true
do_not_submit_1989 = true
```

第一次 Condor（`9289091`）因 nlohmann `json::value(key, nullptr)` 在 `chi2_prof` 为 number 时崩溃。已改为 typed helpers。**未改 optimizer。**

## 结束授权（全部保持 false）

```
restart_invariance_established = false
restart_invariance_authorized = false
full_sample_authorized = false
b15_authorized = false
measurement_model_v2_authorized = false
```

即使机制已经定位，也只授权下一本：**修理 B14M-R 的 termination / globalization 逻辑**。统计模型不变。

只有三个 failure identities 重新通过四 restart 的 objective + prediction invariance 以后，才重新讨论 full-sample preflight。

## 下一本

```
repair exactly:
  flat_direction 不得在 ||g_R|| 或 predicted GN descent 仍大时触发成功终止
  以及 joint 5D 在冻结 LS 全拒时过早停（ν-only 已证明同一 α 还能大幅降 χ²）

然后:
  重跑原来的四个 restart
  不得把 min(R0..R3) 写成新合同
```

不在下一本重开 Jacobian。不固定 q/p。不加 prior / ridge。不提交 1989。

## 产物

```
configs/b14m_profile_basin_diagnosis_v1.yaml
datasets/b14m_profile_basin_diagnosis.py
scripts/audit_b14m_profile_basin_diagnosis.py
tests/test_b14m_profile_basin_diagnosis.py          # 9 passed
alignment/leave_target_out_dump/ProfileBasinDiagnosis.inc
outputs/b14m_profile_basin_diagnosis_v1/sbb14ms_profile_basin_20260909T214204Z_45f7bda5/
  COMPLETE.json
  b14m_profile_basin_decision.json
  endpoint_stationarity_audit.json
  flat_direction_termination_audit.json
  frozen_linesearch_replay.json
  fixed_alpha_cross_start_matrix.json
  profile_continuation_forward.json
  profile_continuation_backward.json
  profile_basin_topology.json
  profile_gradient_audit.json
  negative_control_stationarity.json
  target_exclusion_audit.json
  inherited_stage.json
```
