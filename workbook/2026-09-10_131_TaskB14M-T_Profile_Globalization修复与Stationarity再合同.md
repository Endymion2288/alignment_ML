# Workbook 131: Task B14M-T Profile Globalization 修复与 Stationarity 再合同

日期：2026-09-10
状态：**实现已冻结、48-run smoke 已提交、官方判定待 dump 完成** —— 以 **WB130** 为最新冻结。当前主线是 **B14M-T**。这不是 full-sample preflight，不是 1989。

本任务**只修** optimizer numerical globalization / termination。统计模型、measurement likelihood、Jacobian、mean transport 全部冻结。没有把 `min(R0..R3)` 当作输出。没有用 held-out target 做 globalization。未提交 1989。未进 B15 / V4 C/D / Measurement Model V2 / alignment / ML。未覆盖 WB109 / WB114–WB130 / 旧 `b14m_smoke` dumps。`FieldGradientDefaultExtension.hpp` SHA 未变。

## Incoming freeze（WB130）

```
decision = flat_direction_termination_not_stationary
verdict  = DIAGNOSIS
run_id   = sbb14ms_profile_basin_20260909T214204Z_45f7bda5
decision_sha256 = cb8d178ea67870ddf04bf6babd2957f5be6c29cb0709d31b35696ac535904d91
config_sha256   = 7678eaa12cde9942cb7314959432a6f9249e2b3890217e797411fc61dfb1b042

nuisance_profile_multibasin = false
profile_objective_multimodality_established = false
profile_hysteresis_established = false
```

失败机制已经定位为：joint 5D Gauss–Newton 在冻结 line-search `λ = 1 … 1/128` 全部拒绝后，把

```
termination = flat_direction
```

当作成功停止，即使 `||g_R||`、`||g_α_prof||` 和 predicted GN descent 都很大。这不是合法 stationary termination。

WB129 的 restart-invariance PASS 仍作为历史结果保留，但在发现 termination bug 之后，**不能继续把 restart invariance 单独作为 optimizer validity certificate**。尤其 37/T3：四个 restart 一致地停在同一个非 stationary 点。

## 冻结统计模型（一字不改）

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(theta)     = Σ r_i^T R_i^{-1} r_i
chi2_prof(alpha) = min_ν chi2(alpha, ν)
R_i = (0.08 mm)^2 / 12
```

禁止项全部保持：prior / ridge-as-information / truth q/p / fixed q/p / deleted q/p / target-in-optimizer / full-track CKF Cin / WB109 empirical covariance / new measurement update。

```
shadow_mean_contract_established = true
jacobian_contract_established = true
```

未重开 transport / Jacobian diagnosis。

## 预注册 trust-region（任何 37/86 新结果之前写入 config）

项目里没有可复用的标准 trust-region 常数。alignment 0.15 是另一套线性门，不复用。

```
coordinate system: frozen scaled z
Delta0     = min(1.0, ||delta_GN,R||)
Delta_max  = 10.0
Delta_min  = 1e-8
accept     : rho >= 0.10
expand     : rho >= 0.75 AND step >= 0.8 Delta
shrink     = 0.25
expand     = 2.0
max TR trials / outer = 20
max outer  = 50
max inner  = 50
pinv_relative = 1e-8
gradient_norm_z = 1e-6
step_norm_z = 1e-8
relative_chi2_decrease = 1e-8
```

```
rho = (actual chi2 decrease) / (predicted quadratic decrease)
```

若内部用 LM 等价形式求 trust-region subproblem：

```
lambda = numerical globalization multiplier
NOT prior
NOT ridge
NOT information
```

正式 acceptance 永远使用原始 `chi2` / `chi2_prof`，不是 damped objective。

Config SHA（结果出现前已冻结）：

```
395289a77b6a78d4533ab34ef12eb6617f5aba812922486c3493268cce0d6fc4
```

## 新 solver（并行路径，不改旧 `runOneProfileNumerics` 本体）

```
explicit inner nuisance profile
  ν*(α) = argmin_ν chi2(α, ν)     range-space TR in 3D scaled ν
then
outer profile GN on α
  g_α_prof = g_a − H_an H_nn⁺ g_n
  every α trial re-profiles ν from the last accepted ν*
```

首点使用对应 R0/R1/R2/R3 的预注册 nuisance seed。之后 warm-start 只用上一轮接受的 `ν*`。四个 seed 的最低 χ² **不是**正式算法。

Termination：

```
supported_stationary     : ||g_R|| / ||g_n,R|| / ||g_α_prof|| <= 1e-6
step_converged           : ||delta_R|| <= 1e-8 且没有可解析 supported descent
objective decrease       : 仅当 supported/profile gradient 也不大
globalization_failure    : Delta < Delta_min 且 gradient 仍大
                           valid_solution = false
                           不得再叫 flat_direction
```

`flat_direction` 描述的是 Hessian 的信息几何，不是“这一步走不出去”的同义词。

## Restart 语义（不变）

```
R0 nominal
R1 loc1 + 1 mm
R2 phi + 1e-3
R3 q/p × 1.1
```

只是 optimizer initial state。R3 不是 momentum prior。输出不是 `min(R0..R3)`。

PASS 顺序：

```
stationarity
  ↓
objective invariance          chi2_rel_tolerance = 0.01
  ↓
prediction invariance         0.1 mm
  ↓
transport / branch invariance
```

WB114 冻结容差未改。

## 必须重新认证的 12 identities

```
100043/0, 1, 37
100048/86
× target 1/2/3
× R0/R1/R2/R3
= 48 runs
```

Gate-first（写进审计，不得只修三个旧失败点就宣称 PASS）：

```
37/T1   不得再停在 chi2 ~ 6.9e4–7.6e4
37/T2   旧 nIter=1 stop 必须消失
86/T1   不得再 full GN → 8 次 backtrack → all reject → stop
37/T3   restart invariant 但必须变成 stationary
```

负对照：`0/T1`、`1/T1`、`86/T2`。修旧失败却破坏 already-stable controls → Case F。

## HTCondor smoke（不是 1989）

```
cluster     = 9291953
schedd      = bigbird28.cern.ch
n_jobs      = 2
flavour     = tomorrow
cwd         = /afs/cern.ch/user/x/xcheng/work/b14mt_repair_condor
submit file 不含 /eos
plugin_sha256 = 14c93af3db23beac3035dc343f8ce3bf51c85252918aa7f54cb70c563be188f6
```

作业：

```
mc24_100043_00400_00499  events 0,1,37  → part_evt0_1_37.jsonl
mc24_100048_00000_00049  event 86       → part_evt86.jsonl
```

Smoke root：`outputs/leave_target_out_dump_v1/b14mt_repair_smoke`  
未覆盖 `b14m_reopen_smoke` / `b14ms_basin_smoke` / `b14m_smoke`。

## 允许的最终分类

```
A  profile_globalization_and_restart_contract_established   PASS
B  trust_region_globalization_still_fails                   FAIL
C  inner_nuisance_profile_not_converged                     FAIL
D  profile_outer_optimizer_restart_sensitive                FAIL
E  profile_nuisance_nonidentifiability_prediction_stable    PASS-compatible
F  optimizer_repair_regression                              FAIL
G  mixed_or_inconclusive                                    FAIL
```

即使 Case A：

```
b14m_smoke_passed = true
restart_invariance_established = true
restart_invariance_authorized = true
full_sample_authorized = false
```

下一本才是 WB132 full-sample preflight。只有那本再 PASS，才 `full_sample_authorized = true`，然后才能提交 1989。

## 单元测试

```
tests/test_b14m_profile_globalization_repair.py   8 passed
```

覆盖：WB130 继承、预注册 TR 常数、禁止更小 λ / ridge / min(R0..R3)、stationarity-before-invariance、Case B/C/D/E/F、Case E 与 Case A 兼容。

## 官方判定

待 48-run dump 完成后由 `scripts/audit_b14m_profile_globalization_repair.py` 写入：

```
outputs/b14m_profile_globalization_repair_v1/sbb14mt_profile_globalization_*/
```

在 dump 完成前不得回写 trust-region 常数，也不得宣称 Case A。

## 本任务明确没有做的事

```
do_not_change_statistical_model = true
do_not_add_smaller_line_search_lambda = true
do_not_take_min_of_four_restarts_as_fix = true
do_not_use_held_out_for_globalization = true
do_not_reopen_jacobian = true
do_not_call_lm_lambda_ridge = true
do_not_submit_1989 = true
```

## 产物

```
configs/b14m_profile_globalization_repair_v1.yaml
datasets/b14m_profile_globalization_repair.py
scripts/run_b14m_profile_globalization_repair.py
scripts/audit_b14m_profile_globalization_repair.py
scripts/run_b14m_profile_globalization_repair_condor.sh
scripts/submit_b14m_profile_globalization_repair_condor.py
tests/test_b14m_profile_globalization_repair.py
alignment/leave_target_out_dump/ProfileGlobalizationRepair.inc
```
