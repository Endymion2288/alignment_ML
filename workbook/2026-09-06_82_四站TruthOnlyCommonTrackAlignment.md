# Workbook 82 — Truth-only Common-Track Alignment Qualification

日期： 2026-09-06
分支： `4station`
任务： 冻结 association，降级旧 pairwise WLS，建立 common-track likelihood 合同、求解器、解析/SE(3)/场测试，以及一条 truth-only smoke。

---

## 0. 结论（先写）

```text
association_default_system =
    frozen_W64_raw_energy_plus_exact_solver

physics_constrained_calibration_supported = false
continue_v5a_frozen_head       = false
continue_route_energy_rewrite  = false
continue_hybrid_expansion      = false
continue_residual_calibration  = false
continue_to_15d_relative_wls   = false

final_blind_eval_authorized    = false
sealed_test_accessed           = false
development_00350_used         = false

alignment_oracle_qualified     = false
```

WB81b 是 association correction / replacement / hybrid 的正式终点。本 workbook **不再**设计 route-energy MLP、correction head、Arm C、domain adaptation，也不扫 bound / slack / association loss。

旧 `alignment/physical_jacobian.py` 与 route-selected relative closure 只保留为 **response / regression diagnostic**。它们不是 certified alignment solver、不是 full alignment closure、不是 downstream oracle。

WB82 第一阶段交付的是合同 + 求解器 + hermetic tests + **一条** truth-only smoke。smoke 即使数值上收敛，也 **不能** 把 `alignment_oracle_qualified` 写成 true。批量独立 replica closure 尚未做。

本阶段只准备回答、但还不能最终回答：

> 给定 perfect truth association，新的 common-track alignment solver 是否已经足够可信，可以作为后续 association 的 downstream oracle？

现在的状态是：**合同和玩具资格测试已立；S3-only translation smoke 走通；oracle 尚未 qualified。**

```text
smoke_pass                     = true
alignment_oracle_qualified     = false
phase1_smoke_chart             = S3 (dx_mm, dy_mm) only
```

`smoke_pass` 只覆盖与注入匹配的 S3 平移 chart。S1–S3 的 `a+bz` 斜率，以及 free `ry` 与 `q/p` 的剪切，都是固定外场下的物理 `JG`，不是代数 `g_i^{-1} g_j`。它们 **没有** 在本阶段被 qualified。不得把这条 smoke 写成 full-pose oracle。

---

## 1. 冻结的 association 接口

```text
association = frozen_W64_raw_energy_plus_exact_solver
WB82 phase 1 实际使用 = truth_only
```

不修改 W64 checkpoint、`raw_energy_v1`、exact solver、route accounting、WB74–WB81 artifact。truth-only alignment 资格不依赖 ML association 成功。

---

## 2. 旧 alignment backend 的降级

| 模块 | 现在的合法称呼 | 禁止再称 |
|---|---|---|
| `alignment/physical_jacobian.py` | response / regression diagnostic | certified solver / full closure / oracle |
| route-selected relative closure | same-event conditional response | `selection → solve → update → refit` |
| 旧 15D pairwise WLS | 关闭 | 重新打开的 alignment backend |

`continue_to_15d_relative_wls = false` 保持 false。WB82 是新 qualification 路线，不是把旧 WLS 改名交付。

---

## 3. Common-track 测量模型

详见 `alignment/measurement_likelihood_contract.md`。

```text
per-track ξ = (x, y, tx, ty, q/p)
        +
global left-SE(3) station poses
        ↓
joint residual / unique-hit covariance
        ↓
Schur elimination
        ↓
damped global update
```

禁止把多条 pairwise residual 当独立 likelihood 相加。`measurement_id` 相同的 tracklet 只能进一次。  
`fixed_dz` 与 `finite_survey_prior` 是两个 mode。  
有限旋转误差用 Lie-log，不用 Euler subtraction。  
固定外场下的共同运动是物理 `JG`，不是 observable gauge；真正的坐标变换必须让 field / material / surface 一起变。

Phase-1 smoke Newton chart（预注册，与注入对齐）：

```text
free stations    = S3
free components  = dx_mm, dy_mm
held             = S0, S1, S2, all rotations, dz
q/p              = local nuisance + σ=1e-3 beam/truth prior
```

---

## 4. 第一阶段测试

必须覆盖：

1. linear-Gaussian 与解析 / 联合解一致，pull 有限  
2. Schur 与 dense joint 数值一致  
3. 共享 measurement 不双计  
4. left/right SE(3)、非零 nominal、reference rewrite、Lie-log  
5. 固定场 `JG` ≠ 代数 `g_i^{-1} g_j`；场与表面可共变  
6. `h/2, h, 2h` FD 稳定  
7. `fixed_dz` 与 finite prior 语义分开  
8. 奇异 / NaN / 非 SPD fail-closed  

迭代：最多 10 步，固定阻尼，到第 10 步不自动 PASS。

---

## 5. Smoke（单条件，truth-only）

数字只来自 `outputs/mc24_four_station_wb82_truth_only_smoke_v1/smoke.json`。  
注入 `S3 dx = 0.5 mm`，`B_y = 0.35` toy 场，seed `20260906`，24 fit + 8 held-out truth tracks，`fixed_dz`，S3 `(dx,dy)` chart。

| 量 | 值 |
|---|---|
| `smoke_pass` | `true` |
| `converged` | `true`（2 步，不是第 10 步自动 PASS） |
| `s3_dx_lie_error_mm` | `-0.003710481843057334` |
| `identifiable_x_kink_error_mm` | `-0.003710481843057334` |
| `identifiable_y_kink_error_mm` | `0.002913963786257401` |
| `validation_chi2_before` | `5139.745608623545` |
| `validation_chi2_after` | `102.09368223214577` |
| `n_retained` / `n_dropped` | 2 / 0 |
| `w64_used` / `ml_association_used` | `false` / `false` |
| `alignment_oracle_qualified` | `false` |
| git SHA | `afc8eeba122b376dc2978b0ae4ff8723e49f455b`（工作树脏） |

`|s3_dx_lie_error| < 0.05` 与 `after_val < before_val` 均成立。这是玩具场上的 `inject → truth hits → solve → update → refit ξ → held-out residual`，**不是** Calypso 实物 refit，**不是** 独立 replica coverage，**不是** S1–S3 全平移 / 全姿态资格。

```text
alignment_oracle_qualified = false
```

---

## 6. 明确停止 / 仍禁止

- 回到 association 调参、hybrid、Arm C、residual calibration  
- 放宽 WB81b 的 fake slack / `δ_max` / `+0.01`  
- 打开 00350 / 00800 / 100116 / 100117  
- 把 pairwise same-event counterfactual 写成最终科学验证  
- 在 truth-only FAIL 时用 ML selection 或 robust loss 遮掩  
- 宣称 production / alignment readiness  

---

## 7. 下一允许步骤

仅当 hermetic tests 与 smoke 的统计合同都成立后，才授权 **批量独立 truth-only replica closure**（HTCondor，独立事件，不是 overlay 复本）。在那之前 `alignment_oracle_qualified = false`，禁止用 alignment 结果评价 association。
