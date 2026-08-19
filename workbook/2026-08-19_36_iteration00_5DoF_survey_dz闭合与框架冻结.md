# 2026-08-19 (36) iteration-00 5-DoF+survey-dz：单步 route-selected 闭合与框架冻结

## 生产

Condor cluster 998322：10 train + 8 validation × 17 点全部 `completion_status=accepted`，
零 `failure.json`。两个转换产物截断（`100043_00400` 的 `fd_dx_m/propagations.root`，
`100048_00000` 的 `fd_rz_p/tracklets.root`），均从完好 Athena `enhanced_tracklets.root`
本地重转修复。test 未打开。

随机起点 severity 0.12（dz≡0）：
`(dx, dy, rx, ry, rz) = (+0.147 mm, −0.475 mm, +3.70 mrad, +0.893 mrad, −1.35 mrad)`。

## 预注册 dual gate 下的 truth-selected 步

`run_multisource_refit_multidof_local_step.py`，`--prior-sigma ift_dz_mm:5.0`，
capture JSON 为条目 35 冻结的 train-only 判据。train 2346 边 / validation 1792 边。

| 参数 | expΔ | train recΔ | train err | σ | pull_fit | eng | stat | cov | prior 信息份额 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dx | −0.147 | −0.164 | −0.018 | 0.0069 | −2.56 | ✓ | ✓ | ✓ | 0 |
| dy | +0.475 | +0.415 | −0.060 | 0.026 | −2.30 | ✓ | ✓ | ✓ | 0 |
| rx | −3.70 | −3.54 | +0.158 | 0.034 | **+4.63** | ✓ | ✓ | ✗ | 0 |
| ry | −0.893 | −0.897 | −0.004 | 0.0041 | −1.00 | ✓ | ✓ | ✓ | 0 |
| rz | +1.35 | +1.53 | +0.181 | 0.153 | +1.19 | ✓ | ✓ | ✓ | 0 |
| dz | 0 | −0.48 | −0.48 | 1.34 | −0.36 | — | — | — | 0.07 |

- train **engineering=true**，**framework=false**（仅 rx 的 this-fit coverage：0.158 mrad
  远小于 1 mrad engineering，但 4.6σ_fit）。这是 10 源统计把 σ 收到 0.034 mrad 后，
  残差系统项被 pull 放大；不是线性域失败。
- 剩余 5-DoF severity **0.013**，已在线性盆地内。held-out 响应 χ² 下降 98.7%。
- 数据 6 参数条件数 2.3×10⁷：dz 列仍是弱方向。逐源 dz std **12.6 mm**（train）/
  **9.8 mm**（val），比 pooled σ 1.34 mm 大一个量级 → pooled dz 不是稳定的
  径迹测量，尽管对角信息上 prior 份额只有 7%。
- 近简并：dy–rx −0.53、dx–ry +0.36（与 identifiability map 一致）。
- 已知 validation 离群源 `100047_00150` 仍在，未剔除。

## 冻结 V2 route-selected（unknown association，dedup）

ungated 冻结 backbone；`anchor_selected_field_edge` + `physical_edge_deduplicated`；
同一 dz prior 与 capture JSON。

| | unique edges | 5-DoF remaining | framework | dz prior 份额 | dz pull |
| --- | --- | --- | --- | --- | --- |
| train | 574（2173 replica） | **0.010** | **true** | 0.53（prior 主导） | −0.09 |
| validation | 460（1702 replica） | **0.018** | **true** | 0.26 | −0.29 |

train 全自由参数 |pull_fit| ≤ 0.63；validation ≤ 0.76（rz −0.44 mrad / 0.57）。
engineering、statistical、coverage 全部通过。dy–rx 后验相关 −0.97（小样本 + 近简并，
点估计仍闭合）。

Association（MC audit，未调阈）：

| split | complete-track ε | purity | fake |
| --- | --- | --- | --- |
| train | 0.883 | 0.976 | 0.023 |
| validation | 0.744 | 0.972 | 0.031 |

candidate complete-truth-chain recall 在 ungated 图上 ~0.997；χ² gate=25 的
physical-candidate 报表仍是 ~0.2–0.5（已知覆盖瓶颈，本阶段不改 gate）。

## 线性域与多轮

从随机 5-DoF severity 0.12 出发，**单轮** route-selected Newton 步在 train 与
validation 上都进入 remaining severity 0.01–0.02 并通过对偶 capture。
因此 **不启动 iteration-01 生产**。truth-selected 的 rx coverage 失败不否定
这一操作闭环；它说明 this-fit σ 在高统计下会暴露协方差/源间系统项，
engineering 门仍通过。

Held-out linear 0.10/0.14 与 stress 0.20 点已随 bank 产出，本条目主闭合是
anchor→nominal；stress 点只作线性边界库存，不是单步成功标准。

## 框架冻结

**冻结 FASER station-level「5 个径迹约束 DoF + 1 个 survey 约束 DoF」alignment 框架。**

- 自由：`dx, dy, rx, ry, rz`；观测语义 `physical_edge_deduplicated`；
  mode-0；V2 backbone；route policy 全部保持冻结。
- `dz`：法方程中保留 5 mm survey prior。route-selected train 上 prior 主导；
  逐源散布 ~10 mm。**禁止把任何 dz 后验当成径迹测量。** 写下一轮 payload 时
  dz 保持 survey 值（本步为 0），不把 −0.3/−1.0 mm 的噪声步写进 geometry。
- 单步工作点：归一化 5-DoF severity ≤ 0.15；~0.2 仅作监测；禁止 0.5–1.0。
- capture：engineering 与 3σ_registered ∩ 3σ_fit 双报；推进以
  `framework_capture_success` 为准（条目 35 预注册，validation 未参与选容差）。
- 下一步是 **station/layer hierarchy**，不是新的 Transformer。

## 产物

- 物理 bank：`outputs/mc24_ift_5dof_survey_dz_iteration00_trainval_physical_v1/`
- truth-selected：`outputs/mc24_ift_5dof_survey_dz_iteration00_trainval_closure_v1/`
- synthetic / backbone / route-selected：`outputs/mc24_ift_5dof_survey_dz_iteration00_{synthetic,v2_backbone_{train,validation},route_selected_{train,validation}}_v1/`
- 预注册判据：`outputs/mc24_ift_5dof_survey_dz_capture_criteria_train_v1/capture_criteria.json`
