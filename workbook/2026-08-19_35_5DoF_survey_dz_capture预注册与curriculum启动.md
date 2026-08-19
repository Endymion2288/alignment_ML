# 2026-08-19 (35) 5-DoF + survey-dz curriculum：train-only capture 预注册与 iteration-00 生产启动

## 任务

pilot identifiability gate 已通过（条目 33/34）。本阶段进入 station-level
**5 个径迹约束 DoF + 1 个 survey 约束 DoF** 的 source-disjoint physical
curriculum / 迭代环，严格沿用该结论：

- 自由拟合：`dx, dy, rx, ry, rz`
- `dz` **不是**自由 alignment 参数，以冻结 5 mm survey prior 进入法方程；
  prior 主导的后验不得解释为径迹测量
- 冻结：mode-0、V2 backbone、route policy、`physical_edge_deduplicated`、
  sealed test；不调 χ² / covariance / ML
- 联合随机采样 5-DoF，不做轴向 grid；单步 Newton 的归一化 severity 主区
  `<= 0.15`，少量 `~0.2` validation stress；禁止 0.5–1.0 作为单步目标
- **任何正式 curriculum closure 产生之前**，只用 train pilot 的
  route-selected 不确定性预注册参数级判据；validation 不参与容差选择

## 预注册 dual capture（train-only，validation 未打开）

来源：`outputs/mc24_ift_6dof_sensitivity_pilot_v2_route_selected_closure_c_v1`
（3 个 train source，193 unique physical edges，线性点 c）。点 d 只作成像
核对（σ 与 c 一致到 0.5% 以内），不进入注册值。

产物：`outputs/mc24_ift_5dof_survey_dz_capture_criteria_train_v1/capture_criteria.json`

| 参数 | 角色 | engineering | registered σ | 3σ_reg |
| --- | --- | --- | --- | --- |
| dx | free | 0.1 mm | 0.0289 mm | 0.087 mm |
| dy | free | 0.1 mm | **0.313 mm** | 0.940 mm |
| rx | free | 1.0 mrad | 0.384 mrad | 1.15 mrad |
| ry | free | 1.0 mrad | 0.111 mrad | 0.334 mrad |
| rz | free | 1.0 mrad | 0.502 mrad | 1.51 mrad |
| dz | survey | （不参与 capture） | — | — |

**Dual gate 定义（冻结）：**

- `engineering_capture`：`|error| ≤` 上表绝对容差（保留历史 0.1 mm / 1 mrad）
- `statistical_capture`：`|error| ≤ 3 × σ_registered`（train pilot 冻结）
- `coverage_capture`：`|error| ≤ 3 × σ_this-fit`
- **`framework_capture_success`（推进迭代的正式判据）**：全部自由参数
  同时通过 statistical **且** coverage；dz 排除
- engineering 始终报告，单独不决定推进/拒绝

注册集自检（点 c 本身）：dy 误差 +0.102 mm 相对 0.1 mm **engineering=false**，
相对 0.313 mm 为 0.33σ **statistical+coverage=true** →
`framework_capture_success=true`。这正是旧绝对门被替换的理由。

## 联合随机采样（seed 20260819）

5-DoF severity 球面采样，dz 恒为 0。起点 severity **0.12**（线性域）：

`(dx, dy, rx, ry, rz) = (+0.147 mm, −0.475 mm, +3.70 mrad, +0.893 mrad, −1.35 mrad)`

Held-out（全 18 源共享几何，split 只按 xAOD 源划分）：

- linear 0.10、linear 0.14
- stress 0.20（线性边界监测，不是单步成功标准）

每源 17 点：nominal + 随机起点 + 12 个 6 参数 FD（含 ±dz 以便 prior 列存在）
+ 3 个 held-out = **18 源 × 17 点 = 306 次真实 refit**。

## 生产

```text
outputs/mc24_ift_5dof_survey_dz_iteration00_trainval_physical_v1/
  10 train + 8 validation original xAOD
  test_data_accessed: false
```

每个点独立执行 `/Tracker/Align → SegmentFitRefit → SegmentsRefit →
NtupleDumper → Acts(mode 0)`。完成后：

1. truth-selected `run_multisource_refit_multidof_local_step.py`
   `--prior-sigma ift_dz_mm:5.0 --capture-criteria <frozen JSON>`
2. 冻结 V2 route-selected `anchor_selected_field_edge` +
   `physical_edge_deduplicated`，同一 prior 与 capture JSON
3. 报告 prior/posterior contribution；若 dz `prior_dominated` 则不得写成
   径迹测量
4. 若 framework capture 未过，用 `--update-json` 做 iteration-01 再线性化
   （仍禁止跳到 0.5–1.0 severity）

## 尚未做（等待 Condor）

- 306 点物理生产
- truth-selected / route-selected 闭合数字
- 多轮 associate→align→payload→refit→reassociate 轨迹

这些数字产生后写入条目 36；本条目只冻结判据并启动生产。
