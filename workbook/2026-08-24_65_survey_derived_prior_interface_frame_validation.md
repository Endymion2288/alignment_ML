# 2026-08-24 Survey-Derived Prior Interface & Frame Validation V1

## 任务

条目 61–64 已证明 track-only 无法稳定解除 `ry↔C_dx`，幻灯片也已经给出
C_dx 中心值和若干 tilt candidate，但 frame 未唯一确定。本阶段不再扩展
ML、module map 或 collision track statistics，只做 external-constraint
interface：把 2021/2022 summary 收成 candidate database，完成 frame
validation（禁止未确认 tilt 映射到 `ry`），并扫描
`σ(ry)` / `σ(C_dx)` 对 identifiability 的影响。不写 geometry，不写
alignment payload。

## 答案

冻结
`survey_prior_interface_ready_candidates_lack_validated_ry_mapping_and_measurement_covariance`。

Survey candidates **不具备**解除 degeneracy 的信息需求。

- 数据库记录了每个量的 value / source / frame / parameter mapping /
  provenance。没有任何 candidate 被允许映射到 Calypso station `ry`：
  2021 I/F normal 的 CAD `ry` 默认映像是 `rz`；page 6 in-plane yaw 没有
  数字；2022 `layer_mean_coherent_tilt_candidate` 与 `C_dx_slide` 是同一
  DoF。
- 唯一可以挂到 prior 槽的是 `2022_ift_C_dx_slide_mm = +0.2541169 mm`，
  frame 是 survey-adjusted FASER。官方 `ift_C_dx` 槽为
  `feasibility_only`，`sigma=null`，因此现在 **不会进入 Fisher**。
- 预声明网格上 `σ(ry)=20 mrad` 或 `σ(C_dx)=0.345 mm`（冻结阈值 0.315 mm
  同样解锁）足以解除 degeneracy。缺的是真实 metrology covariance，不是
  更多 track。
- 向硬件/alignment 团队索取：带 measurement covariance 的 IFT L0/L2
  `dx` 或 `C_dx`，以及 Calypso global `ry`（或唯一的 CAD→Calypso
  transform + 量化的 I/F in-plane yaw）。不要把 2022 station width、
  LAYERPITCH、conditions constants 或未确认 tilt 当测量交来。

报告：`outputs/survey_derived_prior_interface_frame_validation_v1/`。
