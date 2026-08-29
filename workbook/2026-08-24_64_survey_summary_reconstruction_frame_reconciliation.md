# 2026-08-24 Survey-Summary Reconstruction & Frame-Reconciliation V1

## 任务

条目 63 确认 `/Tracker/Align` 里的 IFT constants 没有独立 survey
provenance。本阶段不等原始 Excel，也不把 PPT 数字写成 prior 或
geometry。目标是把
`docs/FAS_IFT_Dec10-2021.pdf` 与 `docs/04Nov2022_Survey.pdf`
里所有可恢复的 summary 做成 machine-readable evidence，并做坐标
reconciliation；只在 `slide_summary_feasibility_only` 下扫描
`σ(ry)` / `σ(C_dx)`。

## 答案

冻结
`slide_summaries_define_ift_l0l2_contrast_and_tilt_candidates_frames_not_uniquely_reconciled_remaining_uncertainty_is_measurement_covariance`。

只回答这一题：两份幻灯片能否给出彼此坐标一致、物理含义明确的 IFT
global tilt 与 internal L0/L2 contrast，并把剩余不确定性收敛到缺
measurement covariance？

- **Internal L0/L2 contrast：能。** 2022 layer 0/1/2 的平均精确复现
  Station-0 `(0.727,-0.982,-27.772) mm`，因此 Layer 0–2 就是 IFT。
  `C_dx_slide=(x_L0-x_L2)/2=+0.2541169 mm`，
  `Δx_L0-L2=+0.5082339 mm`。这已经在 survey-adjusted FASER frame。
- **Global tilt：每份幻灯片各自有 candidate，但不是同一个量，也还不能比。**
  2021 I/F 四点 plane 的 measured normal
  `(-0.00306,-0.003954,-0.999987)` 相对 `(0,0,-1)` 给出约 5.0 mrad
  总 tilt，CAD 下 provisional `rx≈-3.95 mrad`、`ry≈+3.06 mrad`，标记
  `frame_not_yet_reconciled`。默认轴置换后这是 Calypso `rx/rz`，不是
  station `ry`；page 6 的 in-plane yaw 才可能是 `ry`，但幻灯片没有数字。
  2022 三层 x 对 `LAYERPITCH=31.5 mm` 的 slope 约 `-8.07 mrad`，只能叫
  `layer_mean_coherent_tilt_candidate`，并且和 `C_dx_slide` 是同一
  L0/L2 contrast，不是第二条独立约束。
- **彼此坐标一致：还没有。** 2022 已按 Stations 1–3 front-sensor 平均
  平移到近似 Calypso；2021 CAD 的轴符号和原点不能从幻灯片唯一确定。
  因此禁止把 2021 tilt、2022 layer trend 和 `/Tracker/Align`
  existing-conditions 当成同一个数来比。2022 与 conditions 的数字对照
  只标 `slide_summary` vs `existing_conditions_state`。
- **剩余不确定性是缺 measurement covariance，不是未知的 track
  degeneracy。** Jacobian 的 `ry↔C_dx` 已经量化。2022 station/module
  width 是 `population_spread_not_measurement_sigma`，禁止当 Gaussian
  prior σ。若这些 slide 中心值后来被原始 survey 证实，预声明网格上
  预声明网格上 `σ(ry)=20 mrad` 或 `σ(C_dx)=0.345 mm`（以及冻结的
  0.315 mm）足以解除 degeneracy；有用精度仍是 `0.5 mrad / 0.080 mm`。
  没有 geometry candidate。

报告：`outputs/survey_summary_reconstruction_frame_reconciliation_v1/`。
