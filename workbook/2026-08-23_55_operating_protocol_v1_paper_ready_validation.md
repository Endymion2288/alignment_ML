# 2026-08-23 (55) Operating Protocol V1 paper-ready validation

## 任务

条目 54 已完成真实数据方法学闭环。本条目不再增加 alignment mode、不重训
V2、不改报警阈值，也不再寻找能“救回” geometry write 的数据子集。只把
冻结的 `outputs/operating_protocol_v1_final_real_data_closure_v1/`
JSON/CSV 转成论文和组会可用的科学证据。

## 三层结论

1. **关联可迁移。** 冻结 V2 从 MC 转到 2024 r0022。100-event selected
   图为空是 statistics-limited；full-segment 上 14973–14976 恢复，7 个
   独立 expansion run 同样恢复。
2. **关联可用 ≠ alignment 可解。** Station Jacobian 有几乎纯 `dz` 的
   规范弱方向；14974 的 5-DoF 弱方向与冻结 `A` 余弦 0.994。reduced
   `{dy,rx,rz}` Fisher 可识别但不能跨 calibration run 搬运。拒绝写
   geometry 是 identifiability 与跨层歧义的实证结论，不是阈值过严。
3. **monitoring 稳定。** 当前官方 geometry + 冻结 V2 在独立 r0022 run
   上全部 nominal。没有持续 `dy/rx` 漂移。14977 是统计不足，不是
   alignment 异常。

Residual / χ² 下降只标 **DQ observable**。Implied `C_dx` 不是测量值。
“不写 geometry”本身就是经过真实数据验证的科学结论，不是未完成状态。

## 产出

配置：`configs/operating_protocol_v1_paper_ready_validation_v1.yaml`

代码：`alignment/operating_protocol_v1_paper_ready.py`，
`scripts/report_operating_protocol_v1_paper_ready.py`，
`tests/test_operating_protocol_v1_paper_ready.py`

文稿与图：

- `docs/operating_protocol_v1_paper_ready/paper_results_summary.md`
- `docs/operating_protocol_v1_paper_ready/paper_results_summary_cn.md`
- `docs/operating_protocol_v1_paper_ready/figure_caption_drafts.md`
- `docs/operating_protocol_v1_paper_ready/main_claims_and_limitations.json`
- `docs/operating_protocol_v1_paper_ready/reviewer_risk_checklist.md`
- `docs/operating_protocol_v1_paper_ready/figures/fig00`–`fig06`（PDF/PNG）

分析副本：`outputs/operating_protocol_v1_paper_ready_validation_v1/`
