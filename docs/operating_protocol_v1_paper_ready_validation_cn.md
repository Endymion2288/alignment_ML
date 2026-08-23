# Operating Protocol V1 论文级验证包

Workbook 55 / 2026-08-23。对条目 54 冻结证据包的只读论文分析。不再打开
新的 alignment mode。

论文文件在
[`docs/operating_protocol_v1_paper_ready/`](operating_protocol_v1_paper_ready/)：

- [`paper_results_summary.md`](operating_protocol_v1_paper_ready/paper_results_summary.md)
- [`paper_results_summary_cn.md`](operating_protocol_v1_paper_ready/paper_results_summary_cn.md)
- [`figure_caption_drafts.md`](operating_protocol_v1_paper_ready/figure_caption_drafts.md)
- [`main_claims_and_limitations.json`](operating_protocol_v1_paper_ready/main_claims_and_limitations.json)
- [`reviewer_risk_checklist.md`](operating_protocol_v1_paper_ready/reviewer_risk_checklist.md)
- [`figures/`](operating_protocol_v1_paper_ready/figures/)（`fig00`–`fig06`）

决策流：

`MC transfer PASS → real-data association PASS → Station calibration REJECT → reduced calibration REJECT → residual/DQ monitoring PASS`

“不写 geometry”是经过真实数据验证的科学结论，不是未完成状态。
Residual / χ² 下降只是 DQ observable。Implied `C_dx` 不是测量值。
