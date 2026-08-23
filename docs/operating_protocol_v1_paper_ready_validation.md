# Operating Protocol V1 paper-ready validation

Workbook 55 / 2026-08-23. Read-only publication analysis of the frozen
entry-54 evidence package. No new alignment mode is opened.

The paper files live in
[`docs/operating_protocol_v1_paper_ready/`](operating_protocol_v1_paper_ready/):

- [`paper_results_summary.md`](operating_protocol_v1_paper_ready/paper_results_summary.md)
- [`paper_results_summary_cn.md`](operating_protocol_v1_paper_ready/paper_results_summary_cn.md)
- [`figure_caption_drafts.md`](operating_protocol_v1_paper_ready/figure_caption_drafts.md)
- [`main_claims_and_limitations.json`](operating_protocol_v1_paper_ready/main_claims_and_limitations.json)
- [`reviewer_risk_checklist.md`](operating_protocol_v1_paper_ready/reviewer_risk_checklist.md)
- [`figures/`](operating_protocol_v1_paper_ready/figures/) (`fig00`–`fig06`)

Decision flow:

`MC transfer PASS → real-data association PASS → Station calibration REJECT → reduced calibration REJECT → residual/DQ monitoring PASS`

Not writing geometry is the validated result, not an unfinished state.
Residual or χ² decrease is a DQ observable only. Implied `C_dx` is not a
measurement.
