# Workbook 83b — WB83 consistency audit and physical replica contract

日期： 2026-09-07
分支： `4station`
任务： 独立核验 Workbook 83 声称的工作是否真实完成；然后只做 `calypso_physical_independent_v1` 生产合同的只读设计。

---

## 冻结状态（未改）

```text
association_default_system =
    frozen_W64_raw_energy_plus_exact_solver

WB83_v1 = official_FAIL
WB83_v2 = final_solver_fix_FAIL

solver_implementation_fixed = true

common_track_solver_qualified_under_toy_model = false
alignment_oracle_qualified_for_physical_FASER = false

further_identical_rerun_authorized = false
ml_alignment_eval_authorized = false
final_blind_eval_authorized = false
```

本日志 **没有** 重跑 WB83 qualification，**没有** 建立新的 association / alignment 模型实验。

---

## 1. Consistency audit

对照仓库和 artifacts，不对照 Workbook 83 正文。10 项要求全部通过，无需为了改科学结论去修 artifact。

1. `outputs/mc24_four_station_wb83_final_closure_v1/wb83_final_closure.json` 存在；`immutable=true`；SHA256 `5b580f764829fff80f9de70bd74c6b691e369db724501de734904fbc6fe05599`。EOS 上 `chmod 0444` 只是提示：属主仍可写。不可变证据是内容哈希。
2. v1 root 与 v2 root 都在。v1 gate 内容 SHA 仍为 closure 记录的 `a5ecf9ea…`。audit 探测只写位时曾把 mtime 碰成 `14:37`，内容未改，mtime 已恢复 `2026-09-07T00:57:22.824541`。
3. matrix SHA256 在 v1 / v2 / closure 中均为 `ca895e44082d4021fe6a3f9d008cf594e03c6330044466a883ca6355e8ac2720`。
4. v2：14 cells × 100 replicas，`coverage_report.n_rows = 1400`。v1 同样 14 × 100。
5. `alignment/common_track_solver.py` 含 `symmetrize_normal` / `inverse_spd_after_symmetrize`（local normal 无 `1e-12 I` ridge）以及 `absolute_survey_prior_terms`（`a_current + theta - a_survey`）。源 SHA 与 closure 一致。
6. `tests/test_wb83_v2_solver_fix.py` 验证 true-indefinite rejection、analytic prior MAP、zero prior residual、fixed-dz parity。本 audit 重跑这组测试，5 passed。
7. 剩余正式 FAIL 只有：
   ```text
   C3_weak_0.5x_fixed_dz            pull_mean = -0.26289344212158866
   C3_weak_1x_finite_survey_prior   coverage = 89/100
   ```
8. 禁止资产未访问。v2 1400 replica 的 overlay / W64 / ML association 标志全为 0。
9. 没有 v3，没有第三次 identical-seed DAG。正式 cluster 仍是 `1109787` 与 `1109806`。
10. toy qualification 与 physical FASER oracle 均保持 false / FAIL。

---

## 2. Physical replica production contract

只输出：

```text
outputs/mc24_four_station_calypso_physical_replica_design_v1/calypso_physical_replica_production_protocol.md
outputs/mc24_four_station_calypso_physical_replica_design_v1/calypso_physical_replica_production_plan.json
outputs/mc24_four_station_calypso_physical_replica_design_v1/physical_replica_capacity_estimate.json
outputs/mc24_four_station_calypso_physical_replica_design_v1/provenance_contract.json
```

未开始生产，未提交长 Calypso job，未生成新 replica，未开始 qualification，未做 ML-vs-truth，未返回 association。

### A–F 摘要

- **Independence：** `source_id:run_id:event_id` 全局唯一；一次重建、一个 payload、一次噪声。禁止 overlay、禁止同事件重复重建当 replica、禁止同一 truth 换 geometry 当独立样本。
- **Chain：** Athena 24.0.41 + 工作区 Calypso `40892527…`（dirty，授权时必须重录）+ `/Tracker/Align` + FASER field/material maps + `FaserActsExtrapolationTool` mode 0 + 导出 4×4 物理 covariance + MC truth match。
- **Injection：** payload → Calypso refit → ACTS → common-track export。禁止再用 toy `uniform_By` 做 physical qualification。
- **Capacity：** 8 conditions × 200 replicas，yield 0.50 → 3200 独立事件。n 来自 0.25σ / Holm m=3 功效，不是 WB83 failed cells。
- **Source policy：** 六源 train-range 可用于 truth-only alignment，但必须披露 W64 已见过；不能称为 source-unseen ML evaluation。
- **Forbidden：** 00350 / 00800–00849 / 100116 / 100117 / Final Blind / sealed 未打开。只确认了六源授权 xAOD 文件存在。

---

## 3. 最终回答

> 在当前允许的数据和 CERN production infrastructure 下，能否构造一套足够独立、provenance 完整的 `calypso_physical_independent_v1` 数据，用于未来 physical alignment oracle qualification？

**能构造，但今天没有现成语料。**

```text
physical_replica_production_feasible             = true
physical_alignment_qualification_blocked_by_data = true
alignment_oracle_qualified                       = false
```

实际 production 必须再单独授权。
