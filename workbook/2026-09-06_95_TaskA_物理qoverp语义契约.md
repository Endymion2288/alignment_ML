# Workbook 95: Task A 物理 q/p 不确定度语义

日期：2026-09-06
状态：**完成** —— 只盘点冻结 WB87 划分上已有 export；未新建 source campaign，未提交 18 源作业，未写 payload。

**最终判定：`FAIL`**

- `decision = physical_qoverp_semantics_not_established`
- `mechanism = reconstruction_chain_qoverp_uncertainty_not_exported`
- `contract_established_for_process_noise = false`
- `measurement_model_v2_entered = false`
- `unconstrained_tracker_only_stopped = true`

物理先验需要同时具备：重建链上的 q/p 估计、不确定度、以及与原生状态其余分量的相关。当前 export 不满足。

## 起始状态

HEAD（任务开始时）：`f31cfa10a2dfda5808a141cfacbf50e2b28bdd16`（WB94 / T12）

相对 WB94：仅新增本任务文件，未重跑 T00–T12，未改冻结负结果。

输入：

- 同一 WB87 construction / validation 源划分
- 冻结 WB87 / WB93 / WB94 决策 SHA
- Calypso pin `40892527e9c65409afd2378a2abfc25ddbddac03`

## 盘点结果

| 来源 | 估计 | 不确定度 | 相关 | 结论 |
| --- | --- | --- | --- | --- |
| SegmentFit dummy `GetState` | 恒为 `1e-5 /MeV` | 恒为 `5e-6 /MeV²` | 0 | 拒绝，非物理 |
| ntuple mode 1/2 truth q/p | 有 | 无 | 无 | 拒绝，禁止作 real-data 解 |
| `CKFTrackCollection` → `Track_p0` | 有（construction 424 + validation 353） | **未导出** | **未导出** | 不完整 |

Tracklets：construction 1871 / validation 1471，全部精确等于 dummy。  
CKF 均值动量不是 100 GeV 常数，但不能当作先验：NtupleDumper 只写了 mean p，丢掉了 5×5。

## 禁止项（均保持）

不用 truth q/p 作 real-data 解；不删 q/p 列；不 rescale 协方差；不进入 alignment / Measurement Model V2；不新建 source campaign。

## 下一步

Task A 未通过，**不进入 Task B**，也不进入 Measurement Model V2。

无约束 tracker-only 主线保持停止。备选仍是：

```text
真实数据 residual / DQ monitoring
+ 可复现负结果
```

若以后要补 CKF 5×5，必须另开预注册 export 计划，不能从本任务自动续跑，也不能把 truth q/p 当部署种子。

## 工程产物

- 配置：`configs/qoverp_semantics_contract_v1.yaml`（SHA `ccd6f8d8…`）
- 模块：`datasets/qoverp_semantics.py`
- 驱动：`scripts/audit_qoverp_semantics_contract.py`
- 测试：`tests/test_qoverp_semantics_contract.py`（8 个测试）
- 文档：`docs/qoverp_semantics_contract.md`、`docs/qoverp_semantics_contract_cn.md`

Run ID：`sba_qoverp_semantics_20260906T135614Z_9e948c39`

Config SHA：`ccd6f8d8c7bc15daf5f5eef41340a4e5d25ad4a62397872c18ccf1afd5f5d29a`

产物（EOS，不入 git）：`outputs/qoverp_semantics_contract_v1/sba_qoverp_semantics_20260906T135614Z_9e948c39/`

| 产物 | SHA256 |
| --- | --- |
| `qoverp_semantics_contract.json` | `d4508fc6728d92fdacfde72758f081c54b24df121028d0d75fc0d4d2571b70e2` |
| `source_inventory.json` | `f55e1d275fec66f80bc277d909edb91d2734e2b89d062a40531d538f0f9087c3` |
| `dummy_segmentfit_audit.json` | `b93777e098cc0f4e244945e0927161964e604eb0410b0be3883d7203d6ce8946` |
| `calypso_provenance.json` | `14f14c7e95d1a4284fe3b071bca6c35d7025d8a532df14a32d21166d258eb482` |
| `COMPLETE.json` | `4639cb3eb5c0b68734bc9669fa9ecd3f81a33a3b848ec4b96aad8798b1a16aa4` |
