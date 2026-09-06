# Workbook 96: Task A2 CKF 物理 q/p 协方差导出契约

日期：2026-09-06
状态：**完成** —— 只从 WB87/95 已用 xAOD 读取持久化 `CKFTrackCollection` 5×5；未重跑 SegmentFit，未写 payload，未进入 alignment / Task B / Measurement Model V2，未新建 source campaign。

**最终判定：`PASS`**

- `decision = ckf_qoverp_covariance_export_established`
- `mechanism = null`
- `contract_established_for_process_noise = true`
- `measurement_model_v2_entered = false`
- `geometry_write_allowed = false`
- `unconstrained_tracker_only_stopped = true`
- 冻结 WB87：`faseracts_transport_covariance_not_validated`（未改写）
- 冻结 WB95：`physical_qoverp_semantics_not_established` / `reconstruction_chain_qoverp_uncertainty_not_exported`（未改写）

WB95 的缺口是 ntuple 只写了 CKF 均值动量。本任务证明同一重建链上的 `CKFTrackCollection` **已经持久化完整 5×5**，包括带符号 q/p、σ(q/p) 以及与其余状态分量的交叉项。这不是 dummy SegmentFit，也不是 truth q/p。

## 起始状态

HEAD（任务开始时）：`77c99b1c5b1df0d7284b835cc780f60ed0237b01`（WB95 / Task A）

相对 WB95：仅新增本任务文件，未重跑 WB87–WB95，未改冻结负结果，未打开 sealed test，未写 geometry。

输入：

- 同一 WB87 construction / validation 源划分（9 个已有 xAOD）
- 冻结 WB87 decision SHA `3f88a67b…`
- 冻结 WB95 contract SHA `d4508fc6…`
- Calypso pin `40892527e9c65409afd2378a2abfc25ddbddac03`，Athena 24.0.41，ACTS 32.0.2

## 验收标准

| 项 | 要求 | 结果 |
| --- | --- | --- |
| 状态定义 | 原生 `(loc1, loc2, phi, theta, q/p)`，curvilinear / bound；q/p 带符号，单位 `1/MeV`，不与 GeV 混用 | 通过。导出态 `(x, y, tx, ty, q/p)`，q/p 列恒等 |
| 协方差正定 | `C = C^T`，特征值 > 0；失败必须记录 | construction 495 / validation 400 全部 SPD；`n_not_symmetric = 0`；最小特征值 > 0 |
| q/p 合理性 | 不能恒定、不能等于 dummy、不能来自 truth | dummy q/p = 0，dummy var = 0，truth = 0；q/p 与 σ(q/p) 均有分布 |
| 交叉项 | 不能只导出对角 σ | 两划分全部 5×5 都有非零 q/p 交叉项 |
| provenance | Calypso / ACTS / geometry / field / conditions | 已写入 contract |

## 导出结果

Athena 从 `Trk::TrackCollection_tlp6` 读出 `CurvilinearParametersT<5,Trk::Charged,Trk::PlaneSurface>`。uproot 无法反序列化该 EDM，因此 dump 走 Calypso + HTCondor，每源 100 事件，不新建 MC campaign。

| 划分 | 源数 | 轨迹 | 完整 5×5 | SPD 失败 | dummy / truth |
| --- | ---: | ---: | ---: | ---: | ---: |
| construction | 5 | 495 | 495 | 0 | 0 / 0 |
| validation | 4 | 400 | 400 | 0 | 0 / 0 |

q/p（`1/MeV`）：

- construction：min `-7.41e-4`，median `-7.53e-7`，max `2.48e-4`
- validation：min `-1.88e-4`，median `-1.21e-7`，max `3.39e-4`

σ(q/p)（`1/MeV`）：

- construction：min `1.14e-7`，median `8.95e-7`，max `3.06e-5`
- validation：min `1.14e-7`，median `5.68e-7`，max `2.11e-5`

对照 SegmentFit dummy：`q/p = 1e-5 /MeV`，`var = 5e-6 /MeV²`，相关 0。本导出全部拒绝该常数。

平均相关（construction，最后一行/列为 q/p）：

`Corr(loc1,q/p) ≈ -0.028`，`Corr(loc2,q/p) ≈ -0.008`，`Corr(phi,q/p) ≈ 0.012`，`Corr(theta,q/p) ≈ 0.0003`。

## 长任务

HTCondor schedd：`bigbird24.cern.ch`

- cluster `1109754`：5 个 mumi 源成功
- cluster `1109755`：4 个 mupl 源重提成功（初提误用了 `mumi` 文件名，未改科学输入，只修正路径）

## 禁止项（均保持）

不用 truth q/p 作 real-data 解；不用 dummy SegmentFit 协方差；不注入合成 q/p；不 rescale 协方差；不进入 alignment；不进入 Measurement Model V2；不写 `/Tracker/Align`；不打开 sealed test。

WB87 的 transport covariance 仍未验证。A2 只补上中间接口：

```text
reconstruction chain
        ↓
CKF track state 5×5
        ↓
(q/p, covariance, cross terms)
```

## 下一步

A2 已通过，**允许以后打开 Task B**：ACTS MaterialInteractor / process-noise contract。

本 workbook **不进入 Task B**，也不进入 Measurement Model V2。无约束 tracker-only 主线继续停止。

若 Task B 通过，下一阶段才是：

```text
physical q/p covariance
        +
ACTS process noise
        ↓
transport covariance validation V3
        ↓
Measurement Model V2
```

## 工程产物

- 配置：`configs/qoverp_covariance_export_v1.yaml`
- 模块：`datasets/qoverp_covariance_export.py`
- Athena dump：`scripts/dump_ckf_qoverp_covariance.py`
- Condor：`scripts/submit_ckf_qoverp_covariance_export_condor.py`、`scripts/run_ckf_qoverp_covariance_export_condor.sh`
- 审计：`scripts/audit_qoverp_covariance_export.py`
- 测试：`tests/test_qoverp_covariance_export.py`（6 个测试）
- 文档：`docs/qoverp_covariance_export.md`、`docs/qoverp_covariance_export_cn.md`

起始 HEAD：`77c99b1c5b1df0d7284b835cc780f60ed0237b01`

Config SHA：`697b71309aaef97565861297c42f7e8ffe7e39382e28533954fd319ae5ba4f4a`

Run ID：`sba2_qoverp_covariance_20260906T144317Z_56c7e891`

产物（EOS，不入 git）：`outputs/qoverp_covariance_export_v1/sba2_qoverp_covariance_20260906T144317Z_56c7e891/`

| 产物 | SHA256 |
| --- | --- |
| `qoverp_covariance_contract.json` | `c722ed7b3462d8d56817beec4e23a3c7b8a4b3a179c6192d957cf6d69a8048e0` |
| `source_inventory.json` | `c87c3e97875488181f7901c333e29454c3bd41fba6b01f4909b8e6c3fccb365e` |
| `state_definition.json` | `cc850d057ddfb384600f905f38cb6b4275a27b9f7697db7b56171677964a42a6` |
| `inherited_stage.json` | `01abc0bfaeb98f862de16ed6291c92f6d577e7c07f5b529f40cf791d6fb7ed31` |
| `COMPLETE.json` | `bca7a378539c5e801e3893a421a4209f3f8aa9ac7c8934bd9d382ec45f8784d0` |

geometry_hash：`4e965f3631dd4ff6e04b141ac36efc49603a1dfc0625de5ea7558dd929486b15`  
field_hash：`60432de5864cfba361f91f47a694dc111bbb06dcec44434e9682548680460e9c`  
conditions_hash：`d30733216d6eb392dbdbcf5a252fed01ff56a045ade59731ca8419febce7a63d`
